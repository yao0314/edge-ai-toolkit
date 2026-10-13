"""Experiment 23c (K2c): where does the 1.24 step spend its time, on an identical graph?

Experiment 23a showed the static MobileViT file executes the same optimized graph, node for node,
under ONNX Runtime 1.23.2 and 1.24.4, yet runs 31% slower at four threads on 1.24.4 and not at all
slower at one. So the step is in kernels or threading, not in the graph. This profiles that file
under both releases at four threads and attributes the difference to op types.

Why four threads is allowed here when node_profile.py refuses it: that rule exists because under a
PARALLEL executor node durations overlap. This runs ORT_SEQUENTIAL with inter_op = 1, so nodes
execute one after another and each node's duration is wall time; intra-op threads only
parallelise inside a node. The script checks rather than assumes: it reports the per-inference
node-time sum against measured wall time, and a ratio far from 1 means the attribution is void.

Decision rule, registered in FINDINGS.md before running: >= 70% of the delta in one op type =>
"a 4-thread-only regression in <op> on an identical optimized graph"; diffuse => "a
thread-pool-level regression".

    python scripts/paper/node_profile_version.py --scratch <dir with ortNNN venvs>

Writes results/paper/node_profile_version.csv (version x session x op_type).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.cpu_topology import pick_cpus  # noqa: E402

# Runs inside each side venv (onnxruntime + numpy + psutil; no onnx package there).
CHILD = textwrap.dedent('''
    import json, os, sys, time
    from collections import defaultdict
    import numpy as np, psutil, onnxruntime as ort
    mp, t, aff, shape, warm, runs, prefix = (sys.argv[1], int(sys.argv[2]), sys.argv[3],
        [int(x) for x in sys.argv[4].split(",")], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
    if aff:
        psutil.Process().cpu_affinity([int(c) for c in aff.split(",")])
    so = ort.SessionOptions()
    so.intra_op_num_threads = t
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    so.enable_profiling = True
    so.profile_file_prefix = prefix
    s = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
    feed = {s.get_inputs()[0].name:
            np.random.default_rng(0).standard_normal((1, *shape)).astype(np.float32)}
    for _ in range(warm):
        s.run(None, feed)
    wall = []
    for _ in range(runs):
        a = time.perf_counter(); s.run(None, feed); wall.append((time.perf_counter() - a) * 1e3)
    prof = s.end_profiling()
    ev = json.load(open(prof, encoding="utf-8"))
    os.remove(prof)
    nodes = [e for e in ev if e.get("cat") == "Node" and str(e.get("name", "")).endswith("_kernel_time")]
    nodes.sort(key=lambda e: e.get("ts", 0))
    per_iter = len(nodes) // (warm + runs)
    nodes = nodes[per_iter * warm:]          # warmup is where one-off allocation is paid
    by_op = defaultdict(float)
    for e in nodes:
        by_op[(e.get("args") or {}).get("op_name", "unknown")] += float(e.get("dur", 0))
    print(json.dumps({"ort": ort.__version__, "per_iter": per_iter,
                      "wall_ms": float(np.median(wall)),
                      "ops_ms": {k: v / 1e3 / runs for k, v in by_op.items()}}))
''')


def interpreters(scratch: Path, want: list[str]) -> dict:
    """Resolve by REPORTED version: directory names do not reliably match what is installed."""
    pys = {}
    cands = [d / "Scripts" / "python.exe" for d in sorted(scratch.glob("ort*"))]
    cands.append(ROOT / ".venv" / "Scripts" / "python.exe")
    for py in cands:
        if not py.exists():
            continue
        r = subprocess.run([str(py), "-c", "import onnxruntime as o; print(o.__version__)"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip() in want:
            pys[r.stdout.strip()] = py
    return pys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", required=True, help="directory holding the ortNNN side venvs")
    ap.add_argument("--versions", default="1.23.2,1.24.4", help="before,after")
    ap.add_argument("--model", default="results/classification_mobilevit/models/p00_int8.onnx")
    ap.add_argument("--shape", default="3,64,64")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3, help="fresh sessions per version")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/paper/node_profile_version.csv")
    a = ap.parse_args()

    want = [v.strip() for v in a.versions.split(",")]
    scratch = Path(a.scratch)
    pys = interpreters(scratch, want)
    if len(pys) != len(want):
        print(f"need interpreters for {want}; found {sorted(pys)}", file=sys.stderr)
        return 1
    work = scratch / "_nodeprof"
    work.mkdir(parents=True, exist_ok=True)
    child = work / "child.py"
    child.write_text(CHILD, encoding="utf-8")

    cells = [(v, r) for v in want for r in range(a.repeats)]
    random.Random(a.seed).shuffle(cells)       # version decorrelated from time, as in the bisect
    aff = ",".join(map(str, pick_cpus(a.threads, "auto") or []))
    rows, totals = [], []
    for i, (v, rep) in enumerate(cells, 1):
        r = subprocess.run([str(pys[v]), str(child), str(ROOT / a.model), str(a.threads), aff,
                            a.shape, str(a.warmup), str(a.runs), str(work / f"p_{v}_{rep}")],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"  [{i}/{len(cells)}] {v} r{rep} FAILED: {r.stderr.strip()[-200:]}")
            continue
        j = json.loads(r.stdout.strip().splitlines()[-1])
        node_sum = sum(j["ops_ms"].values())
        totals.append({"ort": v, "rep": rep, "wall_ms": j["wall_ms"], "node_sum_ms": node_sum})
        for op, ms in j["ops_ms"].items():
            rows.append({"ort": v, "rep": rep, "op_type": op, "ms_per_inf": ms,
                         "wall_ms": j["wall_ms"], "node_sum_ms": node_sum, "threads": a.threads,
                         "nodes_per_iter": j["per_iter"], "pinned_cpus": aff})
        print(f"  [{i}/{len(cells)}] ort {v:8s} r{rep}  wall {j['wall_ms']:.3f} ms  "
              f"node sum {node_sum:.3f} ms  ({j['per_iter']} nodes/iter)", flush=True)

    if not rows:
        return 1
    df = pd.DataFrame(rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    t = pd.DataFrame(totals).groupby("ort")[["wall_ms", "node_sum_ms"]].median()
    t["coverage"] = t["node_sum_ms"] / t["wall_ms"]
    print("\nper-inference medians (profiling on):\n" + t.round(3).to_string())
    per = df.groupby(["op_type", "ort"])["ms_per_inf"].median().unstack("ort").fillna(0)
    before, after = want[0], want[-1]
    per["delta_ms"] = per[after] - per[before]
    total_delta = per["delta_ms"].sum()
    per["share_of_delta"] = per["delta_ms"] / total_delta if total_delta else float("nan")
    print(f"\nper op type, ms per inference, {before} -> {after} (total delta {total_delta:+.3f} ms):")
    print(per.sort_values("delta_ms", ascending=False).round(4).head(15).to_string())
    top = per["share_of_delta"].max()
    print(f"\nlargest single op-type share of the delta: {top:.0%}  "
          f"(registered rule: >= 70% => op-specific; otherwise thread-pool-level)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
