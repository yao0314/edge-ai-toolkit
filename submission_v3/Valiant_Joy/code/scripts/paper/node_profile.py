"""Experiment 2: where does the time actually go in a slow INT8 model?

The static-vs-dynamic finding currently rests on a correlation between a graph statistic (416 Q/DQ
nodes against 90 compute ops; 36% of dequantizes reaching arithmetic) and a measured slowdown.
That is suggestive, not attribution. ONNX Runtime's own profiler gives per-node durations, so the
share of wall-clock spent on QuantizeLinear / DequantizeLinear can be measured directly.

If MobileViT static spends ~30% of its time on Q/DQ while its dynamic sibling spends ~0%, the
causal chain stops being a hypothesis.

    python scripts/paper/node_profile.py --models "results/classification_mobilevit/models/p00_int8*.onnx"

Writes results/paper/node_profile.csv (one row per model x op_type) and
results/paper/node_profile_summary.csv.

TWO RULES THIS SCRIPT ENFORCES, both from EXPERIMENT_PLAN.md:

1. ONE THREAD ONLY. Under threading, per-node durations overlap and their sum exceeds wall-clock,
   so "Q/DQ is X% of runtime" would not be a well-defined quantity. The script refuses --threads > 1.
2. PROFILING IS A SEPARATE RUN. It never touches the harness that produces the paper's headline
   latencies, and it reports its own overhead by timing the same model with profiling off, so the
   reader can see the distortion.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402

QDQ = {"QuantizeLinear", "DequantizeLinear"}


def input_spec(sess):
    i = sess.get_inputs()[0]
    shape = [1 if (isinstance(d, str) or d is None) else d for d in i.shape]
    dt = {"tensor(float)": np.float32, "tensor(uint8)": np.uint8,
          "tensor(int8)": np.int8, "tensor(int64)": np.int64}.get(i.type, np.float32)
    return i.name, shape, dt


def plain_latency(path: Path, runs: int, warmup: int) -> float:
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name, shape, dt = input_spec(s)
    x = (np.random.rand(*shape) * (255 if dt != np.float32 else 1)).astype(dt)
    for _ in range(warmup):
        s.run(None, {name: x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        s.run(None, {name: x})
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000)


def profile(path: Path, runs: int, warmup: int, out_dir: Path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_profiling = True
    so.profile_file_prefix = str(out_dir / f"prof_{path.stem}")
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name, shape, dt = input_spec(s)
    x = (np.random.rand(*shape) * (255 if dt != np.float32 else 1)).astype(dt)
    for _ in range(warmup + runs):
        s.run(None, {name: x})
    prof = Path(s.end_profiling())

    ev = json.loads(prof.read_text(encoding="utf-8"))
    # Keep only per-node kernel events, and only from the MEASURED iterations. ORT emits
    # model-load and session-level events too, and profiling is session-wide so the warmup
    # iterations land in the same file. Warmup is where lazy allocation, first-touch page faults
    # and any one-off kernel setup are paid, so leaving it in inflates both the profiler-overhead
    # figure and any op whose cost is front-loaded - which is exactly the Q/DQ ops this experiment
    # is trying to size. Events carry timestamps, so the fix is to sort and drop the first
    # `warmup` iterations' worth.
    nodes = [e for e in ev
             if e.get("cat") == "Node" and str(e.get("name", "")).endswith("_kernel_time")]
    nodes.sort(key=lambda e: e.get("ts", 0))
    iters = max(warmup + runs, 1)
    per_iter = len(nodes) // iters
    dropped = 0
    if per_iter and warmup:
        dropped = per_iter * warmup
        nodes = nodes[dropped:]
    by_op: dict = defaultdict(lambda: {"dur_us": 0.0, "calls": 0})
    total = 0.0
    for e in nodes:
        args = e.get("args") or {}
        op = args.get("op_name") or "unknown"
        by_op[op]["dur_us"] += float(e.get("dur", 0))
        by_op[op]["calls"] += 1
        total += float(e.get("dur", 0))
    prof.unlink(missing_ok=True)      # the JSON is large; the aggregate is what we keep
    return by_op, total, len(nodes), per_iter, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="results/classification_mobilevit/models/p00_int8*.onnx")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results/paper/node_profile.csv")
    a = ap.parse_args()

    if a.threads != 1:
        print("Refusing --threads != 1. Per-node durations overlap under threading and their sum\n"
              "exceeds wall-clock, so an op-type percentage would not be well defined.",
              file=sys.stderr)
        return 2

    files = sorted(ROOT.glob(a.models))
    if not files:
        print(f"no models matched {a.models}", file=sys.stderr)
        return 1
    tmp = ROOT / "results" / "paper" / "_prof_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} model(s), 1 thread, {a.warmup} warmup + {a.runs} profiled iterations\n")

    rows, summary = [], []
    for f in files:
        base_ms = plain_latency(f, a.runs, a.warmup)
        by_op, total_us, n_events, per_iter, dropped = profile(f, a.runs, a.warmup, tmp)
        if not total_us:
            print(f"  {f.stem:26s} no node events in the profile; skipped", file=sys.stderr)
            continue
        qdq_us = sum(v["dur_us"] for k, v in by_op.items() if k in QDQ)
        prof_ms = total_us / 1000 / max(a.runs, 1)   # warmup iterations were dropped above
        for op, v in sorted(by_op.items(), key=lambda kv: -kv[1]["dur_us"]):
            rows.append({"model": f.stem, "run": f.parent.parent.name, "op_type": op,
                         "calls": v["calls"], "dur_us": v["dur_us"],
                         "share_pct": 100 * v["dur_us"] / total_us, "is_qdq": op in QDQ})
        summary.append({"model": f.stem, "run": f.parent.parent.name,
                        "latency_ms_no_profiler": base_ms,
                        "latency_ms_from_profile": prof_ms,
                        "profiler_overhead_x": prof_ms / base_ms if base_ms else float("nan"),
                        "qdq_share_pct": 100 * qdq_us / total_us,
                        "n_op_types": len(by_op), "n_events": n_events,
                        "nodes_per_iteration": per_iter, "warmup_events_dropped": dropped})
        print(f"  {f.stem:26s} Q/DQ {100 * qdq_us / total_us:5.1f}% of node time   "
              f"latency {base_ms:.3f} ms (profiled {prof_ms:.3f}, {prof_ms / base_ms:.2f}x)", flush=True)
        top = sorted(by_op.items(), key=lambda kv: -kv[1]["dur_us"])[:4]
        print("      top ops: " + ", ".join(f"{k} {100 * v['dur_us'] / total_us:.1f}%" for k, v in top))

    pd.DataFrame(rows).to_csv(ROOT / a.out, index=False)
    sdf = pd.DataFrame(summary)
    sp = ROOT / a.out.replace(".csv", "_summary.csv")
    sdf.to_csv(sp, index=False)
    try:
        tmp.rmdir()
    except OSError:
        pass
    print(f"\nwrote {ROOT / a.out} and {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
