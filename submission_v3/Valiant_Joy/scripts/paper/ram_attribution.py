"""Experiment 4: why do several slow INT8 models use MORE peak RAM than their fp32 sibling?

MobileViT p50 INT8 uses 2.07x the peak RSS of fp32 while also being slower. That is unexplained,
and a reviewer will ask. Three candidates:

  arena        ONNX Runtime's CPU allocator arena deliberately over-allocates and grows in chunks.
               "INT8 uses more memory" may simply be "the arena grew differently", which is a
               different claim. `enable_cpu_mem_arena = False` separates them, and that control is
               the point of this experiment.
  boundaries   at an int8/fp32 boundary both representations of a tensor can be live at once, so a
               graph with many boundaries holds more simultaneous buffers.
  intermediates quantized graphs carry extra scale/zero-point initialisers and, with per-channel
               quantization, per-channel parameter tensors.

Peak RSS is a lifetime high-water mark, so a transient during session construction counts toward
it; steady-state RSS after warmup is reported alongside so a gap localises the growth to setup.

    python scripts/paper/ram_attribution.py

Writes results/paper/ram_attribution.csv.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import onnx  # noqa: E402
import pandas as pd  # noqa: E402

# Measured in a fresh subprocess per configuration: peak RSS is a per-process high-water mark, so
# two configurations measured in one process would report the maximum of both.
CHILD = textwrap.dedent('''
    import json, sys, numpy as np, onnxruntime as ort, resource_shim
    path, threads, arena = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = threads
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = arena
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    shape = [1 if (isinstance(d, str) or d is None) else d for d in i.shape]
    x = np.random.rand(*shape).astype(np.float32)
    after_load = resource_shim.rss_mb()
    for _ in range(20):
        s.run(None, {i.name: x})
    print(json.dumps({"after_load_mb": after_load, "steady_mb": resource_shim.rss_mb(),
                      "peak_mb": resource_shim.peak_rss_mb()}))
''')

SHIM = textwrap.dedent('''
    import sys, psutil
    def rss_mb():
        return psutil.Process().memory_info().rss / 2**20
    def peak_rss_mb():
        if sys.platform == "win32":
            return psutil.Process().memory_info().peak_wset / 2**20
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (r if sys.platform == "darwin" else r * 1024) / 2**20
''')


def graph_stats(p: Path) -> dict:
    g = onnx.load(str(p), load_external_data=False).graph
    ops = [n.op_type for n in g.node]
    init_bytes = sum(len(i.raw_data) or 0 for i in g.initializer)
    return {"n_initializers": len(g.initializer), "initializer_bytes": init_bytes,
            "n_q": ops.count("QuantizeLinear"), "n_dq": ops.count("DequantizeLinear"),
            "file_bytes": p.stat().st_size}


def measure(py: Path, work: Path, model: Path, threads: int, arena: bool) -> dict:
    r = subprocess.run([str(py), str(work / "child.py"), str(model), str(threads),
                        "1" if arena else "0"],
                       capture_output=True, text=True, cwd=str(work))
    line = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("{")]
    if not line:
        return {"error": (r.stderr or r.stdout)[-200:]}
    return json.loads(line[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="classification_mobilevit,classification_deit,stress_resnet50,classification")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/paper/ram_attribution.csv")
    a = ap.parse_args()

    work = ROOT / "results" / "paper" / "_ram"
    work.mkdir(parents=True, exist_ok=True)
    (work / "child.py").write_text(CHILD, encoding="utf-8")
    (work / "resource_shim.py").write_text(SHIM, encoding="utf-8")
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)

    rows = []
    for run in a.runs.split(","):
        d = ROOT / "results" / run / "models"
        if not d.is_dir():
            continue
        models = sorted(p for p in d.glob("p00_*.onnx"))
        print(f"\n=== {run} ===")
        for m in models:
            gs = graph_stats(m)
            for arena in (True, False):
                r = measure(py, work, m, a.threads, arena)
                if "error" in r:
                    print(f"  {m.stem:24s} arena={int(arena)}  FAILED {r['error'][:70]}")
                    continue
                rows.append({"run": run, "variant": m.stem, "arena": arena,
                             "threads": a.threads, **r, **gs})
                print(f"  {m.stem:24s} arena={int(arena)}  peak {r['peak_mb']:7.1f} MB  "
                      f"steady {r['steady_mb']:7.1f}  after_load {r['after_load_mb']:7.1f}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / a.out
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("INT8 peak RSS relative to the fp32 sibling, with the arena on and off:\n")
    print(f"{'run':28s} {'variant':22s} {'arena ON':>9s} {'arena OFF':>10s}")
    for run, g in df.groupby("run"):
        fp = g[g["variant"].str.endswith("fp32")]
        if fp.empty:
            continue
        base = {bool(r["arena"]): float(r["peak_mb"]) for _, r in fp.iterrows()}
        for _, r in g[g["variant"].str.contains("int8")].iterrows():
            if not bool(r["arena"]):
                continue
            off = g[(g["variant"] == r["variant"]) & (~g["arena"].astype(bool))]
            on_x = r["peak_mb"] / base.get(True, float("nan"))
            off_x = (float(off["peak_mb"].iloc[0]) / base.get(False, float("nan"))) if len(off) else float("nan")
            print(f"{run:28s} {r['variant']:22s} {on_x:9.2f} {off_x:10.2f}")
    print("\nIf the ratio drops toward 1.0 with the arena OFF, the growth is allocator behaviour and")
    print("must be reported as such rather than as INT8 needing more memory.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
