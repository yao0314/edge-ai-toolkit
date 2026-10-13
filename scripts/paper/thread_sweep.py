"""Experiment 3: is the INT8 slowdown a parallel-scaling effect?

Experiment 12 (scripts/paper/scaling_deficit.py) established the effect on the existing data and
then nearly took it away again: INT8 scales worse 1T->4T in 66/71 sibling-matched pairs and 23/23
runs, BUT fp32 models' own scaling tracks their latency at Spearman +0.958, so most of that is the
speedup being counted twice. Controlled for absolute speed the deficit survives in 19/23 runs, and
it is architecture-specific: large on transformers, moderate on detection, and REVERSED on ResNet50
and Imagenette.

This script supplies what experiment 12 could not. Two points cannot distinguish a gentle slope from
a cliff, and the existing sweeps were not purpose-built for a scaling measurement. PRE-REGISTERED:
if the deficit is a real parallel-scaling effect, the curve should bend between 2 and 6 threads for
MobileViT/DeiT and stay near-linear for ResNet50. If ResNet50 bends just as hard, the
architecture-specific story is refuted and the residual pattern needs another explanation.

This machine is heterogeneous: 6 P-cores + 4 E-cores, 16 logical. `pick_cpus` takes the fastest
core class first, so 1-6 threads all land on P-cores and are mutually comparable, while 8 and
above spill onto E-cores or SMT siblings. Those are TWO DIFFERENT CONDITIONS and this script keeps
them apart: putting 8 on the same curve as 4 would confound parallel scaling with core
heterogeneity. The pinned CPU ids are recorded per point so the reader can check.

    python scripts/paper/thread_sweep.py --run classification_deit --threads 1,2,4,6 --spill 8,12

Writes results/paper/thread_sweep.csv.
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402

from core.cpu_topology import cpu_layout, pick_cpus  # noqa: E402
from core.measure import runtime_metrics  # noqa: E402


def example_for(path: Path) -> np.ndarray:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    shape = [1 if (isinstance(d, str) or d is None) else d for d in i.shape]
    return np.random.rand(*shape).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="classification_deit")
    ap.add_argument("--threads", default="1,2,4,6", help="within one core class: comparable")
    ap.add_argument("--spill", default="8,12", help="crosses into E-cores / SMT: separate condition")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--out", default="results/paper/thread_sweep.csv")
    a = ap.parse_args()

    layout = cpu_layout() or []
    classes = sorted({d["perf"] for d in layout}, reverse=True)
    fast = [d["cpu"] for d in layout if d["perf"] == classes[0]] if classes else []
    print(f"topology: {len(layout)} logical CPUs, {len(classes)} core class(es), "
          f"{len(fast)} in the fastest class\n")

    models = sorted((ROOT / "results" / a.run / "models").glob("*.onnx"))
    if not models:
        print(f"no models under results/{a.run}/models", file=sys.stderr)
        return 1

    rows = []
    for t, cond in [(int(x), "same-core-class") for x in a.threads.split(",") if x] + \
                   [(int(x), "spills-core-classes") for x in (a.spill.split(",") if a.spill else []) if x]:
        cpus = pick_cpus(t, "auto") or []
        # A request that exceeds the fastest class is not a bigger version of the same experiment.
        spilled = any(c not in fast for c in cpus) if fast else False
        for m in models:
            x = example_for(m)
            with tempfile.TemporaryDirectory():
                # Same timed window as the headline measurements (min_time_s 2.0). This is a
                # SCALING claim: at 4-6 threads a short window is exactly where sync overhead and
                # counter granularity distort the number the claim rests on.
                r = runtime_metrics(m, x, {"threads": [t], "warmup": 10, "runs": a.runs,
                                           "min_time_s": 2.0, "repeats": a.repeats})
            key = f"latency_p50_ms_t{t}"
            rows.append({"run": a.run, "model": m.stem, "threads": t, "condition": cond,
                         "spilled_out_of_fastest_class": spilled,
                         "pinned_cpus": ",".join(map(str, cpus)),
                         "latency_p50_ms": r.get(key), "latency_sd_ms": r.get(f"latency_p50_ms_sd_t{t}"),
                         "peak_rss_mb": r.get(f"peak_rss_mb_t{t}"),
                         "cpu_ms_per_inf": r.get(f"cpu_ms_per_inf_t{t}"),
                         "other_load_pct": r.get("other_load_pct")})
            print(f"  t={t:2d} [{cond:20s}] {m.stem:26s} "
                  f"{r.get(key, float('nan')):8.3f} ms  sd {r.get(f'latency_p50_ms_sd_t{t}') or 0:.3f}  "
                  f"load {r.get('other_load_pct') or 0:.1f}%", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # Speedup against each variant's OWN fp32 sibling, within the comparable condition only.
    print("")
    print("INT8 speedup vs its OWN fp32 sibling, same core class only:")
    same = df[df["condition"] == "same-core-class"]
    for t, g in same.groupby("threads"):
        lat = dict(zip(g["model"], g["latency_p50_ms"]))
        for _, r in g.iterrows():
            if "int8" not in str(r["model"]):
                continue
            # p25_int8_dyn_mm -> p25_fp32, 320_p00_int8 -> 320_p00_fp32. Pairing every INT8 variant
            # with the FIRST fp32 in the group would compare a PRUNED int8 model against an UNPRUNED
            # fp32 one and charge the whole difference to quantization.
            sib = re.sub("_int8.*$", "_fp32", str(r["model"]))
            base = lat.get(sib)
            if base is None or not r["latency_p50_ms"]:
                print(f"  t={t:2d}  {r['model']:26s} no fp32 sibling ({sib}) - skipped")
                continue
            print(f"  t={t:2d}  {r['model']:26s} {base / r['latency_p50_ms']:.3f}x  (vs {sib})")
    if not df[df["spilled_out_of_fastest_class"]].empty:
        print("\nPoints marked spilled_out_of_fastest_class used cores from a slower class and are")
        print("NOT comparable with the rest; report them as a separate condition.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
