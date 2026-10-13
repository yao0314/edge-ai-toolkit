"""How much of each model is actually parallel, and what quantization does to it.

Parallel efficiency tables are hard to read across models because a model with almost no parallel
work looks "bad at scaling" in a way that says nothing about quantization. Amdahl's law turns the
whole curve into one interpretable number:

    speedup(n) = 1 / ((1 - p) + p/n)          p = the parallel fraction of the work

p is what the argument is really about. A model with p = 0.02 is a serial workload and there is no
parallel scaling to destroy, whatever the efficiency column says - the challengers' sharpest point
about MobileViT, and this script is how to check it rather than assert it.

Fitted by least squares on log-speedup across the same-core-class thread counts only. Counts that
spill onto E-cores or SMT siblings are a different machine and are excluded by construction.

    python scripts/paper/parallel_fraction.py

Writes results/paper/parallel_fraction.csv.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def fit_p(threads, speedups) -> tuple:
    """Least-squares p for speedup(n) = 1/((1-p) + p/n), plus the fit's worst residual.

    Solved on the reciprocal, where the model is linear: 1/S = 1 - p(1 - 1/n). A p outside [0, 1]
    is reported rather than clipped - it means Amdahl does not describe this curve (superlinear
    speedup from cache effects, or a measurement problem), and clipping would hide that.
    """
    n = np.asarray(threads, dtype=float)
    s = np.asarray(speedups, dtype=float)
    keep = (n > 1) & (s > 0)
    if keep.sum() < 2:
        return float("nan"), float("nan")
    x = (1.0 - 1.0 / n[keep])
    y = 1.0 - 1.0 / s[keep]
    p = float((x @ y) / (x @ x))            # through the origin: at n=1 the model is exact
    pred = 1.0 / (1.0 - p * x)
    worst = float(np.max(np.abs(pred - s[keep]) / s[keep])) if len(pred) else float("nan")
    return p, worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/paper/thread_sweep_*.csv")
    ap.add_argument("--out", default="results/paper/parallel_fraction.csv")
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(ROOT / a.glob))):
        run = Path(f).stem.replace("thread_sweep_", "")
        d = pd.read_csv(f)
        same = d[d["condition"] == "same-core-class"]
        for model, g in same.groupby("model"):
            g = g.sort_values("threads")
            base = g[g["threads"] == 1]["latency_p50_ms"]
            if base.empty or not float(base.iloc[0]):
                continue
            b = float(base.iloc[0])
            p, worst = fit_p(g["threads"], b / g["latency_p50_ms"])
            rows.append({"run": run, "model": str(model), "parallel_fraction": p,
                         "worst_fit_error": worst, "latency_1t_ms": b,
                         "serial_ms": b * (1 - p) if p == p else float("nan"),
                         "n_points": len(g)})
    if not rows:
        print("no thread sweeps found", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    out = ROOT / a.out
    # A narrow --glob used to clobber the full file: running this for one run wrote a one-run CSV
    # over the fits for every run. Merge on (run, model) instead, so a focused re-fit updates its
    # own rows and leaves the rest alone.
    if out.exists() and a.glob != ap.get_default("glob"):
        try:
            prev = pd.read_csv(out)
            keep = prev.merge(df[["run", "model"]], on=["run", "model"], how="left", indicator=True)
            keep = keep[keep["_merge"] == "left_only"].drop(columns="_merge")
            df = pd.concat([keep, df], ignore_index=True)
            print(f"  merged into the existing {out.name} ({len(keep)} rows kept)")
        except Exception as e:
            print(f"  could not merge with {out.name} ({e}); writing fresh")
    df.to_csv(out, index=False)

    for run, g in df.groupby("run"):
        print(f"=== {run} ===")
        print(f"  {'model':26s} {'parallel p':>10s} {'1T ms':>8s} {'serial ms':>10s} {'fit err':>8s}")
        for _, r in g.sort_values("model").iterrows():
            print(f"  {r['model']:26s} {r['parallel_fraction']:10.3f} {r['latency_1t_ms']:8.3f} "
                  f"{r['serial_ms']:10.3f} {r['worst_fit_error'] * 100:7.1f}%")
        # the within-model contrast is the only one that means anything: p depends on model size
        base = g[g["model"].str.endswith("p00_fp32")]
        if not base.empty:
            bp = float(base["parallel_fraction"].iloc[0])
            print(f"\n  fp32 p = {bp:.3f}.", end=" ")
            if bp < 0.25:
                print("BELOW 0.25 - this is a near-serial workload. Differences in its")
                print("  'scaling' are differences between two numbers that barely exist, and no")
                print("  quantization claim should rest on them.")
            else:
                print("Enough parallel work for the contrast below to be meaningful.")
            for _, r in g.iterrows():
                if r["model"].endswith("p00_fp32") or not r["model"].startswith("p00"):
                    continue
                print(f"    {r['model']:26s} p = {r['parallel_fraction']:.3f}  "
                      f"({r['parallel_fraction'] - bp:+.3f} vs fp32)")
        print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
