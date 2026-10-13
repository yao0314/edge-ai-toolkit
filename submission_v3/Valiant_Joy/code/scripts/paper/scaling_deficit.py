"""The INT8 parallel-scaling deficit, measured on every pair instead of on the sign flips.

The finding was being carried as "6 variants are faster at 1 thread and slower at 4, with no case
in the opposite direction". That framing is weak in a way a reviewer will find immediately: 6 out
of 71 INT8 variants is 8%, it looks like a handful of special cases, and the "no counter-examples"
argument rests on 6 observations drawn from only 3 runs.

But a sign flip is not the effect. A flip happens only when INT8's single-thread advantage is small
enough for the scaling deficit to overturn it, so counting flips measures how many models happen to
sit near the crossover - not how general the deficit is. The effect itself is the SCALING RATIO:

    scaling = latency(1 thread) / latency(4 threads)     higher = parallelises better

and the question is whether INT8 scales worse than its own fp32 sibling. That is a paired
comparison available on every variant, not just the ones that cross over.

Two units of analysis are reported, because they answer different questions and the weaker one is
the one that must be quoted:

  per PAIR   n=71. Overstates significance: variants inside one run share an architecture, a
             dataset and a baseline, so they are not independent observations.
  per RUN    n=23. The honest unit. Each run contributes the mean delta over its own pairs.

    python scripts/paper/scaling_deficit.py

Writes results/paper/scaling_deficit.csv.
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


def collect(a_from: int, a_to: int, include_quick: bool) -> pd.DataFrame:
    lo, hi = f"latency_p50_ms_t{a_from}", f"latency_p50_ms_t{a_to}"
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / "*" / "sweep.csv"))):
        run = Path(f).parent.name
        if run.startswith("_") or (run.endswith("_quick") and not include_quick):
            continue
        d = pd.read_csv(f)
        if lo not in d or hi not in d:
            continue
        lat = {str(r["variant"]): (r[lo], r[hi]) for _, r in d.iterrows()}
        for v, (t_lo, t_hi) in lat.items():
            if "int8" not in v:
                continue
            # p25_int8_dyn_mm -> p25_fp32. The sibling must share the prune ratio, or the pruning
            # cost is charged to quantization.
            sib = re.sub("_int8.*$", "_fp32", v)
            if sib not in lat:
                continue
            b_lo, b_hi = lat[sib]
            if not all(pd.notna(x) and x for x in (t_lo, t_hi, b_lo, b_hi)):
                continue
            rows.append({"run": run, "variant": v, "fp32_sibling": sib,
                         "fp32_lo_ms": b_lo, "fp32_hi_ms": b_hi,
                         "int8_lo_ms": t_lo, "int8_hi_ms": t_hi,
                         "scale_fp32": b_lo / b_hi, "scale_int8": t_lo / t_hi,
                         "int8_speedup_lo": b_lo / t_lo, "int8_speedup_hi": b_hi / t_hi})
    df = pd.DataFrame(rows)
    if len(df):
        df["delta"] = df["scale_int8"] - df["scale_fp32"]
        df["sign_flip"] = (df["int8_speedup_lo"] > 1.0) & (df["int8_speedup_hi"] < 1.0)
    return df


def speed_control(df: pd.DataFrame) -> pd.DataFrame:
    """Control for the obvious confound: INT8 models are FASTER, and a faster model has less work
    per thread, so it may scale worse for reasons that have nothing to do with quantization.

    Measured on the fp32 arms alone - unquantized, so whatever latency/scaling relation exists there
    is the pure "less work scales worse" effect. It is strong (Spearman ~ +0.96), so the confound
    cannot be waved away; it has to be subtracted.

    The control fits scaling ~ ln(latency) on the fp32 models and asks: what scaling would an
    UNQUANTIZED model as fast as this INT8 model be expected to reach? The residual is the part of
    the deficit that quantization has to answer for.
    """
    fp = df.drop_duplicates("fp32_sibling")[["fp32_sibling", "fp32_lo_ms", "scale_fp32"]]
    coef = np.polyfit(np.log(fp["fp32_lo_ms"].values), fp["scale_fp32"].values, 1)
    out = df.copy()
    out["expected_scale_at_its_own_speed"] = np.polyval(coef, np.log(out["int8_lo_ms"]))
    out["residual"] = out["scale_int8"] - out["expected_scale_at_its_own_speed"]
    out.attrs["fp32_fit"] = coef
    out.attrs["fp32_n"] = len(fp)
    try:
        from scipy.stats import spearmanr
        r = spearmanr(fp["fp32_lo_ms"], fp["scale_fp32"])
        out.attrs["fp32_rho"] = (float(r.statistic), float(r.pvalue))
    except Exception:
        out.attrs["fp32_rho"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-threads", type=int, default=1)
    ap.add_argument("--to-threads", type=int, default=4)
    ap.add_argument("--include-quick", action="store_true", help="include *_quick smoke runs")
    ap.add_argument("--out", default="results/paper/scaling_deficit.csv")
    a = ap.parse_args()

    df = collect(a.from_threads, a.to_threads, a.include_quick)
    if df.empty:
        print("no sibling-matched pairs found", file=sys.stderr)
        return 1
    df = speed_control(df)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n, worse = len(df), int((df["delta"] < 0).sum())
    print(f"{a.from_threads}T -> {a.to_threads}T scaling, INT8 vs its own fp32 sibling")
    print(f"  {n} sibling-matched pairs across {df['run'].nunique()} runs")
    print(f"  mean fp32 scaling {df['scale_fp32'].mean():.3f}x    "
          f"mean INT8 scaling {df['scale_int8'].mean():.3f}x")
    print(f"  INT8 scales worse in {worse}/{n} pairs; mean delta {df['delta'].mean():+.3f}")
    print(f"  sign flips (fast at {a.from_threads}T, slow at {a.to_threads}T): "
          f"{int(df['sign_flip'].sum())} - the OLD framing, and the weakest view of the effect")

    per_run = df.groupby("run")["delta"].agg(["size", "mean"]).rename(
        columns={"size": "pairs", "mean": "mean_delta"})
    neg = int((per_run["mean_delta"] < 0).sum())
    print("")
    print("per run (the honest unit - pairs inside a run are not independent):")
    print(f"  {neg}/{len(per_run)} runs have a negative mean delta")

    rho = df.attrs.get("fp32_rho")
    coef = df.attrs["fp32_fit"]
    print("")
    print("=" * 78)
    print("CONFOUND CONTROL - is this just 'faster models scale worse'?")
    if rho:
        print(f"  fp32 arms only (n={df.attrs['fp32_n']}): 1T latency vs scaling  "
              f"rho={rho[0]:+.3f} p={rho[1]:.3g}")
    print(f"  fp32 trend: scaling ~ {coef[0]:+.3f} * ln(latency_ms) + {coef[1]:.3f}")
    print("  The confound is REAL and strong. Slower models genuinely parallelise better, so the")
    print("  raw deficit above is partly just the speedup showing up twice. Residual below is the")
    print("  part quantization has to answer for.")
    res = df.groupby("run")["residual"].mean().sort_values()
    rneg = int((res < 0).sum())
    print("")
    print(f"  INT8 scaling minus an fp32 model of the SAME 1T latency: mean {df['residual'].mean():+.3f}")
    print(f"  negative in {int((df['residual'] < 0).sum())}/{n} pairs, {rneg}/{len(res)} runs")
    print(res.to_string(float_format=lambda v: f"{v:+.3f}"))

    try:
        from scipy.stats import binomtest, wilcoxon
        w = wilcoxon(df["scale_int8"], df["scale_fp32"])
        wr = wilcoxon(per_run["mean_delta"])
        print("")
        print(f"  RAW      per pair Wilcoxon p={w.pvalue:.2g} (clustered, inflated)   "
              f"per run p={wr.pvalue:.2g}  sign p={binomtest(neg, len(per_run), 0.5).pvalue:.2g}")
        wc = wilcoxon(res)
        print(f"  CONTROLLED per run Wilcoxon p={wc.pvalue:.2g}   "
              f"sign p={binomtest(rneg, len(res), 0.5).pvalue:.2g}   <- QUOTE THIS ONE")
    except Exception as e:
        print(f"  scipy unavailable ({e}); report the counts only")
    print("")
    print("Read the per-run residuals, not just the p-value: the deficit is not universal once")
    print("speed is controlled. Where it is large, the node profile should show a large Q/DQ share;")
    print("where it is positive, it should not. That is the falsifiable link to experiment 2.")
    print(f"")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
