"""Does an INT8 speedup drift more often than either latency it is made of?

A latency is one number; a speedup is a ratio of two that move independently, so it should move
more often than either term. That is the one claim about drift which is INT8-specific by
construction, and it is the evidence a reviewer asked for when the map's per-variant rates did not
separate INT8 from FP32.

Reads the drift map already measured (results/expB/batches/*.csv). No new measurement. The rule and
the predictions were registered in results/paper/s_speedup_drift_registered.md before this ran.

    .venv/Scripts/python.exe scripts/paper/speedup_drift.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BATCHES = ROOT / "results" / "expB" / "batches" / "*.csv"
OUT = ROOT / "results" / "paper" / "s_speedup_drift.json"

FLOOR = 0.07
# The paper's eight single-minor steps; 1.28.0 -> 1.30.0 skips a version and is reported apart.
STEPS = [("1.20.1", "1.21.1"), ("1.21.1", "1.22.1"), ("1.22.1", "1.23.2"), ("1.23.2", "1.24.4"),
         ("1.24.4", "1.25.1"), ("1.25.1", "1.26.0"), ("1.26.0", "1.27.0"), ("1.27.0", "1.28.0")]
SKIP_STEP = ("1.28.0", "1.30.0")


def cells(df: pd.DataFrame) -> dict:
    """(family, variant, threads, ort) -> (median, min, max) over the five sessions."""
    out = {}
    g = df.groupby(["family", "variant", "threads", "ort"])["p50_ms"]
    for key, s in g:
        if len(s) >= 3:                    # a cell needs its sessions; the map has five
            out[key] = (float(s.median()), float(s.min()), float(s.max()))
    return out


def verdict(a: tuple, b: tuple) -> str:
    """MOVED / UNRESOLVED / STILL, by the paper's rule: 7% and disjoint ranges."""
    change = abs(b[0] - a[0]) / a[0]
    if change < FLOOR:
        return "STILL"
    disjoint = a[2] < b[1] or b[2] < a[1]
    return "MOVED" if disjoint else "UNRESOLVED"


def speedup(fp32: tuple, int8: tuple) -> tuple:
    """Ratio of medians, with the conservative range from both terms' session extremes.

    The sessions of the two terms are not paired, so no tighter interval is defensible. This one is
    wider than a single latency's, which makes MOVED harder to reach for the speedup, not easier.
    """
    return (fp32[0] / int8[0], fp32[1] / int8[2], fp32[2] / int8[1])


def main() -> int:
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(BATCHES))], ignore_index=True)
    df = df[(df["status"] == "OK") & df["p50_ms"].notna()]
    df["ort"] = df["ort"].astype(str)
    df["threads"] = df["threads"].astype(int)
    c = cells(df)

    pairs = sorted({(fam, th) for (fam, var, th, _) in c if var == "p00_fp32"}
                   & {(fam, th) for (fam, var, th, _) in c if var == "p00_int8"})
    print("families with both terms: %s" % sorted({f for f, _ in pairs}))
    print("(family, threads) pairs : %d\n" % len(pairs))

    rows, counts = [], {q: {"MOVED": 0, "UNRESOLVED": 0, "STILL": 0} for q in ("fp32", "int8", "speedup")}
    for a, b in STEPS:
        for fam, th in pairs:
            k = {q: [(fam, v, th, r) for r in (a, b)] for q, v in
                 (("fp32", "p00_fp32"), ("int8", "p00_int8"))}
            if not all(key in c for keys in k.values() for key in keys):
                continue
            ca, cb = {}, {}
            for q in ("fp32", "int8"):
                ca[q], cb[q] = c[k[q][0]], c[k[q][1]]
            ca["speedup"], cb["speedup"] = speedup(ca["fp32"], ca["int8"]), speedup(cb["fp32"], cb["int8"])
            row = {"step": f"{a}->{b}", "family": fam, "threads": th}
            for q in ("fp32", "int8", "speedup"):
                v = verdict(ca[q], cb[q])
                counts[q][v] += 1
                row[q] = v
                row[f"{q}_pct"] = round(100 * (cb[q][0] - ca[q][0]) / ca[q][0], 1)
            rows.append(row)

    n = len(rows)
    print("denominator (same pairs for all three quantities): %d\n" % n)
    print("%-9s %7s %12s %9s   %s" % ("quantity", "MOVED", "UNRESOLVED", "STILL", "bracket"))
    rates = {}
    for q in ("fp32", "int8", "speedup"):
        m, u, s = counts[q]["MOVED"], counts[q]["UNRESOLVED"], counts[q]["STILL"]
        lo, hi = 100 * m / n, 100 * (m + u) / n
        rates[q] = (lo, hi)
        print("%-9s %7d %12d %9d   %4.1f%% - %4.1f%%" % (q, m, u, s, lo, hi))

    sp, fp, i8 = rates["speedup"], rates["fp32"], rates["int8"]
    p1 = sp[0] > fp[0] and sp[0] > i8[0]
    p2 = sp[1] > fp[1] and sp[1] > i8[1]
    compound = [r for r in rows if r["speedup"] == "MOVED"
                and abs(r["fp32_pct"]) < 100 * FLOOR and abs(r["int8_pct"]) < 100 * FLOOR]
    print("\nprediction 1 (MOVED rate above both terms)      : %s" % ("HELD" if p1 else "FAILED"))
    print("prediction 2 (bracket above both terms)         : %s" % ("HELD" if p2 else "FAILED"))
    print("prediction 3 (a move neither term makes alone)  : %s (%d cases)"
          % ("HELD" if compound else "FAILED", len(compound)))
    for r in compound[:6]:
        print("    %-18s %-8s %dT  fp32 %+5.1f%%  int8 %+5.1f%%  speedup %+5.1f%%"
              % (r["step"], r["family"], r["threads"], r["fp32_pct"], r["int8_pct"], r["speedup_pct"]))
    if not p1:
        print("\nREFUTED: the speedup does not move more often than its terms. The paper must drop"
              "\nthe claim of a higher rate and report this null result.")

    OUT.write_text(json.dumps(
        {"denominator": n, "counts": counts, "rates_pct": rates,
         "predictions": {"1_moved_rate_above_both": p1, "2_bracket_above_both": p2,
                         "3_compound_moves": len(compound)},
         "compound_examples": compound[:10], "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
