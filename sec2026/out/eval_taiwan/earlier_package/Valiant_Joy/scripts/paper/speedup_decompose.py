"""Where does a speedup's drift come from: the part both terms share, or the part only one has?

The registered test (scripts/paper/speedup_drift.py) refuted the prediction that a speedup, being a
ratio of two drifting numbers, moves more often than either. It moves LESS. This asks why, and the
answer is the one INT8-specific statement the map can support: a release usually moves both terms
in the same direction, the ratio cancels that common part, and what survives is the differential —
a release repricing a path that only one of the two files is on. That is precisely what quantization
selects.

Decomposition, per pair-cell. With f and i the fractional changes of the FP32 and INT8 latency:

    common       = the smaller of the two in magnitude, when they share a sign (else 0)
    differential = what is left, which is what the ratio keeps

Reads results/paper/s_speedup_drift.json, written by the registered script. No new measurement.

    .venv/Scripts/python.exe scripts/paper/speedup_decompose.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "paper" / "s_speedup_drift.json"
OUT = ROOT / "results" / "paper" / "s_speedup_decompose.json"
FLOOR = 7.0


def classify(f: float, i: float) -> str:
    """How the two terms moved, before asking what the ratio did."""
    big = max(abs(f), abs(i))
    if big < FLOOR:
        return "neither moved"
    if abs(f) >= FLOOR and abs(i) >= FLOOR:
        return "both moved, same way" if f * i > 0 else "both moved, opposite ways"
    return "one moved, one held"


def main() -> int:
    rows = json.loads(SRC.read_text(encoding="utf-8"))["rows"]
    n = len(rows)

    groups: dict = {}
    for r in rows:
        k = classify(r["fp32_pct"], r["int8_pct"])
        g = groups.setdefault(k, {"n": 0, "speedup_moved": 0, "examples": []})
        g["n"] += 1
        if abs(r["speedup_pct"]) >= FLOOR:
            g["speedup_moved"] += 1
            g["examples"].append(r)

    print("%d pair-cells, seven model families, eight single-release steps\n" % n)
    print("%-26s %6s %16s %9s" % ("how the two terms moved", "cells", "speedup >= 7%", "share"))
    order = ["neither moved", "both moved, same way", "both moved, opposite ways", "one moved, one held"]
    for k in order:
        g = groups.get(k)
        if not g:
            continue
        print("%-26s %6d %16d %8.0f%%" % (k, g["n"], g["speedup_moved"],
                                          100 * g["speedup_moved"] / g["n"]))

    moved = [r for r in rows if abs(r["speedup_pct"]) >= FLOOR]
    diff = [r for r in moved if classify(r["fp32_pct"], r["int8_pct"]) in
            ("one moved, one held", "both moved, opposite ways")]
    print("\nspeedups that moved by 7%% or more : %d of %d (%.1f%%)" % (len(moved), n, 100 * len(moved) / n))
    print("of those, driven by a differential: %d of %d (%.0f%%)"
          % (len(diff), len(moved), 100 * len(diff) / max(len(moved), 1)))

    same = [r for r in rows if r["fp32_pct"] * r["int8_pct"] > 0]
    big_common = [r for r in same if min(abs(r["fp32_pct"]), abs(r["int8_pct"])) >= FLOOR]
    kept = [r for r in big_common if abs(r["speedup_pct"]) >= FLOOR]
    print("\ncommon-mode cancellation: of %d cells where BOTH terms moved >= 7%% the same way,"
          "\n  the speedup still moved >= 7%% in %d." % (len(big_common), len(kept)))

    print("\nthe clearest differentials:")
    for r in sorted(diff, key=lambda r: -abs(r["speedup_pct"]))[:6]:
        print("  %-18s %-8s %dT  fp32 %+6.1f%%  int8 %+6.1f%%  ->  speedup %+6.1f%%"
              % (r["step"], r["family"], r["threads"], r["fp32_pct"], r["int8_pct"], r["speedup_pct"]))

    OUT.write_text(json.dumps(
        {"pair_cells": n,
         "by_term_motion": {k: {"n": v["n"], "speedup_moved": v["speedup_moved"]}
                            for k, v in groups.items()},
         "speedup_moved": len(moved), "differential_driven": len(diff),
         "common_mode_cells": len(big_common), "common_mode_survived": len(kept),
         "examples": sorted(diff, key=lambda r: -abs(r["speedup_pct"]))[:10]},
        indent=2), encoding="utf-8")
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
