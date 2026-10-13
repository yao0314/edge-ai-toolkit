"""Experiment B, §5.5 and P6: did the map's verdicts survive an independent run?

Binding rules, results/paper/b_design_registered.md §5.5 and §6 P6, verbatim in substance:

  * The replication run is its own stratum. Its verdict comes from its own five-a-side sessions
    alone, by the §4.2 rule (|delta| >= 7 % AND disjoint five-session ranges).
  * The map's verdict is never revised, and the two runs' sessions are never pooled.
  * Label per pair-cell: REPLICATED (MOVED in both runs, same sign), NOT-REPLICATED (MOVED in the map,
    not in the replication run), RESOLVED-MOVED / RESOLVED-STILL for cells UNRESOLVED in the map,
    NOT-TESTED if truncated by the cap.
  * P6 predicts >= 80 % of MOVED with |delta| >= 20 % replicate, >= 45 % of all MOVED replicate, and a
    higher rate in the >= 20 % band than in 7-20 %. < 25 % refutes the framing: the map's share is
    then published as an upper bound only.

Where the registration is silent, the gap is printed, not filled: a map-UNRESOLVED cell that is
UNRESOLVED again has no registered label and is reported as UNRESOLVED-AGAIN.

The registered stage is a PREFIX of this run (results/expB/replication_note.md): the first five models
by priority. Both the prefix and the whole run are reported.

    .venv/Scripts/python.exe scripts/paper/b_replicate_analyse.py
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPB = os.path.join(ROOT, "results", "expB")
FLOOR = 7.0
REGISTERED_PREFIX = ("c3_299x299", "c3_112x112", "c3_224x224", "ae", "mv256")


def verdict(a: list[float], b: list[float]) -> tuple[str, float]:
    d = 100.0 * (st.median(b) / st.median(a) - 1.0)
    if abs(d) < FLOOR:
        return "STILL", d
    return ("MOVED" if (min(b) > max(a) or max(b) < min(a)) else "UNRESOLVED"), d


def main() -> int:
    sealed = json.load(open(os.path.join(EXPB, "moves_sealed.json"), encoding="utf-8"))
    rep = defaultdict(list)
    status = Counter()
    for r in csv.DictReader(open(os.path.join(EXPB, "b_drift_replicate.csv"), encoding="utf-8")):
        status[r.get("status") or "OK"] += 1
        if (r.get("status") or "OK") == "OK":
            rep[(r["family"], r["variant"], int(r["threads"]), r["ort"])].append(float(r["p50_ms"]))

    out = []
    for p in sealed["pair_cells"]:
        m, v, t = p["model"], p["variant"], int(p["threads"])
        a, b = rep.get((m, v, t, p["release_a"])), rep.get((m, v, t, p["release_b"]))
        rec = {**p, "prefix": m in REGISTERED_PREFIX}
        if not a or not b:
            rec.update(label="NOT-TESTED", rep_verdict=None, rep_delta=None)
        elif len(a) != 5 or len(b) != 5:
            rec.update(label="VOID", rep_verdict="VOID (%d,%d sessions)" % (len(a), len(b)), rep_delta=None)
        else:
            rv, rd = verdict(a, b)
            rec.update(rep_verdict=rv, rep_delta=rd)
            if p["verdict"] == "MOVED":
                same = rv == "MOVED" and (rd > 0) == (p["delta"] > 0)
                rec["label"] = "REPLICATED" if same else "NOT-REPLICATED"
            else:
                rec["label"] = {"MOVED": "RESOLVED-MOVED", "STILL": "RESOLVED-STILL"}.get(rv, "UNRESOLVED-AGAIN")
        out.append(rec)

    def share(rows):
        n = len(rows)
        k = sum(1 for r in rows if r["label"] == "REPLICATED")
        return k, n, (100.0 * k / n if n else float("nan"))

    print("=" * 92)
    print("EXPERIMENT B - REPLICATION STAGE (§5.5) AND P6")
    print("replication rows: %s" % dict(status))
    print("=" * 92)
    for scope_name, keep in (("REGISTERED PREFIX (75-min cap, first five models)", lambda r: r["prefix"]),
                             ("WHOLE RUN (prefix + extension)", lambda r: True)):
        rows = [r for r in out if keep(r)]
        moved = [r for r in rows if r["verdict"] == "MOVED"]
        unres = [r for r in rows if r["verdict"] == "UNRESOLVED"]
        print("\n-- %s: %d pair-cells (%d map-MOVED, %d map-UNRESOLVED)" % (
            scope_name, len(rows), len(moved), len(unres)))
        print("   labels: %s" % dict(Counter(r["label"] for r in rows)))
        k, n, s = share(moved)
        print("   P6  all MOVED replicated:           %3d / %3d = %5.1f %%   (predicted >= 45 %%, refuted < 25 %%)" % (k, n, s))
        big = [r for r in moved if abs(r["delta"]) >= 20]
        small = [r for r in moved if abs(r["delta"]) < 20]
        kb, nb, sb = share(big)
        ks, ns, ss = share(small)
        print("   P6  MOVED with |d| >= 20 %%:          %3d / %3d = %5.1f %%   (predicted >= 80 %%)" % (kb, nb, sb))
        print("   P6  MOVED with 7 <= |d| < 20 %%:       %3d / %3d = %5.1f %%   (predicted lower than the >= 20 %% band)" % (ks, ns, ss))
        sm = [r for r in moved if r["single_minor"]]
        k1, n1, s1 = share(sm)
        print("   primary denominator only (single-minor MOVED): %d / %d = %.1f %%" % (k1, n1, s1))
        for arm in sorted({r["arm"] for r in moved}):
            ka, na, sa = share([r for r in moved if r["arm"] == arm])
            print("      %-9s %3d / %3d = %5.1f %%" % (arm, ka, na, sa))
        diffs = [r["delta"] - r["rep_delta"] for r in moved if r["rep_delta"] is not None]
        if diffs:
            ad = sorted(abs(x) for x in diffs)
            print("   delta_map - delta_replication on MOVED cells: median |diff| %.1f pp, p90 %.1f pp, max %.1f pp" % (
                st.median(ad), ad[int(len(ad) * 0.9)], ad[-1]))

    whole = [r for r in out if r["verdict"] == "MOVED"]
    k, n, s = share(whole)
    kb, nb, sb = share([r for r in whole if abs(r["delta"]) >= 20])
    ks, ns, ss = share([r for r in whole if abs(r["delta"]) < 20])
    verdicts = {
        "all MOVED >= 45 %": s >= 45,
        "|d| >= 20 % band >= 80 %": sb >= 80,
        ">= 20 % band higher than 7-20 % band": sb > ss,
        "framing refuted (< 25 %)": s < 25,
    }
    print("\nP6 against its pre-registered predictions (whole run):")
    for kname, ok in verdicts.items():
        print("   %-40s %s" % (kname, "YES" if ok else "no"))

    print("\nNOT-REPLICATED cells, largest map |delta| first:")
    for r in sorted([r for r in whole if r["label"] == "NOT-REPLICATED"], key=lambda r: -abs(r["delta"]))[:15]:
        print("   %-9s %-12s %-18s t%d %-17s map %+7.1f%%  rep %+7.1f%% %s" % (
            r["arm"], r["model"], r["variant"][:18], r["threads"], r["transition"],
            r["delta"], r["rep_delta"] if r["rep_delta"] is not None else float("nan"), r["rep_verdict"]))

    path = os.path.join(EXPB, "analysis", "b_replication.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"rules": "b_design_registered.md §5.5, §6 P6", "registered_prefix": list(REGISTERED_PREFIX),
               "p6_whole_run": {"replicated": k, "moved": n, "share": s, "band_ge20": [kb, nb, sb],
                                "band_7_20": [ks, ns, ss], "checks": verdicts},
               "pair_cells": out}, open(path, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
