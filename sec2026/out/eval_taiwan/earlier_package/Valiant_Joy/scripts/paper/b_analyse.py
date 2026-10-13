"""Experiment B analysis: turn the fifteen batch CSVs into the registered verdicts.

Binding protocol: results/paper/b_design_registered.md. This file implements the ANALYSIS side of it
and computes no measurement. Every rule below carries the section it comes from; nothing here is
invented, and where the registration is silent the gap is printed rather than filled.

  4.1  a cell is (arm, model, variant, threads, release); its quantity is the median of five
       session p50 values, and a cell with anything other than five OK sessions is VOID, as is every
       pair-cell touching it.
  4.2  MOVED iff |delta| >= 7 % AND the five-session ranges are disjoint; STILL iff |delta| < 7 %;
       UNRESOLVED iff |delta| >= 7 % with overlapping ranges.
  4.3  every share is a bracket: MOVED/N to (MOVED+UNRESOLVED)/N.
  4.4  persistence of a MOVED pair-cell: PERSISTENT / REVERTED / MIXED / NOT-ASSESSABLE.
  4.5  two consecutive MOVED pair-cells of opposite sign in one series are one SPIKE event.
  2.2  adjacent transitions are primary; baseline-anchored against 1.23.2 is secondary.
  2.3  the primary denominator is the EIGHT single-minor transitions; the two gapped ones
       (1.16.3 -> 1.20.1, 1.28.0 -> 1.30.0) are reported separately and never pooled.
  3.1  BREADTH rolls up to the architecture family: MOVED if any file moved, else UNRESOLVED if any
       is unresolved, else STILL.

    .venv/Scripts/python.exe scripts/paper/b_analyse.py [--out results/expB/analysis]

Safe to run while a batch is being measured: csv and json only, no numpy, no matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPB = os.path.join(ROOT, "results", "expB")
FLOOR = 7.0
BASELINE = "1.23.2"
GAPPED = {("1.16.3", "1.20.1"), ("1.28.0", "1.30.0")}
SESSIONS_PER_CELL = 5


def ladder() -> list[str]:
    plan = json.load(open(os.path.join(EXPB, "batch_plan.json"), encoding="utf-8"))
    return [str(r) for r in plan["releases"]]


def arch_families() -> dict[str, str]:
    """variant -> architecture family, from the A2 manifest (3.1: 15 families over 18 files)."""
    p = os.path.join(ROOT, "results", "expA2", "manifest.json")
    if not os.path.exists(p):
        return {}
    man = json.load(open(p, encoding="utf-8"))
    return {r["variant"]: (r.get("architecture_family") or r["variant"])
            for r in (man.get("measured_set") or [])}


def load_rows(lad: list[str]) -> tuple[list[dict], list[str]]:
    # A failed session never reached the runtime, so the harness writes no `ort` for it. The
    # interpreter directory is always there, and it names the release: without this, every failed
    # cell of one series collapses into a single release-less bucket and V8's "listed by name"
    # cannot be honoured.
    by_env = {"ort" + r.replace(".", ""): r for r in lad}
    rows, files = [], sorted(glob.glob(os.path.join(EXPB, "batches", "b[0-9][0-9]_*.csv")))
    for p in files:
        m = re.match(r"b(\d+)_(\w+?)_(.+)\.csv$", os.path.basename(p))
        arm = (m.group(2) if m else "unknown").upper()
        for r in csv.DictReader(open(p, encoding="utf-8")):
            r["arm"], r["batch_file"] = arm, os.path.basename(p)
            if not (r.get("ort") or "").strip():
                r["ort"] = by_env.get((r.get("env_dir") or "").strip(), "?")
            rows.append(r)
    return rows, [os.path.basename(p) for p in files]


def classify_failure(details: list[str]) -> str:
    """Why a cell has no usable sessions. Kept coarse on purpose; the text is printed with it."""
    blob = " ".join(details).lower()
    if "not_implemented" in blob or "could not find an implementation" in blob:
        return "NOT-APPLICABLE (kernel missing on this release)"
    if "multiarray" in blob or "numpy" in blob:
        return "ENVIRONMENT (numpy ABI against this runtime build)"
    if "timeout" in blob:
        return "TIMEOUT"
    return "OTHER"


def build_cells(rows: list[dict]) -> tuple[dict, dict]:
    by = defaultdict(list)
    for r in rows:
        key = (r["arm"], r["family"], r["variant"], int(r["threads"]), r["ort"] or "?")
        by[key].append(r)
    cells, dead = {}, {}
    for key, rs in by.items():
        ok = [x for x in rs if (x.get("status") or "OK") == "OK"]
        if len(ok) == SESSIONS_PER_CELL:
            v = sorted(float(x["p50_ms"]) for x in ok)
            cells[key] = {"m": st.median(v), "lo": v[0], "hi": v[-1], "n": len(v),
                          "delivery": st.median(float(x["core_delivery"]) for x in ok
                                                if x.get("core_delivery"))}
        elif not ok:
            dead[key] = {"reason": classify_failure([x.get("detail") or "" for x in rs]),
                         "n_rows": len(rs)}
        else:
            dead[key] = {"reason": "VOID (%d OK sessions, the rule needs exactly %d)"
                                   % (len(ok), SESSIONS_PER_CELL), "n_rows": len(rs)}
    return cells, dead


def verdict(a: dict, b: dict) -> tuple[str, float]:
    d = 100.0 * (b["m"] / a["m"] - 1.0)
    if abs(d) < FLOOR:
        return "STILL", d
    disjoint = b["lo"] > a["hi"] or b["hi"] < a["lo"]
    return ("MOVED" if disjoint else "UNRESOLVED"), d


def pair_cells(cells: dict, lad: list[str]) -> list[dict]:
    """Adjacent pairs on the ladder, plus every release against the 1.23.2 baseline (2.2)."""
    series = defaultdict(dict)
    for (arm, fam, var, th, rel), c in cells.items():
        series[(arm, fam, var, th)][rel] = c
    out = []
    for skey, byrel in series.items():
        for i in range(len(lad) - 1):
            a, b = lad[i], lad[i + 1]
            if a in byrel and b in byrel:
                v, d = verdict(byrel[a], byrel[b])
                out.append({"series": skey, "kind": "adjacent", "a": a, "b": b,
                            "single_minor": (a, b) not in GAPPED, "verdict": v, "delta": d,
                            "m_a": byrel[a]["m"], "m_b": byrel[b]["m"]})
        if BASELINE in byrel:
            for rel, c in byrel.items():
                if rel == BASELINE:
                    continue
                v, d = verdict(byrel[BASELINE], c)
                out.append({"series": skey, "kind": "baseline", "a": BASELINE, "b": rel,
                            "single_minor": False, "verdict": v, "delta": d,
                            "m_a": byrel[BASELINE]["m"], "m_b": c["m"]})
    return out


def spikes(pcs: list[dict], lad: list[str]) -> list[dict]:
    """4.5: consecutive MOVED pair-cells of opposite sign inside one series are ONE event."""
    idx = {r: i for i, r in enumerate(lad)}
    by = defaultdict(list)
    for p in pcs:
        if p["kind"] == "adjacent" and p["verdict"] == "MOVED":
            by[p["series"]].append(p)
    found = []
    for skey, ps in by.items():
        ps.sort(key=lambda p: idx[p["a"]])
        for x, y in zip(ps, ps[1:]):
            if y["a"] == x["b"] and (x["delta"] > 0) != (y["delta"] > 0):
                found.append({"series": skey, "at": x["b"],
                              "in": x["delta"], "out": y["delta"]})
    return found


def persistence(pcs: list[dict], cells: dict, lad: list[str]) -> Counter:
    """4.4, for MOVED adjacent pair-cells: does the file ever come back inside the floor?"""
    idx = {r: i for i, r in enumerate(lad)}
    byseries = defaultdict(dict)
    for (arm, fam, var, th, rel), c in cells.items():
        byseries[(arm, fam, var, th)][rel] = c
    counts = Counter()
    for p in pcs:
        if p["kind"] != "adjacent" or p["verdict"] != "MOVED":
            continue
        byrel = byseries[p["series"]]
        later = [r for r in lad[idx[p["b"]] + 1:] if r in byrel]
        if len(later) < 2:
            counts["NOT-ASSESSABLE"] += 1
            continue
        s = 1 if p["delta"] > 0 else -1
        rel_to_a = [(100.0 * (byrel[r]["m"] / p["m_a"] - 1.0)) for r in later]
        if all((1 if d > 0 else -1) == s and abs(d) >= FLOOR for d in rel_to_a):
            counts["PERSISTENT"] += 1
        elif abs(rel_to_a[-1]) < FLOOR and any(abs(d) < FLOOR for d in rel_to_a):
            counts["REVERTED"] += 1
        else:
            counts["MIXED"] += 1
    return counts


def family_rollup(pcs: list[dict], fams: dict) -> list[dict]:
    """3.1: a family moved at (threads, transition) if ANY of its files moved there."""
    order = {"MOVED": 2, "UNRESOLVED": 1, "STILL": 0}
    by = defaultdict(list)
    for p in pcs:
        arm, famkey, var, th = p["series"]
        if arm != "BREADTH" or p["kind"] != "adjacent":
            continue
        by[(fams.get(var, var), th, p["a"], p["b"], p["single_minor"])].append(p)
    out = []
    for (fam, th, a, b, single), ps in by.items():
        best = max(ps, key=lambda p: (order[p["verdict"]], abs(p["delta"])))
        out.append({"family": fam, "threads": th, "a": a, "b": b, "single_minor": single,
                    "verdict": best["verdict"], "delta": best["delta"], "n_files": len(ps)})
    return out


def bracket(items, key=lambda x: x["verdict"]) -> str:
    n = len(items)
    if not n:
        return "no cells"
    c = Counter(key(x) for x in items)
    lo = 100.0 * c["MOVED"] / n
    hi = 100.0 * (c["MOVED"] + c["UNRESOLVED"]) / n
    return ("%5.1f%% to %5.1f%%   (MOVED %d, UNRESOLVED %d, STILL %d, N=%d)"
            % (lo, hi, c["MOVED"], c["UNRESOLVED"], c["STILL"], n))


def sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seal_moves(pcs: list[dict], files: list[str]) -> str:
    """Write the replication stage's input: every adjacent MOVED / UNRESOLVED pair-cell (5.5).

    `b_run.py --stage replicate` reads results/expB/moves_sealed.json and needs one record per
    pair-cell with model / variant / threads / release_a / release_b / verdict / abs_delta
    (b_run.py:970-990). The list is hash-stamped against the CSVs it was computed from, so a
    replication run can always be tied to the exact map it was derived from.
    """
    want = [p for p in pcs if p["kind"] == "adjacent" and p["verdict"] in ("MOVED", "UNRESOLVED")]
    recs = []
    for p in want:
        arm, fam, var, th = p["series"]
        recs.append({"arm": arm, "model": fam, "variant": var, "threads": int(th),
                     "release_a": p["a"], "release_b": p["b"],
                     "transition": "%s->%s" % (p["a"], p["b"]),
                     "verdict": p["verdict"], "delta": round(p["delta"], 3),
                     "abs_delta": round(abs(p["delta"]), 3),
                     "single_minor": p["single_minor"],
                     "m_a": round(p["m_a"], 4), "m_b": round(p["m_b"], 4)})
    recs.sort(key=lambda r: (-r["abs_delta"], r["arm"], r["model"], r["variant"], r["threads"]))
    src = {f: sha256(os.path.join(EXPB, "batches", f))[:16] for f in files}
    out = {"experiment": "B", "stage": "5.5 replication input",
           "note": "Every adjacent MOVED and UNRESOLVED pair-cell of the map. Baseline-anchored "
                   "pairs are not listed separately: they share their cells with the adjacent ones, "
                   "and the replication invocation is per model over the union of releases.",
           "n_pair_cells": len(recs),
           "n_moved": sum(1 for r in recs if r["verdict"] == "MOVED"),
           "n_unresolved": sum(1 for r in recs if r["verdict"] == "UNRESOLVED"),
           "analyst_script_sha256": sha256(os.path.abspath(__file__))[:16],
           "source_csv_sha256": src, "pair_cells": recs}
    path = os.path.join(EXPB, "moves_sealed.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(EXPB, "analysis"))
    ap.add_argument("--seal", action="store_true",
                    help="also write results/expB/moves_sealed.json for --stage replicate")
    a = ap.parse_args()
    lad = ladder()
    fams = arch_families()
    rows, files = load_rows(lad)
    if not rows:
        print("no batch CSV yet")
        return 1
    cells, dead = build_cells(rows)
    pcs = pair_cells(cells, lad)
    adj = [p for p in pcs if p["kind"] == "adjacent"]
    single = [p for p in adj if p["single_minor"]]
    gap = [p for p in adj if not p["single_minor"]]
    base = [p for p in pcs if p["kind"] == "baseline"]
    fam_rows = family_rollup(pcs, fams)
    fam_single = [f for f in fam_rows if f["single_minor"]]

    print("=" * 96)
    print("EXPERIMENT B - PARTIAL ANALYSIS" if len(files) < 15 else "EXPERIMENT B - ANALYSIS")
    print("batches on disk: %d of 15 (%s)" % (len(files), ", ".join(files)))
    print("=" * 96)
    print("cells with exactly %d OK sessions: %d" % (SESSIONS_PER_CELL, len(cells)))
    reasons = Counter(d["reason"] for d in dead.values())
    for r, n in reasons.most_common():
        print("   %-58s %4d cells" % (r, n))
    rel_missing = Counter(k[4] for k in dead)
    if rel_missing:
        print("   unusable cells by release: " + ", ".join("%s %d" % kv for kv in sorted(rel_missing.items())))
    print()
    print("-- 4.2 verdicts, pair-cell unit " + "-" * 62)
    print("adjacent, SINGLE-MINOR (the primary denominator, 2.3): " + bracket(single))
    print("adjacent, GAPPED (reported separately, never pooled): " + bracket(gap))
    print("baseline-anchored against %s (secondary, 2.2):        %s" % (BASELINE, bracket(base)))
    print()
    for arm in sorted(set(p["series"][0] for p in single)):
        print("   %-10s %s" % (arm, bracket([p for p in single if p["series"][0] == arm])))
    print()
    if fam_single:
        print("-- 3.1 BREADTH rolled up to the architecture family " + "-" * 45)
        print("family x threads x single-minor transition:            " + bracket(fam_single))
        print()
    print("-- 4.5 spikes and 4.4 persistence " + "-" * 62)
    sp = spikes(pcs, lad)
    print("spikes (one aberrant release, counted as one event): %d" % len(sp))
    for s in sp[:12]:
        print("   %-42s at %-8s in %+6.1f%% out %+6.1f%%" % ("/".join(map(str, s["series"])),
                                                             s["at"], s["in"], s["out"]))
    pers = persistence(pcs, cells, lad)
    print("persistence of MOVED adjacent pair-cells: " + (", ".join("%s %d" % kv for kv in pers.most_common()) or "none"))
    print()
    print("-- the largest single-minor moves " + "-" * 62)
    for p in sorted(single, key=lambda p: -abs(p["delta"]))[:14]:
        arm, fam, var, th = p["series"]
        print("   %-9s %-14s %-18s t%-2d %-8s -> %-8s %+7.1f%%  %8.3f -> %8.3f ms  %s"
              % (arm, fam, var, th, p["a"], p["b"], p["delta"], p["m_a"], p["m_b"], p["verdict"]))

    os.makedirs(a.out, exist_ok=True)
    out = {"batches": files, "n_cells": len(cells),
           "unusable_cells": {"%s|%s|%s|t%d|%s" % k: v for k, v in dead.items()},
           "counts": {"single_minor": Counter(p["verdict"] for p in single),
                      "gapped": Counter(p["verdict"] for p in gap),
                      "baseline": Counter(p["verdict"] for p in base),
                      "family_single_minor": Counter(f["verdict"] for f in fam_single),
                      "persistence": pers},
           "spikes": [{**s, "series": "/".join(map(str, s["series"]))} for s in sp],
           "pair_cells": [{**p, "series": "/".join(map(str, p["series"]))} for p in pcs]}
    path = os.path.join(a.out, "b_analysis.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=lambda o: dict(o))
    print()
    print("wrote %s" % path)
    if a.seal:
        sealed = seal_moves(pcs, files)
        n = sum(1 for p in pcs if p["kind"] == "adjacent" and p["verdict"] in ("MOVED", "UNRESOLVED"))
        print("sealed %d MOVED/UNRESOLVED adjacent pair-cells -> %s" % (n, sealed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
