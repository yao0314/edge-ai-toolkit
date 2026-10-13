"""Experiment E2: apply the registered decision rule to results/paper/e2_counterfactual.csv.

The rule is fixed in <scratchpad>/e2_design.md and in FINDINGS.md BEFORE any timing. This script only
executes it; it contains no judgement of its own and no timing.

    python scripts/paper/e2_analyse.py results/paper/e2_counterfactual.csv

Writes results/e2_counterfactual/verdict.json and prints the verdict and every number behind it.
A missing cell, a missing session or a NaN never passes a check silently: it makes the run
INCONCLUSIVE.

Revision 2 (2026-09-20, after independent review, still BEFORE any timing). Changes, all registered
in e2_design.md section 5 before the run:
  * V4 is the absolute 0.034 ms bound registered in Exp 27/27b, not an own-floor bound, and it gains
    V4b: the estimand's own numerator measured at ONE thread must be null.
  * V5 (the rewrite's own cost on 1.23.2) is REPORTED, not voiding - it cancels out of f. The gating
    part is V5b, thread-consistency of that cost.
  * V3 is two-sided: a control that AMPLIFIES the step is as disqualifying as one that loses it.
  * FIRM also requires the envelope of D(e2,4T), not only the envelope of f.
  * V7a gates the eight FOUR-THREAD cells only; at one thread the same ratio is reported, not gated.
  * --family (default e2) and --replace-cell stop foreign rows and re-run rows from being pooled.
  * Envelope combinations with a non-positive denominator are skipped and counted, never divided by.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "e2_counterfactual"
MODEL_DIR = OUT_DIR / "models"

ORIG, E2, SHAM, PLAC = "p00_int8", "p00_int8_e2", "p00_int8_sham", "p00_int8_plac"
VARIANTS = [ORIG, E2, SHAM, PLAC]
OLD, NEW = "1.23.2", "1.24.4"

# ---- registered constants -------------------------------------------------------------------
S_MS = 0.48            # the step this experiment is about (2.037 - 1.553 ms at 4T, 31%)
FLOOR_REL = 0.07       # the study's between-session latency floor
SMALL_MS = 0.07 * S_MS  # 0.0336 ms - the ABSOLUTE "small" of Exp 27 section 4.5
F_CONFIRM = 0.70       # share of the step the rewrite must remove (the 23c / Exp 27 threshold)
F_PARTIAL = 0.30       # below this the rewrite removed essentially nothing
CTRL_KEEP = 0.70       # share of the step the sham and the placebo must KEEP
CTRL_KEEP_HI = 1.0 / CTRL_KEEP   # 1.4286 - and may not AMPLIFY it beyond this
REWRITE_COST_REL = FLOOR_REL     # |R(e2,T)| reported against 0.07 * m(orig,1.23.2,T)
MIN_SESSIONS = 5
GATE = 0.95


def _cells(df):
    out = {}
    for (v, r, t), g in df.groupby(["variant", "ort", "threads"]):
        out[(v, r, int(t))] = g.sort_values("rep")
    return out


def _med(cells, key):
    g = cells.get(key)
    if g is None or len(g) == 0 or g["p50_ms"].isna().any():
        return None
    return float(g["p50_ms"].median())


def _envelope(cells, fn, keys, pct=(5, 95)):
    """min/max (and 5-95 percentile, reported only) of fn over every choice of one session p50 per
    cell. fn returns None for a combination whose denominator is not positive; those are skipped
    and counted, never divided by."""
    pools = []
    for k in keys:
        g = cells.get(k)
        if g is None or len(g) == 0:
            return None
        pools.append(list(g["p50_ms"]))
    vals, skipped = [], 0
    for c in itertools.product(*pools):
        v = fn(*c)
        if v is None:
            skipped += 1
        else:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    lo_i = max(0, min(n - 1, int(round((pct[0] / 100.0) * (n - 1)))))
    hi_i = max(0, min(n - 1, int(round((pct[1] / 100.0) * (n - 1)))))
    return {"min": vals[0], "max": vals[-1], "p05": vals[lo_i], "p95": vals[hi_i],
            "n": n, "skipped_nonpositive_denominator": skipped}


def _ratio(num_a, num_b, den_a, den_b):
    """(num_b - num_a) / (den_b - den_a) with the denominator guarded."""
    den = den_b - den_a
    if den <= 0:
        return None
    return (num_b - num_a) / den


def analyse(csv_paths, gate=GATE, out_path=None, step=S_MS, family="e2", replace=()):
    frames = []
    file_rows = []
    for p in csv_paths:
        d = pd.read_csv(p)
        d["_src"] = str(p)
        file_rows.append({"csv": str(p), "rows": int(len(d))})
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if family and family != "all" and "family" in df.columns:
        df = df[df["family"] == family]
    df = df[df["variant"].isin(VARIANTS) & df["ort"].isin([OLD, NEW])]

    # a re-run cell is REPLACED, never pooled: only rows from the LAST csv given survive for it
    replaced = []
    if replace:
        last = str(csv_paths[-1])
        for spec in replace:
            v, r, t = spec.split(",")
            t = int(t)
            mask = ((df["variant"] == v) & (df["ort"] == r) & (df["threads"] == t)
                    & (df["_src"] != last))
            replaced.append({"cell": f"{v}|{r}|{t}", "dropped_rows": int(mask.sum()),
                             "kept_from": last})
            df = df[~mask]

    rep = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "csv": [str(p) for p in csv_paths], "files": file_rows, "family_filter": family,
           "replaced_cells": replaced, "rows_used": int(len(df)), "gate": gate, "S_ms": step,
           "registered": {"F_CONFIRM": F_CONFIRM, "F_PARTIAL": F_PARTIAL,
                          "CTRL_KEEP": CTRL_KEEP, "CTRL_KEEP_HI": CTRL_KEEP_HI,
                          "FLOOR_REL": FLOOR_REL, "SMALL_MS": SMALL_MS,
                          "REWRITE_COST_REL": REWRITE_COST_REL,
                          "MIN_SESSIONS": MIN_SESSIONS, "GATE": gate,
                          "rule": "e2_design.md section 5, revision 2"}}

    # ---- V1: the pre-timing checks, and the files that were timed -----------------------------
    v1 = {"manifest": False, "sha256_match": False, "static_check": False,
          "output_check_bit_identical": False, "executed_check": False}
    man_p = OUT_DIR / "manifest.json"
    if man_p.exists():
        man = json.loads(man_p.read_text())
        v1["manifest"] = True
        v1["sha256_match"] = all(
            (MODEL_DIR / f"{k}.onnx").exists()
            and hashlib.sha256((MODEL_DIR / f"{k}.onnx").read_bytes()).hexdigest() == s
            for k, s in man["sha256"].items())
    sp = OUT_DIR / "static_check.json"
    if sp.exists():
        st = json.loads(sp.read_text())
        v1["static_check"] = all(x["node_census_identical_to_original"] and x["only_intended_changed"]
                                 for x in st.values())
    op = OUT_DIR / "output_check.json"
    if op.exists():
        oc = json.loads(op.read_text())
        v1["output_check_bit_identical"] = bool(oc) and all(x["bit_identical"] for x in oc.values())
    ep = OUT_DIR / "executed_check.json"
    if ep.exists():
        ec = json.loads(ep.read_text())
        ok = True
        for name, per in ec.items():
            ok &= bool(per.get("node_ids_identical_across_releases"))
            for v in (OLD, NEW):
                if v in per and name != ORIG:
                    ok &= bool(per[v]["only_intended_changed"]) and bool(per[v]["same_transpose_name_set"])
        v1["executed_check"] = ok
        # the dispatch predictions that make this a counterfactual at all
        try:
            rep["predicted_dispatches_4t"] = {
                n: {v: ec[n][v]["predicted_dispatches_this_release"] for v in (OLD, NEW)}
                for n in VARIANTS}
        except KeyError:
            pass
    v1["all"] = all(v1[k] for k in ("manifest", "sha256_match", "static_check",
                                    "output_check_bit_identical", "executed_check"))

    # ---- V6: machine precondition, recorded by the runner before timing ------------------------
    pre_p = OUT_DIR / "precondition.json"
    v6 = {"recorded": pre_p.exists()}
    if pre_p.exists():
        pre = json.loads(pre_p.read_text())
        v6.update(pre)
        v6["ok"] = bool(pre.get("ok"))
    else:
        v6["ok"] = False

    results = {}
    for arm in ("ungated", "gated"):
        # the gate is a CORE-DELIVERY measure, so it excludes only 4-thread sessions; at 1 thread
        # cpu_ms/(1*p50) is a CPU-accounting ratio and is reported, not gated (V7 in the design).
        d = df if arm == "ungated" else df[(df["threads"] == 1) | (df["core_delivery"] >= gate)]
        cells = _cells(d)
        full = _cells(df)

        def med(v, r, t):
            return _med(cells, (v, r, t))

        q = {}
        missing = []
        for v in VARIANTS:
            for t in (1, 4):
                for r in (OLD, NEW):
                    g = cells.get((v, r, t))
                    n = 0 if g is None else len(g)
                    q[f"n|{v}|{r}|{t}"] = n
                    if n < MIN_SESSIONS and arm == "ungated":
                        missing.append(f"{v}|{r}|{t}={n}")
                q[f"D|{v}|{t}"] = (None if med(v, NEW, t) is None or med(v, OLD, t) is None
                                   else med(v, NEW, t) - med(v, OLD, t))
            for r in (OLD, NEW):
                for t in (1, 4):
                    q[f"m|{v}|{r}|{t}"] = med(v, r, t)
        for t in (1, 4):
            for v in (E2, SHAM, PLAC):
                a, b = med(v, OLD, t), med(ORIG, OLD, t)
                q[f"R|{v}|{t}"] = None if a is None or b is None else a - b
            q[f"own_floor|orig|{t}"] = (None if med(ORIG, OLD, t) is None
                                        else FLOOR_REL * med(ORIG, OLD, t))
            q[f"own_floor|e2|{t}"] = (None if med(E2, OLD, t) is None
                                      else FLOOR_REL * med(E2, OLD, t))
            for v in VARIANTS:
                q[f"own_floor|{v}|{t}"] = (None if med(v, OLD, t) is None
                                           else FLOOR_REL * med(v, OLD, t))

        Do, De, Ds, Dp = (q[f"D|{v}|4"] for v in VARIANTS)
        have = all(x is not None for x in (Do, De, Ds, Dp)) and Do is not None and Do > 0
        q["f"] = None if not have else (Do - De) / Do
        q["keep_sham"] = None if not have else Ds / Do
        q["keep_plac"] = None if not have else Dp / Do
        q["step_removed_ms"] = None if not have else Do - De
        q["us_per_removed_dispatch"] = None if not have else (Do - De) * 1000.0 / 832.0
        # V4b: the estimand's numerator measured at ONE thread, where nothing dispatches on either
        # release, so it must be null.
        q["numerator_1T"] = (None if q["D|%s|1" % ORIG] is None or q["D|%s|1" % E2] is None
                             else q["D|%s|1" % ORIG] - q["D|%s|1" % E2])

        env = {}
        if have:
            env["f"] = _envelope(cells, lambda a, b, c, e: (None if (b - a) <= 0
                                                            else ((b - a) - (e - c)) / (b - a)),
                                 [(ORIG, OLD, 4), (ORIG, NEW, 4), (E2, OLD, 4), (E2, NEW, 4)])
            env["D_orig_4"] = _envelope(cells, lambda a, b: b - a, [(ORIG, OLD, 4), (ORIG, NEW, 4)])
            env["D_e2_4"] = _envelope(cells, lambda a, b: b - a, [(E2, OLD, 4), (E2, NEW, 4)])
            env["keep_sham"] = _envelope(cells, lambda a, b, c, e: _ratio(c, e, a, b),
                                         [(ORIG, OLD, 4), (ORIG, NEW, 4), (SHAM, OLD, 4), (SHAM, NEW, 4)])
            env["keep_plac"] = _envelope(cells, lambda a, b, c, e: _ratio(c, e, a, b),
                                         [(ORIG, OLD, 4), (ORIG, NEW, 4), (PLAC, OLD, 4), (PLAC, NEW, 4)])

        # ---- validity ------------------------------------------------------------------------
        val = {}
        val["V8_all_cells_have_%d_sessions" % MIN_SESSIONS] = (not missing) if arm == "ungated" else True
        val["V8_missing"] = missing
        val["V2_step_reproduced"] = (Do is not None and q["own_floor|orig|4"] is not None
                                     and Do >= F_CONFIRM * step and Do > q["own_floor|orig|4"])
        # V4a: ABSOLUTE, as registered in Exp 27 section 4.5 and 27b. The own-floor value is
        # reported beside it (q["own_floor|v|1"]) but is not the test.
        val["V4a_no_1thread_step"] = all(
            q[f"D|{v}|1"] is not None and abs(q[f"D|{v}|1"]) < SMALL_MS for v in VARIANTS)
        val["V4b_1thread_numerator_null"] = (q["numerator_1T"] is not None
                                             and abs(q["numerator_1T"]) < SMALL_MS)
        # V5: REPORTED, not voiding - R cancels out of f algebraically.
        v5_ok = all(q[f"R|{E2}|{t}"] is not None and q[f"own_floor|orig|{t}"] is not None
                    and abs(q[f"R|{E2}|{t}"]) <= q[f"own_floor|orig|{t}"] for t in (1, 4))
        val["V5_rewrite_cost_within_own_floor_REPORTED"] = v5_ok
        # V5b: GATING - the added serial cost is thread-independent by construction.
        r4, r1 = q[f"R|{E2}|4"], q[f"R|{E2}|1"]
        v5b_bound = max(SMALL_MS, q["own_floor|orig|4"] or 0.0)
        val["V5b_rewrite_cost_thread_consistent"] = (r4 is not None and r1 is not None
                                                     and abs(r4 - r1) <= v5b_bound)
        q["V5b_bound_ms"] = v5b_bound
        # V3: TWO-SIDED. A control that amplifies the step is as disqualifying as one that loses it.
        val["V3_sham_keeps_step"] = (q["keep_sham"] is not None
                                     and CTRL_KEEP <= q["keep_sham"] <= CTRL_KEEP_HI)
        val["V3_placebo_keeps_step"] = (q["keep_plac"] is not None
                                        and CTRL_KEEP <= q["keep_plac"] <= CTRL_KEEP_HI)

        # core delivery, computed on the FULL frame regardless of arm
        cd = {}
        worst4 = 0
        worst1 = 0
        for v in VARIANTS:
            for r in (OLD, NEW):
                for t in (1, 4):
                    g = full.get((v, r, t))
                    if g is None:
                        continue
                    below = int((g["core_delivery"] < gate).sum())
                    cd[f"{v}|{r}|{t}"] = {"n": len(g), "below_gate": below,
                                          "min": float(g["core_delivery"].min()),
                                          "max": float(g["core_delivery"].max()),
                                          "gating": t == 4}
                    if t == 4:
                        worst4 = max(worst4, below)
                    else:
                        worst1 = max(worst1, below)
        val["V7a_at_most_1_below_gate_per_4thread_cell"] = worst4 <= 1
        val["V7_worst_4thread_cell_below_gate"] = worst4
        val["V7_worst_1thread_cell_below_gate_REPORTED"] = worst1

        # ---- band ----------------------------------------------------------------------------
        f = q["f"]
        e2_floor = max(SMALL_MS, q["own_floor|e2|4"] or 0.0)
        q["e2_floor_ms"] = e2_floor
        e2_small = (De is not None and abs(De) < e2_floor)
        if None in (Do, De, Ds, Dp):
            band = "INCONCLUSIVE (missing cells)"
        elif not val["V2_step_reproduced"]:
            band = "INCONCLUSIVE (the 1.24.4 step did not reproduce in this run)"
        elif not (val["V3_sham_keeps_step"] and val["V3_placebo_keeps_step"]):
            band = "CONFOUNDED (a control did not keep the step: the edit itself, not the dispatch path)"
        elif not val["V4a_no_1thread_step"]:
            band = "INCONCLUSIVE (a 1-thread step appeared)"
        elif not val["V4b_1thread_numerator_null"]:
            band = "INCONCLUSIVE (the rewrite changed the release gap at 1 thread, where nothing dispatches)"
        elif not val["V5b_rewrite_cost_thread_consistent"]:
            band = "INCONCLUSIVE (the rewrite's own cost on 1.23.2 differs between 1 and 4 threads)"
        elif f >= F_CONFIRM and e2_small:
            band = "ROOT-CAUSE CONFIRMED"
        elif f >= F_CONFIRM and not e2_small:
            band = "PARTIAL (share above threshold but the rewritten file keeps a step of its own)"
        elif f >= F_PARTIAL:
            band = "PARTIAL"
        else:
            band = "REFUTED"

        # ---- firmness --------------------------------------------------------------------------
        firm, firm_why = None, []
        if env.get("f"):
            lo, hi = env["f"]["min"], env["f"]["max"]
            same_band = _band_of(lo) == _band_of(hi) == _band_of(f)
            if not same_band:
                firm_why.append(f"f envelope {lo:.3f}..{hi:.3f} crosses a band edge")
            e2_env_ok = True
            if band.startswith("ROOT-CAUSE CONFIRMED") and env.get("D_e2_4"):
                elo, ehi = env["D_e2_4"]["min"], env["D_e2_4"]["max"]
                e2_env_ok = max(abs(elo), abs(ehi)) < e2_floor
                if not e2_env_ok:
                    firm_why.append(
                        f"D(e2,4T) envelope {elo:.3f}..{ehi:.3f} ms leaves the {e2_floor:.3f} ms floor")
            firm = "FIRM" if (same_band and e2_env_ok) else "PROVISIONAL"
        results[arm] = {"quantities": q, "envelopes": env, "validity": val,
                        "core_delivery": cd, "band": band, "firmness": firm,
                        "firmness_why_not": firm_why,
                        "rewritten_step_within_floor": e2_small}

    # ---- final ---------------------------------------------------------------------------------
    u, g = results["ungated"], results["gated"]
    agree = u["band"] == g["band"]
    hard = (v1["all"] and v6["ok"] and u["validity"]["V8_all_cells_have_%d_sessions" % MIN_SESSIONS]
            and u["validity"]["V7a_at_most_1_below_gate_per_4thread_cell"] and agree)
    final = u["band"] if hard else "INCONCLUSIVE"
    reasons = []
    if not v1["all"]:
        reasons.append("V1 pre-timing verification did not pass")
    if not v6["ok"]:
        reasons.append("V6 machine precondition not recorded as passed")
    if not u["validity"]["V8_all_cells_have_%d_sessions" % MIN_SESSIONS]:
        reasons.append("V8 missing sessions: " + ", ".join(u["validity"]["V8_missing"]))
    if not u["validity"]["V7a_at_most_1_below_gate_per_4thread_cell"]:
        reasons.append("V7a more than one session of a 4-thread cell below the core-delivery gate")
    if not agree:
        reasons.append(f"V7b gated ({g['band']}) and ungated ({u['band']}) verdicts differ")
    caveats = []
    if not u["validity"]["V5_rewrite_cost_within_own_floor_REPORTED"]:
        caveats.append(
            "V5 (reported, not voiding): the rewrite's own cost on 1.23.2, R(e2,T) = "
            f"{u['quantities'].get('R|%s|4' % E2)} ms at 4T and "
            f"{u['quantities'].get('R|%s|1' % E2)} ms at 1T, is outside 7% of the original's "
            "1.23.2 median. R cancels out of f, so this is a caveat on the rewrite's size, not on "
            "the estimand.")
    if u["validity"]["V7_worst_1thread_cell_below_gate_REPORTED"] > 1:
        caveats.append(
            "A 1-thread cell has more than one session with cpu_ms/(1*p50) below 0.95 "
            "(possible preemption). Reported, not gating: at 1 thread this ratio is not a "
            "core-delivery measure, and a contaminated 1-thread cell can only void the run "
            "through V4, never create a verdict.")
    ext = (u["firmness"] == "PROVISIONAL")
    rep["validity_V1"] = v1
    rep["validity_V6"] = v6
    rep["arms"] = results
    rep["final"] = {"verdict": final, "firmness": u["firmness"],
                    "ungated_band": u["band"], "gated_band": g["band"],
                    "inconclusive_because": reasons, "caveats": caveats,
                    "firmness_why_not": u["firmness_why_not"],
                    "extension_run_triggered": bool(ext and hard),
                    "paper_wording_key": _wording_key(final, u["firmness"])}

    out = Path(out_path) if out_path else OUT_DIR / "verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    _print(rep)
    print(f"\nwrote {out}")
    return rep


def _band_of(f):
    if f is None:
        return None
    if f >= F_CONFIRM:
        return "confirm"
    if f >= F_PARTIAL:
        return "partial"
    return "refuted"


def _wording_key(final, firm):
    if final.startswith("ROOT-CAUSE CONFIRMED"):
        return "W1_root_cause" if firm == "FIRM" else "W2_partial"
    if final.startswith("PARTIAL"):
        return "W2_partial"
    if final.startswith("REFUTED"):
        return "W3_refuted"
    if final.startswith("CONFOUNDED"):
        return "W4_confounded"
    return "W5_inconclusive"


def _e(d, k):
    if not d:
        return "-"
    return f"[{d['min']:.3f}, {d['max']:.3f}] (p05-p95 {d['p05']:.3f}..{d['p95']:.3f})"


def _print(rep):
    for arm in ("ungated", "gated"):
        a = rep["arms"][arm]
        q = a["quantities"]
        print(f"\n===== {arm.upper()} =====")
        print(f"{'variant':16s} {'1.23.2 4T':>10s} {'1.24.4 4T':>10s} {'D(4T) ms':>9s} {'D(4T) %':>8s} "
              f"{'D(1T) ms':>9s}")
        for v in VARIANTS:
            m0, m1 = q.get(f"m|{v}|{OLD}|4"), q.get(f"m|{v}|{NEW}|4")
            d4, d1 = q.get(f"D|{v}|4"), q.get(f"D|{v}|1")
            pct = None if not m0 or d4 is None else 100 * d4 / m0
            print(f"{v:16s} {_f(m0):>10s} {_f(m1):>10s} {_f(d4):>9s} {_f(pct):>8s} {_f(d1):>9s}")
        print(f"  f (share of the step removed by the rewrite) = {_f(q.get('f'), 4)}"
              f"   envelope {_e(a['envelopes'].get('f'), 'f')}")
        print(f"  sham keeps {_f(q.get('keep_sham'), 3)} {_e(a['envelopes'].get('keep_sham'), 'k')}")
        print(f"  placebo keeps {_f(q.get('keep_plac'), 3)} {_e(a['envelopes'].get('keep_plac'), 'k')}")
        print(f"  D(e2,4T) {_f(q.get(f'D|{E2}|4'), 4)} ms, floor {_f(q.get('e2_floor_ms'), 4)} ms, "
              f"envelope {_e(a['envelopes'].get('D_e2_4'), 'd')}")
        print(f"  rewrite cost on 1.23.2 (reported): 4T {_f(q.get(f'R|{E2}|4'), 4)} ms, "
              f"1T {_f(q.get(f'R|{E2}|1'), 4)} ms, own floor 4T {_f(q.get('own_floor|orig|4'), 4)} ms")
        print(f"  1T numerator D(orig,1T)-D(e2,1T) = {_f(q.get('numerator_1T'), 4)} ms "
              f"(bound {SMALL_MS:.4f})")
        print(f"  step removed {_f(q.get('step_removed_ms'), 4)} ms "
              f"= {_f(q.get('us_per_removed_dispatch'), 3)} us per removed dispatch (832)")
        print(f"  band: {a['band']}  ({a['firmness']})")
        for w in a["firmness_why_not"]:
            print(f"    not firm: {w}")
        for k, v in a["validity"].items():
            if k.startswith("V"):
                print(f"    {k}: {v}")
    print(f"\nV1 pre-timing: {rep['validity_V1']}")
    print(f"V6 precondition: {rep['validity_V6'].get('ok')}")
    print(f"rows used {rep['rows_used']} from {rep['files']}; replaced {rep['replaced_cells']}")
    print(f"\nFINAL: {rep['final']['verdict']} ({rep['final']['firmness']}) "
          f"-> paper wording {rep['final']['paper_wording_key']}")
    for r in rep["final"]["inconclusive_because"]:
        print(f"   ! {r}")
    for c in rep["final"]["caveats"]:
        print(f"   ~ {c}")
    if rep["final"]["extension_run_triggered"]:
        print("   > PROVISIONAL: the registered extension run (section 5.6) is triggered.")


def _f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--step", type=float, default=S_MS)
    ap.add_argument("--family", default="e2",
                    help="keep only rows of this family (default e2; 'all' disables)")
    ap.add_argument("--replace-cell", action="append", default=[], metavar="VARIANT,ORT,THREADS",
                    help="for this cell keep only the rows of the LAST csv given (a V7a re-run); "
                         "repeatable")
    a = ap.parse_args()
    analyse([Path(c) for c in a.csv], gate=a.gate, out_path=a.out, step=a.step,
            family=a.family, replace=tuple(a.replace_cell))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
