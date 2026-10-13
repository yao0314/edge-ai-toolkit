"""Experiment 27b analysis, written by the independent referee BEFORE any 27b timing.

    python probe_followup_analyse.py <exp27 csv> <exp27b csv> [--out <json>]

Part A = the registered Exp 27 V3 clause ("rerun those cells"), applied unchanged: the 27b sessions
of every Exp 27 decisive cell with > 1/3 of its sessions below the 0.95 gate stand in for the excluded
ones. Part B = the 27b verdict: Exp 27 sections 4.2-4.5 unchanged, 4T from 27b, 1T from Exp 27, with
a class-appropriate validity check (see ruling). NaN never passes a comparison silently here.
"""
import argparse
import itertools
import json
import math
import sys

import numpy as np
import pandas as pd

S, FLOOR, GATE, Z_HI = 0.48, 0.07, 0.95, 1.5
SMALL, LOC, REF = FLOOR * S, 0.70 * S, 0.15 * S
R0, R1 = "1.23.2", "1.24.4"
VARIANTS = ["tp_u8_all", "tp_u8_w2", "tp_u8_rest", "tp_u8_ctrl",
            "tp_f32_all", "tp_f32_ctrl", "tp_f16_w2", "tp_f32_w4"]
# Class Z = predicted 0 thread-pool dispatches per inference at 4T (manifest.json, fixed 13:35 on
# 2026-09-19, before any timing). Every other 4T cell is class P.
Z = {("tp_u8_ctrl", R0), ("tp_u8_ctrl", R1), ("tp_f32_ctrl", R0), ("tp_f32_ctrl", R1),
     ("tp_u8_w2", R0), ("tp_f16_w2", R0)}


def load(p):
    d = pd.read_csv(p)
    d["ort"] = d["ort"].astype(str)
    d = d[d["variant"].isin(VARIANTS) & d["ort"].isin([R0, R1])].copy()
    d["busy"] = d["cpu_ms_per_inf"] / d["p50_ms"]
    return d


def class_ok(r):
    if int(r["threads"]) == 4 and (r["variant"], r["ort"]) in Z:
        return GATE <= r["busy"] < Z_HI
    return r["core_delivery"] >= GATE


def band(x):
    return "LOCALISED" if x >= LOC else "PARTIAL" if x >= REF else "REFUTED"


def verdict(cells):
    """cells: {(variant, release, threads): [p50, ...]}. Exp 27 sections 4.2-4.5, unchanged."""
    need = [(v, r, t) for v in VARIANTS for r in (R0, R1) for t in (1, 4)]
    miss = [k for k in need if not cells.get(k)]
    if miss:
        return {"P": f"INCONCLUSIVE (missing cells: {miss})", "M": "not evaluated"}
    m = lambda v, r, t: float(np.median(cells[(v, r, t)]))
    D = lambda v, t: m(v, R1, t) - m(v, R0, t)

    def contrast(v, c, t):
        ks = [(v, R1, t), (v, R0, t), (c, R1, t), (c, R0, t)]
        vals = [a - b - (x - y) for a, b, x, y in itertools.product(*[cells[k] for k in ks])]
        return D(v, t) - D(c, t), min(vals), max(vals)

    v2 = all(abs(D(c, t)) <= max(FLOOR * m(c, R0, t), SMALL)
             for c in ("tp_u8_ctrl", "tp_f32_ctrl") for t in (1, 4))
    du8, lo, hi = contrast("tp_u8_all", "tp_u8_ctrl", 4)
    du8_1 = contrast("tp_u8_all", "tp_u8_ctrl", 1)[0]
    df32 = contrast("tp_f32_all", "tp_f32_ctrl", 4)[0]
    floor_ok = D("tp_u8_all", 4) > FLOOR * m("tp_u8_all", R0, 4)
    out = {"V2": v2, "delta_u8_4T": du8, "envelope": [lo, hi], "delta_u8_1T": du8_1,
           "delta_f32_4T": df32, "D_u8_all_4T": D("tp_u8_all", 4), "u8_all_above_floor": floor_ok,
           "D_ctrl_4T": {c: D(c, 4) for c in ("tp_u8_ctrl", "tp_f32_ctrl")},
           "share_of_0.48": du8 / S}
    if not v2:
        p = "INCONCLUSIVE (V2 control drift)"
    elif not floor_ok or du8 < REF:
        p = "REFUTED"
    elif du8_1 >= SMALL:
        p = "INCONCLUSIVE (steps at 1 thread)"
    elif df32 >= SMALL:
        p = "INCONCLUSIVE (float Transposes step too)"
    else:
        p = band(du8)
    firm = band(lo) == band(hi) == band(du8)
    out["firm"] = firm
    if p in ("LOCALISED", "PARTIAL", "REFUTED") and not firm:
        lower = min(band(lo), band(du8), key=["REFUTED", "PARTIAL", "LOCALISED"].index)
        out["paper_band"] = lower
        p += f" (NOT FIRM: envelope {lo:+.3f}..{hi:+.3f} crosses a band edge; paper uses {lower} wording)"
    if p.startswith("LOCALISED") and du8 > 1.5 * S:
        p += " [exceeds the step: 'reproduces', no share above 100%]"
    out["P"] = p
    mech = "not evaluated"
    if p.startswith(("LOCALISED", "PARTIAL")):
        dw2 = contrast("tp_u8_w2", "tp_u8_ctrl", 4)[0]
        dw2_1 = contrast("tp_u8_w2", "tp_u8_ctrl", 1)[0]
        drest = contrast("tp_u8_rest", "tp_u8_ctrl", 4)[0]
        df16, dw4 = D("tp_f16_w2", 4), D("tp_f32_w4", 4)
        cost = (m("tp_f32_w4", R0, 4) - m("tp_f32_w4", R0, 1)) / 832.0
        out.update({"delta_w2_4T": dw2, "delta_w2_1T": dw2_1, "delta_rest_4T": drest,
                    "D_f16_w2_4T": df16, "D_f32_w4_4T": dw4, "dispatch_cost_pred_ms": 832 * cost,
                    "additivity": D("tp_u8_w2", 4) + D("tp_u8_rest", 4) - D("tp_u8_ctrl", 4)
                    - D("tp_u8_all", 4)})
        if dw2 < 0.30 * du8 or abs(dw4) >= 0.5 * dw2 or df16 < SMALL:
            mech = "M-REFUTED"
        elif (dw2 >= 0.70 * du8 and drest < 0.30 * du8 and df16 >= 0.50 * dw2
              and abs(dw4) < SMALL and dw2_1 < SMALL):
            mech = "M-CONFIRMED"
        else:
            mech = "M-MIXED"
    out["M"] = mech
    return out


def cells_of(d):
    return {k: g["p50_ms"].tolist() for k, g in d.groupby(["variant", "ort", "threads"])}


def word(v):
    return v["P"].split(" (")[0].split(" [")[0], v["M"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp27")
    ap.add_argument("exp27b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    e27, fu = load(a.exp27), load(a.exp27b)
    res = {}

    # ---------- Part A: registered Exp 27 V3 rerun (0.95 gate, unchanged)
    e27["below"] = e27["core_delivery"] < GATE
    frac = e27.groupby(["variant", "ort", "threads"])["below"].mean()
    rerun = sorted(k for k, f in frac.items() if f > 1 / 3)
    un27 = verdict(cells_of(e27))
    fu_r = fu.set_index(["variant", "ort", "threads"]).sort_index()
    still = {}
    for k in rerun:
        s = fu_r.loc[[k]] if k in fu_r.index else fu_r.iloc[0:0]
        still[str(k)] = {"n": int(len(s)), "below_gate": int((s["core_delivery"] < GATE).sum()),
                         "cd": [round(x, 3) for x in s["core_delivery"]]}
    cleared = all(v["n"] >= 2 and v["below_gate"] <= v["n"] / 3 for v in still.values())
    if cleared:
        g = e27[~e27["below"]]
        g = g[~g.set_index(["variant", "ort", "threads"]).index.isin(rerun)]
        add = fu[(fu["core_delivery"] >= GATE)
                 & fu.set_index(["variant", "ort", "threads"]).index.isin(rerun)]
        gv = verdict(cells_of(pd.concat([g, add])))
        a_final = un27["P"] if word(gv) == word(un27) else "INCONCLUSIVE (V3: gated and ungated differ after rerun)"
        a_mech = un27["M"] if word(gv) == word(un27) else None
    else:
        gv, a_final, a_mech = None, "INCONCLUSIVE (V3: rerun cells again below the 0.95 gate)", None
    res["A_exp27_registered"] = {"rerun_cells": [str(k) for k in rerun], "rerun_sessions": still,
                                 "cleared": cleared, "gated_after_rerun": gv,
                                 "final": a_final, "mechanism": a_mech}

    # ---------- Part B: Exp 27b
    fu4 = fu[fu["threads"] == 4].copy()
    fu4["class"] = ["Z" if (v, r) in Z else "P" for v, r in zip(fu4["variant"], fu4["ort"])]
    fu4["valid"] = fu4.apply(class_ok, axis=1)
    one_t = cells_of(e27[e27["threads"] == 1])          # Exp 27 1T: 48/48 passed the gate
    ung = {**one_t, **cells_of(fu4)}
    gat = {**one_t, **cells_of(fu4[fu4["valid"]])}
    fail = fu4.groupby(["variant", "ort"])["valid"].apply(lambda s: int((~s).sum())).to_dict()
    nper = fu4.groupby(["variant", "ort"]).size().to_dict()
    v3_i = all(n == 3 for n in nper.values()) and len(nper) == 16 and all(f <= 1 for f in fail.values())
    vu, vg = verdict(ung), verdict(gat)
    v3_ii = word(vu) == word(vg)
    if not v3_i:
        final = "INCONCLUSIVE (V3b: a cell has > 1/3 sessions failing its class check, or cells missing)"
    elif not v3_ii:
        final = "INCONCLUSIVE (V3b: class-gated and ungated verdicts differ)"
    else:
        final = vu["P"]
    zc, pc = fu4[fu4["class"] == "Z"], fu4[fu4["class"] == "P"]
    struct = bool(len(zc) == 18 and (zc["core_delivery"] < 0.5).all()
                  and ((zc["busy"] >= GATE) & (zc["busy"] < Z_HI)).all()
                  and len(pc) == 30 and (pc["core_delivery"] >= GATE).sum() >= 28)
    res["B_exp27b"] = {"ungated": vu, "class_gated": vg,
                       "failing_sessions_per_cell": {f"{k[0]}@{k[1]}": v for k, v in fail.items() if v},
                       "V3b_i": v3_i, "V3b_ii": v3_ii, "final": final,
                       "mechanism": vu["M"] if final == vu["P"] else None,
                       "Z_cd_range": [float(zc["core_delivery"].min()), float(zc["core_delivery"].max())] if len(zc) else None,
                       "Z_busy_range": [float(zc["busy"].min()), float(zc["busy"].max())] if len(zc) else None,
                       "P_cd_range": [float(pc["core_delivery"].min()), float(pc["core_delivery"].max())] if len(pc) else None,
                       "structural_prediction_held": struct}
    txt = json.dumps(res, indent=1, default=float)
    print(txt)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
