"""Append the paper-analysis sheets to results/edge_ai_toolkit_report.xlsx.

make_report_xlsx.py builds the deliverable-oriented report (what was shipped and why). These extra
sheets hold the analysis the TAAI paper is built on, which is a different question: how the answer
moves when the accuracy budget moves, and whether pruning strategies actually differ.

Run make_report_xlsx.py first, then this.

    python scripts/paper/append_paper_sheets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from openpyxl import load_workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

XLSX = ROOT / "results" / "edge_ai_toolkit_report.xlsx"

HEAD = ["classification", "classification_deit", "classification_mobilevit", "anomaly",
        "detection", "detection_640", "stress_imagenette", "stress_resnet50"]
BUDGETS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
# Same base config, same ratio grid [0.0, 0.25], same 20-epoch finetune: a controlled comparison.
STRATEGIES = {
    "classification_prune_global": "global pruning",
    "classification_prune_long": "uniform L2 (baseline)",
    "classification_prune_alloc": "DP knapsack allocation",
    "classification_prune_bnscale": "BN-scale importance",
    "classification_prune_both": "BN-scale + global",
}


def _sweep(run: str) -> pd.DataFrame | None:
    f = ROOT / "results" / run / "sweep.csv"
    return pd.read_csv(f) if f.exists() else None


def budget_sweep() -> pd.DataFrame:
    """Best achievable compression as the accuracy budget moves, per setting."""
    rows = []
    for run in HEAD:
        d = _sweep(run)
        if d is None or "accuracy" not in d:
            continue
        b = d[d["variant"].astype(str).str.contains("fp32")].iloc[0]
        row = {"setting": run, "metric": d.get("metric_name", pd.Series(["?"])).iloc[0],
               "baseline": round(float(b["accuracy"]), 4)}
        prev = None
        for bud in BUDGETS:
            ok = d[(d["accuracy"] >= b["accuracy"] - bud / 100) & (d["variant"] != b["variant"])]
            if ok.empty:
                row[f"{bud}pt"] = None
                row[f"{bud}pt winner"] = "none eligible"
                continue
            cr = b["size_bytes"] / ok["size_bytes"]
            i = cr.idxmax()
            row[f"{bud}pt"] = round(float(cr.max()), 2)
            w = str(d.loc[i, "variant"])
            row[f"{bud}pt winner"] = w + ("  <-- CHANGED" if prev and w != prev else "")
            prev = w
        vals = [v for k, v in row.items() if k.endswith("pt") and v]
        row["max/min"] = round(max(vals) / min(vals), 2) if vals else None
        rows.append(row)
    return pd.DataFrame(rows)


def strategy_comparison() -> pd.DataFrame:
    """The five structured strategies at the SAME ratio, and the budget each one needs."""
    rows = []
    for run, desc in STRATEGIES.items():
        d = _sweep(run)
        if d is None:
            continue
        b = float(d[d["variant"] == "p00_fp32"]["accuracy"].iloc[0])
        rec = {"strategy": desc, "config": run, "baseline_top1": round(b, 4)}
        best_drop = None
        for v in ("p25_fp32", "p25_int8"):
            r = d[d["variant"] == v]
            if not len(r):
                continue
            drop = (float(r["accuracy"].iloc[0]) - b) * 100
            rec[f"{v} top1"] = round(float(r["accuracy"].iloc[0]), 4)
            rec[f"{v} drop_pt"] = round(drop, 2)
            best_drop = drop if best_drop is None else max(best_drop, drop)
        rec["min_budget_to_be_admissible_pt"] = round(-best_drop, 2) if best_drop is not None else None
        rec["epochs"] = int(d["train_epochs_run"].max()) if "train_epochs_run" in d else None
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values("min_budget_to_be_admissible_pt")
    return df


def feasible_set_control() -> pd.DataFrame:
    """The control that killed the first version of the 'strategies are equivalent' claim: at a
    tight budget almost nothing is eligible, so agreement is degeneracy, not equivalence."""
    runs = list(STRATEGIES) + ["classification_prune_a10", "classification_prune_a15",
                               "classification_prune_a20", "classification_prune_u35",
                               "classification_unstructured", "classification_unstructured_long"]
    rows = []
    for bud in BUDGETS:
        tot, per, wins = 0, [], set()
        for run in runs:
            d = _sweep(run)
            if d is None or "accuracy" not in d:
                continue
            b = d[d["variant"].astype(str).str.contains("fp32")].iloc[0]
            ok = d[(d["accuracy"] >= b["accuracy"] - bud / 100) & (d["variant"] != b["variant"])]
            tot += len(ok)
            per.append(len(ok))
            if len(ok):
                cr = b["size_bytes"] / ok["size_bytes"]
                wins.add(str(d.loc[cr.idxmax(), "variant"]))
        rows.append({"budget_pt": bud, "runs": len(per), "eligible_variants_total": tot,
                     "mean_per_run": round(sum(per) / max(len(per), 1), 2),
                     "min_per_run": min(per) if per else 0, "distinct_winners": len(wins)})
    return pd.DataFrame(rows)


# Every claim this project made and then killed, with the control that killed it. This sheet exists
# because the retractions are the most reusable thing here: each one is a sampling or eligibility
# condition that manufactured a result, and each was caught only by a control written down BEFORE
# looking at the outcome. Sourced from results/paper/FINDINGS.md - keep the two in step.
RETRACTIONS = [
    {"claim": "SQNR predicts task metric at rho=+0.395",
     "where": "layer_sensitivity docstring, core/prune.py",
     "killed_by": "experiment 11 (proxy vs task metric, cross-task)",
     "why_it_was_wrong": "6 hand-picked layers. Evenly sampling n=16 gives +0.688 classification / "
                         "+0.703 detection, both p<0.005 - the sampling made the effect look WEAKER "
                         "than it is, which is the same error in the opposite direction.",
     "status": "corrected in code, number replaced"},
    {"claim": "21 of 23 winners prune zero, so pruning earns nothing",
     "where": "memory + draft narrative",
     "killed_by": "experiment 9 (measurement variance)",
     "why_it_was_wrong": "wrong denominator: ratio variations of one strategy counted as separate "
                         "strategies and _quick smoke runs were mixed in. The honest denominator is "
                         "41 distinct INT8 models in the 8 headline settings.",
     "status": "retracted"},
    {"claim": "budget-dependent argmax flips are the paper's main thesis",
     "where": "Budget sweep sheet",
     "killed_by": "experiment 9 (measurement variance)",
     "why_it_was_wrong": "22 of 48 cells are decided inside a 0.28 pt combined SD, and the "
                         "cross-method comparisons are n=1.",
     "status": "demoted - magnitude kept, argmax dropped"},
    {"claim": "the five pruning strategies are equivalent",
     "where": "Strategy comparison sheet",
     "killed_by": "feasible-set control (this workbook)",
     "why_it_was_wrong": "at 0.5 pt the mean eligible set is ~1.3 variants per run. Agreement was "
                         "degeneracy of the search space, not equivalence of method.",
     "status": "retracted"},
    {"claim": "Q/DQ boundary density explains the INT8 slowdown",
     "where": "experiment 1, and the mechanism sentence of claim A",
     "killed_by": "experiment 14 (independent challengers) - optimized_model_filepath dumps",
     "why_it_was_wrong": "the counts (416 Q / 599 DQ / 90 compute) were parsed from the .onnx FILE. "
                         "ORT deletes ~90% of them before the first inference (MobileViT 416/599 -> "
                         "40/97, ResNet50 74/182 -> 1/1). Recounted on the EXECUTED graph the "
                         "variable points the wrong way: the faster family runs more surviving Q/DQ.",
     "status": "retracted; script repaired to dump the executed graph"},
    {"claim": "static covers 26 op types and dynamic covers 7, and that explains the gap",
     "where": "experiment 1b config headers, claim A",
     "killed_by": "experiment 14 - two lines of config",
     "why_it_was_wrong": "the comparison was never run. The dynamic arm is hand-written "
                         "[MatMul, Gemm] and Gemm is not in IntegerOpsRegistry at all, so it was 26 "
                         "op types against TWO, with calibration mode confounded with op coverage.",
     "status": "retracted; config headers rewritten in place"},
    {"claim": "no case in the opposite direction, so the scaling effect is real",
     "where": "claim B",
     "killed_by": "experiment 14, verified here to 6.7e-16 over 71 pairs",
     "why_it_was_wrong": "sp4 = sp1 x eff_ratio is an exact identity. A reverse flip needs sp1<1 AND "
                         "eff_ratio>1 together; only 2 of 71 pairs could ever have produced one. The "
                         "absence was entailed by the result it was offered as evidence for.",
     "status": "retracted"},
    {"claim": "pruning earns nothing inside the 2.0-pt budget; 3.4-3.9x is the ceiling",
     "where": "memory, HANDOVER, draft narrative",
     "killed_by": "experiment 10 (finetuning schedule control)",
     "why_it_was_wrong": "prune.recovery_lr_mult sat at its default 1.0 for every run in the "
                         "project. Sweeping it 1->20 moves ratio-0.25 global pruning from -2.08 pt "
                         "to -0.93 pt. This is the Wang et al. (arXiv:2301.05219) failure mode, "
                         "reproduced on ourselves.",
     "status": "retracted - but see the 5.56x row below"},
    {"claim": "tuned pruning reaches 5.56x compression vs quantize-only's 3.40x",
     "where": "experiment 10 write-up, memory, and the published report",
     "killed_by": "experiment 10b (fair-schedule control)",
     "why_it_was_wrong": "THE WORST ERROR IN THIS PROJECT. The accuracy came from "
                         "classification_prune_global; the 1,603,509-byte figure came from "
                         "classification (uniform L2) - a DIFFERENT STRATEGY - and that model is "
                         "5.19 pt below baseline, wholly inadmissible. The compression of an "
                         "inadmissible model was paired with the accuracy of a different, "
                         "admissible one.",
     "status": "retracted; correct figure is 4.35x vs 3.40x = a 1.28x advantage at -1.57 pt"},
    {"claim": "INT8 uses 2.07x the peak RAM of fp32 on MobileViT",
     "where": "experiment 4 premise, memory",
     "killed_by": "experiment 4 (RAM attribution) - the premise failed to reproduce",
     "why_it_was_wrong": "peak_rss_mb is BASELINE-SUBTRACTED. 7.0 -> 14.6 MB is a ratio of two small "
                         "remainders above a ~54.5 MB empty-session footprint. On total process RSS "
                         "the same models are 61.5 -> 69.0 MB = 1.12x. Measured properly INT8 "
                         "REDUCES peak memory on plain CNNs (ResNet50 0.36x).",
     "status": "retracted; quote peak_rss_total_mb for any 'needs N MB' sentence"},
    {"claim": "the INT8 slowdown is an architecture property of transformers and mobile nets",
     "where": "claim A and B headline, the published report v1-v7",
     "killed_by": "experiment 15/15b (ORT version bisect, 11 runtimes, 396 measurements)",
     "why_it_was_wrong": "byte-identical files run at parity on ORT 1.20.1-1.23.2 (0.976-1.006x) and "
                         "collapse on 1.24.4 (0.742x) and 1.25.1 (0.631x), stable through 1.30.0. "
                         "Everything this project measured ran on 1.30.0. The 1.24 step costs 24% at "
                         "4 threads and NOTHING at 1 thread; 1.25 costs a further 15%/10%.",
     "status": "restated as an ONNX Runtime regression report, not an architecture claim"},
    {"claim": "DeiT has a real, version-independent static-INT8 speed penalty",
     "where": "my own reading of the 4-version bisect, reported to the user",
     "killed_by": "experiment 15b (the same bisect extended to 11 versions)",
     "why_it_was_wrong": "on 4 versions 0.884 -> 0.977 looked like a trend and I called it "
                         "architectural. On 11 versions it oscillates 0.93-1.05 around parity with "
                         "no trend, inside the 7% noise floor. Four points suggested what eleven "
                         "points deny.",
     "status": "retracted"},
    {"claim": "fusion blocking is the mechanism behind the static-QDQ slowdown",
     "where": "experiment 2 write-up, and the challengers' own replacement mechanism",
     "killed_by": "experiment 16 - the head-to-head it was designed to win",
     "why_it_was_wrong": "restricting op_types_to_quantize to [Conv, MatMul] should have been FASTER "
                         "if fusion blocking dominated. It was 8-9% SLOWER on MobileViT. The "
                         "registered scaling prediction also failed: static_cm's parallel fraction "
                         "is LOWER (0.366 vs 0.430) and its serial time unchanged.",
     "status": "retracted as the dominant mechanism; no confirmed mechanism remains"},
    {"claim": "restricting op types gives a 20% four-thread win on DeiT",
     "where": "reported to the user from experiment 16",
     "killed_by": "experiment 16b (same models, 3 repeats instead of n=1)",
     "why_it_was_wrong": "n=1 said t4 1.201x; n=3 says 1.044x - inside the 7% noise floor. Wrong by "
                         "16 percentage points on the headline number. The real, smaller effect is "
                         "~10% at ONE thread.",
     "status": "retracted; recommendation rescoped to single-threaded deployment"},
    {"claim": "cpu_ms_per_inf/(threads x latency) is 'the instrument that works'",
     "where": "my replacement for other_load_pct, queued as an unconditional gate",
     "killed_by": "the thread sweep, measuring outside the range I had validated",
     "why_it_was_wrong": "it reads as a delivery fraction only at or below the physical core count. "
                         "At 8 threads on 6 P-cores it measures 2.43 - impossible for a fraction - "
                         "because the pool spin-waits and cpu_ms_per_inf is a mean while "
                         "latency_p50_ms is a median. An instrument validated only where you already "
                         "have data is not validated.",
     "status": "gate corrected before it shipped; other_load_pct criticism stands"},
    {"claim": "the 2-point (1T/4T) serial decomposition is unusable",
     "where": "my own over-broad retraction of the corpus-wide serial table",
     "killed_by": "experiment 21b (detection's real 4-point curve)",
     "why_it_was_wrong": "the two methods agree almost exactly where p is away from 1 (detection: "
                         "2-point p=0.890 serial 3.109 vs 4-point p=0.887 serial 3.228). The failure "
                         "is confined to p near 1, where ResNet50 sits. I retracted far more than "
                         "the evidence required.",
     "status": "retraction narrowed - usable below p~0.9, unusable above"},
    {"claim": "DP-knapsack allocation at ratio 0.20 is an admissible 6.15x win (-1.48 pt)",
     "where": "reported to the user 2026-09-16 morning as a pruning breakthrough",
     "killed_by": "three independent challengers - and by this repo's own commit 5eb0574 of 09-13",
     "why_it_was_wrong": "A CLAIM THIS PROJECT HAD ALREADY WITHDRAWN THREE DAYS EARLIER. "
                         "allocate_ratios() encoded 'leave this layer alone' as ABSENCE from the "
                         "plan, so the uniform 0.20 was applied on top of the plan: it asked for "
                         "19.7% of MACs and the exported model lost 34.1% of MACs and 49.5% of "
                         "parameters, narrowing 50 of 53 weight tensors from an 11-layer plan. "
                         "Against the tuned reference it is -2.13 pt, OVER budget. All four "
                         "allocator runs (a10/a15/a20/alloc) predate the fix and none is "
                         "reproducible from its config.",
     "status": "retracted (again). ROOT CAUSE: the 09-13 retraction lived only in a commit message "
               "and was never written into FINDINGS.md, so a CSV audit found the number with "
               "nothing beside it saying 'withdrawn'."},
]

# Per-file notes for the raw CSVs. A file with no entry still gets a sheet, with its producing
# script named - an un-annotated sheet is better than a missing one, worse than an explained one.
CSV_NOTES = {
    "finetune_schedule.csv":
        "Experiment 10. prune.recovery_lr_mult swept 1->20 at ratio 0.25, classification_prune_global, "
        "20 epochs. Monotone and saturating; the 1.15 pt span is 4x the 0.28 pt measurement SD. This is "
        "the row that retracted 'pruning earns nothing'.",
    "fair_schedule.csv":
        "The control ON experiment 10. recovery_lr_mult applies only after pruning, so the unpruned arm "
        "never received the same extra training. Here BOTH arms get the same epochs and the same learning "
        "rates and both are quantized; the pruned arm is judged against the best TUNED unpruned model, not "
        "against the untrained baseline. If the gain disappears here, experiment 10 goes back on the shelf.",
    "variance_classification.csv":
        "Experiment 9. Repeat measurements of the same variant; the SD columns are the noise floor that "
        "every other claim in this workbook is judged against.",
    "variance_detection.csv": "Experiment 9, detection. Same purpose as variance_classification.csv.",
    "qdq_density.csv":
        "Experiment 1. Q/DQ boundary counts per model. The pooled correlation with latency reverses "
        "inside every architecture (Simpson's paradox) - see the Retractions sheet.",
    "sqnr_classification.csv":
        "Experiment 11, classification. Per-layer SQNR against the task metric, n=16 evenly sampled "
        "layers: Spearman rho=+0.688, p=0.003.",
    "sqnr_detection.csv": "Experiment 11, detection. n=16 evenly sampled layers: rho=+0.703, p=0.002.",
    "sqnr_anomaly.csv": "Experiment 11, anomaly. The third task in the cross-task check.",
    "variant_inventory.csv":
        "Every model file under results/*/models/, hashed. Byte-identical duplicates are flagged: "
        "counting them inflates every prevalence figure. This file fixes the paper's denominator at "
        "41 distinct INT8 models in the 8 headline settings.",
    "ram_attribution.csv":
        "Experiment 4. Peak RSS with the ORT CPU arena ON and OFF. If the INT8/fp32 ratio falls toward "
        "1.0 with the arena off, 'INT8 uses more memory' is allocator behaviour and must be reported "
        "as such rather than as a property of quantization.",
    "quant_factorial.csv":
        "Experiments 5+6+7 as one factorial: quant_format x type pair x reduce_range. ORT's documented "
        "trap is a COMBINATION (QOperator x S8S8), so the axes are crossed rather than swept one at a "
        "time. Calibration is resolved once and pinned so it cannot confound the axes.",
    "thread_sweep.csv":
        "Experiment 3. Latency per thread count, including counts above the physical P-core count, for "
        "the claim that quantization destroys parallel scaling.",
    "scaling_deficit.csv":
        "Experiment 12. Every INT8 model paired with its OWN fp32 sibling (same prune ratio), 71 pairs "
        "across 23 runs. RAW: INT8 scales worse 1T->4T in 66/71 pairs, 23/23 runs, p=2.4e-07. But the "
        "confound is real - fp32 models' own scaling tracks their latency at rho=+0.958, so faster "
        "models genuinely parallelise worse. CONTROLLED for speed the deficit survives at 19/23 runs "
        "(p=0.0026) and becomes specific: large on transformers (MobileViT -1.06, DeiT -0.98) and "
        "detection (-0.65), small on plain CNN classification, and REVERSED on ResNet50 and Imagenette. "
        "The residual column is the quantity to quote; `delta` is the uncontrolled version.",
    "static_vs_dynamic.csv":
        "Experiment 13. Each dynamic INT8 variant against its own fp32 and static INT8 stablemates at "
        "the same prune ratio, on speed AND size AND accuracy. The pooled claim (dynamic beats static "
        "1.45-1.56x at 4T, 6/6) holds, but it is two different mechanisms: on DeiT dynamic is a genuine "
        "win (1.27x faster than fp32, 29% of fp32 bytes vs static's 31%); on MobileViT it is a retreat "
        "(1.08x fp32 - i.e. barely quantized - and it keeps 74% of the bytes, WORSE compression than "
        "static's 63%). MobileViT's 1.56x measures the damage static does, not a benefit dynamic gives. "
        "Read the size columns: without them this is the Dehghani et al. efficiency misnomer.",
    "parallel_fraction.csv":
        "Amdahl fits over the 4-point thread curves. p is the parallel fraction; serial_ms = 1T "
        "latency x (1-p). THE READING: DeiT static QDQ adds 1.342 ms of SERIAL work (2.774 -> 4.116, "
        "+48%) while dynamic adds none (2.745); ResNet50 static adds none (3.269 -> 3.120). MobileViT "
        "fits NEGATIVE p for every variant - it has no parallel region, so none of its scaling "
        "numbers mean anything. Do not compute serial time from 2-point (1T/4T) data: near p->1 a "
        "2.3% measurement shift moves the estimate 4x, and the two methods disagreed by 2.9 ms.",
    "thread_sweep.csv":
        "Experiment 3. Latency per thread count within one core class (1,2,4,6 on P-cores) and, as a "
        "SEPARATE condition, counts that spill onto E-cores/SMT (8,12). The spilled points are worse "
        "than 6T for every variant: on a hybrid CPU do not request more threads than P-cores. INT8 "
        "parallel efficiency is lower than fp32 at EVERY count including 2 - there is no knee, just a "
        "lower slope throughout.",
    "quant_factorial.csv":
        "Experiments 5+6+7 as one factorial: quant_format x type pair x reduce_range, calibration "
        "resolved once and PINNED (recorded per row) so it cannot confound the axes. The trap is an "
        "INTERACTION: QDQ+S8S8 is 0.85x and QOperator+U8S8 is 1.58x, but QOperator+S8S8 is 0.098x - "
        "a 10x collapse neither main effect predicts, which a one-axis-at-a-time sweep would miss. "
        "Also: qoperator+U8S8 is 7.1% smaller (MobileNetV2) / 16.6% (MobileViT) than the shipped "
        "default at identical accuracy, and accuracy sensitivity to configuration is itself "
        "architecture-dependent (0.05 pt spread on MobileNetV2, 3.3 pt on MobileViT).",
    "ram_attribution.csv":
        "Experiment 4. Peak RSS with the ORT arena on and off. The arena is NOT the cause - every "
        "ratio moves by <=0.02. And the premise did not reproduce: the '2.07x' came from "
        "peak_rss_mb, which is BASELINE-SUBTRACTED. On total process RSS the same models are 1.12x, "
        "i.e. +7.5 MB on a 61.5 MB process. Measured properly, INT8 REDUCES peak memory on plain "
        "CNNs (ResNet50 0.36x) and raises it only on MobileViT (8-10%). Quote peak_rss_total_mb for "
        "any sentence of the form 'this model needs N MB'.",
    "node_profile.csv": "Experiment 2. Per-node ORT profiling; attribution for the static/dynamic gap.",
    "bn_recalibration.csv":
        "Pruning leaves BatchNorm holding statistics for channels that no longer exist. At ratio 0.25 the "
        "pruned model sits at 0.1000 - exactly chance for 10 classes - and label-free re-estimation "
        "recovers it to 0.6522. At ratio 0.50 recovery is far weaker (0.1924), so it is not a universal "
        "remedy.",
}

_BAD_SHEET_CHARS = "[]:*?/" + chr(92)


def _sheet_name(stem: str) -> str:
    """Excel caps sheet names at 31 characters and rejects a handful of punctuation marks.

    Truncating from the right collides exactly where these files differ: quant_factorial_
    classification and quant_factorial_classification_mobilevit share their first 31 characters, so
    a right-truncation produced two sheets whose names differed only by a trailing space - which
    Excel then strips. Keep both ends instead: the head says what the file is, the tail says which
    run it belongs to, and those are the two things a reader needs.
    """
    name = stem.replace("_", " ")
    for ch in _BAD_SHEET_CHARS:
        name = name.replace(ch, "-")
    if len(name) <= 31:
        return name
    return name[:15] + "~" + name[-15:]


def csv_sheets() -> list:
    """Every raw CSV under results/paper/, so the workbook carries the whole evidence base rather than
    a summary of it. A reader who distrusts a conclusion can open the numbers it came from, in the
    same file, without cloning the repo."""
    out = []
    for f in sorted((ROOT / "results" / "paper").glob("*.csv")):
        note = CSV_NOTES.get(f.name)
        if note is None:
            # per-run files, e.g. quant_factorial_classification.csv -> the quant_factorial note
            stem = next((k for k in CSV_NOTES if f.name.startswith(k[:-4])), None)
            note = (CSV_NOTES[stem] + "  [file: " + f.name + "]") if stem else (
                "Raw output of scripts/paper/" + f.stem + ".py. No curated note yet - read that "
                "script's docstring, which states the hypothesis and its falsification condition.")
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print("  skipped " + f.name + ": " + type(e).__name__ + ": " + str(e), file=sys.stderr)
            continue
        out.append((_sheet_name(f.stem), df, note))
    return out


def write(ws, df: pd.DataFrame, note: str) -> None:
    ws.append([note])
    ws["A1"].font = Font(italic=True, size=9, color="555555")
    ws["A1"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(len(df.columns), 4))
    ws.row_dimensions[1].height = 42
    ws.append([])
    ws.append(list(df.columns))
    for c in ws[3]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDE6F0")
    for r in df.itertuples(index=False):
        ws.append(["" if pd.isna(v) else v for v in r])
    for i, col in enumerate(df.columns, 1):
        w = max(len(str(col)), *(len(str(v)) for v in df[col].astype(str))) + 2
        ws.column_dimensions[get_column_letter(i)].width = min(w, 34)
    ws.freeze_panes = "A4"


def main() -> int:
    if not XLSX.exists():
        print(f"{XLSX} not found - run scripts/make_report_xlsx.py first", file=sys.stderr)
        return 1
    wb = load_workbook(XLSX)
    sheets = [
        ("Budget sweep", budget_sweep(),
         "Best achievable compression as the accuracy budget moves. The winning variant changes in 6 of 8 "
         "settings, and one extra point of tolerance is worth up to 63% more compression (anomaly: 7.84x at "
         "2.0pt -> 12.77x at 3.0pt). This is the paper's central figure."),
        ("Strategy comparison", strategy_comparison(),
         "Five structured pruning strategies at the SAME ratio (0.25), same grid, same 20-epoch finetune. "
         "READ WITH THE RETRACTIONS SHEET. The spread shown here was measured with prune.recovery_lr_mult at "
         "its default 1.0 for every strategy, and experiment 10 showed that single knob moves this column by "
         "1.15 pt - four times the 0.28 pt measurement SD. So the ORDERING below is not evidence about the "
         "strategies; it is evidence about one untuned hyperparameter shared by all of them."),
        ("Feasible set control", feasible_set_control(),
         "The control that refuted the first version of the equivalence claim. At 0.5pt the mean eligible set is "
         "~1.3 variants per run (minimum 1), so 'all strategies agree' is degeneracy of the search space, not a "
         "structural property. Any equivalence claim must report this."),
    ]
    # The retractions come FIRST in the workbook. Every sheet after them has to be read in their
    # light, and a reader who opens the file and sees only the surviving numbers would be misled by
    # exactly the claims this project has already had to withdraw.
    sheets.insert(0, ("Retractions", pd.DataFrame(RETRACTIONS),
                      "Claims this project made and then killed, with the control that killed each one. "
                      "Every row was caught by a falsification condition written down BEFORE the result "
                      "was looked at; none was caught by re-reading the data afterwards. The common "
                      "mechanism is a sampling or eligibility condition that manufactured the result."))
    sheets += csv_sheets()   # the raw evidence, one sheet per file

    for name, df, note in sheets:
        if df is None or not len(df):
            print(f"  empty, skipped: {name}")
            continue
        # Excel sheet names collide case-insensitively, so a legacy "BN recalibration" would push
        # this run's "bn recalibration" to "bn recalibration1" and the workbook would carry both -
        # one of them stale. Drop any case-insensitive match, not just the exact one.
        for existing in [s for s in wb.sheetnames if s.lower() == name.lower()]:
            del wb[existing]
        write(wb.create_sheet(name), df, note)
    wb.save(XLSX)
    print(f"appended {len(sheets)} sheet(s) -> {XLSX}")
    print("sheets now:", ", ".join(wb.sheetnames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
