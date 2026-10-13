"""Build an Excel report from the sweep results (source of truth: results/<run>/sweep.csv + summary.json).

  python scripts/make_report_xlsx.py [--out results/edge_ai_toolkit_report.xlsx]

Sheets: Summary, All variants, Datasets, Method, Fixes & pitfalls, Limitations.
Derived cells (ratios, deltas, totals) are Excel formulas, so the workbook recalculates if a number changes.
"""
import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: F401  (sets sys.path)

import pandas as pd
from openpyxl import Workbook

from core import AUTHOR, PRODUCER
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

FONT = "Arial"
HDR_FILL = PatternFill("solid", fgColor="1F3864")
HDR_FONT = Font(name=FONT, bold=True, color="FFFFFF", size=10)
TITLE_FONT = Font(name=FONT, bold=True, size=13, color="1F3864")
NOTE_FONT = Font(name=FONT, italic=True, size=9, color="595959")
BASE_FONT = Font(name=FONT, size=10)
BOLD = Font(name=FONT, size=10, bold=True)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
BEST_FILL = PatternFill("solid", fgColor="E2EFDA")
WARN_FILL = PatternFill("solid", fgColor="FCE4D6")

TASK_LABEL = {
    "classification": "Image classification (MobileNetV2)",
    "classification_mobilevit": "Classification (MobileViT-XXS)",
    "classification_deit": "Classification (DeiT-tiny)",
    "anomaly": "Time-series anomaly detection (1D-CNN AE)",
    "detection": "Object detection (YOLOv8n)",
    "detection_prune": "Detection, pruning on 5000 real images (YOLOv8n)",
    "detection_640": "Object detection at 640 px (YOLOv8n)",
    "stress_imagenette": "STRESS: real 224 px photos, ImageFolder (MobileNetV2)",
    "stress_resnet50": "STRESS: 25M parameters (ResNet50)",
}
# The five phase runs come first and in this order; any other completed run (stress tests, extra
# model families, new configs) is appended alphabetically. A hardcoded list silently dropped every
# run that was not on it, so a new config's results never reached the report at all.
ORDER = ["classification", "classification_mobilevit", "classification_deit", "anomaly", "detection"]


def is_investigation(name: str) -> bool:
    """A run whose config lives in configs/experiments/ answered one question; it is evidence, not a
    headline result. Mirrors the config layout rather than inventing a second naming rule."""
    return (ROOT / "configs" / "experiments" / f"{name}.yaml").exists()


def discover(results: Path, investigations: bool = True) -> list[str]:
    extra = sorted(d.name for d in results.iterdir()
                   if d.is_dir() and d.name not in ORDER and not d.name.endswith("_quick")
                   and (d / "sweep.csv").exists()
                   and (investigations or not is_investigation(d.name)))
    return [n for n in ORDER if (results / n / "sweep.csv").exists()] + extra


def load_runs(results: Path):
    runs = {}
    for name in discover(results):
        d = results / name
        if not (d / "summary.json").exists():
            # sweep.csv is written incrementally but summary.json only at the end: this run is still
            # going (or died). Reporting its partial rows would understate it, so say so and skip.
            print(f"  skipping {name}: no summary.json (run still in progress or interrupted)")
            continue
        df = pd.read_csv(d / "sweep.csv")
        if "score" not in df.columns:
            # A run being RE-run is the nastier case: sweep.csv is back to partial rows while the
            # PREVIOUS summary.json still sits there, so "summary exists" is not enough. score/
            # eligible are added by core.sweep.finalize() at the end, so their absence means this
            # csv has not been finalized and must not be mixed with the stale summary.
            print(f"  skipping {name}: sweep.csv has no score column (being re-run right now)")
            continue
        if "error" in df:
            df = df[df["error"].isna()]
        if "skipped" in df:
            df = df[df["skipped"].isna()]
        if df.empty:
            print(f"  skipping {name}: no successful variants")
            continue
        runs[name] = (df.reset_index(drop=True), json.load(open(d / "summary.json", encoding="utf-8")))
    return runs


def style_header(ws, row, ncols, height=28):
    ws.row_dimensions[row].height = height
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill, cell.font = HDR_FILL, HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER


def put_title(ws, title, subtitle=None):
    ws["A1"] = title
    ws["A1"].font = TITLE_FONT
    if subtitle:
        ws["A2"] = subtitle
        ws["A2"].font = NOTE_FONT
    return 4 if subtitle else 3


def autosize(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def write_table(ws, top, headers, rows, widths, number_formats=None):
    for j, h in enumerate(headers, start=1):
        ws.cell(row=top, column=j, value=h)
    style_header(ws, top, len(headers))
    for i, row in enumerate(rows, start=top + 1):
        for j, v in enumerate(row, start=1):
            cell = ws.cell(row=i, column=j, value=v)
            cell.font = BASE_FONT
            cell.border = BORDER
            if number_formats and number_formats.get(j):
                cell.number_format = number_formats[j]
    autosize(ws, widths)
    return top + len(rows)


# ------------------------------------------------------------------ sheets
def sheet_summary(wb, runs):
    ws = wb.create_sheet("Summary")
    top = put_title(ws, "SEC 2026 Minimalist Edge AI Challenge - compression results",
                    "Source: results/<run>/sweep.csv (regenerated 2026-09-12). Ratio/delta columns are formulas. "
                    "Best = highest total score under the DEFAULT EQUAL WEIGHTS (official formula unknown). "
                    f"Produced by {PRODUCER}.")
    headers = ["Task", "Variant", "Role", "Metric", "Score value", "Size (MB)", "MACs (M)",
               "Latency p50 1T (ms)", "Latency p50 4T (ms)", "Peak RSS 4T (MB)", "INT8 SQNR (dB)", "Total score",
               "Size vs FP32", "Speed-up 1T", "Speed-up 4T", "Metric delta"]
    fmts = {5: "0.0000", 6: "0.00", 7: "#,##0.0", 8: "0.000", 9: "0.000", 10: "0.00", 11: "0.0", 12: "0.000",
            13: "0.00x", 14: "0.00x", 15: "0.00x", 16: "+0.0000;-0.0000;0.0000"}
    rows, pairs = [], []
    for name, (df, summ) in runs.items():
        # row 0 is only the FP32 baseline if it survived: an errored baseline is filtered out by
        # load_runs, which would relabel an INT8 file as "baseline (FP32)". Ask for the real thing.
        fp32 = df[(df.get("precision", "") == "fp32") & (df.get("prune_ratio", 0).fillna(0) == 0)]             if "precision" in df else df.iloc[0:0]
        base = fp32.iloc[0] if len(fp32) else df.iloc[0]
        elig = df[df["eligible"]] if "eligible" in df else df
        if elig.empty:
            # every variant failed the accuracy constraint: report the baseline alone rather than
            # crashing on an empty idxmax
            print(f"  {name}: no variant passes the accuracy constraint; Summary shows the baseline only")
            rows.append([TASK_LABEL.get(name, name), base["variant"], "baseline (FP32)",
                         base["metric_name"], float(base["accuracy"]),
                         float(base["size_bytes"]) / 2**20, float(base["macs"]) / 1e6,
                         float(base.get("latency_p50_ms_t1", base["latency_p50_ms"])),
                         float(base.get("latency_p50_ms_t4", float("nan"))),
                         float(base.get("peak_rss_mb_t4", base["peak_rss_mb"])), None,
                         float(base["score"]), None, None, None, None])
            continue
        best = elig.loc[elig["score"].idxmax()]
        for role, r in (("baseline (FP32)", base), ("best eligible by score", best)):
            rows.append([TASK_LABEL.get(name, name), r["variant"], role, r["metric_name"],
                         float(r["accuracy"]), float(r["size_bytes"]) / 2**20, float(r["macs"]) / 1e6,
                         # the 1T column must come from *_t1. latency_p50_ms is the UN-SUFFIXED
                         # column, which follows measure.primary_threads (4 in the shipped configs) -
                         # labelling it 1T is the same mislabel already fixed twice elsewhere.
                         float(r.get("latency_p50_ms_t1", r["latency_p50_ms"])),
                         float(r.get("latency_p50_ms_t4", float("nan"))),
                         float(r.get("peak_rss_mb_t4", r["peak_rss_mb"])),
                         float(r["int8_sqnr_db"]) if pd.notna(r.get("int8_sqnr_db")) else None,
                         float(r["score"]), None, None, None, None])
        pairs.append((len(rows) - 1, len(rows)))   # 1-based offsets of (baseline, best) within rows
    end = write_table(ws, top, headers, rows,
                      [34, 18, 20, 10, 11, 10, 11, 14, 14, 11, 12, 11, 11, 11, 11, 12], fmts)
    for b_off, x_off in pairs:
        b, x = top + b_off, top + x_off          # baseline row, best row
        ws.cell(row=x, column=13, value=f"=F{b}/F{x}").number_format = fmts[13]
        ws.cell(row=x, column=14, value=f"=H{b}/H{x}").number_format = fmts[14]
        ws.cell(row=x, column=15, value=f"=I{b}/I{x}").number_format = fmts[15]
        ws.cell(row=x, column=16, value=f"=E{x}-E{b}").number_format = fmts[16]
        for col in (13, 14, 15, 16):
            ws.cell(row=x, column=col).font = BOLD
            ws.cell(row=x, column=col).border = BORDER
        for col in range(1, len(headers) + 1):
            ws.cell(row=x, column=col).fill = BEST_FILL
    note = end + 2
    ws.cell(row=note, column=1, value="Notes").font = BOLD
    for i, t in enumerate([
        "Metrics are NOT comparable across tasks: top-1 accuracy (classification), ROC AUC (anomaly), "
        "COCO mAP@[.5:.95] (detection). Higher is better in all three.",
        "Latency / RSS were measured on a dev laptop (i7-13620H, ONNX Runtime CPU, pinned to performance "
        "cores, median of 3 runs) and are RELATIVE values for comparing variants - not Arduino UNO Q figures.",
        "SPEED-UP DEPENDS ON THREAD COUNT: INT8 gains shrink with 4 threads (see 'Speed-up 1T' vs "
        "'Speed-up 4T'). The UNO Q has 4 Cortex-A53 cores, so quote the 4T column for target-like settings.",
        "INT8 SQNR is a label-free health check of the quantized model (< ~15 dB means it is broken).",
        "Energy (J/inference) is deliberately empty: no real measurement source was available. "
        "It is never estimated.",
        "'Best eligible by score' ranks ONLY variants that pass the accuracy constraint "
        "(score.max_accuracy_drop, default 0.02 absolute). Use the Candidates sheet to choose - the score "
        "alone, with equal weights, would rank a 20-point-worse variant first.",
    ], start=1):
        c = ws.cell(row=note + i, column=1, value=f"{i}. {t}")
        c.font = NOTE_FONT
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=note + i, start_column=1, end_row=note + i, end_column=16)
        ws.row_dimensions[note + i].height = 26
    ws.freeze_panes = ws.cell(row=top + 1, column=1)
    return ws


def sheet_candidates(wb, runs):
    """The shortlist a human picks from: best eligible scores + Pareto fronts, plus what was excluded."""
    ws = wb.create_sheet("Candidates")
    top = put_title(ws, "Candidates per task - pick one, do not assume the top row",
                    "Eligible = passes score.max_accuracy_drop (0.02 absolute vs the FP32 baseline). "
                    "A weighted score alone is not a selector.")
    # summary.json's candidate rows carry the un-suffixed (primary-thread) latency, so this column
    # is labelled for what it holds rather than assuming 1 thread.
    headers = ["Task", "Variant", "Metric", "Score value", "Size (MB)", "Latency p50, scoring threads (ms)",
               "Latency p50 4T (ms)", "Peak RSS (MB)", "Total score", "Why shortlisted"]
    fmts = {4: "0.0000", 5: "0.00", 6: "0.000", 7: "0.000", 8: "0.00", 9: "0.000"}
    rows = []
    for name, (df, summ) in runs.items():
        for c in summ.get("candidates", []):
            src = df[df["variant"] == c["variant"]]
            r4 = float(src.iloc[0].get("latency_p50_ms_t4", float("nan"))) if len(src) else None
            rows.append([TASK_LABEL.get(name, name), c["variant"], c.get("metric_name"),
                         c.get("accuracy"), c.get("size_bytes", 0) / 2**20, c.get("latency_p50_ms"), r4,
                         c.get("peak_rss_mb"), c.get("score"), c.get("why")])
    end = write_table(ws, top, headers, rows, [34, 20, 10, 11, 10, 15, 15, 11, 11, 34], fmts)
    excl = [[TASK_LABEL.get(n, n), e["variant"], e.get("accuracy"), e.get("ineligible_reason")]
            for n, (_, s) in runs.items() for e in s.get("ineligible", [])]
    if excl:
        t2 = end + 3
        ws.cell(row=t2 - 1, column=1, value="Excluded by the accuracy constraint (never proposed):").font = BOLD
        write_table(ws, t2, ["Task", "Variant", "Score value", "Reason"], excl, [34, 20, 12, 80], {3: "0.0000"})
        for r in range(t2 + 1, t2 + 1 + len(excl)):
            ws.cell(row=r, column=4).alignment = Alignment(wrap_text=True, vertical="top")
            for col in range(1, 5):
                ws.cell(row=r, column=col).fill = WARN_FILL
    ws.freeze_panes = ws.cell(row=top + 1, column=1)
    return ws


def sheet_variants(wb, runs):
    ws = wb.create_sheet("All variants")
    n = sum(len(df) for df, _ in runs.values())          # was hardcoded to 38 and went stale
    top = put_title(ws, f"Every measured variant ({n} rows across {len(runs)} runs)",
                    "One row per compressed model. 'Size vs FP32' is a formula against that task's p00_fp32 row.")
    # Every per-thread number is read from its OWN *_t<N> column. Reading the un-suffixed column here
    # silently mislabelled 4-thread latency as 1T (it follows measure.primary_threads, which the
    # shipped configs set to 4) and paired a 4T value with a 1T spread in the next column.
    headers = ["Task", "Variant", "Prune ratio", "Precision", "Metric", "Score value", "Size (MB)",
               "Params (M)", "MACs (M)", "Size gz (MB)", "Latency p50 1T (ms)", "Latency 1T sd",
               "Latency p50 4T (ms)",
               "Latency 4T sd", "Peak RSS 1T (MB)", "Peak RSS 1T sd", "Peak RSS 4T (MB)",
               "CPU ms/inference 4T", "INT8 SQNR mean (dB)", "INT8 SQNR min (dB)", "Calibration chosen",
               "INT8 ops", "FP32-kept nodes", "Max layer cut", "Total score", "Threads for score",
               "Eligible", "Why not eligible", "Size vs FP32"]
    fmts = {3: "0%", 6: "0.0000", 7: "0.00", 8: "0.00", 9: "#,##0.0", 10: "0.000", 11: "0.000", 12: "0.000",
            13: "0.000", 14: "0.00", 15: "0.00", 16: "0.00", 17: "0.000", 18: "0.0", 19: "0.0",
            21: "#,##0", 22: "#,##0", 23: "0%", 24: "0.000", 25: "0", 28: "0.00x"}
    rows, base_rows = [], {}
    for name, (df, _) in runs.items():
        for _, r in df.iterrows():
            def g(col, scale=1.0):
                v = r.get(col)
                return float(v) * scale if pd.notna(v) else None

            def txt(col):
                v = r.get(col)
                return str(v) if pd.notna(v) else None
            label = TASK_LABEL.get(name, name)
            if is_investigation(name):
                label = f"[investigation] {label}"
            rows.append([label, r["variant"], float(r.get("prune_ratio", 0) or 0),
                         r.get("precision", ""), r["metric_name"], float(r["accuracy"]),
                         g("size_bytes", 1 / 2**20), g("params", 1e-6), g("macs", 1e-6),
                         # magnitude pruning keeps tensor SHAPES, so its entire benefit is invisible
                         # in size_bytes and only appears once compressed
                         g("size_bytes_gz", 1 / 2**20),
                         g("latency_p50_ms_t1"), g("latency_p50_ms_sd_t1"),
                         g("latency_p50_ms_t4"), g("latency_p50_ms_sd_t4"),
                         g("peak_rss_mb_t1"), g("peak_rss_mb_sd_t1"), g("peak_rss_mb_t4"),
                         g("cpu_ms_per_inf_t4"),
                         g("int8_sqnr_db"), g("int8_sqnr_min_db"), txt("calib_method_chosen"),
                         g("int8_quantized_nodes"), g("fp32_nodes"), g("max_layer_cut"),
                         float(r["score"]), g("primary_threads"),
                         "yes" if r.get("eligible", True) else "NO",
                         r.get("ineligible_reason") if pd.notna(r.get("ineligible_reason")) else None, None])
            if name not in base_rows:
                # first row of the run is its baseline; variant names may carry a sweep-dimension prefix
                base_rows[name] = top + len(rows)
    end = write_table(ws, top, headers, rows, [34, 18, 11, 12, 9, 11, 10, 10, 11, 15, 12, 15, 12, 14, 12,
                                               14, 15, 15, 14, 18, 9, 13, 11, 11, 13, 9, 46, 11], fmts)
    idx = top + 1
    for name, (df, _) in runs.items():
        for _ in range(len(df)):
            ws.cell(row=idx, column=28, value=f"=G{base_rows[name]}/G{idx}").number_format = fmts[28]
            ws.cell(row=idx, column=28).font = BASE_FONT
            ws.cell(row=idx, column=28).border = BORDER
            if ws.cell(row=idx, column=26).value == "NO":
                ws.cell(row=idx, column=26).font = Font(name=FONT, size=10, bold=True, color="C00000")
            # shade on the per-channel MINIMUM when we have it: the mean hides one destroyed channel group
            sqnr = ws.cell(row=idx, column=19).value
            if not isinstance(sqnr, (int, float)):
                sqnr = ws.cell(row=idx, column=18).value
            if isinstance(sqnr, (int, float)) and sqnr < 15:     # broken INT8 conversion
                for col in range(1, len(headers) + 1):
                    ws.cell(row=idx, column=col).fill = WARN_FILL
            idx += 1
    ws.cell(row=end + 2, column=1,
            value="Rows shaded orange: INT8 SQNR below 15 dB, i.e. the quantized model is damaged "
                  "(MobileViT static INT8). 'INT8 ops' = 0 would mean nothing was actually quantized. "
                  "'Eligible = NO' means the variant failed the accuracy constraint and is never proposed. "
                  "sd columns are the spread over 3 repeated measurement runs.").font = NOTE_FONT
    ws.freeze_panes = ws.cell(row=top + 1, column=3)
    return ws


def sheet_datasets(wb, runs):
    ws = wb.create_sheet("Datasets")
    top = put_title(ws, "Test data: structure and scale",
                    "What each number in this report was measured on.")
    headers = ["Task", "Dataset", "Sample shape", "Classes", "Train samples", "Eval samples",
               "Total", "Eval protocol", "Notes"]
    rows = []
    d = {name: summ.get("adapter", {}) for name, (_, summ) in runs.items()}
    c = d.get("classification", {})
    rows.append(["Image classification", "CIFAR-10", "3 x 32 x 32 RGB", 10, c.get("n_train"), c.get("n_val"),
                 None, "Full 10,000-image test set, top-1",
                 "Balanced classes; INT8 calibration uses 512 training images"])
    mv = d.get("classification_mobilevit", {})
    rows.append(["Classification (MobileViT)", "CIFAR-10 upscaled", "3 x 64 x 64 RGB", 10, mv.get("n_train"),
                 mv.get("n_val"), None, "Full test set, top-1",
                 "64 px because 32 px leaves 1x1 feature maps in the last stage"])
    dt = d.get("classification_deit", {})
    rows.append(["Classification (DeiT-tiny)", "CIFAR-10", "3 x 32 x 32 RGB", 10, dt.get("n_train"),
                 dt.get("n_val"), None, "Full test set, top-1", "4x4 patches -> 64 tokens; 10 training epochs"])
    a = d.get("anomaly", {})
    rows.append(["Anomaly detection", "ECG5000 (UCR)", "1 x 140 time steps", "normal / anomaly",
                 a.get("n_train"), a.get("n_test"), None, "ROC AUC over reconstruction error",
                 "Training uses NORMAL samples only (70%); test = remaining normal + all anomalies (~70% anomalous)"])
    de = d.get("detection", {})
    rows.append(["Object detection", "coco128 train / unseen COCO val2017", "3 x 320 x 320 letterboxed", 80,
                 de.get("n_train"), de.get("n_val"), None, "COCO mAP@[.5:.95] via pycocotools",
                 "Validation images are HELD OUT (coco128 is inside COCO train2017, so validating on it "
                 "measures images the pretrained detector already saw - that inflated mAP by ~0.10). "
                 "5000 train2017 images were downloaded for the pruning study"])
    rows.append(["Unit tests", "Synthetic + tiny fixtures", "3 x 16 x 16 random tensors", 4, 256, 128,
                 None, "89 tests, no downloads",
                 "Plus 3 KB TFLite / NHWC-ONNX fixtures and 6 randomly initialised model families"])
    end = write_table(ws, top, headers, rows, [26, 24, 24, 14, 14, 14, 11, 30, 62],
                      {4: "General", 5: "#,##0", 6: "#,##0", 7: "#,##0"})
    for r in range(top + 1, end + 1):
        ws.cell(row=r, column=7, value=f"=E{r}+F{r}").number_format = "#,##0"
        ws.cell(row=r, column=7).font = BASE_FONT
        ws.cell(row=r, column=7).border = BORDER
        ws.cell(row=r, column=9).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 30
    ws.freeze_panes = ws.cell(row=top + 1, column=1)
    return ws


def sheet_method(wb, runs):
    ws = wb.create_sheet("Method")
    top = put_title(ws, "Measurement protocol and environment",
                    "Same protocol for every variant; accuracy is always evaluated on the deployable ONNX file.")
    env = list(runs.values())[0][1].get("env", {})
    rows = [
        ["Accuracy / AUC / mAP", "adapter.evaluate() on the exported ONNX file (not the PyTorch model)"],
        ["Latency", "ONNX Runtime CPU EP, batch 1, 10 warm-up runs, then >=100 runs AND >=2 s; p50 and p90"],
        ["Thread counts", "1 and 4 intra-op threads, measured separately"],
        ["CPU pinning", "Pinned to the fastest core class (P-cores). Unpinned, the same model measured "
                        "0.9 ms on a P-core and 2.4 ms on an E-core"],
        ["Peak RAM", "Isolated subprocess, OS peak RSS minus an 'empty session' of the same runtime"],
        ["CPU per inference", "Process CPU time / inferences over the timed window"],
        ["INT8 health (SQNR)", "Mean per-output-channel signal-to-quantization-noise vs FP32 on calibration "
                               "data; label-free"],
        ["Energy", "EMPTY - no real measurement source. Never estimated from a formula"],
        ["Export check", "Every ONNX export is compared against PyTorch outputs before it is measured"],
        ["Competing load", "Every row records other_load_pct - how busy the machine was with anything "
                           "else during that measurement - and warns above measure.other_load_warn_pct "
                           "(8%). Measured here: a saturating competing load shifts a large model's "
                           "latency median 1-16% and inflates the *_sd spread ~8x, while peak RSS is "
                           "unaffected. The sandbox is shared, so this is recorded rather than assumed"],
        ["Compressed size", "size_bytes_gz is the gzipped size alongside the raw size. Magnitude "
                            "(unstructured) pruning keeps tensor shapes, so its entire benefit is "
                            "invisible in size_bytes and only appears here; dense models gzip only "
                            "1.07-1.21x, so the column costs nothing for everything else"],
        ["Training device", "torch_train_device records whether PyTorch training/fine-tuning used CUDA. "
                            "Inference is always measured on the CPU in an isolated subprocess; a "
                            "CPU-only reproduction gives the same metrics but takes far longer "
                            "(--device cpu --quick: 52 s)"],
        ["Machine", f"{env.get('cpu', '')} | {env.get('physical_cores', '')} physical cores "
                    f"({env.get('core_classes', '')})"],
        ["Software", f"torch {env.get('torch', '')}, onnxruntime {env.get('onnxruntime', '')}, "
                     f"Python {env.get('python', '')}"],
    ]
    end = write_table(ws, top, ["Item", "Definition"], rows, [26, 118])
    for r in range(top + 1, end + 1):
        ws.cell(row=r, column=2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 28
    return ws


def sheet_fixes(wb):
    ws = wb.create_sheet("Fixes & pitfalls")
    top = put_title(ws, "Silent failures found and fixed",
                    "Each of these produced a wrong or misleading result during development; each now has a "
                    "default or a guard. Numbers are measured on the models in this report.")
    headers = ["Pitfall", "What went wrong (measured)", "Fix now in place"]
    rows = [
        ["P/E-core scheduling", "Same model: 0.9 ms on a P-core vs 2.4 ms on an E-core, which made pruning "
                                "look 4x faster than it is", "Measurements pinned to the fastest core class"],
        ["round_to over-pruning", "torch-pruning rounds kept channels DOWN: '5%' removed 50% of 16-channel "
                                  "layers, top-1 0.684 -> 0.117", "Kept channels round UP; max_layer_cut reported"],
        ["round_to and grouped convs", "Mixed per-group and total channel counts: a groups=4 conv was pruned "
                                       "0% instead of 39%", "Unit-correct rounding; under-pruning warns"],
        ["No BN re-calibration", "25% pruning without it: top-1 0.099", "BN statistics re-estimated on "
                                 "calibration data (0.099 -> 0.657)"],
        ["INT8 on mixed-range outputs", "Detector output concatenates pixel boxes with 0..1 scores; one shared "
                                        "scale (1.53) zeroed every score < 0.77: mAP 0.4095 -> 0.1608",
         "Output tail kept in FP32 by default (-> 0.4006 with percentile calibration)"],
        ["INT8 that quantizes nothing", "On small models the rule could cover the whole graph: the 'INT8' file "
                                        "was FP32, bigger, and reported a perfect 60 dB SQNR",
         "Rule disables itself; every variant reports int8_quantized_nodes"],
        ["INT8 on SiLU conv blocks", "MobileViT static INT8: 0.9165 -> 0.668 and slower than FP32",
         "int8_dyn_mm variant (0.9160) or auto_keep_fp32: search"],
        ["'entropy' calibration", "ORT's 128/128-bin default makes entropy silently identical to minmax",
         "Real 2048-bin entropy; percentile exposed"],
        ["Fine-tune LR too high", "lr 2e-4 on 64 images dragged the UNPRUNED detector from 0.407 to 0.28",
         "Detection default 2e-5 (control: 0.3951 after 3 epochs)"],
        ["Undetectable INT8 damage", "Nothing warned that an INT8 model was broken until a metric was computed",
         "Label-free int8_sqnr_db on every variant"],
        ["Score alone picked a bad model", "With equal weights accuracy is 25% of the score, so MobileViT's "
                                           "50%-pruned variant (top-1 0.717 vs 0.917) ranked FIRST for being "
                                           "small and fast",
         "Hard accuracy constraint before ranking (score.max_accuracy_drop) + a candidate shortlist instead "
         "of a single winner"],
        ["Speed-up quoted at one thread", "INT8 gains shrink with 4 threads (MobileNetV2 4T: 0.496 vs 0.488 ms "
                                          "- a tie), and the UNO Q has 4 cores",
         "Both 1T and 4T columns are reported side by side; the report says which setting a claim uses"],
        ["RSS repeatability unknown", "Peak RSS can be HIGHER for INT8 on small models (more QDQ nodes), and a "
                                      "single measurement cannot tell that from noise",
         "measure.repeats (3) reports the median plus the spread in *_sd columns"],
        ["minmax calibration on an outlier-heavy model",
         "The 25%-pruned ResNet50's INT8 collapsed to top-1 0.468 from its FP32 0.894 (SQNR 1.5 dB). "
         "percentile 99.99 on the SAME model is lossless: 0.8940 at 13.6 dB. 42.6 accuracy points from "
         "one config line - and not predictable from the architecture, since unpruned MobileNetV2 is fine "
         "on minmax at 24.8 dB",
         "calibrate_method: auto quantizes with each candidate and keeps the best label-free per-channel "
         "SQNR (see the calib_method_chosen / calib_auto_tried columns). Ranking by SQNR reproduced the "
         "ranking by accuracy without using a label"],
        ["Ranking calibrators on the minimum alone",
         "On the detector the two best candidates sat 0.1 dB apart in per-channel minimum SQNR (minmax 9.7 "
         "vs percentile 99.99 9.6) while their means differed by 0.5 dB - and the one the minimum preferred "
         "measured 1.54 mAP WORSE (0.3852 vs 0.4006)",
         "The minimum stays the safety screen. A 1.0 dB tie-break on the mean was tried and REVERTED "
         "(calib_auto_tie_db now defaults to 0): re-running on held-out data showed the candidate the "
         "mean preferred scored 0.2988 against the minimum's 0.3010, so the original evidence for it "
         "was stale. It remains available opt-in"],
        ["--quick silently dropped every resource metric",
         "primary_threads: 4 plus the --quick override threads: [1] left the un-suffixed columns NaN, which "
         "emptied score_resource and both the latency and RSS Pareto fronts - in the mode most likely to be "
         "used under time pressure",
         "primary_thread_count() falls back to the nearest measured count and logs it; the primary_threads "
         "column records which count the reported numbers came from"],
        ["Thread count mislabelled downstream",
         "remeasure.py --rescore-only re-derived the primary thread count from the CONFIG, so a --quick CSV "
         "(only *_t1 columns) asked for t4, copied nothing, and still stamped primary_threads=4 onto 1-thread "
         "numbers. compress.py hardcoded 'latency 1T' while filling it from the 4T column",
         "The count is derived from the columns the CSV actually contains (the file, not the config, is the "
         "source of truth when nothing is re-measured); report headers are labelled from the data"],
        ["A documented threshold nothing enforced",
         "'< ~15 dB SQNR = broken' existed only as prose and one column of a 70-column CSV, so a 1.5 dB model "
         "was produced and reported without comment",
         "quantize.sqnr_warn_db (default 15, 0 disables) warns on the per-channel MINIMUM - the mean hides a "
         "single destroyed channel group. Treat it as a screen, not a verdict: 9.6-13.8 dB measured only "
         "0.41-0.61 points of real loss"],
        ["Training silently used the GPU",
         "PyTorch fine-tuning picks CUDA when a GPU is present, so the documented '--quick in under 5 min on "
         "CPU' had never actually been run CPU-only",
         "env_info() records torch_train_device / cuda_device. Verified: --device cpu --quick finishes in 52 s"],
        ["New results silently missing from this report",
         "The run list in make_report_xlsx.py was hardcoded to the five phase runs, so any new config's "
         "results never appeared here at all - and a run still in progress (sweep.csv written, summary.json "
         "not yet) crashed the generator",
         "Runs are discovered from results/: the five phases first, then any other completed run. An "
         "incomplete run is skipped with a message instead of crashing or being reported half-done"],
    ]
    end = write_table(ws, top, headers, rows, [30, 66, 52])
    for r in range(top + 1, end + 1):
        for col in (1, 2, 3):
            ws.cell(row=r, column=col).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 44
    return ws


def sheet_limits(wb):
    ws = wb.create_sheet("Limitations")
    top = put_title(ws, "What these numbers do NOT show", "Read before quoting anything in this workbook.")
    rows = [
        ["Latency / RAM / CPU", "Dev-laptop x86 values. Use them to compare variants, not as Arduino UNO Q "
                                "(Cortex-A53) figures. scripts/remeasure.py re-measures on another machine "
                                "without retraining"],
        ["Energy", "Not measured at all. Needs the target board plus a USB power meter, or the organisers' tool"],
        ["Detection mAP", "Validated on 500-2000 UNSEEN COCO val2017 images; coco128 is part of COCO "
                          "train2017, so a pretrained detector has already seen it. The 640 px baseline "
                          "reaches 0.3687 against ultralytics' published 0.373, so the evaluation chain "
                          "itself is sound"],
        ["DeiT / MobileViT accuracy", "DeiT-tiny trains 30 epochs, which is what made its 25%-pruned INT8 "
                                      "variant eligible at 6.22x; MobileViT is still short-trained. The "
                                      "compression DELTA is the meaningful part, not the absolute accuracy"],
        ["Total score", "Equal weights over the four official criteria - the official formula and weights are "
                        "unknown. Change score.weights in the YAML and re-score from the CSV"],
        ["Detection pruning", "Excluded on the evidence, not on principle: with 5000 real training images "
                              "and 10 fine-tune epochs, 10/20/30% pruning gives mAP 0.2251/0.1755/0.0907 "
                              "against a 0.3057 baseline. Raising the recovery LR 20x and adding warmup "
                              "changed nothing (0.1755 -> 0.1728), so it is neither the epoch count nor the "
                              "LR. round_to: 8 did fix the latency side (10.26 -> 8.55 ms)"],
        ["Toolchain RAM (not the model's)", "Every peak_rss_mb here is the MODEL's inference memory in "
                                            "an isolated subprocess. Producing the variants needs far "
                                            "more: fine-tuning a mid-size CNN was measured at ~2.3 GB in "
                                            "the trainer plus ~180 MB per dataloader worker. "
                                            "host_peak_rss_mb records it per variant. On a RAM-limited "
                                            "sandbox that, not the optimised model, is the binding "
                                            "constraint - and the sandbox spec is unknown"],
        ["Qualcomm AI Hub", "Wrapper written against the documented API but never run against a real account"],
        ["Output format", "ONNX only. TFLite output would need the TFLite converter path (target_runtime is "
                          "reserved but not implemented)"],
    ]
    end = write_table(ws, top, ["Area", "Limitation"], rows, [26, 118])
    for r in range(top + 1, end + 1):
        ws.cell(row=r, column=2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 32
    return ws


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "edge_ai_toolkit_report.xlsx"))
    args = ap.parse_args()
    runs = load_runs(Path(args.results))
    if not runs:
        raise SystemExit("no sweep results found")
    wb = Workbook()
    wb.properties.creator = AUTHOR
    wb.properties.lastModifiedBy = AUTHOR
    wb.properties.title = 'edge-ai-toolkit compression results'
    wb.remove(wb.active)
    # Summary and Candidates are the DELIVERABLE claim, so they carry only the headline runs -
    # one per model family and task. The investigation runs under configs/experiments/ each answered
    # one question and belong with the evidence, not next to the results a judge reads first.
    headline = {k: v for k, v in runs.items() if not is_investigation(k)}
    n_inv = len(runs) - len(headline)
    if n_inv:
        print(f"  {len(headline)} headline run(s) in Summary/Candidates; "
              f"{n_inv} investigation run(s) appear in All variants only")
    if not headline:
        # do not quietly promote investigations into the deliverable claim; say which they are
        print("  WARNING: every discovered run is an investigation (configs/experiments/); "
              "Summary and Candidates will present them as headline results")
    sheet_summary(wb, headline or runs)
    sheet_candidates(wb, headline or runs)
    sheet_variants(wb, runs)
    sheet_datasets(wb, runs)
    sheet_method(wb, runs)
    sheet_fixes(wb)
    sheet_limits(wb)
    for ws in wb.worksheets:      # a default font for anything not styled explicitly
        for row in ws.iter_rows():
            for cell in row:
                if cell.font is None or cell.font.name is None:
                    cell.font = BASE_FONT
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"wrote {out} ({sum(ws.max_row for ws in wb.worksheets)} rows across {len(wb.worksheets)} sheets)")


if __name__ == "__main__":
    main()
