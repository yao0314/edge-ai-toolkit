"""The four paper figures. Every value is read from a CSV - nothing is typed in by hand.

That rule is not stylistic. The single most frequent defect in this project was a number quoted
from a different source than the sentence it appeared in; seven separate claims died of it. A figure
script that hard-codes its data is the same failure waiting to happen in a place nobody re-checks,
so each figure below derives its numbers from the same files the text cites, and collect() writes
the rows that Table II is built from (paper/figs/_fig_data.csv), so the table and the figures cannot
disagree. Table II has all eight settings; Figs 1-2 plot the rows flagged in_fig (YOLOv8n at 256 px).

Numbering follows the order of first citation in the paper:
  fig1_method_by_class    which INT8 method wins, 1 and 4 threads (Sec. IV)
  fig2_quality_by_model   quality change by model and method (Sec. IV)
  fig3_default_config     format x activation type, the quantizer default (Sec. IV-D)
  fig4_version_sweep      INT8 speedup across eleven ONNX Runtime releases (Sec. V)

Sized for Springer LNCS A4 (122 mm text width). Every text element is >= 7 pt at print size, fonts
are embedded as TrueType (pdf.fonttype 42) rather than Type 3, and line style and marker shape
repeat the colour distinction so the figures survive greyscale printing.

    python scripts/paper/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
BISECT = ROOT / "results" / "paper" / "ort_version_bisect.csv"
PROTOCOL_RELEASE = "1.30.0"

# validated categorical order - assigned in fixed order, never cycled
C = ["#2a6fb8", "#d1600f", "#1f8a6d", "#8e44ad"]
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d6d6d6"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType, not Type 3
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = 4.75     # Springer LNCS A4 text width (122 mm)
ANNOT = 7.0    # smallest annotation size; the figures print at ~1.0x, so this is ~7 pt on paper

ORDER = ["ResNet-50", "MobileNetV2", "YOLOv8n", "1-D conv AE", "DeiT-tiny", "MobileViT-XXS"]
ATTN = {"DeiT-tiny", "MobileViT-XXS"}


def vkey(v):
    return tuple(int(x) for x in str(v).split("."))


TEXTW = 12.2 / 2.54   # LNCS text width in inches; build_tex.py includes every figure at \textwidth


def fit_to_text_width(fig):
    """Widen or narrow the canvas, keeping its height, until the saved (tight-bbox) figure is
    exactly the text width. LaTeX then includes it at 1.0x: fonts print at the sizes set here
    (ANNOT and up, all above LNCS's 6 pt minimum) and the figure is only as tall as drawn,
    instead of being magnified 1.07-1.17x to fill the width."""
    pad = plt.rcParams["savefig.pad_inches"]
    for _ in range(6):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        gap = TEXTW - (bb.width + 2 * pad)
        if abs(gap) < 0.003:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + gap, h)


def save(fig, name):
    fit_to_text_width(fig)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def clean(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        if s not in keep:
            ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ data for Table II, Figs 1-2
# Convolutional models: harness runs in which static and dynamic were measured in the SAME run
# (the *_dyn directories). The earlier standalone static runs are not used: mixing a static value
# from one run with a dynamic value from another was the cross-run defect the review caught.
HARNESS = [
    # run directory, INT8 variant, FP32 sibling, display name, class, method
    ("stress_resnet50_dyn", "p00_int8",              "p00_fp32",     "ResNet-50",   "conv", "static"),
    ("stress_resnet50_dyn", "p00_int8_dyn_conv",     "p00_fp32",     "ResNet-50",   "conv", "dynamic"),
    ("classification_dyn",  "p00_int8",              "p00_fp32",     "MobileNetV2", "conv", "static"),
    ("classification_dyn",  "p00_int8_dyn_conv",     "p00_fp32",     "MobileNetV2", "conv", "dynamic"),
    ("detection_dyn",       "192_p00_int8",          "192_p00_fp32", "YOLOv8n",     "conv", "static"),
    ("detection_dyn",       "192_p00_int8_dyn_conv", "192_p00_fp32", "YOLOv8n",     "conv", "dynamic"),
    ("detection_dyn",       "256_p00_int8",          "256_p00_fp32", "YOLOv8n",     "conv", "static"),
    ("detection_dyn",       "256_p00_int8_dyn_conv", "256_p00_fp32", "YOLOv8n",     "conv", "dynamic"),
    ("detection_dyn",       "320_p00_int8",          "320_p00_fp32", "YOLOv8n",     "conv", "static"),
    ("detection_dyn",       "320_p00_int8_dyn_conv", "320_p00_fp32", "YOLOv8n",     "conv", "dynamic"),
    ("anomaly",             "p00_int8",              "p00_fp32",     "1-D conv AE", "conv", "static"),
]
# Table II carries all three YOLOv8n input sizes; Figs 1-2 show one row per model, at this size.
FIG_YOLO_PX = 256


def input_size(run, row):
    """Input side in px (length in samples for the 1-D autoencoder), read from the run itself:
    the sweep row when the run swept sizes, otherwise the adapter block of the run's summary."""
    if "img_size" in row and pd.notna(row["img_size"]):
        return int(row["img_size"])
    import json
    ad = json.loads((ROOT / "results" / run / "summary.json").read_text(encoding="utf-8"))["adapter"]
    return int(ad.get("img_size") or ad.get("seq_len"))
# Attention / hybrid models: speed from the version protocol (isolated, randomised sessions) on
# PROTOCOL_RELEASE; quality and size are properties of the files, read from the sweep that made them.
PROTOCOL = [
    # bisect family, INT8 variant, sweep run, display name, method
    ("deit",      "p00_int8",        "classification_deit",      "DeiT-tiny",     "static"),
    ("deit",      "p00_int8_dyn_mm", "classification_deit",      "DeiT-tiny",     "dynamic"),
    ("mobilevit", "p00_int8",        "classification_mobilevit", "MobileViT-XXS", "static"),
    ("mobilevit", "p00_int8_dyn_mm", "classification_mobilevit", "MobileViT-XXS", "dynamic"),
]


def collect():
    rows = []
    for run, var, sib, disp, klass, meth in HARNESS:
        d = pd.read_csv(ROOT / "results" / run / "sweep.csv")
        by = {str(r["variant"]): r for _, r in d.iterrows()}
        q, fp = by[var], by[sib]
        px = input_size(run, q)
        rows.append({"model": disp, "klass": klass, "method": meth, "source": "harness",
                     "run": run, "variant": var, "px": px,
                     "in_fig": disp != "YOLOv8n" or px == FIG_YOLO_PX,
                     "metric": q["metric_name"],
                     "calib": q.get("calib_method_chosen", float("nan")),
                     "fp1_ms": fp["latency_p50_ms_t1"], "fp4_ms": fp["latency_p50_ms_t4"],
                     "q1_ms": q["latency_p50_ms_t1"], "q4_ms": q["latency_p50_ms_t4"],
                     "sp1": fp["latency_p50_ms_t1"] / q["latency_p50_ms_t1"],
                     "sp4": fp["latency_p50_ms_t4"] / q["latency_p50_ms_t4"],
                     "drop": (q["accuracy"] - fp["accuracy"]) * 100,
                     "size": q["size_bytes"] / fp["size_bytes"]})
    b = pd.read_csv(BISECT)
    med = b[b["ort"] == PROTOCOL_RELEASE].groupby(["family", "variant", "threads"])["p50_ms"].median()
    for fam, var, run, disp, meth in PROTOCOL:
        d = pd.read_csv(ROOT / "results" / run / "sweep.csv")
        by = {str(r["variant"]): r for _, r in d.iterrows()}
        q, fp = by[var], by["p00_fp32"]
        rows.append({"model": disp, "klass": "attn", "method": meth,
                     "source": f"version protocol {PROTOCOL_RELEASE}",
                     "run": run, "variant": var, "px": input_size(run, q), "in_fig": True,
                     "metric": q["metric_name"],
                     "calib": q.get("calib_method_chosen", float("nan")),
                     "fp1_ms": med[(fam, "p00_fp32", 1)], "fp4_ms": med[(fam, "p00_fp32", 4)],
                     "q1_ms": med[(fam, var, 1)], "q4_ms": med[(fam, var, 4)],
                     "sp1": med[(fam, "p00_fp32", 1)] / med[(fam, var, 1)],
                     "sp4": med[(fam, "p00_fp32", 4)] / med[(fam, var, 4)],
                     "drop": (q["accuracy"] - fp["accuracy"]) * 100,
                     "size": q["size_bytes"] / fp["size_bytes"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Fig 1: which method wins
def fig1(df):
    """Dumbbell per model, one panel per thread count; log x because the dynamic-on-convolution
    points sit at 0.05-0.08x and a linear axis would render every other model as one tick."""
    # heights trimmed 2026-09-19 (all four figures, ~0.65 in in total) to keep the paper at 15 pages
    # after the R1 revisions; every label stays >= 7 pt
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.74), sharey=True)
    for ax, col, title in ((axes[0], "sp1", "1 thread"), (axes[1], "sp4", "4 threads")):
        piv = df.pivot_table(index="model", columns="method", values=col, aggfunc="first")
        piv = piv.reindex([m for m in ORDER if m in piv.index])
        ys = list(range(len(piv)))[::-1]
        rows = [y for y, m in zip(ys, piv.index) if m in ATTN]
        ax.axhspan(min(rows) - 0.5, max(rows) + 0.5, color="#eef2f6", zorder=0)
        ax.axvline(1.0, color=MUTED, lw=0.8, zorder=2)
        for y, (model, r) in zip(ys, piv.iterrows()):
            s, d = r.get("static"), r.get("dynamic")
            if pd.notna(s) and pd.notna(d):
                ax.plot([s, d], [y, y], color="#c8ccd0", lw=1.6, zorder=1, solid_capstyle="round")
            if pd.notna(s):
                ax.scatter(s, y, s=30, marker="o", facecolor=C[0], edgecolor="white", lw=.6,
                           zorder=3, label="static INT8" if y == ys[0] else None)
            if pd.notna(d):
                ax.scatter(d, y, s=30, marker="s", facecolor=C[1], edgecolor="white", lw=.6,
                           zorder=3, label="dynamic INT8" if y == ys[0] else None)
                if d < 0.2:
                    ax.annotate(f"{d:.3f}", (d, y), fontsize=ANNOT, color=C[1],
                                xytext=(0, 5), textcoords="offset points", ha="center")
            if pd.notna(s) and s > 2.0:
                # an opaque backing: at s ~ 3.2 the label sits on the x = 3 gridline, which
                # otherwise runs through the digits and splits "3.21" into "3." and "21"
                ax.annotate(f"{s:.2f}", (s, y), fontsize=ANNOT, color=C[0],
                            xytext=(0, 5), textcoords="offset points", ha="center",
                            bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=4)
            if model == "MobileViT-XXS":
                # the hybrid's two points sit either side of parity: static labelled to the left
                # of its circle, dynamic to the right of its square, level with the markers
                ax.annotate(f"{s:.3f}", (s, y), fontsize=ANNOT, color=C[0], ha="right",
                            va="center", xytext=(-6, 0), textcoords="offset points")
                ax.annotate(f"{d:.3f}", (d, y), fontsize=ANNOT, color=C[1], ha="left",
                            va="center", xytext=(6, 0), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlim(0.035, 5.0)
        ax.set_xticks([0.05, 0.1, 0.3, 1, 3])
        ax.set_xticklabels(["0.05", "0.1", "0.3", "1", "3"])
        ax.minorticks_off()
        ax.set_xlabel(f"speedup vs FP32, {title}")
        ax.set_ylim(-0.6, len(piv) - 0.4)
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        clean(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(ys)
        ax.set_yticklabels(piv.index, fontsize=8)
    # No legend: inside the shaded band its markers read as data points, and the caption already
    # defines circles (static), squares (dynamic) and the gray joining lines.
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig1_method_by_class")


# ---------------------------------------------------------------- Fig 2: quality by model
def fig2(df):
    """Quality change by model and method. The -11.07 pt bar is clipped, and its value is printed
    beside it in dark ink; every other bar carries its own value, so nothing is left to guess."""
    piv = df.pivot_table(index="model", columns="method", values="drop", aggfunc="first")
    piv = piv.reindex([m for m in ORDER if m in piv.index])
    FLOOR = -5.6
    fig, ax = plt.subplots(figsize=(COL, 1.78))
    xs = range(len(piv))
    w = 0.36
    for i, (meth, col, hatch, lab) in enumerate(
            (("static", C[0], "", "static INT8"), ("dynamic", C[1], "///", "dynamic INT8"))):
        vals = piv[meth]
        shown = vals.clip(lower=FLOOR + 0.35)
        pos = [x + (i - .5) * w for x in xs]
        ax.bar(pos, shown, w, color=col, edgecolor="white", linewidth=.6, hatch=hatch,
               label=lab, zorder=3)
        for x, v, sv in zip(pos, vals, shown):
            if pd.isna(v):
                continue
            if v < FLOOR:
                ax.annotate(f"{v:.2f} (clipped)".replace("-", "\u2212"), (x + w * 0.62, FLOOR + 0.55), fontsize=ANNOT,
                            color=INK, ha="left", va="bottom", zorder=4)
            else:
                # The two labels of a model sit apart: the static label ends a quarter bar in
                # from its bar's right edge and runs left, the dynamic label starts a quarter bar
                # in from its bar's left edge and runs right, so no two labels touch.
                lx = x + w * 0.25 if meth == "static" else x - w * 0.25
                ax.annotate(f"{v:+.2f}".replace("-", "\u2212"), (lx, sv), fontsize=ANNOT,
                            color=INK, ha="right" if meth == "static" else "left",
                            va="top" if v < 0 else "bottom",
                            xytext=(0, -2 if v < 0 else 2), textcoords="offset points", zorder=4)
    ax.axhline(0, color=MUTED, lw=0.7, zorder=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([m.replace("MobileViT-", "MobileViT-\n") for m in piv.index], fontsize=8)
    ax.set_ylabel("quality change (pt)")
    ax.set_ylim(FLOOR, 0.9)
    ax.set_yticks([0, -1, -2, -3, -4, -5])
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
    # the empty middle of the band under YOLOv8n / autoencoder / DeiT-tiny, above the clipped-bar
    # label (which sits near the floor, right of MobileNetV2) and left of the MobileViT-XXS bar
    ax.legend(loc="center left", frameon=False, borderaxespad=0.2, labelspacing=0.25,
              bbox_to_anchor=(0.50, 0.42))
    save(fig, "fig2_quality_by_model")


# ------------------------------------------------- Fig 3: format x activation type, 4 threads
def fig3():
    runs = (("quant_factorial_classification.csv", "MobileNetV2"),
            ("quant_factorial_classification_mobilevit.csv", "MobileViT-XXS"))
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.62), sharey=True)
    for ax, (f, title) in zip(axes, runs):
        d = pd.read_csv(ROOT / "results" / "paper" / f)
        d = d[d["reduce_range"] == False]          # noqa: E712 - rr=True repeats the pattern
        xs = [0, 1]
        for fmt, col, ls, mk, lab in (("qdq", C[0], "-", "o", "QDQ"),
                                      ("qoperator", C[1], "--", "s", "QOperator")):
            g = d[d["quant_format"] == fmt].set_index("activation_type")
            ys = [g.loc["uint8", "speedup_t4"], g.loc["int8", "speedup_t4"]]
            ax.plot(xs, ys, ls, color=col, marker=mk, ms=4.2, lw=1.3, label=lab, zorder=3,
                    markeredgewidth=0)
            for x, y in zip(xs, ys):
                if x == 0:     # U8S8: the two formats sit close; label them to the left, apart
                    up = fmt == "qoperator"
                    # the lower label hangs below its point; when the point sits just above the
                    # y = 1 line (MobileNetV2 QDQ, 1.090), drop it clear of the line
                    dy = 1 if up else (-8 if 1.0 < y < 1.15 else -1)
                    ax.annotate(f"{y:.3f}", (x, y), fontsize=ANNOT, color=INK, ha="right",
                                va="bottom" if up else "top",
                                xytext=(-5, dy), textcoords="offset points")
                elif fmt == "qdq":   # S8S8 QDQ is the quantizer's default: label up-right
                    ax.annotate(f"{y:.3f}\n(default)", (x, y), fontsize=ANNOT, color=C[0],
                                ha="left", va="bottom", xytext=(4, 2), textcoords="offset points")
                else:                # S8S8 QOperator: label to the left, under its own line
                    ax.annotate(f"{y:.3f}", (x, y), fontsize=ANNOT, color=INK, ha="right",
                                va="top", xytext=(-6, -1), textcoords="offset points")
        ax.axhline(1.0, color=MUTED, lw=0.7, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(["U8S8", "S8S8"], fontsize=8)
        ax.set_xlim(-0.35, 1.45)
        ax.set_ylim(-0.05, 1.45)
        ax.set_title(title, fontsize=8.5, pad=2)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        clean(ax)
    axes[0].set_ylabel("speedup vs FP32, 4 threads")
    axes[1].legend(loc="upper right", frameon=False, borderaxespad=0.2, labelspacing=0.3)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig3_default_config")


# ---------------------------------------------------------------- Fig 4: version sweep
def fig4():
    d = pd.read_csv(BISECT)
    med = d.groupby(["ort", "family", "variant", "threads"])["p50_ms"].median()
    vers = sorted(d["ort"].unique(), key=vkey)

    def series(fam, var, t):
        mid, lo, hi = [], [], []
        for v in vers:
            fp = med[(v, fam, "p00_fp32", t)]
            s = d[(d.ort == v) & (d.family == fam) & (d.variant == var) & (d.threads == t)]["p50_ms"]
            mid.append(fp / med[(v, fam, var, t)])
            lo.append(fp / s.max())
            hi.append(fp / s.min())
        return mid, lo, hi

    spec = [
        ("mobilevit", "p00_int8",        4, "MobileViT-XXS static, 4 threads", INK,  "-",  "o"),
        ("mobilevit", "p00_int8",        1, "MobileViT-XXS static, 1 thread",  C[0], "--", "s"),
        ("mobilevit", "p00_int8_dyn_mm", 4, "MobileViT-XXS dynamic, 4 threads", C[2], "-.", "^"),
        ("deit",      "p00_int8",        4, "DeiT-tiny static, 4 threads",     C[3], ":",  "D"),
    ]
    fig, ax = plt.subplots(figsize=(COL, 1.74))
    x = list(range(len(vers)))
    ax.axhline(1.0, color=MUTED, lw=0.7, zorder=1)
    i24 = vers.index("1.24.4")
    ax.axvspan(i24 - 0.5, len(vers) - 0.5, color="#f0f0f0", zorder=0)
    for k, (fam, var, t, lab, col, ls, mk) in enumerate(spec):
        mid, lo, hi = series(fam, var, t)
        xo = [xi + (k - 1.5) * 0.06 for xi in x]      # tiny offset so whiskers do not overlap
        ax.errorbar(xo, mid, yerr=[[m - l for m, l in zip(mid, lo)], [h - m for m, h in zip(mid, hi)]],
                    fmt=ls, color=col, marker=mk, ms=3.2, lw=1.2, elinewidth=0.7, capsize=0,
                    label=lab, zorder=3, markeredgewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(vers, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("INT8 speedup vs FP32")
    ax.set_xlabel("ONNX Runtime release")
    # the band below 0.6 holds the legend, clear of the 1.24.4 point (0.742)
    ax.set_ylim(0.48, 1.10)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
    ax.legend(loc="lower left", frameon=False, handlelength=2.4, borderaxespad=0.2,
              labelspacing=0.2, fontsize=7.4)
    save(fig, "fig4_version_sweep")


def main() -> int:
    print("figures ->", OUT)
    df = collect()
    df.to_csv(OUT / "_fig_data.csv", index=False)
    print(f"  {len(df)} model/method rows collected")
    figdf = df[df["in_fig"]]
    fig1(figdf)
    fig2(figdf)
    fig3()
    fig4()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
