"""Figures for the camera-ready rewrite (paper/cr). One style, ten figures, every number from a CSV.

Design rules (paper/cr/OUTLINE.md): each figure carries its own key numbers, so the caption's first
sentence and the figure say the same thing; one palette across the paper - orange = slower,
green = faster, grey = inside the 7 % floor, blue = reference; identity is never colour alone
(position, sign and direct labels carry it too), so the figures survive greyscale print.

    .venv/Scripts/python.exe scripts/paper/cr_figures.py            # all figures
    .venv/Scripts/python.exe scripts/paper/cr_figures.py f3 f6      # some
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "cr" / "figs"
W = 4.8                                    # LNCS text width, 122 mm
INK, MUTED, GRID, BAND = "#1a1a1a", "#5a5a5a", "#d9d9d9", "#ececec"
SLOW, FAST, NEUT, REF = "#d1600f", "#1f8a6d", "#9aa0a6", "#2a6fb8"
SLOW_BG, FAST_BG, REF_BG, NEUT_BG = "#fbe6d8", "#d8efe8", "#dde8f5", "#f1f1f1"
FLOOR = 7.0
MINUS = "−"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def sg(v, f="%+.1f"):
    return (f % v).replace("-", MINUS)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=170)
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter, x=-0.02, y=1.03):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="right", va="bottom")


def cells(path, keycols, val="p50_ms", ok="status"):
    d = defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if ok in r and (r[ok] or "OK") != "OK":
            continue
        d[tuple(r[k] for k in keycols)].append(float(r[val]))
    return d


def box(ax, x, y, w, h, text, fc=NEUT_BG, ec=MUTED, fs=6.8, weight="normal", color=INK, lw=0.7,
        ha="center", pad=0.012):
    """A rounded box whose visible outline is exactly (x, y, w, h): the rounding pad is drawn inside,
    so an arrow placed GAP outside these coordinates never touches the frame."""
    ax.add_patch(FancyBboxPatch((x + pad, y + pad), w - 2 * pad, h - 2 * pad,
                                boxstyle="round,pad=%g,rounding_size=0.018" % pad,
                                fc=fc, ec=ec, lw=lw, transform=ax.transAxes, clip_on=False))
    tx = x + w / 2 if ha == "center" else x + 0.016
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, fontweight=weight, color=color,
            transform=ax.transAxes, linespacing=1.15)


GAP = 0.008                                # clear space between a box outline and an arrow end


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.9, style="-|>", ms=8.5, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color,
                                 lw=lw, transform=ax.transAxes, clip_on=False, shrinkA=0, shrinkB=0,
                                 connectionstyle="arc3,rad=%g" % rad))


def harrow(ax, x_right_of, x_left_of, y, color=MUTED, **kw):
    """Horizontal arrow from the right edge of one box to the left edge of the next, with gaps."""
    arrow(ax, x_right_of + GAP, y, x_left_of - GAP, y, color=color, **kw)


def canvas(h):
    fig, ax = plt.subplots(figsize=(W, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ------------------------------------------------------------------------------------------ F1
def f1():
    """The whole paper in one picture: one file, two releases, two paths; what follows from it."""
    # Latencies are the release sweep's (Section 5, Fig. 3): the same run as Figs. 3, 4 and 10.
    d = cells(ROOT / "results/paper/ort_version_bisect.csv", ("family", "variant", "threads", "ort"))
    before = st.median(d[("mobilevit", "p00_int8", "4", "1.23.2")])
    after = st.median(d[("mobilevit", "p00_int8", "4", "1.24.4")])
    fig, ax = canvas(2.0)
    fx, fy, fw, fh = 0.0, 0.31, 0.17, 0.38
    box(ax, fx, fy, fw, fh, "MobileViT-\nXXS,\nstatic INT8\n\n(the same\nbytes)", fc="white", ec=INK,
        fs=6.6, weight="bold")
    bx, bw, bh = 0.225, 0.235, 0.27
    for y, rel, path, lat, col, bg in ((0.6, "ONNX Runtime 1.23.2", "Transposes of 2-byte\nblocks → scalar loop",
                                        "%.3f ms" % before, INK, NEUT_BG),
                                       (0.05, "ONNX Runtime 1.24.4", "Transposes of 2-byte\nblocks → thread-pool\ntranspose, 832\ndispatches",
                                        "%.3f ms\n+%.0f %%" % (after, (after / before - 1) * 100), SLOW, SLOW_BG)):
        mid = y + bh / 2
        arrow(ax, fx + fw + GAP, fy + fh / 2 + (0.06 if y > 0.3 else -0.06), bx - GAP, mid, color=MUTED)
        ax.text(bx, y + bh + 0.035, rel, fontsize=6.9, fontweight="bold", transform=ax.transAxes, color=col)
        box(ax, bx, y, bw, bh, path, fc=bg, ec=col, fs=6.3, ha="left")
        harrow(ax, bx + bw, 0.505, mid, color=col)
        ax.text(0.505, mid, lat, fontsize=7.4, fontweight="bold", va="center", color=col,
                transform=ax.transAxes, linespacing=1.1)
    ax.plot([0.59, 0.59], [0.02, 0.98], color=GRID, lw=0.8, transform=ax.transAxes)
    finds = (("THE LAW", "Only 2-byte blocks slow down; the\nadded time tracks the dispatches\n(16× dispatches → 13.2× added time).", "§7"),
             ("THE CRITERION", "Readable before timing: 18 third-party\nfiles called safe, none moved.", "§8"),
             ("THE SCALE", "A single-release upgrade moves about one\nunchanged file in eight, FP32 and INT8\nalike; 88 % of those moves reproduce.", "§9"))
    for i, (k, t, sec) in enumerate(finds):
        y = 0.68 - i * 0.33
        ax.text(0.615, y + 0.23, k, fontsize=6.4, fontweight="bold", color=REF, transform=ax.transAxes)
        ax.text(0.995, y + 0.23, sec, fontsize=6.4, color=MUTED, ha="right", transform=ax.transAxes)
        ax.text(0.615, y + 0.19, t, fontsize=6.6, va="top", transform=ax.transAxes, linespacing=1.15)
    save(fig, "f1_overview")


# ------------------------------------------------------------------------------------------ F2
def f2():
    """What INT8 buys: speed at 1 and 4 threads, size, quality, by method and model class."""
    rows = [r for r in csv.DictReader(open(ROOT / "paper/figs/_fig_data.csv", encoding="utf-8"))
            if r["in_fig"] == "True"]
    order = ["ResNet-50", "YOLOv8n", "1-D conv AE", "MobileNetV2", "DeiT-tiny", "MobileViT-XXS"]
    cls = {"ResNet-50": "CNN", "YOLOv8n": "CNN", "1-D conv AE": "CNN", "MobileNetV2": "CNN",
           "DeiT-tiny": "attention", "MobileViT-XXS": "hybrid"}
    get = {(r["model"], r["method"]): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.1), gridspec_kw={"width_ratios": [2.3, 1, 1]})
    y = np.arange(len(order))[::-1]
    ax = axes[0]
    ax.axvline(1, color=INK, lw=0.7)
    for yi, m in zip(y, order):
        s, dnm = get.get((m, "static")), get.get((m, "dynamic"))
        if s:
            ax.plot([float(s["sp4"]), float(s["sp1"])], [yi + 0.14] * 2, color=REF, lw=1.0, alpha=0.5)
            ax.plot(float(s["sp1"]), yi + 0.14, "o", ms=4.2, color=REF, mec="white", mew=0.5)
            ax.plot(float(s["sp4"]), yi + 0.14, "o", ms=4.2, mfc="white", mec=REF, mew=0.9)
            if float(s["sp4"]) < 1:
                ax.annotate("%.2f×" % float(s["sp4"]), (float(s["sp4"]), yi + 0.14), xytext=(0, 4.5),
                            textcoords="offset points", ha="center", va="bottom", fontsize=6.4, color=SLOW)
        if dnm:
            ax.plot(float(dnm["sp1"]), yi - 0.16, "D", ms=3.6, color=NEUT, mec=MUTED, mew=0.5)
            if float(dnm["sp1"]) < 0.2:
                ax.annotate("%.0f× slower" % (1 / float(dnm["sp1"])), (float(dnm["sp1"]), yi - 0.16),
                            xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.4, color=SLOW)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.04, 5)
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2", "4"])
    ax.set_yticks(y)
    ax.set_yticklabels(["%s (%s)" % (m, cls[m]) for m in order], fontsize=6.9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("speedup over FP32 (log)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], ls="", marker="o", color=REF, ms=4, label="static, 1 thread"),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec=REF, ms=4, label="static, 4 threads"),
               plt.Line2D([], [], ls="", marker="D", color=NEUT, mec=MUTED, ms=3.5, label="dynamic, 1 thread")]
    panel(ax, "a")
    for ax, key, lab, letter in ((axes[1], "size", "size (× FP32)", "b"),
                                 (axes[2], "drop", "quality (pt)", "c")):
        for yi, m in zip(y, order):
            for dy, meth, mk in ((0.14, "static", "o"), (-0.16, "dynamic", "D")):
                r = get.get((m, meth))
                if not r:
                    continue
                v = float(r[key])
                c = REF if meth == "static" else NEUT
                if key == "drop" and v < -1:
                    left = v > -8                  # keep the label clear of the zero line
                    ax.annotate(sg(v, "%+.1f"), (v, yi + dy), xytext=(-4 if left else 4, 0),
                                textcoords="offset points", ha="right" if left else "left",
                                va="center", fontsize=6.3, color=SLOW)
                ax.plot(v, yi + dy, mk, ms=3.8 if mk == "o" else 3.3, color=c,
                        mec="white" if mk == "o" else MUTED, mew=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(lab)
        panel(ax, letter)
    axes[1].set_xlim(0.2, 0.72)
    axes[1].axvspan(0.256, 0.322, color=BAND, lw=0, zorder=0)
    axes[1].text(0.289, -0.75, "26–32 %", ha="center", fontsize=6.2, color=MUTED)
    axes[1].set_ylim(-0.95, len(order) - 0.4)
    axes[0].set_ylim(-0.95, len(order) - 0.4)
    axes[2].set_ylim(-0.95, len(order) - 0.4)
    axes[2].axvline(0, color=INK, lw=0.7)
    axes[2].set_xlim(-13.5, 2)
    axes[2].set_xticks([-12, -8, -4, 0])
    fig.tight_layout(w_pad=1.1)
    fig.subplots_adjust(top=0.84)
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=6.6, handletextpad=0.2,
               columnspacing=1.4, bbox_to_anchor=(0.55, 1.0))
    save(fig, "f2_what_int8_buys")


# ------------------------------------------------------------------------------------------ F3
def f3():
    """The same MobileViT file across eleven releases: two steps, two causes."""
    d = cells(ROOT / "results/paper/ort_version_bisect.csv", ("family", "variant", "threads", "ort"))
    rels = sorted({k[3] for k in d if k[0] == "mobilevit"}, key=lambda v: [int(x) for x in v.split(".")])
    med = lambda v, t, r: st.median(d[("mobilevit", v, t, r)])
    x = np.arange(len(rels))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.95), gridspec_kw={"width_ratios": [1.25, 1]})
    q = [med("p00_int8", "4", r) for r in rels]
    f = [med("p00_fp32", "4", r) for r in rels]
    i23, i24, i25 = rels.index("1.23.2"), rels.index("1.24.4"), rels.index("1.25.1")
    for ax in (a1, a2):
        ax.axvspan(i24 - 0.5, i25 + 0.5, color=BAND, lw=0, zorder=0)
    a1.plot(x, q, "-o", color=SLOW, ms=3.2, lw=1.3, label="static INT8")
    a1.plot(x, f, "-s", color=REF, ms=2.8, lw=1.3, label="FP32, same weights")
    a1.annotate("static file\n+31 %", (i24 - 0.5, (q[i23] + q[i24]) / 2), xytext=(i24 + 2.2, 1.78),
                fontsize=6.6, color=SLOW, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=SLOW, lw=0.6))
    a1.annotate("FP32 baseline\n" + MINUS + "15 %", (i25 - 0.5, (f[i24] + f[i25]) / 2), xytext=(i25 + 2.4, 1.52),
                fontsize=6.6, color=REF, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=REF, lw=0.6))
    a1.set_ylabel("median latency, 4 threads (ms)")
    a1.set_ylim(1.2, 2.2)
    a1.legend(frameon=False, loc="upper left", fontsize=6.4)
    sp = [fv / qv for fv, qv in zip(f, q)]
    a2.axhline(1, color=INK, lw=0.7)
    a2.plot(x, sp, "-o", color=INK, ms=3.2, lw=1.3)
    for i, off, ha in ((i23, (-3, -11), "right"), (i24, (5, 1), "left"), (i25, (4, -11), "left")):
        a2.annotate("%.3f×" % sp[i], (x[i], sp[i]), xytext=off, textcoords="offset points", ha=ha,
                    fontsize=6.6, fontweight="bold" if i == i25 else "normal")
    a2.set_ylabel("INT8 speedup over FP32")
    a2.set_ylim(0.5, 1.12)
    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels(rels, rotation=55, ha="right", fontsize=6.2)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=1.0)
    save(fig, "f3_unchanged_file")


# ------------------------------------------------------------------------------------------ F4
def f4():
    """The search for the 1.24.4 step (the submitted paper's Table 3): three explanations fall, the
    profile and the isolation probes narrow the step to four Transposes and one pull request."""
    fig, ax = canvas(2.3)
    box(ax, 0.0, 0.64, 0.29, 0.3, "1.24.4 makes the static\nfile 31 % slower at\nfour threads",
        fc=SLOW_BG, ec=SLOW, fs=6.6, weight="bold")
    sus = (("graph changed", "identical on 1.23.2, 1.24.4, 1.25.1: 698 nodes"),
           ("thread spinning", "step persists with spinning off (+23.7 %)"),
           ("quantized MatMuls", "MatMul-only static file: no step (" + MINUS + "1 %);\nfull file +35 % in the same run"))
    ax.text(0.335, 0.965, "Three explanations, each tested against a result fixed in advance:", fontsize=6.4,
            color=MUTED, transform=ax.transAxes)
    for i, (s, why) in enumerate(sus):
        y = 0.87 - i * 0.105
        ax.text(0.337, y, "×", fontsize=9, color=SLOW, transform=ax.transAxes, va="center", fontweight="bold")
        ax.text(0.362, y, s, fontsize=6.5, transform=ax.transAxes, va="center", fontweight="bold")
        ax.text(0.585, y, why, fontsize=6.2, transform=ax.transAxes, va="center", color=MUTED, linespacing=1.1)
    arrow(ax, 0.145, 0.64 - GAP - 0.004, 0.145, 0.40 + GAP + 0.004, color=MUTED)
    ax.text(0.16, 0.52, "what the profile\nand the probes show", fontsize=6.1, color=MUTED,
            transform=ax.transAxes, va="center")
    chain = (("profile, 4 threads:\nTranspose has 62 %\nof the added time;\nthe rest +9 %", NEUT_BG, MUTED, "normal"),
             ("the 56 uint8\nTransposes alone:\n87 % of the step;\nnone at 1 thread", NEUT_BG, MUTED, "normal"),
             ("the 4 with a\n2-byte block:\n+0.417 ms; the\nother 52: +0.007", SLOW_BG, SLOW, "bold"),
             ("PR #25680\nroutes 2-byte\nblocks to a multi-\nthreaded transpose", "white", INK, "bold"))
    bw, step = 0.228, 0.257
    for i, (t, fc, ec, wt) in enumerate(chain):
        x = 0.0 + i * step
        box(ax, x, 0.02, bw, 0.38, t, fc=fc, ec=ec, fs=6.3, weight=wt)
        if i < 3:
            harrow(ax, x + bw, x + step, 0.2, color=MUTED)
    save(fig, "f4_search")


# ------------------------------------------------------------------------------------------ F5
def f5():
    """The mechanism: how wide a moved block is, and which path each width takes."""
    fig = plt.figure(figsize=(W, 1.95))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # left: one outer loop of the tensor, before and after the transpose; one block and where it goes
    ax.text(0.0, 0.95, "dims [L, 2, N, K], perm [0, 2, 1, 3]", fontsize=7.0, fontweight="bold",
            transform=ax.transAxes)
    bw_, ch, bgap, rgap = 0.042, 0.1, 0.01, 0.045        # block width, row height, gaps

    def block(x, y, hot, lab):                             # one block = K contiguous elements
        ax.add_patch(Rectangle((x, y), bw_, ch, transform=ax.transAxes, lw=0.9 if hot else 0.6,
                               fc=SLOW_BG if hot else NEUT_BG, ec=SLOW if hot else MUTED))
        ax.text(x + bw_ / 2, y + ch / 2, lab, ha="center", va="center", fontsize=6.3, transform=ax.transAxes,
                color=SLOW if hot else MUTED, fontweight="bold" if hot else "normal")
    step_ = bw_ + bgap
    ix, iy = 0.0, 0.6                                      # input: 2 rows x N = 4 blocks
    for r in range(2):
        for n in range(4):
            block(ix + n * step_, iy - r * (ch + rgap), (r, n) == (0, 1), "%s%d" % ("ab"[r], n))
    ax.text(ix + 2 * step_ - bgap / 2, iy + ch + 0.075, "input: 2 rows of N blocks", ha="center",
            fontsize=6.3, color=MUTED, transform=ax.transAxes)
    ox, oy, orow = 0.265, 0.66, 0.15                       # output: N rows x 2 blocks
    for n in range(4):
        for r in range(2):
            block(ox + r * step_, oy - n * orow, (r, n) == (0, 1), "%s%d" % ("ab"[r], n))
    ax.text(ox + step_ - bgap / 2, oy + ch + 0.04, "output: N rows of 2", ha="center", fontsize=6.3,
            color=MUTED, transform=ax.transAxes)
    src = (ix + step_ + bw_ / 2, iy + ch + GAP)
    dst = (ox - GAP, oy - orow + ch / 2)
    arrow(ax, src[0], src[1], dst[0], dst[1], color=SLOW, lw=1.0, rad=-0.35)
    ax.text(ix + 2 * step_ - bgap / 2, iy - (ch + rgap) - 0.05, "one moved block =\nK elements × b bytes",
            ha="center", va="top", fontsize=6.6, color=SLOW, fontweight="bold", transform=ax.transAxes,
            linespacing=1.1)
    ax.text(0.0, 0.12, "one index of L = one outer loop", fontsize=6.3, color=MUTED, transform=ax.transAxes)
    # right: the routing table
    tx = 0.405
    cols = (("moved\nblock", 0.0, 0.115), ("1.23.2", 0.125, 0.2), ("1.24.4,\nafter PR #25680", 0.335, 0.235))
    for lab, dx, w in cols:
        ax.text(tx + dx + w / 2, 0.925, lab, ha="center", va="center", fontsize=6.9, fontweight="bold",
                color=SLOW if "PR" in lab else INK, transform=ax.transAxes, linespacing=1.1)
    rows = (("1 byte", "MLAS, thread pool", "MLAS, thread pool", False),
            ("2 bytes", "scalar loop", "MLAS, thread pool", True),
            ("4 bytes", "MLAS, thread pool", "MLAS, thread pool", False),
            ("8 bytes", "scalar loop", "scalar loop", False))
    for i, (w, a, b, hot) in enumerate(rows):
        y = 0.68 - i * 0.165
        for (lab, dx, wd), txt in zip(cols, (w, a, b)):
            is_new = hot and "PR" in lab
            fc = SLOW_BG if is_new else (NEUT_BG if "scalar" in txt else "white")
            box(ax, tx + dx, y, wd, 0.13, txt, fc=fc, ec=SLOW if is_new else GRID, fs=6.8,
                weight="bold" if (is_new or "block" in lab) else "normal", lw=1.1 if is_new else 0.6)
    save(fig, "f5_mechanism")


# ------------------------------------------------------------------------------------------ F6
def f6():
    """The width law on constructed files: ladder, dose response, matched dispatch count."""
    w1 = cells(ROOT / "results/expW/batches/w01_main.csv", ("variant", "threads", "ort"))
    w2 = cells(ROOT / "results/expW/batches/w02_dose.csv", ("variant", "threads", "ort"))
    m = lambda d, v, t, r: st.median(d[(v, t, r)])
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W, 2.05), gridspec_kw={"width_ratios": [1.45, 1, 0.95]})
    # "c" = the Transpose-free control of each dtype: the release made every sub-0.04 ms file 4-10 %
    # faster, Transpose or not, so the control is drawn beside the ladder it calibrates.
    ladder = [("ctrl_u8", "u8", "c"), ("m_u8_w1", "u8", 1), ("m_u8_w2", "u8", 2), ("m_u8_w4", "u8", 4),
              ("m_u8_w8", "u8", 8), ("ctrl_f16", "f16", "c"), ("m_f16_w2", "f16", 2), ("m_f16_w4", "f16", 4),
              ("m_f16_w8", "f16", 8), ("ctrl_f32", "f32", "c"), ("m_f32_w4", "f32", 4), ("m_f32_w8", "f32", 8)]
    xs, labels, xpos, prev, groups = [], [], 0.0, None, defaultdict(list)
    for var, dt, wid in ladder:
        if prev and dt != prev:
            xpos += 0.6
        r = m(w1, var, "4", "1.24.4") / m(w1, var, "4", "1.23.2")
        col = SLOW if wid == 2 else ("white" if wid == "c" else NEUT)
        a.bar(xpos, r - 1, bottom=1, width=0.72, color=col, ec=MUTED if wid == "c" else col, lw=0.6,
              hatch="////" if wid == "c" else None, zorder=2)
        if wid == 2:
            a.text(xpos, r * 1.13, "%.1f×" % r, ha="center", fontsize=6.6, color=SLOW, fontweight="bold")
        xs.append(xpos)
        labels.append(str(wid))
        groups[dt].append(xpos)
        prev = dt
        xpos += 1
    a.set_yscale("log")
    a.yaxis.set_minor_formatter(NullFormatter())
    a.set_ylim(0.7, 30)
    a.axhspan(0.93, 1.07, color=BAND, lw=0, zorder=0)
    a.axhline(1, color=INK, lw=0.7, zorder=1)
    a.set_xticks(xs)
    a.set_xticklabels(labels, fontsize=6.4)
    a.set_xlabel("moved block (bytes); c = no Transpose", labelpad=13)
    for dt, name in (("u8", "uint8"), ("f16", "float16"), ("f32", "float32")):
        g = groups[dt]
        a.text((g[0] + g[-1]) / 2, -0.2, name, ha="center", fontsize=6.4, color=MUTED,
               transform=a.get_xaxis_transform())
    a.set_ylabel("latency ratio, 1.24.4 / 1.23.2")
    a.set_yticks([1, 2, 5, 10, 20])
    a.set_yticklabels(["1", "2", "5", "10", "20"])
    a.grid(axis="x", visible=False)
    panel(a, "a")
    disp = {"d_u8_w2_L64": 256, "d_u8_w2_L256": 1024, "d_u8_w2_L1024": 4096}
    nn = np.array([180, 6000])
    b.plot(nn, nn * 0.43e-3, "--", color=MUTED, lw=0.8, zorder=1)
    for t, mk, fc in (("4", "o", SLOW), ("2", "s", "white")):
        pts = [(n, m(w2, v, t, "1.24.4") - m(w2, v, t, "1.23.2")) for v, n in disp.items()]
        b.plot([p[0] for p in pts], [p[1] for p in pts], mk, ms=4.2, mfc=fc, mec=SLOW, mew=0.9,
               label="%s threads" % t, zorder=3)
    b.text(2300, 2300 * 0.43e-3 * 0.52, "0.43 µs\nper dispatch", fontsize=6.0, color=MUTED, ha="left")
    b.set_xscale("log")
    b.set_yscale("log")
    for axis in (b.xaxis, b.yaxis):
        axis.set_minor_formatter(NullFormatter())
    b.set_xlim(170, 7000)
    b.set_ylim(0.07, 3)
    b.set_xlabel("dispatches added per inference")
    b.set_ylabel("latency added (ms)")
    b.set_xticks([256, 1024, 4096])
    b.set_xticklabels(["256", "1024", "4096"])
    b.set_yticks([0.1, 0.3, 1, 2])
    b.set_yticklabels(["0.1", "0.3", "1", "2"])
    b.text(0.04, 0.96, "16× dispatches\n→ 13.2×\nadded time", transform=b.transAxes, va="top",
           fontsize=6.6, fontweight="bold")
    b.legend(frameon=False, loc="lower right", fontsize=6.2, handletextpad=0.1)
    panel(b, "b")
    vals = [("1-byte\n(always\npool)", m(w2, "d_u8_w1_L1024", "4", "1.23.2"),
             m(w2, "d_u8_w1_L1024", "4", "1.24.4"), NEUT),
            ("2-byte\n(re-\nrouted)", m(w2, "d_u8_w2_L1024", "4", "1.23.2"),
             m(w2, "d_u8_w2_L1024", "4", "1.24.4"), SLOW)]
    for i, (lab, before, after, col) in enumerate(vals):
        c.bar(i - 0.19, before, width=0.36, color=col, alpha=0.35, zorder=2)
        c.bar(i + 0.19, after, width=0.36, color=col, zorder=2)
        if i == 0:
            c.text(i, max(before, after) + 0.06, "%.2f → %.2f" % (before, after), ha="center", fontsize=6.0)
        else:
            c.text(i - 0.19, before + 0.06, "%.3f" % before, ha="center", fontsize=6.2, color=MUTED)
            c.text(i + 0.19, after + 0.06, "%.2f" % after, ha="center", fontsize=6.4, fontweight="bold", color=SLOW)
    c.set_xticks([0, 1])
    c.set_xticklabels([v[0] for v in vals], fontsize=6.1, linespacing=1.0)
    c.set_ylabel("latency, 4 threads (ms)")
    c.set_ylim(0, 2.75)
    c.set_xlim(-0.6, 1.6)
    c.grid(axis="x", visible=False)
    c.legend(handles=[Rectangle((0, 0), 1, 1, fc=MUTED, alpha=0.35, lw=0), Rectangle((0, 0), 1, 1, fc=MUTED, lw=0)],
             labels=["1.23.2", "1.24.4"], loc="upper center", ncol=2, frameon=False, fontsize=6.0,
             handlelength=0.9, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)
    panel(c, "c")
    fig.tight_layout(w_pad=0.8)
    save(fig, "f6_width_law")


# ------------------------------------------------------------------------------------------ F7
def f7():
    """Eighteen third-party files the criterion called safe; none moved. Controls for scale."""
    d = cells(ROOT / "results/paper/a2_thirdparty_exploratory.csv", ("family", "variant", "threads", "ort"))
    ctl = {"mobilevit": "MobileViT-XXS (ours, flagged)", "deit": "DeiT-tiny (ours)", "cls": "MobileNetV2 (ours)"}
    pts = []
    for (fam, var, th, ort), v in d.items():
        if th != "4" or ort != "1.23.2" or (fam, var, th, "1.24.4") not in d:
            continue
        pct = (st.median(d[(fam, var, th, "1.24.4")]) / st.median(v) - 1) * 100
        pts.append((pct, ctl.get(fam, var), fam in ctl))
    fig, ax = plt.subplots(figsize=(W, 1.8))
    ax.axvspan(-FLOOR, FLOOR, color=BAND, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.6)
    third = [p for p in pts if not p[2]]
    ctrl = sorted([p for p in pts if p[2]], key=lambda p: p[0])
    rng = np.random.default_rng(3)
    ax.plot([p[0] for p in third], rng.uniform(-0.26, 0.26, len(third)), "o", ms=4.4, color=NEUT,
            mec="white", mew=0.5, zorder=3)
    for i, (pct, name, _) in enumerate(ctrl):
        yy = 0.95 + 0.3 * i
        col = SLOW if pct >= FLOOR else REF
        ax.plot(pct, yy, "o", ms=5, color=col, mec="white", mew=0.5, zorder=3)
        ax.text(pct + (1.2 if pct < 20 else -1.2), yy, "%s  %s %%" % (name, sg(pct)), va="center",
                ha="left" if pct < 20 else "right", fontsize=6.6, color=SLOW if pct >= FLOOR else INK)
    lo, hi = min(p[0] for p in third), max(p[0] for p in third)
    ax.text(9.5, 0.0, "18 third-party static INT8 files\n15 families · 9 publishers · 11 toolchains\n"
            "all inside the floor: %s %% to %s %%" % (sg(lo), sg(hi)), va="center", fontsize=6.6, linespacing=1.2)
    ax.text(FLOOR + 0.4, 1.78, "7 % floor", ha="left", fontsize=6.2, color=MUTED)
    ax.annotate("flagged, constructed (§7):\n8 of 8 cells moved,\n4.6× to 44×, off scale",
                xy=(38, 0.5), xytext=(36.6, 0.5), ha="right", va="center", fontsize=6.4, color=SLOW,
                fontweight="bold", linespacing=1.15, xycoords="data",
                arrowprops=dict(arrowstyle="-|>", color=SLOW, lw=0.8, mutation_scale=7, shrinkA=2, shrinkB=0))
    ax.set_yticks([0, 1.25])
    ax.set_yticklabels(["called safe\nby the check", "our controls"], fontsize=6.9)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.45, 1.9)
    ax.set_xlim(-12, 38)
    ax.set_xlabel("change in four-thread latency, 1.23.2 → 1.24.4 (%)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "f7_criterion")


# ------------------------------------------------------------------------------------------ F8
MODEL = {"deit224": "DeiT-tiny 224", "mnv2": "MobileNetV2", "mv256": "MobileViT 256", "mv64": "MobileViT 64",
         "r50": "ResNet-50", "y256": "YOLOv8n", "ae": "1-D conv AE"}
VARIANT = {"p00_fp32": "FP32", "p00_int8": "static", "p00_int8_dyn_mm": "dynamic", "p00_int8_dyn_conv": "dynamic"}
SHAPE = {"c1_64x64": "64×64", "c224_224x3": "224×224", "c3_112x112": "112×112",
         "c3_224x224": "224×224", "c3_240x320": "240×320", "c3_299x299": "299×299",
         "c3_320x320": "320×320", "c56_56x3": "56×56"}


def f8():
    """The drift map: every unchanged file against every single release step, four threads."""
    from matplotlib.colors import LinearSegmentedColormap
    a = json.load(open(ROOT / "results/expB/analysis/b_analysis.json", encoding="utf-8"))
    pcs = [p for p in a["pair_cells"] if p["kind"] == "adjacent" and p["series"].endswith("/4")]
    rels = ["1.20.1", "1.21.1", "1.22.1", "1.23.2", "1.24.4", "1.25.1", "1.26.0", "1.27.0", "1.28.0", "1.30.0"]
    trans = list(zip(rels, rels[1:]))
    series = sorted({p["series"] for p in pcs},
                    key=lambda s: ({"CONTRAST": 0, "APPENDIX": 1, "BREADTH": 2}[s.split("/")[0]], s))
    cell = {(p["series"], p["a"], p["b"]): p for p in pcs}
    cap = 50.0
    cmap = LinearSegmentedColormap.from_list("d", [FAST, "#f3f3f3", SLOW])
    fig, ax = plt.subplots(figsize=(W, 3.85))
    M = np.full((len(series), len(trans)), np.nan)
    for i, s in enumerate(series):
        for j, (ra, rb) in enumerate(trans):
            p = cell.get((s, ra, rb))
            if p:
                M[i, j] = p["delta"]
    ax.imshow(np.clip(M, -cap, cap), cmap=cmap, vmin=-cap, vmax=cap, aspect="auto")
    for i, s in enumerate(series):
        for j, (ra, rb) in enumerate(trans):
            p = cell.get((s, ra, rb))
            if not p:
                ax.text(j, i, "·", ha="center", va="center", fontsize=6, color=NEUT)
                continue
            if p["verdict"] == "MOVED":
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.0))
            if abs(p["delta"]) >= FLOOR:
                ax.text(j, i, sg(p["delta"], "%+.0f"), ha="center", va="center", fontsize=5.9,
                        color="white" if abs(p["delta"]) > 30 else INK,
                        fontweight="bold" if p["verdict"] == "MOVED" else "normal")
    names = []
    for s in series:
        arm, fam, var, _ = s.split("/")
        if arm == "BREADTH":
            names.append("%s  %s" % (var, SHAPE.get(fam, fam)))
        else:
            names.append("%s, %s" % (MODEL.get(fam, fam), VARIANT.get(var.replace("256_", ""), var)))
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels(names, fontsize=6.0)
    ax.set_xticks(range(len(trans)))
    ax.set_xticklabels(["%s→\n%s" % t for t in trans], fontsize=5.6)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    nc = sum(1 for s in series if s.startswith("CONTRAST"))
    na = sum(1 for s in series if s.startswith("APPENDIX"))
    for yline in (nc - 0.5, nc + na - 0.5):
        ax.axhline(yline, color=INK, lw=0.8)
    ax.set_xlim(-0.5, len(trans) - 0.5)
    save(fig, "f8_drift_map")


# ------------------------------------------------------------------------------------------ F9
def f9():
    """Replication: the map's change against an independent run's change, every moved cell."""
    r = json.load(open(ROOT / "results/expB/analysis/b_replication.json", encoding="utf-8"))
    pcs = [p for p in r["pair_cells"] if p["rep_delta"] is not None]
    fig, ax = plt.subplots(figsize=(W * 0.6, 2.5))
    ax.axhspan(-FLOOR, FLOOR, color=BAND, lw=0, zorder=0)
    ax.axvspan(-FLOOR, FLOOR, color=BAND, lw=0, zorder=0)
    ax.plot([-60, 105], [-60, 105], color=MUTED, lw=0.7, ls="--", zorder=1)
    groups = (("REPLICATED", lambda p: p["delta"] > 0, "o", SLOW, "replicated, slower"),
              ("REPLICATED", lambda p: p["delta"] < 0, "o", FAST, "replicated, faster"),
              ("NOT-REPLICATED", lambda p: True, "x", INK, "not replicated"),
              ("RESOLVED-MOVED", lambda p: True, "^", REF, "unresolved → moved"),
              ("RESOLVED-STILL", lambda p: True, "v", NEUT, "unresolved → still"))
    for lab, keep, mk, col, txt in groups:
        pts = [p for p in pcs if p["label"] == lab and keep(p)]
        if pts:
            ax.scatter([p["delta"] for p in pts], [p["rep_delta"] for p in pts], marker=mk, s=11, c=col,
                       linewidths=0.8 if mk == "x" else 0.3, edgecolors=None if mk == "x" else "white",
                       zorder=3, label="%s (%d)" % (txt, len(pts)))
    ax.set_xlim(-60, 105)
    ax.set_ylim(-60, 105)
    ax.set_xlabel("change in the map (%)")
    ax.set_ylabel("change in the independent run (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=5.9, handletextpad=0.1, borderaxespad=0.2)
    ax.text(0.97, 0.05, "|Δ| ≥ 20 %: 28 of 28 reproduce\nsingle-release moves: 52 of 59\nall moves: 69 of 78\n"
            "between runs: median 1.7 pp, max 10.7",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, fontweight="bold", linespacing=1.15)
    ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "f9_replication")


# ------------------------------------------------------------------------------------------ F10
def f10():
    """Diagnosing a drift: three signatures, one procedure, a worked case on each branch."""
    def ops(path, a, b):
        by = defaultdict(lambda: defaultdict(list))
        for r in csv.DictReader(open(path, encoding="utf-8")):
            by[r["op_type"]][r["ort"]].append(float(r["ms_per_inf"]))
        d = {op: st.median(v.get(b, [0])) - st.median(v.get(a, [0])) for op, v in by.items()}
        return sorted(d.items(), key=lambda kv: -abs(kv[1]))[:3], sum(d.values())
    cases = (("graph differs", "FUSION FALLBACK",
              "b05, 299 px, 1.21.1 → 1.22.1:\n+96 % at 1 thread; 427 → 491\nnodes: every QLinearConv becomes\na float Conv inside Q/DQ",
              ROOT / "results/expW/b05_profile_t1.csv", "1.21.1", "1.22.1", "1 thread"),
             ("same graph,\nmoves at 1 thread", "KERNEL CHANGE",
              "YOLOv8n, 1.23.2 → 1.24.4:\n" + MINUS + "21 % at 1 thread; 283 nodes on\nboth releases; Resize, 7.3×\nfaster, is 97–99 % of it",
              ROOT / "results/expW/y256_profile_t1.csv", "1.23.2", "1.24.4", "1 thread"),
             ("same graph,\nmoves far more at\nseveral threads", "THREAD-POOL DISPATCH",
              "MobileViT-XXS, 1.23.2 → 1.24.4:\n+31 % at 4 threads; 698 nodes\non both releases; Transpose\nis 62 % of it",
              ROOT / "results/paper/node_profile_version.csv", "1.23.2", "1.24.4", "4 threads (profiled)"))
    fig = plt.figure(figsize=(W, 2.45))
    top = fig.add_axes([0, 0.9, 1, 0.1])
    top.axis("off")
    steps = ("1  compare the executed graphs", "2  compare one thread with several", "3  profile per operator")
    sw, sstep = 0.3, 0.345
    for i, s in enumerate(steps):
        x = 0.0 + i * sstep
        box(top, x, 0.1, sw, 0.8, s, fc="white", ec=INK, fs=6.9, weight="bold")
        if i < 2:
            harrow(top, x + sw, x + sstep, 0.5, color=INK)
    for k, (sig, kind, desc, path, a, b, thr) in enumerate(cases):
        y0 = 0.62 - k * 0.3
        left = fig.add_axes([0, y0, 0.58, 0.26])
        left.axis("off")
        left.set_xlim(0, 1)
        left.set_ylim(0, 1)
        box(left, 0.0, 0.08, 0.26, 0.84, sig, fc=NEUT_BG, ec=MUTED, fs=6.6)
        harrow(left, 0.26, 0.31, 0.5, color=MUTED)
        left.text(0.33, 0.9, kind, fontsize=6.9, fontweight="bold", color=REF, va="top", transform=left.transAxes)
        left.text(0.33, 0.68, desc, fontsize=6.3, va="top", transform=left.transAxes, linespacing=1.15)
        axb = fig.add_axes([0.715, y0 + 0.03, 0.27, 0.2])
        top3, tot = ops(path, a, b)
        names = [o for o, _ in top3][::-1]
        vals = [v for _, v in top3][::-1]
        axb.barh(range(3), vals, color=[SLOW if v > 0 else FAST for v in vals], height=0.62, zorder=2)
        axb.axvline(0, color=INK, lw=0.6)
        axb.set_yticks(range(3))
        axb.set_yticklabels(names, fontsize=6.2)
        axb.tick_params(axis="y", length=0)
        axb.tick_params(axis="x", labelsize=6.0)
        axb.grid(axis="y", visible=False)
        lim = max(abs(v) for v in vals) * 1.55
        axb.set_xlim(-lim if min(vals) < 0 else -lim * 0.3, lim if max(vals) > 0 else lim * 0.3)
        axb.set_title("per-operator change (ms), %s" % thr, fontsize=6.2, color=MUTED, loc="left", pad=2)
    save(fig, "f10_diagnosis")


FIGS = {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "f6": f6, "f7": f7, "f8": f8, "f9": f9, "f10": f10}

if __name__ == "__main__":
    for k in sys.argv[1:] or list(FIGS):
        FIGS[k]()
