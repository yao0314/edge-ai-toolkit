"""Presentation figures for the jury pitch. One message per figure, readable from the back row.

These are not the report's figures. A reader can sit with a two-panel curve; an audience gets a few
seconds per slide, so each of these carries exactly one claim, states it in the title as a sentence
rather than a label, and puts the number that proves it directly on the mark instead of in a legend
the eye has to travel to.

Palette is Okabe-Ito, every series differs by more than hue, and nothing relies on a colour the
projector might not reproduce.

  python scripts/sec_slides.py --out-dir submission/slides
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e8e8e8"
BASE, OURS, HARD, WARN = "#9a9a9a", "#0072B2", "#D55E00", "#CC79A7"
BARE = False   # --bare omits title and subtitle: the slide carries them
CLASSES = ["AnnualCrop", "Forest", "Herbaceous", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


def frame(ax, title, subtitle=None):
    if BARE:
        title, subtitle = None, None
    ax.set_facecolor("white")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=11, length=0)
    # the subtitle sits BELOW the title, so the title's pad has to clear both lines; placing it at
    # a small offset above the axes put it straight through the title on every figure
    if title:
        ax.set_title(title, fontsize=17, color=INK, loc="left", pad=34 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=11.5, color=MUTED,
                va="bottom")


def fig_headline(out, m):
    """One slide: what the submission is worth, as multiples of the model we were given."""
    rows = [("Parameters", m["par"]), ("File size", m["size"]),
            ("Latency on the target device", m["aihub_lat"]), ("Energy per inference", m["energy"])]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    frame(ax, "Valiant Joy, as multiples of the model we were handed",
          f"and it keeps {m['retained']:.2f}% of that model's accuracy")
    y = range(len(rows))[::-1]
    vals = [v for _, v in rows]
    ax.barh(list(y), vals, height=0.52, color=OURS, zorder=3)
    ax.axvline(1.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
    ax.text(1.12, -0.58, "the provided baseline", fontsize=10.5, color=MUTED, ha="left", va="top")
    ax.set_ylim(-0.95, len(rows) - 0.4)
    for yi, (label, v) in zip(y, rows):
        ax.text(v + 0.18, yi, f"{v:.2f}×", fontsize=19, color=INK, va="center", fontweight="bold")
        ax.text(-0.25, yi, label, fontsize=13, color=INK, va="center", ha="right")
    ax.set_yticks([])
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xticks([])
    ax.grid(False)
    for s in ("bottom",):
        ax.spines[s].set_visible(False)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def fig_recovery(out, hard, kd, base_worst):
    """One slide: the same cut survives or collapses depending only on what it learned from."""
    ratios = [50, 60, 70]
    h = [hard[hard.variant == f"p{r}_int8"].iloc[0]["worst_class_recall"] * 100 for r in ratios]
    k = [kd[kd.variant == f"p{r}_int8"].iloc[0]["worst_class_recall"] * 100 for r in ratios]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    frame(ax, "Same cut. Same architecture. Seven points of difference.",
          "weakest of the ten per-class recalls, after pruning that fraction of the channels")
    x = range(len(ratios))
    w = 0.34
    ax.bar([i - w / 2 for i in x], h, w, color=HARD, label="recovered on hard labels", zorder=3)
    ax.bar([i + w / 2 for i in x], k, w, color=OURS, label="recovered from the provided teacher",
           zorder=3)
    ax.axhline(base_worst * 100, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
    ax.text(-0.46, base_worst * 100 - 0.3, "the baseline's own weakest class", fontsize=10.5,
            color=MUTED, ha="left", va="top")
    for i, (a, b) in enumerate(zip(h, k)):
        ax.text(i - w / 2, a + 0.45, f"{a:.1f}", ha="center", fontsize=13, color=HARD)
        ax.text(i + w / 2, b + 0.45, f"{b:.1f}", ha="center", fontsize=13, color=OURS,
                fontweight="bold" if i == 1 else "normal")
    # call out the one that decided the submission
    # the gap between the groups is the only region with nothing in it at any y
    ax.annotate(f"+{k[1] - h[1]:.1f} points,\nfor four lines\nof config",
                xy=(1 + w / 2 - 0.08, k[1] - 0.5), xytext=(0.42, 88.9), fontsize=12.5, color=OURS,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=OURS, linewidth=1.6,
                                connectionstyle="arc3,rad=0.15"))
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r}% pruned" for r in ratios], fontsize=13, color=INK)
    ax.set_ylim(82, 97.5)
    ax.set_ylabel("weakest class  (%)", fontsize=12, color=MUTED)
    ax.legend(frameon=False, fontsize=12, loc="upper right", ncol=1)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def fig_device(out, m):
    """One slide: the development machine understates what compression is worth."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    frame(ax, "Our laptop said 3.1×. The edge device said 8.9×.",
          "speed-up of the submitted model over the provided baseline, same two ONNX files")
    labels = ["Development laptop\ni7-13620H, 4 threads", "Target-class device\nQualcomm QCS6490"]
    vals = [m["lat"], m["aihub_lat"]]
    bars = ax.bar(labels, vals, width=0.44, color=[BASE, OURS], zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.22, f"{v:.2f}×", ha="center", fontsize=24,
                color=INK, fontweight="bold")
    # start the arrow clear of the "3.13x" label rather than through it
    ax.annotate("", xy=(0.93, vals[1] - 0.45), xytext=(0.25, vals[0] + 0.1),
                arrowprops=dict(arrowstyle="->", color=WARN, linewidth=2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(0.5, vals[0] + 2.4, "a big core hides what\ncompression buys", fontsize=12.5,
            color=WARN, ha="center", va="center")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_yticks([])
    ax.grid(False)
    ax.tick_params(labelsize=12.5)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def fig_perclass(out, base_pc, final_pc):
    """One slide: where the accuracy actually went. Stated, not buried."""
    order = sorted(range(10), key=lambda i: final_pc[i])
    fig, ax = plt.subplots(figsize=(13.0, 4.2) if BARE else (11.5, 5.4))
    frame(ax, "What it cost, class by class",
          "per-class recall. One class carries most of the loss, and we say so")
    x = range(10)
    w = 0.36
    ax.bar([i - w / 2 for i in x], [base_pc[j] * 100 for j in order], w, color=BASE,
           label="provided baseline", zorder=3)
    ax.bar([i + w / 2 for i in x], [final_pc[j] * 100 for j in order], w, color=OURS,
           label="Valiant Joy", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([CLASSES[j] for j in order], fontsize=10.5, color=INK, rotation=28,
                       ha="right")
    ax.set_ylim(88, 102.6)
    ax.set_ylabel("recall  (%)", fontsize=12, color=MUTED)
    # above the plot, not inside it: with ten pairs of bars there is no interior region a legend
    # or a callout can occupy without landing on data
    ax.legend(frameon=False, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2)
    worst = order[0]
    ax.annotate(f"the one that pays:\n{final_pc[worst] * 100:.1f}%  against  {base_pc[worst] * 100:.1f}%",
                xy=(0 + w / 2, final_pc[worst] * 100 + 0.25), xytext=(0.52, 97.6),
                fontsize=12.5, color=OURS, ha="center", va="center", zorder=6,
                arrowprops=dict(arrowstyle="->", color=OURS, linewidth=1.5,
                                connectionstyle="arc3,rad=0.18"))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", default="results/sec2026_real/sweep.csv")
    ap.add_argument("--deep", default="results/sec2026_deep/sweep.csv")
    ap.add_argument("--kd", default="results/sec2026_kd/sweep.csv")
    ap.add_argument("--out-dir", default="submission/slides")
    ap.add_argument("--bare", action="store_true", help="no titles: for embedding in a slide")
    a = ap.parse_args()
    global BARE
    BARE = a.bare

    real, deep, kd = (pd.read_csv(p) for p in (a.real, a.deep, a.kd))
    hard = pd.concat([real, deep], ignore_index=True)
    b = real[real.variant == "p00_fp32"].iloc[0]
    f = kd[kd.variant == "p60_int8"].iloc[0]

    m = {
        "par": b["params"] / f["params"],
        "size": 9198475 / f["size_bytes"],          # the provided .pt, as section 5.2 measures it
        "lat": b["latency_p50_ms"] / f["latency_p50_ms"],
        "aihub_lat": 27.969 / 3.127,
        "energy": b["energy_j"] / f["energy_j"],
        "retained": f["accuracy"] / b["accuracy"] * 100,
    }
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_headline(out / "1_headline.png", m)
    fig_recovery(out / "2_recovery.png", hard, kd, b["worst_class_recall"])
    fig_device(out / "3_device.png", m)
    fig_perclass(out / "4_perclass.png",
                 ast.literal_eval(b["per_class_recall"]), ast.literal_eval(f["per_class_recall"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
