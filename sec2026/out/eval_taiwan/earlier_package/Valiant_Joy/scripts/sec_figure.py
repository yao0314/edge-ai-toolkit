"""The tradeoff figure for the submission report.

Two panels over one x-axis and two recovery methods, because the figure has to carry the claim the
report is built on: the collapse at 60% looks like a capacity limit and is a RECOVERY limit. The
same prune, recovered against the provided teacher's soft targets instead of hard labels, keeps a
model the hard-label run threw away.

The top panel is the mean everyone optimises, with the +/-0.36-point band on a difference between
two variants over 5,400 images drawn on it - without that band a reader sees the bump at 10% and
concludes pruning improved the model. The bottom panel is the weakest of the ten per-class recalls,
which is where both the cliff and the recovery actually show up.

  python scripts/sec_figure.py --csv results/sec2026_real/sweep.csv \
      --kd-csv results/sec2026_kd/sweep.csv --chosen p60_int8 --out submission/tradeoff.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Okabe-Ito. The two recovery methods also differ by dash and marker, so the panels survive
# greyscale printing and the common colour-vision deficiencies without relying on hue.
INK, HARD, KD, WARN = "#1a1a1a", "#888888", "#0072B2", "#D55E00"


def series(frames, prec):
    out = []
    for label, d in frames:
        s = d[d["variant"].astype(str).str.endswith(prec)].copy()
        s["ratio"] = s["variant"].astype(str).str.extract(r"p(\d+)_").astype(float) / 100.0
        out.append((label, s.sort_values("ratio")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="the hard-label sweep")
    ap.add_argument("--deep-csv", help="its extension past 50%")
    ap.add_argument("--kd-csv", required=True, help="the teacher-recovered sweep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chosen", default="p60_int8")
    ap.add_argument("--baseline", default="p00_fp32")
    ap.add_argument("--bare", action="store_true", help="no title: for embedding in a slide")
    a = ap.parse_args()

    hard = pd.read_csv(a.csv)
    if a.deep_csv and Path(a.deep_csv).exists():
        hard = pd.concat([hard, pd.read_csv(a.deep_csv)], ignore_index=True)
    kd = pd.read_csv(a.kd_csv)
    base = hard[hard["variant"] == a.baseline].iloc[0]
    chosen = kd[kd["variant"] == a.chosen].iloc[0]

    n = 5400
    half = 1.96 * math.sqrt(2 * base["accuracy"] * (1 - base["accuracy"]) / n) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.9), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#cccccc")
        ax.grid(axis="y", color="#eeeeee", linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#555555", labelsize=9)

    ax1.axhspan((base["accuracy"] * 100) - half, (base["accuracy"] * 100) + half,
                color="#bbbbbb", alpha=0.22, linewidth=0, zorder=0)
    ax1.axhline(base["accuracy"] * 100, color="#999999", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax1.text(0.705, base["accuracy"] * 100 + half + 0.08,
             f"indistinguishable from the provided baseline  (±{half:.2f} pp, n=5,400)",
             fontsize=8, color="#666666", ha="right", va="bottom")

    styles = [("recovered on hard labels", HARD, "o", (0, (5, 2)), 1.8),
              ("recovered from the teacher (KD)", KD, "s", "-", 2.2)]
    for (label, colour, marker, style, lw), (_, s) in zip(styles, series([("h", hard), ("k", kd)], "int8")):
        for ax, col in ((ax1, "accuracy"), (ax2, "worst_class_recall")):
            ax.plot(s["ratio"], s[col] * 100, linestyle=style, color=colour, marker=marker,
                    markersize=5, linewidth=lw, label=label, zorder=3)

    ax1.set_ylabel("top-1 accuracy  (%)", fontsize=9, color="#333333")
    ax1.set_ylim(94.9, 99.85)
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    if not a.bare:
        ax1.set_title("The collapse at 60% is a recovery limit, not a capacity limit",
                      fontsize=11.5, color=INK, loc="left", pad=10)

    ax2.axhline(base["worst_class_recall"] * 100, color="#999999", linewidth=1,
                linestyle=(0, (4, 3)), zorder=1)
    ax2.text(0.008, base["worst_class_recall"] * 100 + 0.35,
             "the provided baseline's own weakest class", fontsize=8, color="#666666", va="bottom")
    ax2.set_ylabel("weakest of the 10\nper-class recalls  (%)", fontsize=9, color="#333333")
    ax2.set_xlabel("structured prune ratio  (fraction of channels removed)", fontsize=9,
                   color="#333333")
    ax2.set_ylim(82.5, 99.6)
    ticks = sorted(set(hard["variant"].astype(str).str.extract(r"p(\d+)_")[0].dropna().astype(int)))
    ax2.set_xticks([t / 100 for t in ticks])
    ax2.set_xticklabels([f"{t}%" for t in ticks])

    for ax, col in ((ax1, "accuracy"), (ax2, "worst_class_recall")):
        ax.plot([0.6], [chosen[col] * 100], marker="o", markersize=13, markerfacecolor="none",
                markeredgecolor=INK, markeredgewidth=1.7, zorder=4)
    ax1.annotate("submitted:  Valiant Joy", xy=(0.6, chosen["accuracy"] * 100 - 0.15),
                 xytext=(0.6, 95.2), fontsize=9.5, color=INK, ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=INK, linewidth=1, shrinkA=2, shrinkB=7))

    # the gap the teacher closes, drawn where it happens
    h60 = hard[hard["variant"] == "p60_int8"]
    if len(h60):
        lo, hi = h60.iloc[0]["worst_class_recall"] * 100, chosen["worst_class_recall"] * 100
        ax2.annotate("", xy=(0.6, hi - 0.4), xytext=(0.6, lo + 0.4),
                     arrowprops=dict(arrowstyle="<->", color=KD, linewidth=1.4))
        ax2.text(0.573, (lo + hi) / 2, f"+{hi - lo:.1f} pp\nfrom the\nteacher", fontsize=8.5,
                 color=KD, ha="right", va="center")

    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}  (band ±{half:.3f} pp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
