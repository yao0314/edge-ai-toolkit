"""Fig. 3 of the rewrite: who the 1.24 release affects, one row per model.

Each row is one model's static INT8 file: the change in its four-thread latency from 1.23.2 to
1.24.4 (bar), next to the same change for its FP32 sibling (open circle). The shaded band is the
study's 7% latency floor. A skimming reader should see three things: only the hybrid model's static
files leave the band upwards, their FP32 siblings stay inside it, and one convolutional model leaves
it downwards.

Same rule as make_figures.py: every value is recomputed here from the scope run's CSV. The
autoencoder is left out: 1.24.4 cut the latency of its FP32 and INT8 files alike, a per-call cost
that says nothing about the INT8 path (the text reports it).

    python scripts/paper/make_fig_scope.py
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

from scripts.paper.make_figures import ANNOT, C, COL, GRID, INK, MUTED, fit_to_text_width  # noqa: E402

SCOPE = ROOT / "results" / "paper" / "ort_version_scope.csv"
FIGS = ROOT / "paper" / "figs"
FLOOR = 7.0

# (family, static variant, FP32 variant, label) - top to bottom
ROWS = [
    ("mv256", "p00_int8", "p00_fp32", "MobileViT-XXS, 256 px"),
    ("mv64", "p00_int8", "p00_fp32", "MobileViT-XXS, 64 px"),
    ("deit224", "p00_int8", "p00_fp32", "DeiT-tiny, 224 px"),
    ("r50", "p00_int8", "p00_fp32", "ResNet-50"),
    ("mnv2", "p00_int8", "p00_fp32", "MobileNetV2"),
    ("y256", "256_p00_int8", "256_p00_fp32", "YOLOv8n, 256 px"),
]


def step(d: pd.DataFrame, fam: str, var: str) -> tuple[float, bool]:
    """Median four-thread latency change 1.23.2 -> 1.24.4 in %, and whether sessions are disjoint."""
    g = d[(d.family == fam) & (d.variant == var) & (d.threads == 4)]
    a = g[g.ort == "1.23.2"].p50_ms
    b = g[g.ort == "1.24.4"].p50_ms
    return (b.median() / a.median() - 1) * 100, bool(b.min() > a.max() or b.max() < a.min())


def main() -> int:
    d = pd.read_csv(SCOPE)
    vals = [(lab, *step(d, f, s), step(d, f, p)[0]) for f, s, p, lab in ROWS]

    fig, ax = plt.subplots(figsize=(COL, 1.72))
    ys = list(range(len(vals)))[::-1]
    ax.axvspan(-FLOOR, FLOOR, color=GRID, alpha=0.55, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    for y, (lab, s, disj, fp) in zip(ys, vals):
        beyond = abs(s) > FLOOR and disj
        color = (C[1] if s > 0 else C[2]) if beyond else "#b8b8b8"
        ax.barh(y, s, height=0.56, color=color, edgecolor=INK, linewidth=0.5, zorder=3)
        ax.plot(fp, y, marker="o", ms=4.2, mfc="white", mec=INK, mew=0.8, zorder=4, ls="none")
        if abs(s) > FLOOR:     # rows inside the floor need no number: the point is that they did not move
            ax.annotate(f"{s:+.0f}%".replace("-", "−"), (s, y), xytext=(4 if s >= 0 else -4, 0), textcoords="offset points",
                        ha="left" if s >= 0 else "right", va="center", fontsize=ANNOT, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([v[0] for v in vals], fontsize=7.8)
    ax.set_xlabel("four-thread latency at 1.24.4 against 1.23.2 (%)", fontsize=8)
    ax.set_xlim(-40, 58)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # a two-entry key: the bar is the static INT8 file, the circle its FP32 sibling
    ax.plot([], [], marker="s", ms=5, mfc="#b8b8b8", mec=INK, mew=0.5, ls="none", label="static INT8 file")
    ax.plot([], [], marker="o", ms=4.2, mfc="white", mec=INK, mew=0.8, ls="none", label="its FP32 sibling")
    ax.legend(loc="lower right", frameon=False, fontsize=7.2, handletextpad=0.3, borderaxespad=0.1)
    ax.text(0, len(vals) - 0.35, "7% floor", ha="center", va="bottom", fontsize=ANNOT, color=MUTED)
    fit_to_text_width(fig)
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig6_scope.{ext}")
    plt.close(fig)
    print("  wrote fig6_scope.pdf / .png:", ", ".join(f"{v[0]} {v[1]:+.1f}%" for v in vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
