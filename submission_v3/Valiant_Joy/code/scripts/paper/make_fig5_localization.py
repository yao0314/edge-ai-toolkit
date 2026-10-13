"""Fig. 5: where the 1.24 four-thread regression lives.

The paper's strongest result, the localization of Section V-C, is the one result with no figure:
a reader who skims the figures sees the step (Fig. 4) but not the evidence that four Transposes
carry it. This draws that evidence as one panel.

Same rule as make_figures.py: nothing is typed in. The probe deltas come from the registered
verdict (results/transpose_probe/verdict_followup.json, the class-gated block the paper leads
with) and the full-model step is recomputed here from the sweep CSV.

    python scripts/paper/make_fig5_localization.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.paper.make_figures import (  # noqa: E402
    ANNOT, C, COL, GRID, INK, MUTED, fit_to_text_width,
)

FIGS = ROOT / "paper" / "figs"
VERDICT = ROOT / "results" / "transpose_probe" / "verdict_followup.json"
SWEEP = ROOT / "results" / "paper" / "ort_version_bisect.csv"


def full_model_step() -> float:
    """The static MobileViT-XXS file's four-thread step, 1.23.2 -> 1.24.4, from the sweep."""
    d = pd.read_csv(SWEEP)
    d = d[(d.family == "mobilevit") & (d.variant == "p00_int8") & (d.threads == 4)]
    m = d.groupby("ort").p50_ms.median()
    return float(m["1.24.4"] - m["1.23.2"])


def main() -> int:
    v = json.loads(VERDICT.read_text(encoding="utf-8"))["B_exp27b"]["class_gated"]
    step = full_model_step()

    # (label, value, group) - group 0 the measured step, 1 the probe, 2 the controls
    rows = [
        ("Full static file, four threads", step, 0),
        ("All 56 uint8 Transposes", v["delta_u8_4T"], 1),
        ("   of those, the four with a 2-byte block", v["delta_w2_4T"], 1),
        ("   of those, the other 52", v["delta_rest_4T"], 1),
        ("Same four shapes, float16", v["D_f16_w2_4T"], 2),
        ("Same four shapes, float32", v["D_f32_w4_4T"], 2),
        ("All 56, one thread", v["delta_u8_1T"], 2),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    groups = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(COL, 1.62))
    ys = list(range(len(rows)))[::-1]
    face = {0: INK, 1: C[0], 2: C[1]}
    hatch = {0: "", 1: "", 2: "//"}
    for y, val, g in zip(ys, vals, groups):
        ax.barh(y, val, height=0.62, color=face[g], edgecolor=INK, linewidth=0.5,
                hatch=hatch[g], zorder=3)
    ax.axvline(step, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=2)

    for y, val, g in zip(ys, vals, groups):
        lab = f"{val:+.3f}".replace("-", "−")   # a true minus, as on the axis
        if val > 0.3 * step:   # long bars: the value sits inside, on a solid patch over any hatching
            ax.annotate(lab, (val, y), xytext=(-4, 0), textcoords="offset points",
                        va="center", ha="right", fontsize=ANNOT, color="white", fontweight="bold",
                        bbox=dict(boxstyle="square,pad=0.15", fc=face[g], ec="none"))
        else:
            ax.annotate(lab, (max(val, 0.0), y), xytext=(4, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=ANNOT, color=INK)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.set_xlabel("added latency at 1.24.4 against 1.23.2 (ms)", fontsize=8)
    ax.set_xlim(-0.06, max(vals) * 1.34)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.annotate("the step to be explained", (step, -0.62), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=ANNOT, color=MUTED)
    ax.set_ylim(-1.05, len(rows) - 0.35)
    fit_to_text_width(fig)
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig5_localization.{ext}")
    plt.close(fig)
    print(f"  wrote fig5_localization.pdf / .png   (step {step:+.3f} ms, "
          f"probe {v['delta_u8_4T']:+.3f}, four {v['delta_w2_4T']:+.3f}, "
          f"rest {v['delta_rest_4T']:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
