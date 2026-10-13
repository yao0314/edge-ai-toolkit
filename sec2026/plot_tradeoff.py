"""Tradeoff curve for the report: full-val accuracy vs model size / parameters, from measure.py results.

    python sec2026/plot_tradeoff.py --out submission_v2/tradeoff.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POINTS = [  # (results file, label, pruning ratio, precision)
    ("baseline_onnx_accuracy", "baseline FP32", 0.0, "fp32"),
    ("p00_int8_pct_rr_accuracy", "INT8", 0.0, "int8"),
    ("p20_fp32_accuracy", "prune 20% FP32", 0.2, "fp32"),
    ("p20_int8_accuracy", "prune 20% + INT8", 0.2, "int8"),
    ("p30_fp32_accuracy", "prune 30% FP32", 0.3, "fp32"),
    ("p30_int8_accuracy", "prune 30% + INT8", 0.3, "int8"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for f, label, r, prec in POINTS:
        p = HERE / "results" / f"{f}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            assert d["accuracy"]["full_val_split"]
            rows.append((label, r, prec, d["size_mb"], d["params"], 100 * d["accuracy"]["accuracy"],
                         100 * d["accuracy"]["worst_class_accuracy"]))
    base_size = next(x[3] for x in rows if x[0] == "baseline FP32")
    base_acc = next(x[5] for x in rows if x[0] == "baseline FP32")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, idx, title in ((axes[0], 5, "top-1 accuracy"), (axes[1], 6, "worst-class accuracy")):
        for prec, mark in (("fp32", "o-"), ("int8", "s--")):
            pts = sorted((base_size / x[3], x[idx], x[0]) for x in rows if x[2] == prec)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], mark, label=prec.upper())
            for x, y, lab in pts:
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7)
        if idx == 5:
            ax.axhline(0.97 * base_acc, color="red", lw=0.8, ls=":", label="97% of baseline")
        ax.set_xlabel("file-size reduction vs baseline FP32 ONNX (x)")
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(f"{title}, full val split (5,400 images)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    for x in rows:
        print(f"{x[0]:20s} size {x[3]:.2f} MiB ({base_size / x[3]:.2f}x) params {x[4] / 1e6:.3f} M "
              f"acc {x[5]:.2f} worst {x[6]:.2f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
