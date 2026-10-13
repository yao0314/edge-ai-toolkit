"""Experiment 17: why dynamic quantization wins on transformers and is catastrophic on CNNs.

Stage 1b produced a number that demands an explanation rather than a table row: on plain
classification, `int8_dyn_conv` (dynamic, [Conv, MatMul]) measured **0.054x** - eighteen times
SLOWER than fp32 - at -11.07 pt accuracy, while static INT8 on the same model was 1.12x faster at
4 threads for -0.14 pt.

The mechanism is not exotic. Dynamic quantization has no calibration, so it inserts a
`DynamicQuantizeLinear` before each quantized op and computes that activation tensor's min/max
**at every inference** - a full reduction over the activation, per op, per run - and then rescales
the ConvInteger output through unfused Cast/Mul/Add kernels. What decides whether that is affordable
is the RATIO of that overhead to the op it wraps:

    transformer   few, large MatMuls; overhead amortises over a lot of arithmetic  -> dynamic wins
    CNN           many Convs, each wrapped individually                            -> dynamic loses

and the ratio is set by how much arithmetic each wrapped op does, NOT by how large the activation
is - see the two prediction versions below, the first of which this project registered and lost.

This script tests that prediction instead of asserting it. For every model it sums the output
element count of the ops the dynamic arm was asked to quantize, and relates that to the measured
dynamic penalty across runs whose input resolutions differ by more than an order of magnitude
(CIFAR-scale classification, ImageNet-scale ResNet50, 320px detection).

PREDICTION, VERSION 1 - REGISTERED AND THEN FALSIFIED. The original text read: "the dynamic
penalty should rise monotonically with activation volume per quantized op." Stage 1b refuted it:
MobileNetV2 at 32px measured 0.060x and ResNet50, with far more activation volume, measured 0.068x.
Flat, and if anything slightly better. Recorded rather than quietly rewritten, because the failure
is what produced the correct version.

PREDICTION, VERSION 2 - what the mechanism actually implies. DynamicQuantizeLinear does O(N) work
over an N-element activation while the convolution it feeds does O(N*K). The RATIO is therefore set
by kernel size and fusion, not by N, so the penalty should be roughly SCALE-INVARIANT - near 15x
across resolutions. Version 1 confused absolute overhead with relative overhead.

FALSIFICATION for version 2: a run whose penalty is far from ~15x. Detection at 320px is the third
point. Note that the accuracy damage is a SEPARATE axis and is not explained by this cost model at
all - see experiment 18, where -11.07 pt on MobileNetV2 against -0.46 pt on ResNet50 is attributed
to calibration, not to volume.

    python scripts/paper/dynamic_quant_cost_model.py

Writes results/paper/dynamic_cost_model.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import onnx  # noqa: E402
import pandas as pd  # noqa: E402
from onnx import shape_inference  # noqa: E402

DYN_TARGETS = {"Conv", "MatMul", "Gemm", "Attention", "LSTM", "Gather", "Transpose"}


def activation_volume(path: Path) -> dict:
    """Total output elements of the ops a dynamic quantizer would touch, plus a node census.

    Shape inference is required: an exported graph carries shapes only on inputs and outputs, and
    the interior is where the large feature maps live. A dimension that stays symbolic after
    inference is counted as 1, which UNDER-counts - so a positive result here is conservative.
    """
    m = onnx.load(str(path), load_external_data=False)
    try:
        m = shape_inference.infer_shapes(m, strict_mode=False)
    except Exception:
        pass
    dims = {}
    for vi in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input):
        tt = vi.type.tensor_type
        if not tt.shape.dim:
            continue
        n = 1
        for d in tt.shape.dim:
            n *= d.dim_value if d.dim_value > 0 else 1
        dims[vi.name] = n
    vol, n_dyn, unknown = 0, 0, 0
    census = {}
    for node in m.graph.node:
        census[node.op_type] = census.get(node.op_type, 0) + 1
        if node.op_type not in DYN_TARGETS:
            continue
        n_dyn += 1
        out = node.output[0] if node.output else None
        v = dims.get(out)
        if v is None or v <= 1:
            unknown += 1
        vol += v or 0
    return {"dyn_target_ops": n_dyn, "activation_elems": vol,
            "elems_per_dyn_op": vol / n_dyn if n_dyn else float("nan"),
            "shape_unknown_ops": unknown, "n_nodes": len(m.graph.node),
            "n_conv": census.get("Conv", 0), "n_matmul": census.get("MatMul", 0),
            "n_gemm": census.get("Gemm", 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="classification_dyn,stress_resnet50_dyn,detection_dyn,"
                                      "classification_mobilevit,classification_deit")
    ap.add_argument("--out", default="results/paper/dynamic_cost_model.csv")
    a = ap.parse_args()

    rows = []
    for run in a.runs.split(","):
        sweep = ROOT / "results" / run / "sweep.csv"
        models = ROOT / "results" / run / "models"
        if not sweep.exists() or not models.is_dir():
            print(f"  {run}: no sweep.csv yet, skipped")
            continue
        d = pd.read_csv(sweep)
        by = {str(r["variant"]): r for _, r in d.iterrows()}
        # the fp32 arm carries the shapes the dynamic quantizer would have to scan at run time
        fp32_name = next((v for v in by if v.endswith("fp32")), None)
        if fp32_name is None:
            continue
        fp32_model = models / f"{fp32_name}.onnx"
        if not fp32_model.exists():
            continue
        stats = activation_volume(fp32_model)
        fp = by[fp32_name]
        for v, r in by.items():
            if "dyn" not in v:
                continue
            for t in (1, 4):
                lo, hi = f"latency_p50_ms_t{t}", None
                if lo not in d:
                    continue
                rows.append({"run": run, "variant": v, "threads": t, **stats,
                             "fp32_ms": fp[lo], "dyn_ms": r[lo],
                             "dyn_speedup_vs_fp32": fp[lo] / r[lo] if r[lo] else float("nan"),
                             "dyn_penalty_x": r[lo] / fp[lo] if fp[lo] else float("nan")})
    if not rows:
        print("no runs with a dynamic variant and a sweep.csv yet", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("Dynamic quantization has no calibration: it recomputes each activation's min/max at")
    print("every inference, so its overhead scales with ACTIVATION VOLUME, not parameter count.\n")
    show = df[df["threads"] == 4].sort_values("elems_per_dyn_op")
    print(f"{'run':26s} {'variant':20s} {'dyn ops':>8s} {'elems/op':>12s} {'vs fp32':>9s}")
    for _, r in show.iterrows():
        print(f"{r['run']:26s} {r['variant']:20s} {r['dyn_target_ops']:8.0f} "
              f"{r['elems_per_dyn_op']:12,.0f} {r['dyn_speedup_vs_fp32']:9.3f}x")

    if len(show) > 2:
        from scipy.stats import spearmanr
        s = spearmanr(show["elems_per_dyn_op"], show["dyn_speedup_vs_fp32"])
        print(f"\nSpearman(elements per dynamic op, dynamic speedup) = {s.statistic:+.3f} "
              f"p={s.pvalue:.3g}  n={len(show)}")
        print("NOTE: a strong rho here is NOT the prediction any more. Version 1 of this")
        print("prediction (penalty rises with volume) was falsified by stage 1b - MobileNetV2 0.060x")
        print("vs ResNet50 0.068x, flat across a large volume difference. The live prediction is")
        print("SCALE INVARIANCE: penalties clustered near ~15x regardless of volume, because the")
        print("O(N) min/max pass and the O(N*K) convolution scale together. Read the spread of the")
        print("speedup column, not the correlation.")
        sp = show["dyn_speedup_vs_fp32"]
        print(f"  penalty spread: {1/sp.max():.1f}x .. {1/sp.min():.1f}x slower than fp32 "
              f"(tight = scale-invariant = supports the mechanism)")
        u = int(show["shape_unknown_ops"].sum())
        if u:
            print(f"\nNote: {u} op(s) had an unresolved output shape and were counted as 1 element.")
            print("That under-counts volume, so it biases against the hypothesis, not toward it.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
