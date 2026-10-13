"""Section 6.2 of the task sheet: export to ONNX, then static (QDQ) INT8 quantization.

    python sec2026/export_quant.py --model starter_kit/student_mobilenetv2_distilled.pt --out sec2026/out/p00
    python sec2026/export_quant.py --model sec2026/out/p20/pruned.pt --out sec2026/out/p20 --calib 512

Writes <out>/student_model.onnx (FP32) and <out>/student_model_int8.onnx, plus quant.json with the
settings and a label-free health check (output SQNR and top-1 agreement vs FP32 on held-out
calibration-style images). Follows the sheet: TorchScript exporter (dynamo=False), opset 13,
input "input" / output "output" of shape (1, 3, 224, 224), quantization.preprocess before
quantize_static, QDQ format. Calibration images come from the TRAIN split, never from val.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OPSET = 13


# ----------------------------------------------------------------------------- export
def export_fp32(model: torch.nn.Module, out_path, input_shape=common.INPUT_SHAPE, opset: int = OPSET) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.randn(*input_shape)
    torch.onnx.export(model, dummy, str(out_path), input_names=["input"], output_names=["output"],
                      opset_version=opset, dynamo=False)
    strip_output_value_info(out_path)
    onnx.checker.check_model(str(out_path))
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = common.onnx_predictor(out_path)(dummy)
    err = float(np.abs(ref - got).max())
    if err > 1e-3:
        raise RuntimeError(f"ONNX export differs from PyTorch: max abs error {err:.2e}")
    return out_path


def strip_output_value_info(path) -> int:
    """Section 7: drop graph.value_info entries that collide with graph outputs (AI Hub rejects
    them). Returns how many were removed; rewrites the file only if there were any."""
    m = onnx.load(str(path))
    outs = {o.name for o in m.graph.output}
    keep = [v for v in m.graph.value_info if v.name not in outs]
    removed = len(m.graph.value_info) - len(keep)
    if removed:
        del m.graph.value_info[:]
        m.graph.value_info.extend(keep)
        onnx.save(m, str(path))
    return removed


# ----------------------------------------------------------------------------- graph facts
def onnx_params(path) -> int:
    """Weight + bias elements of Conv/Gemm/MatMul, following DequantizeLinear back to the stored
    initializer so FP32 and QDQ files of the same network report the same count. Quantization
    scales and zero points are not parameters and are not counted."""
    g = onnx.load(str(path)).graph
    dims = _stored_tensors(g)
    producer = {o: n for n in g.node for o in n.output}
    # counted per layer, not per unique tensor: the exporter merges byte-identical constants (e.g.
    # two all-zero folded biases), and deduplicating would make the count depend on the weights' values
    total = 0
    for n in g.node:
        if n.op_type in ("Conv", "Gemm", "MatMul"):
            for name in n.input[1:]:
                s = _source(g, name, producer)
                if s in dims:
                    total += int(np.prod(dims[s])) if dims[s] else 1
    return total


def _stored_tensors(g) -> dict:
    """name -> dims of every stored tensor: initializers and Constant-node values (the TorchScript
    exporter emits some folded weights as Constant nodes rather than initializers)."""
    d = {i.name: list(i.dims) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    d[n.output[0]] = list(a.t.dims)
    return d


def _source(g, name, producer=None):
    """Follow Identity / DequantizeLinear back to the tensor that is actually stored."""
    producer = producer or {o: n for n in g.node for o in n.output}
    n = producer.get(name)
    while n is not None and n.op_type in ("Identity", "DequantizeLinear"):
        name = n.input[0]
        n = producer.get(name)
    return name


def onnx_macs(path) -> int:
    """Multiply-accumulates of Conv and Gemm/MatMul from shape-inferred tensor shapes (batch 1).
    Same convention as thop: one MAC per weight use, element-wise ops not counted."""
    m = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
    g = m.graph
    shapes = {}
    for v in list(g.value_info) + list(g.output) + list(g.input):
        dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
        shapes[v.name] = dims
    stored = _stored_tensors(g)
    producer = {o: n for n in g.node for o in n.output}

    def wshape(name):
        s = _source(g, name, producer)
        return stored.get(s) or shapes.get(s) or shapes.get(name)

    total = 0
    for n in g.node:
        out = shapes.get(n.output[0])
        if not out:
            continue
        if n.op_type == "Conv":
            w = wshape(n.input[1])
            if w:
                total += int(np.prod(out)) * int(np.prod(w[1:]))
        elif n.op_type in ("Gemm", "MatMul"):
            w = wshape(n.input[1])
            if w:
                k = w[0] if n.op_type == "MatMul" else (w[1] if any(a.name == "transB" and a.i for a in n.attribute) else w[0])
                total += int(np.prod(out)) * int(k)
    return total


# ----------------------------------------------------------------------------- quantization
class ImageReader:
    """CalibrationDataReader over a list of (1, 3, H, W) float32 arrays."""

    def __init__(self, arrays, input_name="input"):
        self._it = iter([{input_name: a} for a in arrays])

    def get_next(self):
        return next(self._it, None)


def calibration_arrays(dataset, n: int, seed: int = 1) -> list:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    return [dataset[int(i)][0].unsqueeze(0).numpy().astype(np.float32) for i in idx]


METHODS = {"minmax": "MinMax", "percentile": "Percentile", "entropy": "Entropy"}


def quantize_static_qdq(fp32_path, out_path, calib_arrays, method: str = "minmax",
                        per_channel: bool = True, reduce_range: bool = False,
                        nodes_to_exclude=None, percentile: float = 99.999) -> Path:
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp32_path, out_path = Path(fp32_path), Path(out_path)
    pre = out_path.with_name(out_path.stem + "_pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))          # the sheet's "Gotcha": preprocess first
    extra = {"ActivationSymmetric": False, "WeightSymmetric": True}
    if method == "percentile":
        extra["CalibPercentile"] = percentile
    if method == "entropy":                               # ORT's 128-bin default makes entropy == minmax
        extra.update(CalibNumBins=2048, CalibNumQuantizedBins=128)
    quantize_static(str(pre), str(out_path), ImageReader(calib_arrays),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,   # U8S8
                    per_channel=per_channel, reduce_range=reduce_range,
                    calibrate_method=getattr(CalibrationMethod, METHODS[method]),
                    nodes_to_exclude=list(nodes_to_exclude or []), extra_options=extra)
    pre.unlink(missing_ok=True)
    strip_output_value_info(out_path)
    return out_path


def health_check(fp32_path, int8_path, arrays) -> dict:
    """Label-free: SQNR of the INT8 logits against FP32, and how often the top-1 class agrees.
    Below ~15 dB, or agreement well under 99%, the conversion is damaged."""
    f32, i8 = common.onnx_predictor(fp32_path), common.onnx_predictor(int8_path)
    sig = err = 0.0
    agree = 0
    for a in arrays:
        x = torch.from_numpy(a)
        r, q = f32(x).astype(np.float64), i8(x).astype(np.float64)
        sig += float((r ** 2).sum())
        err += float(((r - q) ** 2).sum())
        agree += int(r.argmax() == q.argmax())
    return {"sqnr_db": round(10 * np.log10(sig / max(err, 1e-30)), 2),
            "top1_agreement": round(agree / max(len(arrays), 1), 4), "n_images": len(arrays)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="official state_dict or a pruned whole-module .pt "
                                    "(default: the kit's student)")
    ap.add_argument("--kit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", type=int, default=256, help="calibration images from the train split")
    ap.add_argument("--check", type=int, default=128, help="held-out images for the health check")
    ap.add_argument("--method", choices=sorted(METHODS), default="minmax")
    ap.add_argument("--percentile", type=float, default=99.999)
    ap.add_argument("--reduce-range", action="store_true",
                    help="7-bit weights; can help accuracy on CPUs without VNNI")
    ap.add_argument("--no-per-channel", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[], help="ONNX node names to keep in FP32")
    ap.add_argument("--fp32-only", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out)
    t0 = time.time()
    model = common.build_student(a.model, a.kit)
    fp32 = export_fp32(model, out / "student_model.onnx")
    rec = {"source": str(a.model or common.kit_file(common.STUDENT, a.kit)), "opset": OPSET,
           "fp32": {"file": str(fp32), "params": onnx_params(fp32), "macs": onnx_macs(fp32),
                    "size_mb": fp32.stat().st_size / 2**20}}
    if not a.fp32_only:
        train, _ = common.subsets(a.kit)
        arrays = calibration_arrays(train, a.calib + a.check, seed=1)
        calib, check = arrays[:a.calib], arrays[a.calib:]
        int8 = quantize_static_qdq(fp32, out / "student_model_int8.onnx", calib, a.method,
                                   per_channel=not a.no_per_channel, reduce_range=a.reduce_range,
                                   nodes_to_exclude=a.exclude, percentile=a.percentile)
        rec["int8"] = {"file": str(int8), "params": onnx_params(int8), "macs": onnx_macs(int8),
                       "size_mb": int8.stat().st_size / 2**20,
                       "settings": {"format": "QDQ", "activations": "uint8", "weights": "int8",
                                    "per_channel": not a.no_per_channel, "reduce_range": a.reduce_range,
                                    "calibrate_method": a.method, "calib_images": len(calib),
                                    "excluded_nodes": a.exclude,
                                    **({"percentile": a.percentile} if a.method == "percentile" else {})},
                       "health": health_check(fp32, int8, check)}
        h = rec["int8"]["health"]
        flag = "" if h["sqnr_db"] >= 15 else "   <-- below 15 dB: quantization damaged"
        print(f"INT8 health: SQNR {h['sqnr_db']} dB, top-1 agreement {h['top1_agreement']:.2%}{flag}")
    rec["seconds"] = round(time.time() - t0, 1)
    common.save_json(rec, out / "quant.json")
    print(json.dumps({k: v for k, v in rec.items() if k != "source"}, indent=2))


if __name__ == "__main__":
    main()
