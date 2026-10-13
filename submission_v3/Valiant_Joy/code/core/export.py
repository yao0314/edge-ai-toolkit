"""Export a torch model to ONNX and verify the ONNX output against PyTorch."""
from __future__ import annotations

import copy
import inspect
import logging
import warnings
from pathlib import Path

import numpy as np

# torch is imported lazily. core/quantize.py does `from .export import stamp_onnx`, so a module-level
# torch import here makes INT8 quantization require torch even when the input is already a .onnx -
# the exact rescue path that has to work in a sandbox that cannot install a 2-3 GB torch.

log = logging.getLogger(__name__)

# Ops that ONNX Runtime's CPU EP (x86 + ARM/NEON) executes efficiently and that the
# INT8 quantizer understands. Anything else triggers a warning, not an error.
CPU_FRIENDLY_OPS = {
    "Conv", "ConvTranspose", "Gemm", "MatMul", "Relu", "Clip", "LeakyRelu", "Sigmoid", "HardSigmoid",
    "HardSwish", "Tanh", "Add", "Sub", "Mul", "Div", "Concat", "Reshape", "Flatten", "Transpose",
    "Squeeze", "Unsqueeze", "GlobalAveragePool", "AveragePool", "MaxPool", "BatchNormalization",
    "Softmax", "Resize", "Upsample", "Pad", "Slice", "Split", "Shape", "Gather", "Constant",
    "Identity", "ReduceMean", "Pow", "Sqrt", "Exp", "Erf", "Cast", "Expand", "Where", "Equal",
    "ConstantOfShape", "Range", "Unsqueeze", "Neg", "Min", "Max", "Tile", "ArgMax",
    "QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul", "QLinearAdd",
}


class ExportMismatchError(RuntimeError):
    pass


def stamp_onnx(path: str | Path) -> Path:
    """Write authorship/provenance into the ONNX file itself, so it travels with the model.

    `producer_name` is the standard ONNX field; the rest go into metadata_props. Re-stamping is safe
    (entries are replaced, not duplicated).
    """
    import onnx

    from . import PRODUCER, provenance

    path = Path(path)
    # load_external_data=False is load-bearing, not an optimisation. Stamping only rewrites
    # producer fields and metadata_props; the tensors are never read or changed. Loading them
    # would (a) clear each tensor's external_data, so the re-save inlines multi-GB weights into
    # the protobuf and orphans the sidecar - which measure.py then counts a second time in
    # size_bytes - and (b) make the re-save go through onnx's external-data writer, which opens
    # the sidecar "r+b" and seeks to EOF, APPENDING a fresh full copy of every weight on every
    # call without ever truncating. Keeping the references intact leaves the sidecar untouched on
    # disk and works whatever it is named, instead of guessing "<model>.onnx.data".
    m = onnx.load(str(path), load_external_data=False)
    m.producer_name = PRODUCER
    m.producer_version = provenance()["toolkit_version"]
    fields = provenance()
    keep = [p for p in m.metadata_props if p.key not in fields]
    del m.metadata_props[:]
    m.metadata_props.extend(keep)
    for k, v in fields.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, str(v)
    onnx.save(m, str(path))
    return path


def _flatten(out):
    if isinstance(out, (tuple, list)):
        res = []
        for o in out:
            res.extend(_flatten(o))
        return res
    return [out]


def export_onnx(model: nn.Module, example_input: torch.Tensor, path: str | Path, *,
                input_names=("input",), output_names=("output",), opset: int = 17,
                dynamic_batch: bool = True, verify: bool = True, atol: float = 1e-3,
                rtol: float = 1e-3, allowed_ops: set | None = None) -> Path:
    """Export `model` to a single-file ONNX at `path`, then check ORT matches PyTorch.

    Raises ExportMismatchError when max |torch - onnx| exceeds atol + rtol*|torch|.
    """
    import onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(model).cpu().eval()
    x = example_input.detach().cpu().float()
    input_names, output_names = list(input_names), list(output_names)
    import torch

    with torch.no_grad():
        ref = [t.numpy() for t in _flatten(m(x))]
    if len(output_names) != len(ref):
        output_names = [f"output{i}" if i else "output" for i in range(len(ref))]

    dyn = None
    if dynamic_batch:
        dyn = {n: {0: "batch"} for n in input_names + output_names}
    kwargs = dict(input_names=input_names, output_names=output_names, opset_version=opset,
                  dynamic_axes=dyn, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # TorchScript exporter: stable, single file, readable node names
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(m, (x,), str(path), **kwargs)

    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)
    stamp_onnx(path)
    ops = {n.op_type for n in proto.graph.node}
    unusual = sorted(ops - (allowed_ops or CPU_FRIENDLY_OPS))
    if unusual:
        log.warning("ONNX graph uses ops outside the CPU-friendly list: %s", unusual)

    if verify:
        verify_onnx(path, x, ref, atol=atol, rtol=rtol)
    return path


def verify_onnx(path, x: torch.Tensor, ref: list, atol: float = 1e-3, rtol: float = 1e-3) -> float:
    from .runtime import ort_session

    sess = ort_session(path)
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})
    worst = 0.0
    for r, g in zip(ref, got):
        if r.shape != g.shape:
            raise ExportMismatchError(f"shape mismatch torch={r.shape} onnx={g.shape}")
        diff = np.abs(r - g)
        excess = diff - (atol + rtol * np.abs(r))
        worst = max(worst, float(diff.max()) if diff.size else 0.0)
        if diff.size and excess.max() > 0:
            raise ExportMismatchError(
                f"{path}: ONNX output differs from PyTorch (max abs diff {diff.max():.3e}, "
                f"atol={atol}, rtol={rtol})")
    log.info("export verified: %s (max abs diff %.2e)", path, worst)
    return worst
