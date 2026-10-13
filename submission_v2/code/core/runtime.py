"""Uniform inference wrappers so adapters evaluate torch, ONNX and TFLite models the same way.

Foreign models (handed to us as ONNX/TFLite) often differ from the adapter's data in two
ways that are fixed here, not in every adapter:
  * fixed batch size 1        -> the batch is run sample by sample and concatenated
  * NHWC instead of NCHW      -> 4-D inputs are transposed when that makes the shape match
"""
from __future__ import annotations

import sys

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

# torch is imported lazily: the ONNX and TFLite predictors need only numpy + their runtime, and the
# degraded competition-day path (a supplied .onnx, no torch in the sandbox) must be able to run
# inference to compute accuracy. Only TorchPredictor genuinely requires it.

log = logging.getLogger(__name__)


def ort_session(path: str | os.PathLike, threads: int | None = None, spinning: bool = True,
                opt_level: str = "all", cpu_mem_arena: bool = True, mem_pattern: bool = True):
    """CPU ONNX Runtime session. threads=None/0 lets ORT use all cores.

    `cpu_mem_arena` and `mem_pattern` are ORT's two memory strategies, both on by default. The
    arena allocates a pool up front and grows it without giving memory back, which inflates peak
    RSS — and peak RSS is a reported number, so this has to be a knob rather than a default we
    never questioned. Turning either off usually costs some latency; measure both before choosing.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = bool(cpu_mem_arena)
    so.enable_mem_pattern = bool(mem_pattern)
    so.graph_optimization_level = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[opt_level]
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    if threads:
        so.intra_op_num_threads = int(threads)
    if not spinning:
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def is_onnx(model: Any) -> bool:
    if isinstance(model, (str, Path)):
        from .model_io import model_format

        return model_format(model) == "onnx"
    return type(model).__name__ == "InferenceSession"


def is_tflite(model: Any) -> bool:
    if isinstance(model, (str, Path)):
        from .model_io import model_format

        return model_format(model) == "tflite"
    return False


def adapt_layout(x: np.ndarray, expected: list) -> np.ndarray:
    """Transpose NCHW <-> NHWC when the model's expected (non-batch) shape asks for it."""
    exp = [d if isinstance(d, int) and d > 0 else None for d in expected[1:]]

    def fits(a):
        return a.ndim == len(expected) and all(e is None or e == s for e, s in zip(exp, a.shape[1:]))

    if x.ndim != 4 or fits(x):
        return x
    for perm in ((0, 2, 3, 1), (0, 3, 1, 2)):
        y = x.transpose(perm)
        if fits(y):
            return np.ascontiguousarray(y)
    return x


def fixed_batch(expected: list) -> int | None:
    b = expected[0] if expected else None
    return b if isinstance(b, int) and b > 0 else None


def _run_batched(fn, x: np.ndarray, batch: int | None):
    """Call fn on chunks of size `batch` (when the model has a fixed batch) and concatenate."""
    if not batch or x.shape[0] == batch:
        return fn(x)
    outs = [fn(x[i:i + batch]) for i in range(0, x.shape[0], batch)]
    if isinstance(outs[0], list):
        return [np.concatenate([o[k] for o in outs]) for k in range(len(outs[0]))]
    return np.concatenate(outs)


def _to_numpy(x):
    """Anything array-like -> numpy, without needing torch to be importable. `.detach()` is the
    torch-tensor tell; numpy arrays and lists fall through to np.asarray."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x


class TorchPredictor:
    def __init__(self, model: nn.Module, device: str | None = None):
        import torch

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

    def __call__(self, x):
        torch = self._torch
        with torch.no_grad():
            x = torch.as_tensor(x).to(self.device, dtype=torch.float32)
            out = self.model(x)
        if isinstance(out, (tuple, list)):
            return [o.detach().float().cpu().numpy() for o in out]
        return out.detach().float().cpu().numpy()


class OnnxPredictor:
    def __init__(self, model: Any, threads: int | None = None):
        self.sess = ort_session(model, threads=threads) if isinstance(model, (str, Path)) else model
        inp = self.sess.get_inputs()[0]
        self.input_name, self.shape = inp.name, list(inp.shape)
        self.batch = fixed_batch(self.shape)
        self.n_outputs = len(self.sess.get_outputs())

    def _run(self, x):
        out = self.sess.run(None, {self.input_name: x})
        return out if self.n_outputs > 1 else out[0]

    def __call__(self, x):
        x = _to_numpy(x)
        x = adapt_layout(np.ascontiguousarray(x, dtype=np.float32), self.shape)
        return _run_batched(self._run, x, self.batch)


class TFLitePredictor:
    """LiteRT (ai-edge-litert) interpreter. Handles int8/uint8 I/O via the tensors' quantization params."""

    def __init__(self, path: str | Path, threads: int | None = None):
        from ai_edge_litert.interpreter import Interpreter

        self.interp = Interpreter(model_path=str(path), num_threads=threads or os.cpu_count())
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()
        self.shape = [int(d) for d in self.inp["shape"]]
        self.batch = self.shape[0] if self.shape and self.shape[0] > 0 else 1

    @staticmethod
    def quantize_input(x: np.ndarray, detail: dict) -> np.ndarray:
        dt = detail["dtype"]
        if np.issubdtype(dt, np.integer):
            scale, zp = detail["quantization"]
            if scale:
                info = np.iinfo(dt)
                return np.clip(np.round(x / scale + zp), info.min, info.max).astype(dt)
            return x.astype(dt)
        return x.astype(dt)

    @staticmethod
    def dequantize(y: np.ndarray, detail: dict) -> np.ndarray:
        scale, zp = detail["quantization"]
        if np.issubdtype(y.dtype, np.integer) and scale:
            return (y.astype(np.float32) - zp) * scale
        return y.astype(np.float32)

    def _run(self, x):
        self.interp.set_tensor(self.inp["index"], self.quantize_input(x, self.inp))
        self.interp.invoke()
        outs = [self.dequantize(self.interp.get_tensor(o["index"]), o) for o in self.outs]
        return outs if len(outs) > 1 else outs[0]

    def __call__(self, x):
        x = _to_numpy(x)
        x = adapt_layout(np.asarray(x, dtype=np.float32), self.shape)
        return _run_batched(self._run, x, self.batch)


def as_predictor(model: Any, device: str | None = None):
    """Return a callable f(batch) -> np.ndarray (or list of arrays for multi-output models).

    Accepts nn.Module / ScriptModule, a path to .onnx or .tflite, or an onnxruntime.InferenceSession.
    """
    if isinstance(model, (TorchPredictor, OnnxPredictor, TFLitePredictor)):
        return model
    if is_tflite(model):
        return TFLitePredictor(model)
    if is_onnx(model):
        return OnnxPredictor(model)
    if _is_torch_module(model):
        return TorchPredictor(model, device)
    raise TypeError(f"Cannot build predictor for {type(model)}")


def _is_torch_module(model: Any) -> bool:
    """True for nn.Module WITHOUT importing torch when it is absent: on the ONNX-only path the
    dispatch above has already matched, and `import torch` here would turn a working configuration
    into an ImportError."""
    torch_nn = sys.modules.get("torch.nn")
    return torch_nn is not None and isinstance(model, torch_nn.Module)
