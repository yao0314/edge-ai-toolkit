"""Load whatever model file we are handed, and say what can be done with it.

Supported inputs and what each allows:

  format        loaded as                      prune  export  INT8 (ORT)  measure
  ------------  -----------------------------  -----  ------  ----------  -------
  torch         nn.Module (pickled .pt/.pth)     yes    yes      yes        yes
  state_dict    nn.Module via `arch` factory     yes    yes      yes        yes
  checkpoint    dict with a module under "model"/"ema"   (same as torch)
  torchscript   torch.jit.ScriptModule           no     yes      yes        yes
  torch_export  .pt2 -> ExportedProgram.module() no     yes      yes        yes
  onnx          path                             no     -        yes        yes
  tflite        path (LiteRT interpreter)        no     -        no*        yes

  * a float TFLite can go through ONNX when tf2onnx + tensorflow are installed (convert_tflite_to_onnx).
"""
from __future__ import annotations

import importlib
import sys
import logging
import zipfile
from pathlib import Path
from typing import Any


# torch is imported lazily. model_format()/sniff_format() on a PATH only read bytes, and the
# degraded path (a supplied .onnx in a sandbox with no torch) reaches them via runtime.is_tflite().
# Loading an actual torch file of course needs torch, and says so.
log = logging.getLogger(__name__)


def _torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("this model format needs PyTorch, which is not installed; ONNX and "
                           "TFLite files work without it") from e
    return torch

CAPABILITIES = {
    "torch": {"prune": True, "export": True, "quantize": True},
    "torchscript": {"prune": False, "export": True, "quantize": True},
    "torch_export": {"prune": False, "export": True, "quantize": True},
    "onnx": {"prune": False, "export": False, "quantize": True},
    "tflite": {"prune": False, "export": False, "quantize": False},
}


def import_callable(spec: str):
    """'package.module:attr' -> object."""
    mod, _, attr = spec.partition(":")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def sniff_format(path: str | Path) -> str:
    """Format from the file's first bytes: 'tflite' | 'onnx' | 'torch_file' | 'unknown'.

    TFLite flatbuffers carry the identifier b"TFL3" at offset 4; torch.save / TorchScript /
    torch.export files are zip ("PK") or legacy pickle (0x80); an ONNX ModelProto starts with
    field 1 (ir_version) -> tag byte 0x08.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return "unknown"
    if head[4:8] == b"TFL3":
        return "tflite"
    if head[:2] == b"PK" or head[:1] == b"\x80":
        return "torch_file"
    if head[:1] == b"\x08":
        return "onnx"
    return "unknown"


def model_format(model: Any) -> str:
    """Format of an already-loaded model or of a path (file content wins over the extension)."""
    if isinstance(model, (str, Path)):
        suf = Path(model).suffix.lower()
        by_ext = {".onnx": "onnx", ".tflite": "tflite"}.get(suf, "torch_file")
        by_content = sniff_format(model) if Path(model).exists() else "unknown"
        if by_content != "unknown" and by_content != by_ext:
            log.warning("%s: extension says %s but content is %s; using %s",
                        Path(model).name, by_ext, by_content, by_content)
            return by_content
        return by_ext
    if type(model).__name__ == "InferenceSession":
        return "onnx"
    # An in-memory torch object cannot exist unless torch is already imported, so checking
    # sys.modules avoids importing it just to reject a non-torch object.
    torch = sys.modules.get("torch")
    if torch is not None:
        if isinstance(model, torch.jit.ScriptModule):
            return "torchscript"
        if isinstance(model, torch.fx.GraphModule):
            return "torch_export"
        if isinstance(model, torch.nn.Module):
            return "torch"
    raise TypeError(f"unsupported model object {type(model)}")


def _is_torchscript(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return any(n.endswith(("constants.pkl", "code/__torch__.py")) or "/code/" in n for n in z.namelist())
    except zipfile.BadZipFile:
        return False


def load_model(path: str | Path, arch: str | None = None, arch_kwargs: dict | None = None,
               strict: bool = True) -> Any:
    """Return nn.Module / ScriptModule for torch files, the path itself for .onnx / .tflite.

    arch: 'pkg.module:factory' building the architecture when the file holds a state_dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    fmt = model_format(path)
    if fmt in ("onnx", "tflite"):
        return str(path)
    torch = _torch()
    nn = torch.nn
    if path.suffix.lower() == ".pt2":
        log.info("%s: torch.export program (export/quantize/measure only, no pruning)", path.name)
        return torch.export.load(str(path)).module()
    if _is_torchscript(path):
        log.info("%s: TorchScript (export/quantize/measure only, no pruning)", path.name)
        return torch.jit.load(str(path), map_location="cpu").eval()
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.float().eval()
    if isinstance(obj, dict):
        for key in ("model", "ema", "module", "net"):  # common checkpoint layouts
            if isinstance(obj.get(key), nn.Module):
                log.info("%s: using module stored under %r", path.name, key)
                return obj[key].float().eval()
        sd = obj.get("state_dict", obj.get("model_state_dict", obj))
        if not arch:
            raise ValueError(f"{path.name} holds a state_dict: pass arch='pkg.module:factory' "
                             "(config model.arch / --arch) to build the architecture")
        model = import_callable(arch)(**(arch_kwargs or {}))
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if missing or unexpected:
            log.warning("state_dict mismatch: missing=%s unexpected=%s", missing[:5], unexpected[:5])
        return model.eval()
    raise TypeError(f"{path.name}: don't know how to use a {type(obj).__name__}")


def capabilities(model: Any) -> dict:
    fmt = model_format(model)
    return {"format": fmt, **CAPABILITIES.get(fmt, {})}


def convert_tflite_to_onnx(tflite_path: str | Path, onnx_path: str | Path, opset: int = 17) -> Path:
    """Optional: float TFLite -> ONNX with tf2onnx (needs `pip install tf2onnx tensorflow`)."""
    try:
        import tf2onnx
    except ImportError as e:
        raise RuntimeError("TFLite -> ONNX needs `pip install tf2onnx tensorflow`") from e
    tf2onnx.convert.from_tflite(str(tflite_path), opset=opset, output_path=str(onnx_path))
    return Path(onnx_path)
