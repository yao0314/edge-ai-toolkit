"""Get the trained baseline model for an adapter, cached under weights/<task>/."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from .config import weights_dir

log = logging.getLogger(__name__)


def save_model(model: nn.Module, path: str | Path):
    """Save the full module (architecture included), needed for pruned models."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu(), path)


def load_model(path: str | Path) -> nn.Module:
    return torch.load(path, map_location="cpu", weights_only=False)


def get_baseline(adapter, weights: str | Path | None = None, retrain: bool = False,
                 model_cfg: dict | None = None):
    """Return the baseline: nn.Module, TorchScript module, or a path to .onnx / .tflite.

    Priority:
    - `weights` (CLI --model) or `model_cfg['path']` (config `model:`): a file we were handed.
      .onnx/.tflite -> returned as path; .pt -> full module / checkpoint / TorchScript, or a
      state_dict loaded into model_cfg['arch'] (a 'pkg.module:factory') or adapter.build_model().
    - else cached weights/<task>/baseline[_quick].pt if present.
    - else build_model() + adapter.train_baseline(), then cache.
    Every torch module then goes through adapter.prepare_model().
    """
    model = _get_baseline(adapter, weights, retrain, model_cfg or {})
    if isinstance(model, nn.Module) and not isinstance(model, torch.jit.ScriptModule):
        model = adapter.prepare_model(model)
    return model


def _get_baseline(adapter, weights, retrain, model_cfg):
    from .model_io import load_model

    path = weights or model_cfg.get("path")
    if path:
        try:
            return load_model(path, model_cfg.get("arch"), model_cfg.get("arch_kwargs"),
                              strict=model_cfg.get("strict", True))
        except ValueError:  # plain state_dict without arch: use the adapter's architecture
            obj = torch.load(path, map_location="cpu", weights_only=False)
            model = adapter.build_model()
            sd = obj.get("state_dict", obj.get("model_state_dict", obj))
            model.load_state_dict(sd, strict=model_cfg.get("strict", True))
            return model.eval()

    cache = weights_dir(getattr(adapter, "run_name", adapter.name)) / (
        "baseline_quick.pt" if adapter.quick else "baseline.pt")
    if cache.exists() and not retrain:
        log.info("loading cached baseline %s", cache)
        return load_model(cache)
    log.info("training baseline for %s (quick=%s)", adapter.name, adapter.quick)
    model = adapter.train_baseline(adapter.prepare_model(adapter.build_model()))
    save_model(model, cache)
    log.info("saved baseline -> %s", cache)
    return model
