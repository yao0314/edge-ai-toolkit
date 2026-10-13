"""YAML config loading, --quick overrides, adapter instantiation, paths."""
from __future__ import annotations

import copy
import importlib
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    """YAML with optional `extends: other.yaml` (relative to this file), deep-merged underneath."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("extends", None)
    if base:
        cfg = deep_merge(_read_yaml((path.parent / base).resolve()), cfg)
    return cfg


def load_config(path: str | Path, quick: bool = False, overrides: dict | None = None) -> dict:
    """Load YAML (following `extends:`). When quick=True the `quick:` section is deep-merged on top."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    cfg = _read_yaml(path)
    quick_over = cfg.pop("quick", {}) or {}
    if quick:
        cfg = deep_merge(cfg, quick_over)
    cfg = deep_merge(cfg, overrides or {})
    cfg["_quick"] = quick
    cfg["_config_path"] = str(path)
    return cfg


def config_for_task(task: str) -> Path:
    p = ROOT / "configs" / f"{task}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no config for task {task!r}: expected {p}")
    return p


def run_name(cfg: dict) -> str:
    """Name for results/<name>/ and weights/<name>/: config `name`, default the task."""
    return cfg.get("name") or cfg["task"]


def build_adapter(cfg: dict, device: str | None = None):
    """Instantiate `adapter: package.module:ClassName` with cfg['adapter_args']."""
    mod_name, cls_name = cfg["adapter"].split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    adapter = cls(cfg.get("adapter_args", {}), quick=cfg.get("_quick", False), device=device)
    adapter.run_name = run_name(cfg)
    return adapter


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_env():
    """Keep every download inside the repo so later runs work offline."""
    os.environ.setdefault("TORCH_HOME", str(ROOT / "weights" / "torch"))
    os.environ.setdefault("HF_HOME", str(ROOT / "weights" / "hf"))  # timm pretrained weights
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")


def results_dir(task: str, quick: bool = False) -> Path:
    d = ROOT / "results" / (f"{task}_quick" if quick else task)
    d.mkdir(parents=True, exist_ok=True)
    return d


def weights_dir(task: str) -> Path:
    d = ROOT / "weights" / task
    d.mkdir(parents=True, exist_ok=True)
    return d
