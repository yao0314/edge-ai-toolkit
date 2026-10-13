"""TaskAdapter: the only interface between the task-agnostic core and a task.

Everything task-specific (data, model architecture, metric, training loop,
layers that must not be pruned) is implemented by a subclass in adapters/.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class TaskAdapter(ABC):
    #: short task name, used for results/<task>/ and weights/<task>/
    name: str = "task"

    def __init__(self, cfg: dict | None = None, quick: bool = False, device: str | None = None):
        self.cfg = dict(cfg or {})
        self.quick = quick
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Per-epoch training curve, appended by the adapter's training loop and drained by
        # core.sweep after each variant. Task-agnostic: just a list of dicts.
        self.history: list[dict] = []

    # ------------------------------------------------------------------ required
    @abstractmethod
    def load_data(self) -> dict:
        """Return {"train": DataLoader, "val": DataLoader, "calib": DataLoader}.

        "calib" batches are fed to INT8 calibration; the first element of each
        batch must be the model input tensor.
        """

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Return the (untrained or pretrained-init) baseline architecture."""

    @abstractmethod
    def evaluate(self, model: Any) -> dict:
        """Evaluate a torch nn.Module OR an ONNX model (path / ort.InferenceSession).

        Use core.runtime.as_predictor(model) to get a uniform callable.
        Must return {"metric_name": str, "score": float}.

        `score` MUST be HIGHER-is-better. This is not a style preference: score.py's accuracy
        constraint rejects anything that fell below the baseline, and pareto.py keeps points with
        y >= y[i]. Return a raw error (MAE, RMSE, reconstruction loss) and the entire ranking
        inverts silently - the best variant is excluded as "too much accuracy lost" and the worst
        one is selected. Nothing crashes, so nothing else catches it. For an error metric return
        its negation (-mae) or a reciprocal; measure() warns when metric_name looks like one.
        """

    @abstractmethod
    def example_input(self) -> torch.Tensor:
        """One real input with batch=1, used for export, MACs and latency."""

    @abstractmethod
    def finetune(self, model: nn.Module, epochs: int, lr: float | None = None) -> nn.Module:
        """Train `model` for `epochs` (used after pruning). Return the model."""

    @abstractmethod
    def ignored_layers(self, model: nn.Module) -> list:
        """Modules the structured pruner must not touch (e.g. the output layer / task head)."""

    # ------------------------------------------------------------------ optional
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Called on every baseline (built, cached or loaded from a file) before compression.

        Use it for function-preserving rewrites that make a model family prunable/exportable
        (e.g. replacing ops the pruner cannot trace) or to switch it to its export mode.
        """
        return model

    def bn_calibration_loader(self):
        """Unlabelled batches (first element = model input) for BatchNorm re-calibration after pruning.

        Default: the INT8 calibration loader. Override when BN statistics depend on something the
        calib loader does not match (e.g. a detector pretrained at a larger resolution)."""
        return self.load_data()["calib"]

    def train_baseline(self, model: nn.Module) -> nn.Module:
        """Train the baseline from build_model(). Default: finetune for cfg['train_epochs']."""
        return self.finetune(model, int(self.cfg.get("train_epochs", 1)))

    def build_student(self, **kwargs) -> nn.Module:
        """Smaller architecture for distillation. Default: same as build_model()."""
        return self.build_model()

    def distill_loss(self, student_out, teacher_out, batch) -> torch.Tensor | None:
        """Optional task-specific KD loss. Return None to use core's logits KD loss."""
        return None

    def input_names(self) -> list[str]:
        return ["input"]

    def output_names(self) -> list[str]:
        return ["output"]

    def preprocessing(self) -> dict:
        """How a raw sample becomes the tensor the exported model expects.

        This goes into the deliverable. Without it the artifact is not reproducible: a judge handed
        best.onnx and an input shape of [batch, 3, 320, 320] cannot tell RGB from BGR, [0,1] from
        [0,255], or which resize was used, so they cannot reproduce the accuracy we report - and a
        number nobody can reproduce is a claim, not a result.

        Recommended keys (all optional, include what applies):
          layout   "NCHW" | "NLC" | ...        color   "RGB" | "BGR" | None
          dtype    "float32"                   scale   divide raw values by this before normalising
          mean/std per-channel normalisation, or scalars
          resize   {"to": ..., "mode": ..., "pad_value": ...}
          steps    ordered list of human-readable strings - the authoritative description
        Return {} when there is genuinely nothing to declare.
        """
        return {}

    def describe(self) -> dict:
        """Free-form metadata written next to results (dataset, model, sizes)."""
        return {"task": self.name, "quick": self.quick}
