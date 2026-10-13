"""Quantization-aware fine-tuning (QAT), used to make post-training INT8 work on models that PTQ breaks.

Post-training quantization fails when a layer's activations are badly suited to ONE scale per tensor
(e.g. SiLU blocks whose channels span +-90 next to channels within +-1). QAT does not change that scale;
it changes the *weights*, by training the model while it sees the quantization error:

  1. insert fake-quant on the output of every Conv/Linear (matching ORT's scheme: per-tensor ASYMMETRIC
     uint8 activations, the thing that actually hurts),
  2. observe ranges on calibration batches with the network frozen,
  3. fine-tune with a straight-through estimator (gradients flow as if quantization were the identity),
  4. remove the fake-quant and export FP32 as usual - the ordinary static INT8 path then loses less.

Task-agnostic: it only needs adapter.load_data() and adapter.finetune().

Caveat for the INT8 health check: QAT deliberately trains the model to tolerate quantization error, so the
quantized outputs stay far from the FP32 ones and `int8_sqnr_db` stays LOW even when accuracy is fine.
Judge a QAT variant by the task metric, not by SQNR.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class FakeQuant(nn.Module):
    """Per-tensor affine fake quantization with an EMA min/max observer and a straight-through gradient."""

    def __init__(self, bits: int = 8, symmetric: bool = False, momentum: float = 0.1):
        super().__init__()
        self.qmin, self.qmax = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if symmetric else (0, 2 ** bits - 1)
        self.symmetric, self.momentum = symmetric, momentum
        self.register_buffer("lo", torch.tensor(float("inf")))
        self.register_buffer("hi", torch.tensor(float("-inf")))
        self.observing, self.enabled = True, True

    def _update(self, x: torch.Tensor) -> None:
        lo, hi = x.detach().amin().float(), x.detach().amax().float()
        if torch.isinf(self.lo):
            self.lo, self.hi = lo, hi
        else:
            m = self.momentum
            self.lo, self.hi = (1 - m) * self.lo + m * lo, (1 - m) * self.hi + m * hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observing:
            self._update(x)
        if not self.enabled or torch.isinf(self.lo):
            return x
        lo, hi = float(self.lo), float(self.hi)
        if self.symmetric:
            scale = max(abs(lo), abs(hi)) / max(self.qmax, 1)
            zp = 0.0
        else:
            scale = (hi - lo) / (self.qmax - self.qmin)
            zp = self.qmin - lo / scale if scale > 0 else 0.0
        if scale <= 0:
            return x
        q = torch.clamp(torch.round(x / scale + zp), self.qmin, self.qmax)
        dq = (q - zp) * scale
        return x + (dq - x).detach()          # straight-through estimator


def insert_fake_quant(model: nn.Module, types=(nn.Conv1d, nn.Conv2d, nn.Linear), skip: list | None = None,
                      bits: int = 8) -> list:
    """Fake-quantize the OUTPUT of every module of `types`. Returns the hook handles (call .remove())."""
    skip_ids = {id(m) for m in (skip or [])}
    handles, quants = [], []
    for mod in model.modules():
        if isinstance(mod, types) and id(mod) not in skip_ids:
            fq = FakeQuant(bits=bits).to(next(mod.parameters()).device)
            quants.append(fq)
            handles.append(mod.register_forward_hook(lambda m, i, o, fq=fq: fq(o) if torch.is_tensor(o) else o))
    log.info("QAT: fake-quant inserted on %d layer outputs", len(quants))
    return handles, quants


@torch.no_grad()
def observe(model: nn.Module, loader, device: str, batches: int = 8) -> int:
    """Collect activation ranges without training (the network is frozen)."""
    model.to(device).eval()
    n = 0
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(x.to(device, dtype=torch.float32))
        n += 1
    return n


def qat_finetune(adapter, model: nn.Module, epochs: int, cfg: dict | None = None, finetune_fn=None) -> nn.Module:
    """Fake-quantize -> observe ranges -> fine-tune with STE -> remove fake-quant. Returns an FP32 model
    whose weights tolerate INT8 better; quantize it afterwards exactly as usual."""
    cfg = cfg or {}
    skip = []
    if cfg.get("skip_ignored_layers", True):
        try:
            skip = adapter.ignored_layers(model)     # the task head: keep it out of the simulation
        except Exception:
            skip = []
    handles, quants = insert_fake_quant(model, skip=skip, bits=int(cfg.get("bits", 8)))
    try:
        n = observe(model, adapter.bn_calibration_loader(), adapter.device, int(cfg.get("observe_batches", 8)))
        for q in quants:
            q.observing = bool(cfg.get("keep_observing", True))   # keep tracking ranges while training
        log.info("QAT: ranges observed on %d batches; fine-tuning %d epoch(s)", n, epochs)
        model = (finetune_fn or adapter.finetune)(model, epochs)
    finally:
        for h in handles:
            h.remove()                                # the exported model is plain FP32 again
    return model.eval()
