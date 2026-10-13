"""Helpers shared by adapters (not part of core: may know about losses / training)."""
from __future__ import annotations

import logging
import math
import time
import warnings
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

log = logging.getLogger(__name__)


def train_loop(model: nn.Module, loader: DataLoader, loss_fn: Callable, epochs: int, lr: float,
               device: str, weight_decay: float = 4e-5, optimizer: str = "sgd", amp: bool = True,
               max_steps_per_epoch: int | None = None, grad_clip: float | None = None,
               warmup_frac: float = 0.0, history: list | None = None, eval_fn: Callable | None = None,
               eval_every: int = 0, tag: str = "train") -> nn.Module:
    """Generic loop: loss_fn(model, batch, device) -> scalar loss. Cosine LR, optional AMP (CUDA only).

    `warmup_frac` ramps the LR linearly over the first fraction of steps. It matters for recovering a
    freshly pruned model: the weights are damaged and the first full-LR step can push them further
    out, which is one reason a plain constant-LR loop plateaued at a bad point in the pruning study.

    `history` (a list) collects one dict per epoch: loss, lr, seconds and - every `eval_every`
    epochs, if `eval_fn` is given - the task metric. Recording BOTH matters: falling loss does not
    mean a rising metric. In the pruning study the loss kept dropping (0.5747 -> 0.5355 over 18 more
    epochs) while the accuracy gap barely moved (-3.18 -> -2.83 pt), and reading the loss alone
    would have suggested there was headroom left.
    """
    if epochs <= 0:
        return model
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == "adam":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    steps = epochs * (min(len(loader), max_steps_per_epoch) if max_steps_per_epoch else len(loader))
    warm = max(int(steps * max(min(float(warmup_frac), 0.5), 0.0)), 0)

    def lr_at(step):
        if warm and step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # AMP may skip the very first optimizer step (inf grads); the resulting warning is harmless
    warnings.filterwarnings("ignore", message="Detected call of")
    for ep in range(epochs):
        t0, tot, n = time.time(), 0.0, 0
        for i, batch in enumerate(loader):
            if max_steps_per_epoch and i >= max_steps_per_epoch:
                break
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model, batch, device)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + float(loss.detach()), n + 1
        secs, mean_loss = time.time() - t0, tot / max(n, 1)
        metric = None
        if eval_fn and eval_every and ((ep + 1) % eval_every == 0 or ep + 1 == epochs):
            model.eval()
            try:
                metric = float(eval_fn(model))
            except Exception as e:      # a metric is a diagnostic; it must never kill the training
                log.warning("[%s] epoch %d: eval_fn failed: %s", tag, ep + 1, e)
            model.train()
        if history is not None:
            history.append({"tag": tag, "epoch": ep + 1, "epochs": epochs, "loss": mean_loss,
                            "lr": opt.param_groups[0]["lr"], "seconds": round(secs, 1),
                            "metric": metric})
        log.info("[%s] epoch %d/%d loss %.4f%s (%.0fs)", tag, ep + 1, epochs, mean_loss,
                 f" metric {metric:.4f}" if metric is not None else "", secs)
    model.eval()
    return model


def subset(ds, n: int | None, seed: int = 0):
    """First-n random subset (deterministic) or the full dataset when n is None."""
    if not n or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)


def loader_kwargs(num_workers: int) -> dict:
    kw = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    if num_workers > 0:
        kw["persistent_workers"] = True
    return kw
