"""Knowledge distillation (Hinton et al. 2015): soft targets from a teacher + hard labels.

Default loss works on logits:  alpha * T^2 * KL(softmax(t/T) || softmax(s/T)) + (1-alpha) * CE(s, y)
Adapters whose outputs are not logits override TaskAdapter.distill_loss().
The train loader's batches are (input, target, ...); the hard term needs integer targets.

Typical uses: (1) teacher = baseline, student = adapter.build_student(); (2) teacher =
baseline, student = its pruned copy (KD as the recovery step after pruning).
"""
from __future__ import annotations

import copy
import logging
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def kd_loss(student_out, teacher_out, target=None, temperature: float = 4.0, alpha: float = 0.7):
    s, t = student_out, teacher_out
    if isinstance(s, (tuple, list)):
        s, t = s[0], t[0]
    soft = F.kl_div(F.log_softmax(s.float() / temperature, dim=1), F.softmax(t.float() / temperature, dim=1),
                    reduction="batchmean") * temperature ** 2
    if target is None or alpha >= 1:
        return soft
    return alpha * soft + (1 - alpha) * F.cross_entropy(s.float(), target)


def _target(batch):
    if isinstance(batch, (list, tuple)) and len(batch) > 1 and torch.is_tensor(batch[1]) \
            and batch[1].dtype in (torch.int64, torch.int32) and batch[1].dim() == 1:
        return batch[1]
    return None


def distill(teacher: nn.Module, student: nn.Module, adapter, epochs: int, temperature: float = 4.0,
            alpha: float = 0.7, lr: float = 0.01, optimizer: str = "sgd", weight_decay: float = 4e-5,
            max_steps_per_epoch: int | None = None, history: list | None = None,
            eval_fn=None, eval_every: int = 0) -> nn.Module:
    """`history` collects the same per-epoch dicts as common.train_loop, so a KD run produces the
    same curve as a plain one. Without it the sweep recorded train_epochs_run 0 and wrote no curve
    for any distilled variant, which is exactly where a recovery run needs to be readable."""
    if epochs <= 0:
        return student
    dev = adapter.device
    teacher = copy.deepcopy(teacher).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = student.to(dev).train()
    loader = adapter.load_data()["train"]
    params = [p for p in student.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay) if optimizer == "adam"
           else torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True))
    steps = epochs * (min(len(loader), max_steps_per_epoch) if max_steps_per_epoch else len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    use_amp = dev.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for ep in range(epochs):
        t0, tot, n = time.time(), 0.0, 0
        for i, batch in enumerate(loader):
            if max_steps_per_epoch and i >= max_steps_per_epoch:
                break
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(dev, non_blocking=True)
            y = _target(batch)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                with torch.no_grad():
                    t_out = teacher(x)
                s_out = student(x)
                loss = adapter.distill_loss(s_out, t_out, batch)
                if loss is None:
                    loss = kd_loss(s_out, t_out, y.to(dev) if y is not None else None, temperature, alpha)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + float(loss.detach()), n + 1
        secs, mean_loss = time.time() - t0, tot / max(n, 1)
        metric = None
        if eval_fn and eval_every and ((ep + 1) % eval_every == 0 or ep + 1 == epochs):
            student.eval()
            try:
                metric = float(eval_fn(student))
            except Exception as e:      # a metric is a diagnostic; it must never kill the training
                log.warning("[distill] epoch %d: eval_fn failed: %s", ep + 1, e)
            student.train()
        if history is not None:
            history.append({"tag": "distill", "epoch": ep + 1, "epochs": epochs, "loss": mean_loss,
                            "lr": opt.param_groups[0]["lr"], "seconds": round(secs, 1),
                            "metric": metric})
        log.info("[distill] epoch %d/%d loss %.4f%s (%.0fs)", ep + 1, epochs, mean_loss,
                 f" metric {metric:.4f}" if metric is not None else "", secs)
    return student.eval()
