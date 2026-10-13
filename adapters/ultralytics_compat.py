"""Make ultralytics YOLO models prunable and exportable (model-family helper, not task code).

C2f (and subclasses such as C3k2) split their first conv output with `.chunk(2, 1)`, which
torch-pruning cannot track (it indexes past the split). `make_prunable` swaps every C2f for
SplitC2f: the first 1x1 conv is cut into two convs holding the same weights, so the
function is identical but there is no chunk op.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def _half(block: nn.Module, part: int) -> nn.Module:
    """Copy of an ultralytics Conv (conv + bn + act) keeping output-channel half `part` (0/1)."""
    b = copy.deepcopy(block)
    conv, bn = b.conv, getattr(b, "bn", None)
    n = conv.out_channels // 2
    sl = slice(0, n) if part == 0 else slice(n, 2 * n)
    conv.weight = nn.Parameter(conv.weight.data[sl].clone())
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[sl].clone())
    conv.out_channels = n
    if isinstance(bn, nn.BatchNorm2d):
        bn.weight = nn.Parameter(bn.weight.data[sl].clone())
        bn.bias = nn.Parameter(bn.bias.data[sl].clone())
        bn.running_mean = bn.running_mean[sl].clone()
        bn.running_var = bn.running_var[sl].clone()
        bn.num_features = n
    return b


class SplitC2f(nn.Module):
    """C2f without chunk(): cv1 is split into cv0 + cv1 (same weights, same output)."""

    def __init__(self, c2f: nn.Module):
        super().__init__()
        self.cv0, self.cv1 = _half(c2f.cv1, 0), _half(c2f.cv1, 1)
        self.cv2, self.m = c2f.cv2, c2f.m
        self.c = self.cv0.conv.out_channels
        for a in ("f", "i", "type", "np"):  # graph bookkeeping used by ultralytics' forward
            if hasattr(c2f, a):
                setattr(self, a, getattr(c2f, a))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def make_prunable(model: nn.Module) -> nn.Module:
    """In-place: replace every ultralytics C2f/C3k2 by SplitC2f. No-op for other models."""
    try:
        from ultralytics.nn.modules.block import C2f
    except ImportError:
        return model

    def swap(parent):
        for name, child in parent.named_children():
            if isinstance(child, C2f):
                setattr(parent, name, SplitC2f(child))
            else:
                swap(child)
    swap(model)
    return model
