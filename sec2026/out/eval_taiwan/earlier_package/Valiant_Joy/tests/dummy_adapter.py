"""Tiny synthetic adapter used by the smoke tests (runs in seconds on CPU, no downloads)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.adapter import TaskAdapter
from core.runtime import as_predictor


class TinyNet(nn.Module):
    def __init__(self, c=16, n_cls=4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c), nn.BatchNorm2d(2 * c), nn.ReLU(),
        )
        self.head = nn.Linear(2 * c, n_cls)

    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.body(x), 1).flatten(1))


class DummyAdapter(TaskAdapter):
    name = "dummy"

    def __init__(self, cfg=None, quick=True, device="cpu"):
        super().__init__(cfg, quick, device)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(384, 3, 16, 16, generator=g)
        y = (x[:, 0].mean((1, 2)) > 0).long() + 2 * (x[:, 1].mean((1, 2)) > 0).long()
        self.x, self.y = x, y

    def load_data(self):
        tr = TensorDataset(self.x[:256], self.y[:256])
        va = TensorDataset(self.x[256:], self.y[256:])
        return {"train": DataLoader(tr, batch_size=32, shuffle=True),
                "val": DataLoader(va, batch_size=64),
                "calib": DataLoader(tr, batch_size=32)}

    def build_model(self):
        torch.manual_seed(0)
        return TinyNet(c=int(self.cfg.get("width", 16)))   # `width` lets tests sweep an adapter dimension

    def build_student(self, width=8, **_):
        """A genuinely narrower net, so a students sweep changes the parameter count. The base
        class returns build_model() unchanged, which would make such a test vacuous."""
        torch.manual_seed(0)
        return TinyNet(c=int(width))

    def evaluate(self, model):
        pred = as_predictor(model, self.device)
        correct = 0
        for x, y in self.load_data()["val"]:
            correct += int((np.argmax(pred(x), 1) == y.numpy()).sum())
        return {"metric_name": "top1", "score": correct / 128}

    def example_input(self):
        return self.x[256:257]

    def finetune(self, model, epochs, lr=None):
        model.train()
        opt = torch.optim.Adam(model.parameters(), 1e-2)
        for _ in range(epochs):
            for x, y in self.load_data()["train"]:
                opt.zero_grad()
                F.cross_entropy(model(x), y).backward()
                opt.step()
        return model.eval()

    def ignored_layers(self, model):
        return [model.head]
