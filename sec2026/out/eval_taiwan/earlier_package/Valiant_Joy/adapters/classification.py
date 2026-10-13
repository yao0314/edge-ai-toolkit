"""Image classification adapter: MobileNetV2 on CIFAR-10 (or any ImageFolder dataset)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.adapter import TaskAdapter
from core.config import ROOT
from core.runtime import as_predictor

from .common import loader_kwargs, subset, train_loop

log = logging.getLogger(__name__)

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

DEFAULTS = {
    "dataset": "cifar10",       # cifar10 | imagefolder (data_root/train, data_root/val) | eurosat
    "data_root": "data",
    "split_json": None,         # eurosat: {"train": [...], "val": [...]} indices supplied with a task
    "allow_unsplit": False,     # eurosat without that file: exploration only, not comparable
    "augment": True,            # eurosat: flips during training; off makes train and eval identical
    "download": True,           # fetch the dataset if absent
    "img_size": 32,
    "arch": "mobilenet_v2",     # torchvision name (mobilenet_v2, resnet18, ...) or "timm:<name>"
    "timm_kwargs": None,        # extra timm.create_model kwargs, e.g. {img_size: 32, patch_size: 4}
    "pretrained": True,         # ImageNet init (cached in weights/torch)
    "width_mult": 1.0,
    "small_input_stem": True,   # stride-1 first conv for 32x32 inputs
    "batch_size": 128,
    "eval_batch_size": 256,
    "num_workers": 4,
    "train_epochs": 15,
    "lr": 0.05,
    "finetune_lr": 0.01,
    "weight_decay": 4e-5,
    "optimizer": "sgd",         # sgd | adam (AdamW; better for ViT-style models)
    "label_smoothing": 0.1,
    "calib_size": 512,
    "train_subset": None,
    "val_subset": None,
    "max_steps_per_epoch": None,
}


class ClassificationAdapter(TaskAdapter):
    name = "classification"

    def __init__(self, cfg=None, quick=False, device=None):
        super().__init__({**DEFAULTS, **(cfg or {})}, quick, device)
        self._data = None
        self.num_classes = None

    # ------------------------------------------------------------------ data
    def _datasets(self):
        import torchvision.transforms as T
        from torchvision import datasets

        c = self.cfg
        s = c["img_size"]
        root = Path(c["data_root"])
        root = root if root.is_absolute() else ROOT / root
        norm = T.Normalize(MEAN, STD)
        if c["dataset"] == "cifar10":
            aug = [T.RandomCrop(s, padding=4)] if s == 32 else [T.Resize(s), T.RandomCrop(s, padding=s // 8)]
            train_tf = T.Compose(aug + [T.RandomHorizontalFlip(), T.ToTensor(), norm])
            eval_tf = T.Compose(([] if s == 32 else [T.Resize(s)]) + [T.ToTensor(), norm])
            train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
            val = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)
            calib = datasets.CIFAR10(root, train=True, download=True, transform=eval_tf)
            classes = train.classes
        elif c["dataset"] == "imagefolder":
            train_tf = T.Compose([T.RandomResizedCrop(s, scale=(0.5, 1)), T.RandomHorizontalFlip(), T.ToTensor(), norm])
            eval_tf = T.Compose([T.Resize(int(s * 1.14)), T.CenterCrop(s), T.ToTensor(), norm])
            train = datasets.ImageFolder(root / "train", transform=train_tf)
            val = datasets.ImageFolder(root / "val", transform=eval_tf)
            calib = datasets.ImageFolder(root / "train", transform=eval_tf)
            classes = train.classes
        elif c["dataset"] == "eurosat":
            # One dataset, split by a file of indices supplied with the task. The evaluation
            # transform is Resize(s) alone, with no crop: EuroSAT tiles are square, and a centre
            # crop after an aspect-preserving resize would discard a border that the organisers'
            # own transform keeps, which is enough to make our accuracy incomparable to theirs.
            eval_tf = T.Compose([T.Resize(s), T.ToTensor(), norm])
            train_tf = T.Compose([T.Resize(s), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                                  T.ToTensor(), norm]) if c.get("augment", True) else eval_tf
            dl = bool(c.get("download", True))
            train = datasets.EuroSAT(root=str(root), download=dl, transform=train_tf)
            val = datasets.EuroSAT(root=str(root), download=False, transform=eval_tf)
            calib = datasets.EuroSAT(root=str(root), download=False, transform=eval_tf)
            classes = train.classes
            split = c.get("split_json")
            if split:
                import json

                from torch.utils.data import Subset

                p = Path(split)
                idx = json.loads((p if p.is_absolute() else ROOT / p).read_text(encoding="utf-8"))
                missing = {"train", "val"} - set(idx)
                if missing:
                    raise ValueError(f"{split} has no {sorted(missing)} key")
                train = Subset(train, idx["train"])
                val = Subset(val, idx["val"])
                # Calibration draws from the TRAIN indices only: calibrating a quantizer on the
                # scored split would leak it into the model being scored.
                calib = Subset(calib, idx["train"])
            elif not c.get("allow_unsplit", False):
                raise ValueError(
                    "dataset: eurosat without split_json. The task supplies val_split.json and "
                    "scores on it; a split of our own is not comparable. Set allow_unsplit: true "
                    "only to explore before the file exists.")
        else:
            raise ValueError(f"unknown dataset {c['dataset']}")
        self.num_classes = len(classes)
        return train, val, calib

    def load_data(self) -> dict:
        if self._data is None:
            c = self.cfg
            train, val, calib = self._datasets()
            train = subset(train, c["train_subset"])
            val = subset(val, c["val_subset"])
            calib = subset(calib, c["calib_size"], seed=1)
            kw = loader_kwargs(c["num_workers"])
            self._data = {
                "train": DataLoader(train, batch_size=c["batch_size"], shuffle=True, drop_last=True, **kw),
                "val": DataLoader(val, batch_size=c["eval_batch_size"], shuffle=False, **kw),
                "calib": DataLoader(calib, batch_size=32, shuffle=False),
            }
        return self._data

    # ------------------------------------------------------------------ model
    def _make(self, width_mult: float, pretrained: bool, arch: str | None = None) -> nn.Module:
        """arch: a torchvision name (mobilenet_v2, resnet18, ...) or 'timm:<name>' (mobilevit_xxs,
        deit_tiny_patch16_224, ...). timm kwargs come from cfg['timm_kwargs'] (e.g. img_size, patch_size)."""
        if self.num_classes is None:
            self.load_data()
        c = self.cfg
        arch = arch or c["arch"]
        if arch.startswith("timm:"):
            import timm

            return timm.create_model(arch[5:], pretrained=pretrained, num_classes=self.num_classes,
                                     **(c.get("timm_kwargs") or {}))

        import torchvision.models as tvm

        uses_width = arch.startswith("mobilenet_v2")
        kwargs = {"width_mult": width_mult} if uses_width else {}
        # torchvision ships mobilenet_v2 weights for the full width only; the other architectures
        # carry their own and ignore width_mult, so the width guard must not block them.
        weights = "DEFAULT" if pretrained and (not uses_width or width_mult == 1.0) else None
        if pretrained and weights is None:
            # Silently dropping this is how a "pretrained" student ends up at chance accuracy with
            # nothing in the log to explain it: torchvision has no weights for a scaled width, and
            # a few distillation epochs cannot train one from scratch.
            log.warning("no pretrained weights for %s at width_mult=%s - training from scratch. "
                        "Vary the ARCHITECTURE instead of the width to stay pretrained.",
                        arch, width_mult)
        model = getattr(tvm, arch)(weights=weights, **kwargs)
        # replace the last Linear with a num_classes head
        name, last = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], nn.Linear(last.in_features, self.num_classes))
        if c["small_input_stem"] and c["img_size"] <= 64:
            # tiny inputs (e.g. 32x32): keep resolution in the stem (first conv stride 1, no stem max-pool)
            first = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
            first.stride = (1, 1)
            if isinstance(getattr(model, "maxpool", None), nn.MaxPool2d):
                model.maxpool = nn.Identity()
        return model

    def build_model(self) -> nn.Module:
        return self._make(self.cfg["width_mult"], self.cfg["pretrained"])

    def build_student(self, width_mult: float = 1.0, arch: str | None = None,
                      pretrained: bool = False, **_) -> nn.Module:
        """`pretrained` starts the student from public ImageNet weights instead of from scratch.
        Off by default - a student derived only from the baseline is the cleaner claim - but with a
        few KD epochs the difference is large, so it is a per-student choice rather than hard-wired.

        width_mult defaults to 1.0, not to a scaled-down value: naming an architecture and getting
        a half-width version of it is a surprise, and for the one architecture that reads the knob
        it is also the only width with pretrained weights. Shrink by choosing a smaller
        architecture; use width_mult only when training from scratch anyway.
        """
        return self._make(width_mult, pretrained=pretrained, arch=arch)

    def ignored_layers(self, model: nn.Module) -> list:
        return [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == self.num_classes]

    def example_input(self) -> torch.Tensor:
        x = self.load_data()["val"].dataset[0][0]
        return x.unsqueeze(0)

    # ------------------------------------------------------------------ train / eval
    def _loss(self, model, batch, device):
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        return F.cross_entropy(model(x), y, label_smoothing=self.cfg["label_smoothing"])

    def _train(self, model, epochs, lr, tag, warmup_frac=0.0):
        # adapter_args.eval_every > 0 evaluates the task metric during training, so a curve shows
        # whether the metric is still moving - falling loss alone does not prove it is.
        every = int(self.cfg.get("eval_every", 0) or 0)
        ev = (lambda m: self.evaluate(m)["score"]) if every else None
        c = self.cfg
        return train_loop(model, self.load_data()["train"], self._loss, epochs, lr, self.device,
                          weight_decay=c["weight_decay"], optimizer=c["optimizer"],
                          max_steps_per_epoch=c["max_steps_per_epoch"],
                          warmup_frac=warmup_frac, history=self.history,
                          eval_fn=ev, eval_every=every, tag=tag)

    def train_baseline(self, model):
        return self._train(model, int(self.cfg["train_epochs"]), self.cfg["lr"], "baseline")

    def finetune(self, model, epochs, lr=None):
        return self._train(model, int(epochs), lr if lr is not None else self.cfg["finetune_lr"],
                           "finetune", warmup_frac=float(self.cfg.get("finetune_warmup_frac", 0.0)))

    def evaluate(self, model) -> dict:
        """Top-1 plus per-class recall and the worst class.

        A mean hides a collapse: a model that loses one class entirely and gains a little on the
        other nine can read higher overall than one that degrades evenly, while being the worse
        model to deploy. The confusion matrix costs one extra pass over predictions we already
        have, so the per-class figures are always computed, never an opt-in.
        """
        pred = as_predictor(model, self.device)
        n = self.num_classes or 0
        if not n:
            self.load_data()
            n = self.num_classes
        conf = np.zeros((n, n), dtype=np.int64)
        for x, y in self.load_data()["val"]:
            p = np.argmax(pred(x), axis=1)
            for t, q in zip(y.numpy(), p):
                conf[int(t), int(q)] += 1
        support = conf.sum(axis=1)
        seen = support > 0
        recall = np.divide(conf.diagonal(), np.maximum(support, 1), dtype=float)
        total = int(support.sum())
        return {"metric_name": "top1", "score": float(conf.diagonal().sum()) / max(total, 1),
                "per_class_recall": [round(float(v), 6) for v in recall],
                # Only classes the split actually contains: an absent class would read as a
                # worst-class recall of zero and make every model look broken.
                "worst_class_recall": float(recall[seen].min()) if seen.any() else float("nan"),
                "worst_class_index": int(np.argmin(np.where(seen, recall, np.inf))) if seen.any() else -1,
                "class_support": [int(v) for v in support]}

    def preprocessing(self) -> dict:
        """Mirrors eval_tf in _datasets() exactly. If that changes, change this."""
        c = self.cfg
        s = int(c["img_size"])
        if c["dataset"] == "cifar10":
            resize = None if s == 32 else {"to": s, "mode": "bilinear", "note": "shorter side"}
            steps = ([] if s == 32 else [f"resize shorter side to {s} (bilinear)"])
        else:
            resize = {"to": s, "mode": "bilinear", "note": f"resize shorter side to {int(s * 1.14)}, "
                                                          f"then centre-crop {s}"}
            steps = [f"resize shorter side to {int(s * 1.14)} (bilinear)", f"centre-crop {s}x{s}"]
        steps += ["convert to float32 and divide by 255",
                  f"normalise per channel: mean={list(MEAN)}, std={list(STD)}",
                  "arrange as NCHW"]
        return {"layout": "NCHW", "color": "RGB", "dtype": "float32", "scale": 255.0,
                "mean": list(MEAN), "std": list(STD), "resize": resize, "steps": steps}

    def describe(self) -> dict:
        d = self.load_data()
        return {"task": self.name, "quick": self.quick, "dataset": self.cfg["dataset"],
                "arch": self.cfg["arch"], "img_size": self.cfg["img_size"],
                "n_train": len(d["train"].dataset), "n_val": len(d["val"].dataset),
                "num_classes": self.num_classes}
