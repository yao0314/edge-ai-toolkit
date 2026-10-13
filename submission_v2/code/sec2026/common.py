"""Shared pieces for the SEC 2026 EuroSAT task, written to match the official task sheet.

Every function here mirrors a code block of "The Minimalist Edge AI Challenge" (Software Track):
data loading and model construction from Section 4, accuracy and per-class recall from Section 5.1.
Keep them identical to the sheet: judges score with the sheet's method, so any deviation here
would make our numbers disagree with theirs.

All official inputs live in one Starter Kit folder (default ./starter_kit, override with --kit or
the SEC_KIT environment variable):

    <kit>/student_mobilenetv2_distilled.pt   starting point and baseline
    <kit>/teacher_resnet50_eurosat.pt        optional, for distillation during recovery
    <kit>/baseline_metrics.json              official baseline figures
    <kit>/val_split.json                     fixed {"train": [...], "val": [...]} indices
    <kit>/data/                              pre-cached EuroSAT (torchvision layout)
"""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = 10
INPUT_SHAPE = (1, 3, 224, 224)          # every submission must accept exactly this (sheet, Section 3)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------- Starter Kit paths
def kit_dir(kit: str | os.PathLike | None = None) -> Path:
    k = Path(kit or os.environ.get("SEC_KIT") or ROOT / "starter_kit")
    return k if k.is_absolute() else (Path.cwd() / k)


def kit_file(name: str, kit=None, required: bool = True) -> Path:
    """Path of an official Starter Kit file. Never substitutes a stand-in when it is missing."""
    p = kit_dir(kit) / name
    if required and not p.exists():
        raise FileNotFoundError(
            f"official Starter Kit file missing: {p}\n"
            f"Copy the kit handed out at kickoff into {kit_dir(kit)} (or pass --kit / set SEC_KIT).")
    return p


STUDENT = "student_mobilenetv2_distilled.pt"
TEACHER = "teacher_resnet50_eurosat.pt"
BASELINE_METRICS = "baseline_metrics.json"
SPLIT = "val_split.json"


# ----------------------------------------------------------------------------- data (Section 4)
def eval_transform():
    """Exactly the sheet's transform: Resize(224), ToTensor, ImageNet normalisation."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def train_transform():
    """Light augmentation for short recovery fine-tuning. Satellite tiles have no canonical
    orientation, so flips are label-preserving; nothing that changes scale or colour statistics."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def load_split(kit=None) -> dict:
    split = json.loads(kit_file(SPLIT, kit).read_text())
    for key in ("train", "val"):
        if key not in split:
            raise KeyError(f"{SPLIT} has no '{key}' list (keys: {sorted(split)})")
    return split


def eurosat(kit=None, transform=None):
    from torchvision import datasets

    root = kit_dir(kit) / "data"
    if not root.exists():
        raise FileNotFoundError(f"pre-cached EuroSAT not found at {root} (copy the kit's ./data folder)")
    return datasets.EuroSAT(root=root, download=False, transform=transform or eval_transform())


def subsets(kit=None, train_tf=None, limit_train: int | None = None, limit_val: int | None = None,
            seed: int = 0):
    """(train, val) torch Subsets on the official split. `limit_*` take a fixed random sample for
    quick iterations only; reported numbers must use the full val split."""
    from torch.utils.data import Subset

    split = load_split(kit)
    tr_idx, va_idx = list(split["train"]), list(split["val"])
    rng = np.random.default_rng(seed)
    if limit_train:
        tr_idx = sorted(rng.choice(tr_idx, size=min(limit_train, len(tr_idx)), replace=False).tolist())
    if limit_val:
        va_idx = sorted(rng.choice(va_idx, size=min(limit_val, len(va_idx)), replace=False).tolist())
    train = Subset(eurosat(kit, train_tf or eval_transform()), tr_idx)
    val = Subset(eurosat(kit, eval_transform()), va_idx)
    return train, val


def loader(ds, batch_size: int = 32, shuffle: bool = False, num_workers: int = 0):
    from torch.utils.data import DataLoader

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ----------------------------------------------------------------------------- models (Section 4)
def build_student(path=None, kit=None) -> nn.Module:
    """The sheet's construction: torchvision mobilenet_v2(weights=None), 10-class head, official
    state_dict. A pruned model is saved as a whole module instead (its shapes differ), so a .pt that
    unpickles to an nn.Module is returned as is."""
    from torchvision import models

    p = Path(path) if path else kit_file(STUDENT, kit)
    try:   # official checkpoints are plain state_dicts: load them with the safe (tensor-only) loader
        obj = torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        # a whole pruned module saved by prune_sweep.py; full unpickling is only for our own files
        if p.name == STUDENT:
            raise
        obj = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval()
    student = models.mobilenet_v2(weights=None)
    student.classifier[1] = nn.Linear(student.last_channel, NUM_CLASSES)
    student.load_state_dict(obj)
    return student.eval()


def build_teacher(path=None, kit=None) -> nn.Module:
    from torchvision import models

    p = Path(path) if path else kit_file(TEACHER, kit)
    teacher = models.resnet50(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, NUM_CLASSES)
    teacher.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    return teacher.eval()


# ----------------------------------------------------------------------------- predictors
def torch_predictor(model: nn.Module):
    model.eval()

    @torch.no_grad()
    def f(images: torch.Tensor) -> np.ndarray:
        return model(images).numpy()
    return f


def onnx_session(path, threads: int | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def onnx_predictor(path, threads: int | None = None):
    """session.run in place of model(images), as Section 5.4 says. The submission has a fixed batch
    of 1, so a loader batch is fed one image at a time."""
    sess = onnx_session(path, threads)
    name = sess.get_inputs()[0].name
    fixed_one = sess.get_inputs()[0].shape[0] == 1

    def f(images: torch.Tensor) -> np.ndarray:
        x = images.numpy().astype(np.float32)
        if fixed_one and len(x) != 1:
            return np.concatenate([sess.run(None, {name: x[i:i + 1]})[0] for i in range(len(x))])
        return sess.run(None, {name: x})[0]
    return f


def predictor(model_or_path, threads: int | None = None):
    if isinstance(model_or_path, nn.Module):
        return torch_predictor(model_or_path)
    p = Path(model_or_path)
    if p.suffix == ".onnx":
        return onnx_predictor(p, threads)
    return torch_predictor(build_student(p))


# ----------------------------------------------------------------------------- accuracy (Section 5.1)
def evaluate(pred, data_loader, num_classes: int = NUM_CLASSES) -> dict:
    """Top-1 accuracy and the confusion matrix / per-class recall, in one pass. Identical arithmetic
    to the sheet's evaluate() and per_class_accuracy(): argmax, recall = diag / row sum."""
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    correct = total = 0
    for images, labels in data_loader:
        p = torch.from_numpy(np.asarray(pred(images))).argmax(dim=1)
        correct += int((p == labels).sum())
        total += int(labels.size(0))
        for t, q in zip(labels, p):
            confusion[t, q] += 1
    recall = confusion.diag().float() / confusion.sum(dim=1).clamp(min=1)
    return {"accuracy": correct / max(total, 1), "n": total,
            "per_class_recall": [round(float(r), 6) for r in recall],
            "worst_class_accuracy": float(recall.min()), "worst_class": int(recall.argmin()),
            "confusion": confusion.tolist()}


# ----------------------------------------------------------------------------- environment
def machine_info() -> dict:
    import onnxruntime as ort
    import psutil

    info = {"platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "onnxruntime": ort.__version__,
            "physical_cores": psutil.cpu_count(logical=False), "logical_cpus": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)}
    try:
        import cpuinfo

        ci = cpuinfo.get_cpu_info()
        info["cpu"] = ci.get("brand_raw")
        info["avx_vnni"] = any("vnni" in f for f in ci.get("flags", []))
    except Exception:
        info["cpu"] = platform.processor()
    return info


def save_json(obj, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
