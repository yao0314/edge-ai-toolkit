"""Enforce the hard rule: core/ contains no task-specific code."""
import re

from conftest import ROOT

FORBIDDEN = [
    "cifar", "imagenet", "mobilenet", "resnet", "yolo", "coco", "ecg", "mimii", "ucr",
    "classification", "anomaly", "detection", "top1", "auc", "map50", "bbox", "autoencoder",
    "adapters.",
]


def test_core_has_no_task_specific_words():
    offenders = []
    for py in (ROOT / "core").glob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            low = line.lower()
            for w in FORBIDDEN:
                if re.search(rf"\b{re.escape(w)}", low):
                    offenders.append(f"{py.name}:{i}: {w!r}: {line.strip()}")
    assert not offenders, "task-specific code found in core/:\n" + "\n".join(offenders)
