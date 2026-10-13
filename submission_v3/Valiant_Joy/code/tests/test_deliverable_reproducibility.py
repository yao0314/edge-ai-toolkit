"""The deliverable must be usable WITHOUT this repo.

A judge is handed best.onnx and metadata.json. If the preprocessing is not written down they cannot
reproduce the accuracy we report, and an unreproducible number is a claim rather than a result.
Measured gap that motivated this: the shipped deliverables documented input shape but never colour
order, value range, normalisation constants or resize mode.
"""
import json
import sys
from pathlib import Path

import pytest

from core.adapter import TaskAdapter

# scripts/compress.py starts with `from _common import ...`, which needs scripts/ on the path.
# conftest puts ROOT and ROOT/tests there but not ROOT/scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_base_adapter_declares_nothing_by_default(dummy):
    """The contract exists on the base class so every adapter has it; the default is honest emptiness
    rather than a guessed spec."""
    assert TaskAdapter.preprocessing(dummy) == {}


@pytest.mark.parametrize("module,cls", [
    ("adapters.classification", "ClassificationAdapter"),
    ("adapters.detection", "DetectionAdapter"),
    ("adapters.anomaly", "AnomalyAdapter"),
])
def test_shipped_adapters_override_preprocessing(module, cls):
    """Each shipped adapter must declare its own; inheriting the empty default silently ships an
    undocumented artifact."""
    import importlib
    m = importlib.import_module(module)
    k = getattr(m, cls)
    assert k.preprocessing is not TaskAdapter.preprocessing, f"{cls} does not declare preprocessing"


def test_tracking_inherits_the_detection_spec():
    """tracking subclasses detection and uses the same letterbox pipeline, so inheriting is correct
    here - but assert it so a future divergence is caught."""
    from adapters.detection import DetectionAdapter
    from adapters.tracking import TrackingAdapter
    assert TrackingAdapter.preprocessing is DetectionAdapter.preprocessing


def test_packaging_survives_a_broken_preprocessing(dummy):
    """An unusable deliverable is bad; no deliverable at all is worse. A raising preprocessing()
    must degrade to an empty spec, not abort packaging."""
    import scripts.compress as C

    class Boom:
        def preprocessing(self):
            raise RuntimeError("adapter bug")

    assert C._preprocessing(Boom()) == {}
    assert C._preprocessing(dummy) == {}          # dummy inherits the empty default
    assert json.dumps(C._preprocessing(dummy)) == "{}"


def test_input_signature_is_reported_and_never_raises(dummy):
    import scripts.compress as C

    sig = C._input_signature(dummy)
    assert sig["name"] == "input"
    assert sig["shape"][0] == 1, "the signature is reported for a single sample"

    class Boom:
        def example_input(self):
            raise RuntimeError("no example")

        def input_names(self):
            return ["input"]

    assert C._input_signature(Boom()) == {}
