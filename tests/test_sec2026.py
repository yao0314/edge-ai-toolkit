"""Plumbing tests for sec2026/ (the EuroSAT task scripts).

They build a throwaway Starter-Kit-shaped folder in tmp_path: a few synthetic 64x64 images in the
torchvision EuroSAT layout, a split file, and a RANDOMLY initialised MobileNetV2 state_dict. None of
it stands in for the official kit; it only proves every script runs end to end and that the graph
facts (params, MACs) agree between PyTorch, FP32 ONNX and INT8 ONNX.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

SEC = Path(__file__).resolve().parents[1] / "sec2026"
sys.path.insert(0, str(SEC))

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture",
           "PermanentCrop", "Residential", "River", "SeaLake"]


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    from PIL import Image
    from torchvision import models

    k = tmp_path_factory.mktemp("kit")
    rng = np.random.default_rng(0)
    base = k / "data" / "eurosat" / "2750"
    for c in CLASSES:
        (base / c).mkdir(parents=True)
        for i in range(3):
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(base / c / f"{c}_{i}.jpg")
    n = len(CLASSES) * 3
    (k / "val_split.json").write_text(json.dumps({"train": list(range(0, n, 2)), "val": list(range(1, n, 2))}))
    torch.manual_seed(0)
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, 10)
    torch.save(m.state_dict(), k / "student_mobilenetv2_distilled.pt")
    return k


def test_data_and_model_follow_the_sheet(kit):
    import common

    train, val = common.subsets(kit)
    x, y = val[0]
    assert tuple(x.shape) == (3, 224, 224) and 0 <= y < 10
    assert len(train) + len(val) == 30
    student = common.build_student(kit=kit)
    r = common.evaluate(common.predictor(student), common.loader(val, 8))
    assert r["n"] == len(val) and len(r["per_class_recall"]) == 10
    assert r["worst_class_accuracy"] == min(r["per_class_recall"])


def test_missing_kit_file_is_an_error_not_a_stand_in(tmp_path):
    import common

    with pytest.raises(FileNotFoundError, match="official Starter Kit file missing"):
        common.build_student(kit=tmp_path)


def test_export_quantize_and_graph_facts(kit, tmp_path):
    import common
    import export_quant as eq

    eq.main(["--kit", str(kit), "--out", str(tmp_path), "--calib", "6", "--check", "4"])
    rec = json.loads((tmp_path / "quant.json").read_text())
    fp32, int8 = tmp_path / "student_model.onnx", tmp_path / "student_model_int8.onnx"
    assert fp32.exists() and int8.exists()
    import onnx

    g = onnx.load(str(int8)).graph
    assert any(n.op_type == "QuantizeLinear" for n in g.node), "not QDQ"
    assert not any(n.op_type in ("ConvInteger", "DynamicQuantizeLinear") for n in g.node), "dynamic ops"
    assert [d.dim_value for d in g.input[0].type.tensor_type.shape.dim] == [1, 3, 224, 224]
    assert onnx.load(str(fp32)).opset_import[0].version == 13
    # same network, same parameter count whatever the precision; BN folding makes it a bit smaller
    # than the PyTorch count (BN weight+bias become one conv bias)
    assert rec["fp32"]["params"] == rec["int8"]["params"]
    torch_params = sum(p.numel() for p in common.build_student(kit=kit).parameters())
    assert 0.98 * torch_params < rec["fp32"]["params"] <= torch_params
    assert rec["fp32"]["macs"] == rec["int8"]["macs"] and 2.5e8 < rec["fp32"]["macs"] < 3.5e8
    assert rec["int8"]["size_mb"] < rec["fp32"]["size_mb"] / 3


def test_measure_and_metrics_json(kit, tmp_path):
    import export_quant as eq
    import make_metrics
    import measure

    onnx_path = eq.export_fp32(__import__("common").build_student(kit=kit), tmp_path / "m.onnx")
    for name in ("base", "final"):
        measure.main(["--model", str(onnx_path), "--kit", str(kit), "--name", name, "--out-dir", str(tmp_path),
                      "--repeats", "1", "--runs", "3", "--energy-seconds", "0"])
    b = json.loads((tmp_path / "base.json").read_text())
    assert b["latency_ms"] > 0 and b["peak_ram_mb"] > 0 and b["accuracy"]["n"] == 15
    make_metrics.main(["--baseline", str(tmp_path / "base.json"), "--final", str(tmp_path / "final.json"),
                       "--team", "t", "--technique", "x", "--kit", str(kit), "--out", str(tmp_path / "metrics.json")])
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert set(m) == {"team_name", "baseline", "final", "worst_class_accuracy", "codecarbon_mode",
                      "techniques_applied", "ai_hub_profile_summary", "measurement_machine"}
    assert set(m["final"]) == set(make_metrics.KEYS)


def test_prune_sweep_shrinks_the_model(kit, tmp_path):
    import prune_sweep

    prune_sweep.main(["--kit", str(kit), "--out", str(tmp_path), "--ratios", "0", "0.2", "--ft-steps", "1",
                      "--batch-size", "4", "--bn-batches", "2", "--calib", "4", "--dry"])
    import csv

    rows = {float(r["ratio"]): r for r in csv.DictReader(open(tmp_path / "sweep.csv", encoding="utf-8"))}
    assert int(rows[0.2]["params"]) < 0.8 * int(rows[0.0]["params"])
    assert (tmp_path / "p20" / "student_model_int8.onnx").exists() and (tmp_path / "tradeoff.png").exists()
