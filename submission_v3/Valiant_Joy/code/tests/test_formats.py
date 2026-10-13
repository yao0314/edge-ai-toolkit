"""Scenario A/B/C compatibility: models handed to us in any format go through measure and sweep."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from core.measure import measure
from core.model_io import capabilities, load_model, model_format
from core.sweep import run_sweep

FIX = Path(__file__).parent / "fixtures"
FAST = {"threads": [1], "warmup": 2, "runs": 10, "min_time_s": 0.1}
SWEEP = {"task": "dummy", "measure": FAST, "quantize": {"calib_batches": 2}, "prune": {"round_to": None},
         "sweep": {"prune_ratios": [0.0, 0.5], "precisions": ["fp32", "int8"], "finetune_epochs": 1}}
NHWC = torch.randn(1, 16, 16, 3)

# ai-edge-litert is an OPTIONAL extra (requirements-optional.txt), so skip only the tests that
# actually need it. A module-level importorskip silently removed all 15 tests from this file on any
# machine without it - including every Linux box, which is the likely shape of the competition
# sandbox. Six of those tests have nothing to do with TFLite: they are the torch / TorchScript /
# .pt2 / ONNX ingestion tests, i.e. exactly the "model handed to us in an unknown format" scenarios
# this file exists to cover. They were passing on Windows only, and their absence elsewhere did not
# even appear in the pytest summary.
needs_litert = pytest.mark.skipif(
    importlib.util.find_spec("ai_edge_litert") is None,
    reason="TFLite tests need ai-edge-litert (requirements-optional.txt)")


def _tflite(name):
    """A parametrize entry that is skipped when LiteRT is unavailable."""
    return pytest.param(name, marks=needs_litert)


# ---------------------------------------------------------------- loading (scenario A)
def test_load_full_module_checkpoint_statedict_torchscript(tmp_path, trained, dummy):
    x = dummy.example_input()
    ref = trained(x)
    torch.save(trained, tmp_path / "full.pt")
    torch.save({"model": trained, "epoch": 3}, tmp_path / "ckpt.pt")
    torch.save(trained.state_dict(), tmp_path / "sd.pt")
    torch.jit.trace(trained, x).save(str(tmp_path / "ts.pt"))

    for f in ("full.pt", "ckpt.pt"):
        m = load_model(tmp_path / f)
        assert model_format(m) == "torch" and torch.allclose(m(x), ref, atol=1e-5)
    m = load_model(tmp_path / "sd.pt", arch="dummy_adapter:TinyNet")
    assert torch.allclose(m(x), ref, atol=1e-5)
    with pytest.raises(ValueError):
        load_model(tmp_path / "sd.pt")  # state_dict without arch
    ts = load_model(tmp_path / "ts.pt")
    assert model_format(ts) == "torchscript" and not capabilities(ts)["prune"]
    assert torch.allclose(ts(x), ref, atol=1e-5)
    assert load_model(FIX / "tiny_f32.tflite").endswith(".tflite")


# ---------------------------------------------------------------- measuring (scenario B)
@pytest.mark.parametrize("fname", [_tflite("tiny_f32.tflite"), _tflite("tiny_int8.tflite"),
                                   "tiny_nhwc_b1.onnx"])
def test_measure_without_adapter(fname):
    row = measure(None, FIX / fname, None, fname, FAST, example_input=NHWC)
    assert math.isnan(row["accuracy"]) and row["latency_p50_ms"] > 0 and row["size_bytes"] > 0
    assert row["runtime"] == ("tflite" if fname.endswith("tflite") else "onnx")


@pytest.mark.parametrize("fname", [_tflite("tiny_f32.tflite"), _tflite("tiny_int8.tflite"),
                                   "tiny_nhwc_b1.onnx"])
def test_foreign_models_evaluate_with_nchw_adapter(dummy, fname):
    # the adapter yields NCHW batches of 64; these models are NHWC with a fixed batch of 1
    acc = dummy.evaluate(str(FIX / fname))["score"]
    assert acc > 0.6, acc


# ---------------------------------------------------------------- sweeping foreign models
def test_sweep_onnx_only_skips_pruning(tmp_path, dummy):
    df = run_sweep(SWEEP, dummy, str(FIX / "tiny_nhwc_b1.onnx"), tmp_path)
    ok = df[df["error"].isna() & df["skipped"].isna()]
    assert set(ok["variant"]) == {"p00_fp32", "p00_int8"}
    assert df[df["prune_ratio"] == 0.5]["skipped"].str.contains("PyTorch").all()
    assert (ok["accuracy"] > 0.6).all()
    assert ok["score"].notna().all()


def _has(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


@needs_litert
def test_sweep_tflite_measures_as_given(tmp_path, dummy):
    df = run_sweep(SWEEP, dummy, str(FIX / "tiny_f32.tflite"), tmp_path)
    fp = df[df["variant"] == "p00_fp32"].iloc[0]
    assert fp["runtime"] == "tflite" and fp["accuracy"] > 0.6
    int8 = df[df["variant"] == "p00_int8"].iloc[0]
    if _has("tf2onnx") and _has("tensorflow"):  # float TFLite -> ONNX -> ORT INT8
        assert int8["runtime"] == "onnx" and int8["accuracy"] > 0.6, int8.get("error")
    else:
        assert "tf2onnx" in int8["skipped"]


# ---------------------------------------------------------------- format sniffing / RSS baseline
def test_format_detected_from_content(tmp_path):
    import shutil

    from core.model_io import sniff_format

    shutil.copy(FIX / "tiny_f32.tflite", tmp_path / "model.bin")
    shutil.copy(FIX / "tiny_f32.tflite", tmp_path / "misnamed.onnx")
    shutil.copy(FIX / "tiny_nhwc_b1.onnx", tmp_path / "model.dat")
    assert sniff_format(tmp_path / "model.bin") == "tflite"
    assert model_format(str(tmp_path / "misnamed.onnx")) == "tflite"  # content wins over extension
    assert sniff_format(tmp_path / "model.dat") == "onnx"


@needs_litert
def test_a_misnamed_model_is_measured_by_its_real_format(tmp_path):
    """Split out of the sniffing test above: deciding the format is pure byte inspection and must be
    covered everywhere, but actually MEASURING a TFLite file needs the LiteRT runtime. Keeping them
    in one test meant the sniffing assertions were skipped on every machine without LiteRT."""
    import shutil

    shutil.copy(FIX / "tiny_f32.tflite", tmp_path / "misnamed.onnx")
    row = measure(None, tmp_path / "misnamed.onnx", None, "x", FAST, example_input=NHWC)
    assert row["runtime"] == "tflite", "the .onnx extension must not decide the runtime"


@needs_litert
@pytest.mark.parametrize("fname", ["tiny_f32.tflite", "tiny_int8.tflite"])
def test_tflite_rss_baseline_is_an_empty_tflite_model(fname):
    row = measure(None, FIX / fname, None, fname, FAST, example_input=NHWC)
    assert row["rss_baseline"] == ("empty_model" if _has("tensorflow") else "runtime_only")
    assert row["peak_rss_mb"] >= 0
    onnx_row = measure(None, FIX / "tiny_nhwc_b1.onnx", None, "o", FAST, example_input=NHWC)
    assert onnx_row["rss_baseline"] == "empty_session"


def test_target_runtime_tflite_is_reserved(tmp_path, dummy):
    with pytest.raises(NotImplementedError):
        run_sweep({**SWEEP, "target_runtime": "tflite"}, dummy, str(FIX / "tiny_nhwc_b1.onnx"), tmp_path)


def test_sweep_torchscript(tmp_path, dummy, trained):
    ts = torch.jit.trace(trained, dummy.example_input())
    df = run_sweep(SWEEP, dummy, ts, tmp_path)
    ok = df[df["error"].isna() & df["skipped"].isna()]
    assert set(ok["variant"]) == {"p00_fp32", "p00_int8"}, df[["variant", "error", "skipped"]]


def test_torch_export_pt2(tmp_path, dummy, trained):
    x = dummy.example_input()
    torch.export.save(torch.export.export(trained.eval(), (x,)), str(tmp_path / "m.pt2"))
    m = load_model(tmp_path / "m.pt2")
    assert model_format(m) == "torch_export" and torch.allclose(m(x), trained(x), atol=1e-5)
    df = run_sweep({**SWEEP, "sweep": {**SWEEP["sweep"], "prune_ratios": [0.0]}}, dummy, m, tmp_path)
    assert df["error"].isna().all(), df["error"].tolist()
