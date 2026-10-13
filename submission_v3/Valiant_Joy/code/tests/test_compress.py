
import numpy as np
import pandas as pd

from core.distill import distill
from core.export import export_onnx
from core.pareto import pareto_mask
from core.prune import prune_model
from core.quantize import match_nodes, quantize_onnx
from core.score import score_table
from core.sweep import run_sweep

FAST = {"threads": [1], "warmup": 2, "runs": 10, "min_time_s": 0.1}


def test_prune_shrinks_and_keeps_head(dummy, trained):
    x = dummy.example_input()
    p = prune_model(trained, x, 0.5, dummy.ignored_layers(trained), round_to=None)
    n0 = sum(q.numel() for q in trained.parameters())
    n1 = sum(q.numel() for q in p.parameters())
    assert n1 < 0.6 * n0
    assert p.head.out_features == trained.head.out_features
    assert p(x).shape == trained(x).shape
    assert sum(q.numel() for q in trained.parameters()) == n0  # original untouched


def test_static_int8_quantization(tmp_path, dummy, trained):
    fp32 = export_onnx(trained, dummy.example_input(), tmp_path / "fp32.onnx")
    q = quantize_onnx(fp32, tmp_path / "int8.onnx", dummy.load_data()["calib"], {"calib_batches": 2})
    assert q.stat().st_size < fp32.stat().st_size
    a32 = dummy.evaluate(str(fp32))["score"]
    a8 = dummy.evaluate(str(q))["score"]
    assert a8 > a32 - 0.15


def test_keep_fp32_excludes_nodes(tmp_path, dummy, trained):
    import onnx

    fp32 = export_onnx(trained, dummy.example_input(), tmp_path / "fp32.onnx")
    assert match_nodes(fp32, ["head"])
    q = quantize_onnx(fp32, tmp_path / "int8.onnx", dummy.load_data()["calib"],
                      {"calib_batches": 2, "keep_fp32": ["head"]})
    g = onnx.load(str(q)).graph
    # the Gemm of the head must consume a float tensor, i.e. no DequantizeLinear right before its weight
    gemm = [n for n in g.node if n.op_type == "Gemm"]
    assert gemm and all("DequantizeLinear" not in i for n in gemm for i in n.input[1:2])


def test_dynamic_quantization(tmp_path, dummy, trained):
    fp32 = export_onnx(trained, dummy.example_input(), tmp_path / "fp32.onnx")
    q = quantize_onnx(fp32, tmp_path / "dyn.onnx", None, {"method": "dynamic"})
    assert 0 <= dummy.evaluate(str(q))["score"] <= 1


def test_pareto_mask():
    x = np.array([1, 2, 3, 2.5])
    y = np.array([0.5, 0.7, 0.9, 0.6])
    assert pareto_mask(x, y).tolist() == [True, True, True, False]


def test_score_drops_missing_energy():
    df = pd.DataFrame({"variant": ["a", "b"], "accuracy": [0.9, 0.8], "size_bytes": [100, 50],
                       "peak_rss_mb": [10, 5], "cpu_ms_per_inf": [2, 1], "latency_p50_ms": [2, 1],
                       "energy_j": [np.nan, np.nan], "error": [None, None]})
    s = score_table(df)
    assert s["score"].notna().all()
    assert "energy" in s["score_note"].iloc[0]
    # weights: accuracy favours a, compression+resource favour b -> b wins with equal weights
    assert s.loc[1, "score"] > s.loc[0, "score"]


def test_distill_runs(dummy, trained):
    student = dummy.build_model()
    out = distill(trained, student, dummy, epochs=1, lr=0.01)
    assert out(dummy.example_input()).shape == (1, 4)


def test_sweep_continues_after_failure(tmp_path, dummy, trained, monkeypatch):
    cfg = {"task": "dummy", "measure": FAST, "quantize": {"calib_batches": 2},
           "prune": {"round_to": None},
           "sweep": {"prune_ratios": [0.0, 0.5], "finetune_epochs": 1,
                     "precisions": ["fp32", "int8", {"name": "bogus", "method": "nope"}]}}
    df = run_sweep(cfg, dummy, trained, tmp_path)
    assert len(df) == 6
    assert df["error"].notna().sum() == 2  # the two "bogus" rows
    ok = df[df["error"].isna()]
    assert ok["accuracy"].notna().all() and ok["score"].notna().all()
    assert (tmp_path / "sweep.csv").exists() and (tmp_path / "pareto_size_bytes.png").exists()
    # resume: nothing new to do for successful rows
    df2 = run_sweep(cfg, dummy, trained, tmp_path, resume=True)
    assert (df2["error"].isna()).sum() == 4
