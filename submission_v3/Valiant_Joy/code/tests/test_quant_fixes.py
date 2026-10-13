"""Regression tests for the INT8 findings (see core/quantize.py docstring)."""
import math
from pathlib import Path

import pytest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.export import export_onnx
from core.quantize import (DEFAULT_QUANT_CFG, SqnrProbe, calibrate, concat_range_mismatch, output_tail,
                           quantize_onnx)


class BoxesAndScores(nn.Module):
    """A detector-like output: concat([large-valued 'pixel' channels, 0..1 sigmoid channels])."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.box = nn.Conv2d(32, 4, 1)
        self.cls = nn.Conv2d(32, 8, 1)

    def forward(self, x):
        f = self.trunk(x)
        return torch.cat([self.box(f) * 100.0 + 200.0, self.cls(f).sigmoid()], 1).flatten(2)


def _calib(shape=(3, 16, 16), n=64):
    g = torch.Generator().manual_seed(0)
    return DataLoader(TensorDataset(torch.randn(n, *shape, generator=g)), batch_size=16)


def test_output_tail_rescues_mixed_range_outputs(tmp_path):
    torch.manual_seed(0)
    m = BoxesAndScores().eval()
    fp32 = export_onnx(m, torch.randn(1, 3, 16, 16), tmp_path / "m.onnx", atol=1e-2)
    tail = output_tail(fp32)
    import onnx
    ops = {n.name: n.op_type for n in onnx.load(str(fp32)).graph.node}
    assert any(ops[t] == "Concat" for t in tail) and any(ops[t] == "Sigmoid" for t in tail)
    assert not any(ops[t] == "Conv" and "trunk" in t for t in tail)  # the heavy convs stay INT8
    calib = _calib()
    plain, rules = {}, {}
    quantize_onnx(fp32, tmp_path / "plain.onnx", calib, {"auto_keep_fp32": "none", "calib_batches": 4}, info=plain)
    quantize_onnx(fp32, tmp_path / "rules.onnx", calib, {"auto_keep_fp32": "rules", "calib_batches": 4}, info=rules)
    # the worst (score) channels are crushed by a shared scale without the rule
    assert plain["int8_sqnr_min_db"] < 10
    assert rules["int8_sqnr_min_db"] > plain["int8_sqnr_min_db"] + 10
    assert rules["fp32_tail_nodes"] >= 3


def test_concat_range_mismatch_flags_mixed_concat(tmp_path):
    torch.manual_seed(0)
    fp32 = export_onnx(BoxesAndScores().eval(), torch.randn(1, 3, 16, 16), tmp_path / "m.onnx", atol=1e-2)
    ranges = calibrate(fp32, _calib(), {**DEFAULT_QUANT_CFG, "calib_batches": 4}, tmp_path / "c.json")
    assert concat_range_mismatch(fp32, ranges, 32.0)


def test_entropy_is_not_minmax(tmp_path, dummy, trained):
    fp32 = export_onnx(trained, dummy.example_input(), tmp_path / "m.onnx")
    calib = dummy.load_data()["calib"]
    mm = calibrate(fp32, calib, {**DEFAULT_QUANT_CFG, "calibrate_method": "minmax"}, tmp_path / "mm.json")
    en = calibrate(fp32, calib, {**DEFAULT_QUANT_CFG, "calibrate_method": "entropy"}, tmp_path / "en.json")
    pc = calibrate(fp32, calib, {**DEFAULT_QUANT_CFG, "calibrate_method": "percentile", "calib_percentile": 99.0},
                   tmp_path / "pc.json")
    common = set(mm) & set(en) & set(pc)
    assert common
    assert any(abs(mm[k][1] - en[k][1]) > 1e-6 for k in common), "entropy degenerated to minmax"
    assert any(abs(mm[k][1] - pc[k][1]) > 1e-6 for k in common)


def test_quantize_reports_sqnr_for_static_and_dynamic(tmp_path, dummy, trained):
    fp32 = export_onnx(trained, dummy.example_input(), tmp_path / "m.onnx")
    calib = dummy.load_data()["calib"]
    for method in ("static", "dynamic"):
        info = {}
        quantize_onnx(fp32, tmp_path / f"{method}.onnx", calib, {"method": method, "calib_batches": 2}, info=info)
        assert math.isfinite(info["int8_sqnr_db"]) and info["int8_sqnr_db"] > 5
        assert "fp32_nodes" in info
    assert SqnrProbe(fp32, calib, 2).score(fp32)["sqnr_db"] >= 59  # identical model -> clipped max


def test_small_model_is_still_quantized(tmp_path):
    """A model whose weighted ops are all below tail_min_params must not end up 100% FP32."""
    from adapters.anomaly import ConvAE1d

    torch.manual_seed(0)
    m = ConvAE1d(1, (8, 16)).eval()
    x = torch.randn(1, 1, 144)
    fp32 = export_onnx(m, x, tmp_path / "ae.onnx")
    assert output_tail(fp32) == []              # the rule disables itself instead of excluding everything
    info = {}
    q = quantize_onnx(fp32, tmp_path / "ae_int8.onnx", _calib((1, 144), 32), {"calib_batches": 2}, info=info)
    assert info["int8_quantized_nodes"] > 0
    assert q.stat().st_size < fp32.stat().st_size


def test_module_blocks_not_empty_for_shallow_graphs(tmp_path, dummy, trained):
    from core.quantize import module_blocks

    fp32 = export_onnx(trained, dummy.example_input(), tmp_path / "m.onnx")
    blocks = module_blocks(fp32)
    assert blocks and all(v for v in blocks.values())


def test_search_kept_lists_only_blocks_actually_excluded(tmp_path):
    torch.manual_seed(0)
    m = BoxesAndScores().eval()
    fp32 = export_onnx(m, torch.randn(1, 3, 16, 16), tmp_path / "m.onnx", atol=1e-2)
    info = {}
    quantize_onnx(fp32, tmp_path / "s.onnx", _calib(), {"calib_batches": 2, "auto_keep_fp32": "search",
                                                        "search_target_db": 60, "search_max_blocks": 3}, info=info)
    import onnx
    g = onnx.load(str(tmp_path / "s.onnx")).graph
    quantized = {n.name for n in g.node if n.op_type in ("QuantizeLinear", "QLinearConv")}
    for block in info.get("search_kept") or []:
        assert not any(n.startswith(block) for n in quantized), f"{block} reported kept but still quantized"


def test_search_mode_runs_and_never_worse(tmp_path, dummy, trained):
    fp32 = export_onnx(trained, dummy.example_input(), tmp_path / "m.onnx")
    calib = dummy.load_data()["calib"]
    plain, search = {}, {}
    quantize_onnx(fp32, tmp_path / "p.onnx", calib, {"calib_batches": 2}, info=plain)
    quantize_onnx(fp32, tmp_path / "s.onnx", calib, {"calib_batches": 2, "auto_keep_fp32": "search",
                                                     "search_target_db": 60}, info=search)
    assert "search_final_db" in search and search["search_evals"] >= 1
    assert search["int8_sqnr_db"] >= plain["int8_sqnr_db"] - 0.5


def test_low_sqnr_is_warned_not_just_recorded(tmp_path, caplog, monkeypatch):
    """README documents a ~15 dB floor, but the number was only written to a wide CSV column.
    A broken INT8 variant must announce itself in the log and in `info`."""
    import core.quantize as q

    torch.manual_seed(0)
    fp32 = export_onnx(BoxesAndScores().eval(), torch.randn(1, 3, 16, 16), tmp_path / "m.onnx", atol=1e-2)
    calib, scores = _calib(), {}

    class FakeProbe:                                   # drive the threshold, not ORT's real numbers
        def __init__(self, *a, **k):
            pass

        def score(self, _out):
            return dict(scores)

    monkeypatch.setattr(q, "SqnrProbe", FakeProbe)
    base = {"calib_batches": 4}

    scores.update(sqnr_db=30.0, sqnr_min_db=28.0, sqnr_global_db=30.0)
    info = {}
    with caplog.at_level("WARNING"):
        q.quantize_onnx(fp32, tmp_path / "healthy.onnx", calib, base, info=info)
    assert "int8_sqnr_warning" not in info
    assert "looks broken" not in caplog.text

    # the mean is fine but one channel group is destroyed: the mean must not hide it
    scores.update(sqnr_db=24.0, sqnr_min_db=4.3)
    info, _ = {}, caplog.clear()
    with caplog.at_level("WARNING"):
        q.quantize_onnx(fp32, tmp_path / "broken.onnx", calib, base, info=info)
    assert "int8_sqnr_warning" in info and "4.3" in info["int8_sqnr_warning"]
    assert "looks broken" in caplog.text

    info = {}
    q.quantize_onnx(fp32, tmp_path / "off.onnx", calib, {**base, "sqnr_warn_db": 0}, info=info)
    assert "int8_sqnr_warning" not in info              # threshold is configurable, never hardcoded


def test_auto_calibration_picks_the_best_method_by_sqnr(tmp_path, monkeypatch, caplog):
    """calibrate_method: auto must measure, keep the WINNER'S FILE (not just its score), stop early on a
    healthy candidate, and never recurse. The pruned ResNet50 swung 1.5 -> 13.6 dB on this one knob."""
    import core.quantize as q

    torch.manual_seed(0)
    fp32 = export_onnx(BoxesAndScores().eval(), torch.randn(1, 3, 16, 16), tmp_path / "m.onnx", atol=1e-2)
    calib = _calib()
    # label each candidate's output file so we can prove the winner's bytes are the ones kept
    scores = {"minmax": 2.0, "percentile 99.99": 30.0, "percentile 99.9": 8.0, "entropy": 50.0}
    seen, bytes_of = [], {}
    real_q = q._quantize_static_cached

    def tagged(src, out, cache, cfg, exclude):
        real_q(src, out, cache, cfg, exclude)
        label = q._method_label({k: cfg[k] for k in ("calibrate_method", "calib_percentile") if k in cfg})
        seen.append(label)
        Path(out).with_suffix(".tag").write_text(label, encoding="utf-8")
        return out

    class FakeProbe:
        def __init__(self, *a, **k):
            pass

        def score(self, out):
            label = Path(out).with_suffix(".tag").read_text(encoding="utf-8")
            bytes_of[label] = Path(out).read_bytes()      # after stamping: the final candidate file
            db = scores[label]
            return {"sqnr_db": db + 1, "sqnr_min_db": db, "sqnr_global_db": db}

    monkeypatch.setattr(q, "_quantize_static_cached", tagged)
    monkeypatch.setattr(q, "SqnrProbe", FakeProbe)
    cfg = {"calibrate_method": "auto", "calib_batches": 4, "calib_auto_target_db": 20.0}

    info = {}
    out = q.quantize_onnx(fp32, tmp_path / "auto.onnx", calib, cfg, info=info)
    assert info["calib_method_chosen"] == "percentile 99.99"          # best of those actually tried
    assert out.read_bytes() == bytes_of["percentile 99.99"]        # the winner's bytes, not its score
    assert out.read_bytes() != bytes_of["minmax"]
    assert seen == ["minmax", "percentile 99.99"]                     # stopped early: entropy never ran
    # a flat list of strings, so core.sweep's ", ".join() can put it in one CSV cell
    assert info["calib_auto_tried"] == ["minmax=2.0dB", "percentile 99.99=30.0dB"]
    assert ", ".join(info["calib_auto_tried"])          # must not raise on non-string members
    assert "int8_sqnr_warning" not in info

    # nothing healthy: try everything, keep the least-bad, and warn exactly once
    scores.update({"minmax": 2.0, "percentile 99.99": 3.0, "percentile 99.9": 8.0, "entropy": 5.0})
    seen.clear()
    info = {}
    with caplog.at_level("WARNING"):
        q.quantize_onnx(fp32, tmp_path / "bad.onnx", calib, cfg, info=info)
    assert info["calib_method_chosen"] == "percentile 99.9"
    assert len(seen) == 4                                              # no early exit
    assert caplog.text.count("looks broken") == 1                      # losers stay quiet


def test_auto_calibration_cannot_recurse_or_lose_the_output_dir(tmp_path, monkeypatch, caplog):
    """A candidate without an explicit calibrate_method used to inherit "auto" and recurse forever;
    the output's parent directory also has to exist before the winner is copied into it."""
    import core.quantize as q

    torch.manual_seed(0)
    fp32 = export_onnx(BoxesAndScores().eval(), torch.randn(1, 3, 16, 16), tmp_path / "m.onnx", atol=1e-2)
    calib = _calib()

    class FakeProbe:
        def __init__(self, *a, **k):
            pass

        def score(self, _out):
            return {"sqnr_db": 40.0, "sqnr_min_db": 40.0, "sqnr_global_db": 40.0}

    monkeypatch.setattr(q, "SqnrProbe", FakeProbe)
    # a candidate that omits calibrate_method used to inherit "auto" and recurse; now it is rejected
    # loudly instead of being guessed at
    with pytest.raises(ValueError, match="explicit calibrate_method"):
        q.quantize_onnx(fp32, tmp_path / "x.onnx", calib,
                        {"calibrate_method": "auto", "calib_auto_methods": [{"calib_percentile": 99.9}]})

    # the winner is copied into a directory that quantize_onnx has not created yet
    deep = tmp_path / "does" / "not" / "exist" / "auto.onnx"
    info = {}
    out = q.quantize_onnx(fp32, deep, calib, {
        "calibrate_method": "auto", "calib_batches": 4,
        "calib_auto_methods": [{"calibrate_method": "percentile", "calib_percentile": 99.9}]}, info=info)
    assert out.exists() and info["calib_method_chosen"] == "percentile 99.9"

    # no probe to rank with -> take the first candidate and say so, rather than picking arbitrarily
    info = {}
    with caplog.at_level("WARNING"):
        q.quantize_onnx(fp32, tmp_path / "nop.onnx", calib,
                        {"calibrate_method": "auto", "calib_batches": 4, "sqnr_batches": 0}, info=info)
    assert info["calib_method_chosen"] == "minmax" and "needs the SQNR probe" in caplog.text


def test_auto_calibration_breaks_near_ties_on_the_mean(tmp_path, monkeypatch):
    """Ranking on the per-channel minimum alone is too brittle when candidates are within noise:
    measured on a detector, minmax scored 9.7 dB min / 18.9 dB mean and percentile 99.99 scored
    9.6 / 19.4 - a 0.1 dB gap in the minimum that picked the calibrator worth 1.5 mAP LESS."""
    import core.quantize as q

    torch.manual_seed(0)
    fp32 = export_onnx(BoxesAndScores().eval(), torch.randn(1, 3, 16, 16), tmp_path / "m.onnx", atol=1e-2)
    calib, bytes_of = _calib(), {}
    # (min, mean) per method, mirroring the detector measurement
    scores = {"minmax": (9.7, 18.9), "percentile 99.99": (9.6, 19.4),
              "percentile 99.9": (2.4, 13.2), "entropy": (0.3, 5.0)}
    real_q = q._quantize_static_cached

    def tagged(src, out, cache, cfg, exclude):
        real_q(src, out, cache, cfg, exclude)
        label = q._method_label({k: cfg[k] for k in ("calibrate_method", "calib_percentile") if k in cfg})
        Path(out).with_suffix(".tag").write_text(label, encoding="utf-8")
        return out

    class FakeProbe:
        def __init__(self, *a, **k):
            pass

        def score(self, out):
            label = Path(out).with_suffix(".tag").read_text(encoding="utf-8")
            bytes_of[label] = Path(out).read_bytes()
            lo, mean = scores[label]
            return {"sqnr_db": mean, "sqnr_min_db": lo, "sqnr_global_db": mean}

    monkeypatch.setattr(q, "_quantize_static_cached", tagged)
    monkeypatch.setattr(q, "SqnrProbe", FakeProbe)
    base = {"calibrate_method": "auto", "calib_batches": 4}

    # DEFAULT is pure minimum-ranking (calib_auto_tie_db 0): the 1.0 dB margin was calibrated on a
    # stale number, and re-running on unseen data showed the mean's pick scored 0.2988 vs minmax's
    # 0.3010. The mechanism stays available, opt-in, for a model with fresh evidence for it.
    info = {}
    q.quantize_onnx(fp32, tmp_path / "default.onnx", calib, base, info=info)
    assert info["calib_method_chosen"] == "minmax"               # 9.7 dB beats 9.6 dB, full stop

    info = {}
    out = q.quantize_onnx(fp32, tmp_path / "tie.onnx", calib, {**base, "calib_auto_tie_db": 1.0},
                          info=info)
    assert info["calib_method_chosen"] == "percentile 99.99"     # the mean breaks the 0.1 dB tie
    assert out.read_bytes() == bytes_of["percentile 99.99"]

    # a gap wider than the tie margin must still be decided by the minimum, not the mean: a high mean
    # cannot rescue a candidate with one destroyed channel group
    scores["minmax"] = (9.7, 40.0)
    scores["percentile 99.99"] = (2.0, 60.0)
    info = {}
    q.quantize_onnx(fp32, tmp_path / "gap.onnx", calib, {**base, "calib_auto_tie_db": 1.0}, info=info)
    assert info["calib_method_chosen"] == "minmax"

    # tie margin is configurable, and 0 restores pure minimum-ranking
    scores["minmax"] = (9.7, 18.9)
    scores["percentile 99.99"] = (9.6, 19.4)
    info = {}
    q.quantize_onnx(fp32, tmp_path / "strict.onnx", calib, {**base, "calib_auto_tie_db": 0}, info=info)
    assert info["calib_method_chosen"] == "minmax"


def test_ram_preflight_warns_before_a_model_that_will_not_fit(tmp_path, caplog, monkeypatch):
    """Quantization peaks at ~28x the weight bytes (measured: 97 MB -> 2.7 GB, 338 MB -> 8.9 GB, and
    a 1.2 GB model took a 16 GB machine to zero free). None of the obvious knobs reduce it, so the
    useful behaviour is to predict it and name what to turn down instead of dying half way."""
    import psutil

    import core.quantize as q

    torch.manual_seed(0)
    fp32 = export_onnx(BoxesAndScores().eval(), torch.randn(1, 3, 16, 16), tmp_path / "m.onnx", atol=1e-2)
    size = fp32.stat().st_size

    class Mem:
        def __init__(self, avail):
            self.available = avail

    # plenty of room: quiet
    monkeypatch.setattr(psutil, "virtual_memory", lambda: Mem(size * 1000))
    with caplog.at_level("WARNING"):
        r = q.ram_preflight(fp32, {"ram_warn_factor": 28.0})
    assert r["model_bytes"] == size and r["est_peak_bytes"] == int(size * 28)
    assert "may need" not in caplog.text

    # not enough room: warn, and say what to do about it
    caplog.clear()
    monkeypatch.setattr(psutil, "virtual_memory", lambda: Mem(size * 5))
    with caplog.at_level("WARNING"):
        q.ram_preflight(fp32, {"ram_warn_factor": 28.0})
    assert "may need" in caplog.text
    assert "batch_size" in caplog.text and "external_data_format" in caplog.text

    # factor 0 disables it entirely
    caplog.clear()
    with caplog.at_level("WARNING"):
        q.ram_preflight(fp32, {"ram_warn_factor": 0})
    assert caplog.text == ""
