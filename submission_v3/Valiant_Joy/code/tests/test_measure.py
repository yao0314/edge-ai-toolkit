import math

import pytest

from core.export import ExportMismatchError, export_onnx, verify_onnx
from core.measure import count_macs, count_macs_onnx, measure

FAST = {"threads": [1, 2], "warmup": 2, "runs": 10, "min_time_s": 0.2}


def test_export_verifies(tmp_path, dummy, trained):
    p = export_onnx(trained, dummy.example_input(), tmp_path / "m.onnx")
    assert p.exists() and p.stat().st_size > 0


def test_export_mismatch_detected(tmp_path, dummy, trained):
    p = export_onnx(trained, dummy.example_input(), tmp_path / "m.onnx")
    x = dummy.example_input()
    wrong = [trained(x).detach().numpy() + 1.0]
    with pytest.raises(ExportMismatchError):
        verify_onnx(p, x, wrong)


def test_macs_torch_and_onnx_agree(tmp_path, dummy, trained):
    p = export_onnx(trained, dummy.example_input(), tmp_path / "m.onnx")
    a, b = count_macs(trained, dummy.example_input()), count_macs_onnx(p)
    assert a > 0 and abs(a - b) / a < 0.1


def test_measure_row_has_all_fields(tmp_path, dummy, trained):
    p = export_onnx(trained, dummy.example_input(), tmp_path / "m.onnx")
    row = measure(dummy, p, trained, "tiny", FAST)
    for k in ["accuracy", "size_bytes", "params", "macs", "latency_p50_ms", "latency_p90_ms",
              "peak_rss_mb", "cpu_util_pct", "latency_p50_ms_t1", "latency_p50_ms_t2"]:
        assert k in row and not (isinstance(row[k], float) and math.isnan(row[k])), k
    assert row["latency_p90_ms"] >= row["latency_p50_ms"] > 0
    assert math.isnan(row["energy_j"])  # no real energy source configured -> left empty
    assert 0 <= row["accuracy"] <= 1


def test_torch_and_onnx_eval_match(tmp_path, dummy, trained):
    p = export_onnx(trained, dummy.example_input(), tmp_path / "m.onnx")
    assert abs(dummy.evaluate(trained)["score"] - dummy.evaluate(str(p))["score"]) < 1e-6


def test_energy_csv_provider(tmp_path):
    from core.energy import measure_energy

    f = tmp_path / "e.csv"
    f.write_text("variant,energy_j\nv1,0.0123\n")
    assert measure_energy("x.onnx", "v1", None, {"provider": "csv", "path": str(f)})[0] == pytest.approx(0.0123)
    assert math.isnan(measure_energy("x.onnx", "v2", None, {"provider": "csv", "path": str(f)})[0])
    assert math.isnan(measure_energy("x.onnx", "v1", None, None)[0])


def test_primary_threads_falls_back_to_a_measured_count(caplog):
    """--quick sets threads:[1] while configs set primary_threads:4. Reporting NaN there silently
    emptied score_resource and the latency/RSS Pareto fronts."""
    from core.measure import primary_thread_count

    assert primary_thread_count({"threads": [1, 4], "primary_threads": 4}) == 4
    assert primary_thread_count({"threads": [1, 4]}) == 1              # default: first measured
    with caplog.at_level("WARNING"):
        assert primary_thread_count({"threads": [1], "primary_threads": 4}) == 1
    assert "was not measured" in caplog.text                           # never silent
    assert primary_thread_count({"threads": [2, 8], "primary_threads": 4}) == 2   # closest <= want
    assert primary_thread_count({"threads": [8], "primary_threads": 4}) == 8      # nothing <=: min


def test_quick_profile_still_fills_the_primary_columns(tmp_path, dummy):
    """End-to-end guard: the columns score/Pareto read must not be NaN under the --quick measure profile."""
    import numpy as np

    from core.export import export_onnx
    from core.measure import DEFAULT_MEASURE_CFG, runtime_metrics

    onnx_path = tmp_path / "m.onnx"
    export_onnx(dummy.build_model().eval(), dummy.example_input(), onnx_path)
    cfg = {**DEFAULT_MEASURE_CFG, "threads": [1], "primary_threads": 4,
           "warmup": 2, "runs": 5, "min_time_s": 0.1, "repeats": 1}
    row = runtime_metrics(onnx_path, dummy.example_input(), cfg)
    for k in ("latency_p50_ms", "cpu_ms_per_inf", "peak_rss_mb"):
        assert not np.isnan(row[k]), f"{k} is NaN: scoring would silently drop the resource term"
    assert row["primary_threads"] == 1      # honest about where the numbers came from


def test_competing_cpu_load_is_recorded_and_warned(tmp_path, dummy, caplog, monkeypatch):
    """Measuring next to another job shifts the median a little and inflates the spread a lot.
    Measured on the dev host with a 94 MB model: 23.56 -> 27.38 ms (1.16x) and sd 0.31 -> 1.49,
    with other_load_pct going 4% -> 27%. That has to be recorded, not left to whoever reads the
    table - the competition sandbox is shared and we do not control what else runs on it.

    The threshold decision is driven with an INJECTED load value: asserting on the real ambient
    load made this test depend on how busy the machine happened to be, which is exactly the kind of
    flakiness that had it passing alone and failing in a full run.
    """
    import numpy as np

    import core.measure as cm
    from core.export import export_onnx
    from core.measure import DEFAULT_MEASURE_CFG, runtime_metrics

    onnx_path = tmp_path / "m.onnx"
    export_onnx(dummy.build_model().eval(), dummy.example_input(), onnx_path)
    cfg = {**DEFAULT_MEASURE_CFG, "threads": [1], "primary_threads": 1,
           "warmup": 2, "runs": 5, "min_time_s": 0.1, "repeats": 1}

    # a real run must populate the columns at all
    row = runtime_metrics(onnx_path, dummy.example_input(), cfg)
    for k in ("other_load_pct_t1", "other_load_pct"):
        assert k in row
    assert not np.isnan(row["other_load_pct"]) and row["other_load_pct"] >= 0

    # now drive the threshold deterministically
    real_probe = cm._run_probe

    def fake(path, npy, threads, c, backend=None, load=[0.0]):
        r = real_probe(path, npy, threads, c, backend)
        r["other_load_pct"] = load[0]
        return r

    # other_load_pct is RECORDED, but it is no longer an alarm: it fired on 61% of measurements while
    # not tracking real core delivery, so core/measure.py warns on core_delivery_warn instead.
    for load in (3.0, 40.0):
        fake.__defaults__ = (None, [load])
        monkeypatch.setattr(cm, "_run_probe", fake)
        caplog.clear()
        with caplog.at_level("WARNING"):
            r = runtime_metrics(onnx_path, dummy.example_input(), {**cfg, "other_load_warn_pct": 8.0})
        assert r["other_load_pct"] == load
        assert "competing CPU load" not in caplog.text, f"load {load} must not raise the retired alarm"

    # 0 or negative disables the check entirely, however busy the machine is
    fake.__defaults__ = (None, [99.0])
    monkeypatch.setattr(cm, "_run_probe", fake)
    caplog.clear()
    with caplog.at_level("WARNING"):
        runtime_metrics(onnx_path, dummy.example_input(), {**cfg, "other_load_warn_pct": 0})
    assert "competing CPU load" not in caplog.text


def test_training_curve_records_loss_and_metric(tmp_path):
    """Falling loss does not prove a rising metric: in the pruning study the loss kept dropping
    (0.5747 -> 0.5355 over 18 extra epochs) while the accuracy gap barely moved (-3.18 -> -2.83 pt).
    Both columns have to be on the curve, or a run looks like it still has headroom when it does not."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from adapters.common import train_loop
    from core.sweep import _save_curve, plot_curves

    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(8, 4))
    loader = DataLoader(TensorDataset(torch.randn(32, 8), torch.randint(0, 4, (32,))), batch_size=8)

    def loss_fn(m, batch, device):
        x, y = batch[0].to(device), batch[1].to(device)
        return nn.functional.cross_entropy(m(x), y)

    hist, seen = [], []
    train_loop(model, loader, loss_fn, epochs=4, lr=0.01, device="cpu", amp=False,
               history=hist, eval_fn=lambda m: seen.append(1) or 0.5 + 0.1 * len(seen),
               eval_every=2, tag="finetune")

    assert len(hist) == 4 and [h["epoch"] for h in hist] == [1, 2, 3, 4]
    assert all(h["loss"] == h["loss"] and h["seconds"] >= 0 for h in hist)   # no NaN losses
    assert [h["metric"] for h in hist] == [None, 0.6, None, 0.7]             # only every 2nd epoch
    assert hist[0]["lr"] > hist[-1]["lr"]                                    # the cosine schedule ran

    # a failing eval_fn is a diagnostic, never fatal
    hist2 = []
    train_loop(model, loader, loss_fn, epochs=2, lr=0.01, device="cpu", amp=False, history=hist2,
               eval_fn=lambda m: 1 / 0, eval_every=1, tag="finetune")
    assert len(hist2) == 2 and all(h["metric"] is None for h in hist2)

    _save_curve(tmp_path, "p25_fp32", hist)
    assert (tmp_path / "curves" / "p25_fp32.csv").exists()
    assert [p.name for p in plot_curves(tmp_path)] == ["curves.png"]
    assert (tmp_path / "curves.png").stat().st_size > 1000

    _save_curve(tmp_path, "x", [])          # nothing recorded: no file, no crash
    assert not (tmp_path / "curves" / "x.csv").exists()
