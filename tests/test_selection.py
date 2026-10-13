"""The selector must never propose a variant that failed the accuracy constraint, and never a single answer."""
import numpy as np
import pandas as pd
import pytest

from core.score import candidates, eligibility, rank, score_table

BASE = dict(error=None, skipped=None, metric_name="top1", macs=1e6, energy_j=np.nan)


def _df():
    # a small/fast but 20-points-worse variant would win on equal weights without a hard constraint
    return pd.DataFrame([
        {**BASE, "variant": "p00_fp32", "prune_ratio": 0.0, "precision": "fp32", "accuracy": 0.9165,
         "size_bytes": 4_014_419, "latency_p50_ms": 1.34, "peak_rss_mb": 9.5, "cpu_ms_per_inf": 1.38},
        {**BASE, "variant": "p00_int8_dyn_mm", "prune_ratio": 0.0, "precision": "int8_dyn_mm", "accuracy": 0.9160,
         "size_bytes": 2_701_133, "latency_p50_ms": 1.36, "peak_rss_mb": 9.7, "cpu_ms_per_inf": 1.34},
        {**BASE, "variant": "p50_int8_dyn_mm", "prune_ratio": 0.5, "precision": "int8_dyn_mm", "accuracy": 0.7169,
         "size_bytes": 962_588, "latency_p50_ms": 0.82, "peak_rss_mb": 7.4, "cpu_ms_per_inf": 0.84},
    ])


def test_big_accuracy_loss_is_excluded_from_ranking():
    s = score_table(_df(), {"max_accuracy_drop": 0.02})
    row = s.set_index("variant")
    assert row.loc["p50_int8_dyn_mm", "score"] > row.loc["p00_int8_dyn_mm", "score"]  # equal weights prefer it
    assert not row.loc["p50_int8_dyn_mm", "eligible"]                                  # ... but it is not an answer
    assert "below p00_fp32" in row.loc["p50_int8_dyn_mm", "ineligible_reason"]
    assert row.loc["p00_int8_dyn_mm", "eligible"] and row.loc["p00_fp32", "eligible"]
    assert rank(s, 1).iloc[0]["variant"] == "p00_int8_dyn_mm"


def test_min_accuracy_floor():
    # Both drop constraints off, so the floor is the only thing deciding eligibility.
    s = score_table(_df(), {"max_accuracy_drop": None, "max_accuracy_drop_rel": None,
                            "min_accuracy": 0.90})
    assert set(s.loc[s["eligible"], "variant"]) == {"p00_fp32", "p00_int8_dyn_mm"}


def test_constraint_can_be_disabled():
    # Disabling means turning off every constraint: the absolute drop, the relative drop (which
    # defaults to the rubric's 97%-of-baseline band) and the floor.
    s = score_table(_df(), {"max_accuracy_drop": None, "max_accuracy_drop_rel": None,
                            "min_accuracy": None})
    assert s["eligible"].all()


def test_candidates_is_a_shortlist_not_one_answer():
    s = score_table(_df(), {"max_accuracy_drop": 0.02})
    s["pareto_size_bytes"] = [False, True, True]     # as finalize() would flag them
    short = candidates(s, ["size_bytes"], top=2)
    assert len(short) >= 2
    assert "p50_int8_dyn_mm" not in set(short["variant"])   # ineligible rows never appear
    assert all(short["why"].str.len() > 0)


def test_eligibility_handles_missing_accuracy():
    df = _df()
    df.loc[1, "accuracy"] = np.nan
    ok = pd.Series(True, index=df.index)
    elig, reason = eligibility(df, ok, {"max_accuracy_drop": 0.02, "min_accuracy": None})
    assert not elig[1] and reason[1] == "no accuracy"


@pytest.mark.parametrize("reps", [1, 3])
def test_measure_repeats_report_spread(tmp_path, dummy, trained, reps):
    from core.export import export_onnx
    from core.measure import runtime_metrics

    p = export_onnx(trained, dummy.example_input(), tmp_path / "m.onnx")
    row = runtime_metrics(p, dummy.example_input(),
                          {"threads": [1], "warmup": 2, "runs": 10, "min_time_s": 0.1, "repeats": reps})
    assert row["latency_p50_ms"] > 0
    assert ("latency_p50_ms_sd_t1" in row) == (reps > 1)
    if reps > 1:
        assert row["latency_p50_ms_sd_t1"] >= 0 and row["peak_rss_mb_sd_t1"] >= 0


def test_compress_report_states_constraint_and_rejects(tmp_path, dummy):
    """The deliverable report must name the chosen variant, the constraint, and what it refused."""
    import importlib.util
    import sys
    from pathlib import Path

    scripts = str(Path(__file__).resolve().parents[1] / "scripts")   # compress.py imports _common from there
    sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("compress_mod", f"{scripts}/compress.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    s = score_table(_df(), {"max_accuracy_drop": 0.02})
    s["pareto_size_bytes"] = [False, True, True]
    # distinct per row AND per thread count, so a report that prints the wrong column or the wrong
    # row fails instead of coincidentally matching
    s["latency_p50_ms_t1"] = [9.10, 5.20, 3.30]
    s["latency_p50_ms_t4"] = [2.10, 1.20, 0.30]
    s["latency_p50_ms"] = s["latency_p50_ms_t4"]      # un-suffixed follows primary_threads
    for col, val in (("int8_sqnr_db", 30.0), ("int8_quantized_nodes", 12), ("energy_j", float("nan"))):
        s[col] = val
    ok = s[s["eligible"]]
    best = ok.loc[ok["score"].idxmax()]
    rejected = s[(~s["eligible"]) & (s["score"] >= best["score"])]
    files = {"best": tmp_path / "best.onnx"}
    cfg = {"task": "dummy", "score": {"max_accuracy_drop": 0.02}, "_config_path": "configs/x.yaml"}
    p = mod.write_report(tmp_path, cfg,
                         dummy, s.iloc[0], best, candidates(s, ["size_bytes"]), rejected, files)
    text = p.read_text(encoding="utf-8")
    assert "p00_int8_dyn_mm" in text and "at most 0.02 absolute" in text
    assert "p50_int8_dyn_mm" in text and "Rejected despite scoring" in text
    assert "never estimated" in text
    # The thread rows must carry the RIGHT numbers, not merely the right words: asserting only that
    # "1 thread" appears cannot catch the mislabel that shipped three times (an un-suffixed value,
    # which follows primary_threads, printed under a 1-thread heading).
    import re
    for t, col in ((1, "latency_p50_ms_t1"), (4, "latency_p50_ms_t4")):
        m = re.search(rf"\| latency p50, {t} thread[s]? \| ([\d.]+) ms \| ([\d.]+) ms", text)
        assert m, f"no {t}-thread latency row in the report"
        assert abs(float(m.group(1)) - float(s.iloc[0][col])) < 5e-3, f"{t}T baseline is not {col}"
        assert abs(float(m.group(2)) - float(best[col])) < 5e-3, f"{t}T chosen is not {col}"


def test_relative_accuracy_budget_scales_with_the_metric():
    """0.02 absolute is 2% of a 0.92 top-1 but ~5% of a 0.41 mAP: the relative form fixes that."""
    df = _df()
    df["accuracy"] = [0.4095, 0.4006, 0.3950]          # detector-like magnitudes (drops 0, 0.9, 1.5 pt)
    s = score_table(df, {"max_accuracy_drop": 0.02, "max_accuracy_drop_rel": None})
    assert s["eligible"].all()                          # absolute 0.02 lets a 3.5%-relative loss through
    s = score_table(df, {"max_accuracy_drop": None, "max_accuracy_drop_rel": 0.025})
    assert list(s["eligible"]) == [True, True, False]    # 2.5% of 0.4095 = 0.0102
    assert "relative" in s.iloc[2]["ineligible_reason"]
    s = score_table(df, {"max_accuracy_drop": 0.005, "max_accuracy_drop_rel": 0.025})
    assert list(s["eligible"]) == [True, False, False]   # the stricter of the two wins


def test_sweep_grid_runs_one_grid_per_adapter_variant(tmp_path):
    """sweep.adapter_variants (e.g. input resolution) is a real sweep dimension: prefixed rows, one CSV."""
    from core.sweep import ok_mask, run_sweep_grid
    from dummy_adapter import DummyAdapter

    cfg = {"task": "dummy", "measure": {"threads": [1], "warmup": 1, "runs": 5, "min_time_s": 0.05},
           "quantize": {"calib_batches": 1},
           "sweep": {"prune_ratios": [0.0], "precisions": ["fp32"],
                     "adapter_variants": [{"width": 8}, {"width": 16}]}}
    df = run_sweep_grid(cfg, lambda c: DummyAdapter(c.get("adapter_args", {})),
                        lambda ad, c: ad.build_model(), tmp_path)
    ok = df[ok_mask(df)]
    assert set(ok["variant"]) == {"8_p00_fp32", "16_p00_fp32"}
    assert set(ok["width"]) == {8, 16}                       # the dimension is recorded per row
    assert ok.loc[ok["width"] == 8, "size_bytes"].iloc[0] < ok.loc[ok["width"] == 16, "size_bytes"].iloc[0]
    assert (tmp_path / "sweep.csv").exists() and len(pd.read_csv(tmp_path / "sweep.csv")) == 2


def test_rescore_only_never_relabels_threads_it_did_not_measure(tmp_path, monkeypatch):
    """--rescore-only re-derives the primary thread count. It used to read it from the CONFIG, so a
    --quick CSV (only *_t1 columns) asked for t4, copied nothing, and still stamped primary_threads=4
    on 1-thread numbers. The CSV is the source of truth: nothing is re-measured here."""
    import re

    import pandas as pd

    from core.measure import DEFAULT_MEASURE_CFG, primary_thread_count

    def rescore(df, mcfg):
        """The logic of scripts/remeasure.py's --rescore-only branch."""
        measured = sorted({int(m.group(1)) for c in df.columns
                           if (m := re.fullmatch(r"latency_p50_ms_t(\d+)", c))})
        assert measured, "no per-thread column to re-score from"
        p = primary_thread_count({**DEFAULT_MEASURE_CFG, **mcfg, "threads": measured})
        for k in ("latency_p50_ms", "peak_rss_mb"):
            if f"{k}_t{p}" in df:
                df[k] = df[f"{k}_t{p}"]
        df["primary_threads"] = p
        return df

    quick = pd.DataFrame({"variant": ["p00_fp32"], "latency_p50_ms_t1": [2.45], "peak_rss_mb_t1": [22.8],
                          "latency_p50_ms": [2.45], "peak_rss_mb": [22.8], "primary_threads": [1]})
    out = rescore(quick.copy(), {"threads": [1], "primary_threads": 4})
    assert out["primary_threads"].iloc[0] == 1          # not 4: t4 was never measured
    assert out["latency_p50_ms"].iloc[0] == 2.45        # and the value still matches its label

    full = pd.DataFrame({"variant": ["p00_fp32"], "latency_p50_ms_t1": [8.76], "latency_p50_ms_t4": [2.62],
                         "peak_rss_mb_t1": [22.4], "peak_rss_mb_t4": [22.4],
                         "latency_p50_ms": [8.76], "peak_rss_mb": [22.4], "primary_threads": [1]})
    out = rescore(full.copy(), {"threads": [1, 4], "primary_threads": 4})
    assert out["primary_threads"].iloc[0] == 4          # t4 exists, so honour the request
    assert out["latency_p50_ms"].iloc[0] == 2.62        # and actually switch the numbers over


def test_report_labels_latency_with_the_thread_count_it_used():
    """scripts/compress.py hardcoded "latency 1T" in the candidates table while filling it from the
    un-suffixed column, which follows measure.primary_threads (4 with the shipped configs)."""
    from scripts.compress import _nt

    assert _nt({"primary_threads": 4}) == "4 threads"
    assert _nt({"primary_threads": 1}) == "1 thread"
    assert _nt({}) == "1 thread"                        # missing column must not crash the report
    assert _nt({"primary_threads": float("nan")}) == "1 thread"


def test_report_discovers_new_runs_and_skips_incomplete_ones(tmp_path, monkeypatch, capsys):
    """The run list was hardcoded to the five phase runs, so a new config's results never reached the
    Excel report at all; and a run still in progress (sweep.csv written, summary.json not yet) made the
    generator crash on the missing file."""
    import json
    import sys

    import pandas as pd
    from conftest import ROOT

    sys.path.insert(0, str(ROOT / "scripts"))
    from make_report_xlsx import discover, load_runs

    results = tmp_path / "results"
    def mk(name, summary=True, rows=1):
        d = results / name
        d.mkdir(parents=True)
        pd.DataFrame({"variant": ["p00_fp32"] * rows, "accuracy": [0.9] * rows,
                      "metric_name": ["top1"] * rows, "score": [0.5] * rows,
                      "eligible": [True] * rows}).to_csv(d / "sweep.csv", index=False)
        if summary:
            (d / "summary.json").write_text(json.dumps({"candidates": [], "ineligible": []}), encoding="utf-8")

    mk("detection")
    mk("classification")
    mk("stress_resnet50")            # a new run that is not on the hardcoded list
    mk("zz_custom_task")             # any future config
    mk("detection_prune", summary=False)   # still running: no summary.json yet
    mk("classification_quick")       # smoke run: never in the report

    order = discover(results)
    assert order[:2] == ["classification", "detection"]          # phase runs first, in ORDER
    assert order[2:] == ["detection_prune", "stress_resnet50", "zz_custom_task"]   # rest alphabetical
    assert "classification_quick" not in order

    # the nastier case: a run being RE-run has a partial sweep.csv next to the PREVIOUS summary.json,
    # so "summary exists" is not enough to call it complete
    rerun = results / "zz_custom_task"
    pd.DataFrame({"variant": ["p00_fp32"], "accuracy": [0.9], "metric_name": ["top1"]}).to_csv(
        rerun / "sweep.csv", index=False)        # finalize() has not added score/eligible back yet

    runs = load_runs(results)                                     # must not raise on either case
    assert set(runs) == {"classification", "detection", "stress_resnet50"}
    out = capsys.readouterr().out
    assert "detection_prune" in out and "zz_custom_task" in out   # both skipped out loud
