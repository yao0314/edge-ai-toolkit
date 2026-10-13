"""A missing matplotlib must cost the plots, not the run.

finalize() writes sweep.csv, then draws Pareto plots, then writes summary.json. The plotting import
was unguarded, so on a sandbox without matplotlib a COMPLETED sweep raised ModuleNotFoundError after
sweep.csv was on disk: summary.json was never written and scripts/compress.py then refused to
package the run. Every measured number survived and the deliverable was still lost.
"""
import builtins

import pandas as pd
import pytest

from core.pareto import _pyplot, plot_pareto


@pytest.fixture
def no_matplotlib(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib is blocked by this test")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)


def test_pyplot_returns_none_instead_of_raising(no_matplotlib):
    assert _pyplot() is None


def test_plot_pareto_returns_none_without_matplotlib(no_matplotlib, tmp_path):
    df = pd.DataFrame({"variant": ["a", "b"], "size_bytes": [10, 20], "accuracy": [0.9, 0.8],
                       "eligible": [True, True]})
    assert plot_pareto(df, "size_bytes", tmp_path / "p.png") is None
    assert not (tmp_path / "p.png").exists()


def test_plot_curves_returns_empty_without_matplotlib(no_matplotlib, tmp_path):
    from core.sweep import plot_curves

    d = tmp_path / "curves"
    d.mkdir()
    pd.DataFrame({"tag": ["t"], "epoch": [1], "epochs": [1], "loss": [0.5],
                  "lr": [0.01], "seconds": [1.0], "metric": [0.9]}).to_csv(d / "v.csv", index=False)
    assert plot_curves(tmp_path) == []


def test_finalize_still_writes_summary_without_matplotlib(no_matplotlib, tmp_path):
    """The point of the whole fix: the run's outputs survive."""
    import json

    from core.sweep import finalize

    df = pd.DataFrame({
        "variant": ["p00_fp32", "p00_int8"], "accuracy": [0.90, 0.89],
        "metric_name": ["top1"] * 2, "size_bytes": [1000, 250],
        "latency_p50_ms": [10.0, 4.0], "peak_rss_mb": [10.0, 4.0],
        "prune_ratio": [0.0, 0.0], "error": [None, None],
    })
    finalize(df, {"task": "t"}, tmp_path)
    assert (tmp_path / "sweep.csv").exists()
    s = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert s["n_variants"] == 2
    assert s["plots"] == [], "no plots were possible, and that must be recorded rather than fatal"
