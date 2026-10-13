"""An adapter returning a lower-is-better score inverts every ranking, silently.

TaskAdapter.evaluate() is contracted to return a HIGHER-is-better score. score.py's eligibility
check rejects variants that fell below the baseline and pareto.py keeps y >= y[i], so handing back
a raw MAE/RMSE/reconstruction error makes the BEST variant ineligible and lets the WORST one win.
It is a wrong-answer failure, not a crash - and `return {"metric_name": "mae", "score": mae}` is the
natural way to write an adapter under time pressure on competition day.

There is no safe automatic fix (the toolkit cannot know the direction), so the guard is a loud
warning keyed on the metric NAME.
"""
import numpy as np
import pandas as pd
import pytest

from core.measure import _warn_if_metric_looks_inverted, _warned_metrics
from core.score import score_table


@pytest.mark.parametrize("name", ["mae", "MAE", "rmse", "val_mse", "recon_error", "test-loss",
                                  "perplexity", "word_error_rate_wer", "mean_distance"])
def test_lower_is_better_names_are_flagged(name):
    _warned_metrics.discard(name.strip().lower())
    assert _warn_if_metric_looks_inverted(name), f"{name!r} should have been flagged"


@pytest.mark.parametrize("name", ["top1", "accuracy", "mAP50-95", "auc", "f1", "IDF1", "MOTA",
                                  "precision", "recall", "psnr", "miou"])
def test_higher_is_better_names_are_not_flagged(name):
    _warned_metrics.discard(name.strip().lower())
    assert not _warn_if_metric_looks_inverted(name), f"{name!r} is a false alarm"


def test_each_name_warns_only_once():
    """One line per offending metric, not one per variant in the sweep."""
    _warned_metrics.discard("mae")
    assert _warn_if_metric_looks_inverted("mae")
    assert not _warn_if_metric_looks_inverted("mae")


def test_inversion_actually_selects_the_worst_variant():
    """Documents the damage the guard is warning about, so the cost is on record rather than
    asserted in a comment: with an error metric fed in raw, the variant with the WORST error is
    the one that survives the accuracy constraint."""
    df = pd.DataFrame({
        "variant": ["p00_fp32", "good_int8", "bad_int8"],
        "accuracy": [0.10, 0.11, 0.55],          # an MAE: 0.11 is nearly baseline, 0.55 is ruined
        "size_bytes": [1000, 250, 250],
        "peak_rss_mb": [10.0, 4.0, 4.0],
        "latency_p50_ms": [10.0, 4.0, 4.0],
        "prune_ratio": [0.0, 0.0, 0.5],
        "error": [np.nan] * 3,
    })
    out = score_table(df, {"max_accuracy_drop": 0.02}).set_index("variant")

    # The accuracy constraint only rejects values BELOW the baseline, so with an error metric it
    # never fires: a model whose error grew 5.5x sails through.
    assert bool(out.at["bad_int8", "eligible"]),         "the ruined variant passes the accuracy constraint when the metric is inverted"

    # And then the ranking is actively wrong: the ruined variant scores BEST on accuracy, because
    # a bigger number is read as a better model.
    assert out.at["bad_int8", "score_accuracy"] > out.at["good_int8", "score_accuracy"],         "the ruined variant should out-score the good one - that is the damage being demonstrated"
    assert out["score"].idxmax() == "bad_int8",         "the worst variant wins outright once size and latency agree"
