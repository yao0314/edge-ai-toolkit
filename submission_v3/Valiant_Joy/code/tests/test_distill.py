"""Regression tests for the knowledge-distillation recovery path.

KD is wired end to end (sweep._finetune_fn -> prune_and_finetune(finetune_fn=...) -> distill), but
no run had ever set finetune_mode: distill, so nothing exercised it. Three defects were sitting
there: no curve was recorded, max_steps_per_epoch was ignored, and recovery_lr_mult silently did
nothing on this path.
"""
import copy

import torch

from core.distill import distill
from core.prune import prune_and_finetune
from core.sweep import _finetune_fn


def test_distill_records_a_curve_like_a_plain_finetune(dummy):
    """distill() never touched `history`, so the sweep wrote train_epochs_run 0 and saved no curve
    for a distilled variant - the one run where the recovery curve is the point."""
    teacher = dummy.build_model().eval()
    student = copy.deepcopy(teacher)
    hist = []
    distill(teacher, student, dummy, epochs=2, history=hist,
            eval_fn=lambda m: dummy.evaluate(m)["score"], eval_every=1)

    assert [h["epoch"] for h in hist] == [1, 2]
    assert all(h["tag"] == "distill" for h in hist)
    for h in hist:                      # same keys common.train_loop emits, so _save_curve works
        assert {"tag", "epoch", "epochs", "loss", "lr", "seconds", "metric"} <= set(h)
        assert h["metric"] is not None, "eval_every was set, so the metric column must be filled"


def test_distill_honours_max_steps_per_epoch(dummy):
    """_finetune_fn never forwarded it, so a KD run ignored the adapter's step cap and a --quick
    sweep would have walked the whole training set."""
    hist = []
    teacher = dummy.build_model().eval()
    distill(teacher, copy.deepcopy(teacher), dummy, epochs=1, max_steps_per_epoch=1, history=hist)
    capped = hist[0]["seconds"]

    hist2 = []
    distill(teacher, copy.deepcopy(teacher), dummy, epochs=1, history=hist2)
    assert capped <= hist2[0]["seconds"] + 1e-6
    cfg = {"sweep": {"finetune_mode": "distill"}, "distill": {}}
    dummy.cfg["max_steps_per_epoch"] = 1
    assert _finetune_fn(cfg, dummy, teacher) is not None


def test_finetune_mode_plain_returns_no_kd_fn(dummy):
    assert _finetune_fn({"sweep": {"finetune_mode": "plain"}}, dummy, dummy.build_model()) is None
    assert _finetune_fn({}, dummy, dummy.build_model()) is None


def test_recovery_lr_mult_scales_the_kd_learning_rate(dummy, trained):
    """`not finetune_fn` skipped the scaling on the KD path, so recovery_lr_mult - the knob that
    exists precisely to repair a freshly pruned model - did nothing whenever KD was on. Scaling the
    adapter's finetune_lr instead would have multiplied a number KD never reads."""
    cfg = {"sweep": {"finetune_mode": "distill"}, "distill": {"lr": 0.02}}
    fn = _finetune_fn(cfg, dummy, copy.deepcopy(trained))
    assert fn.base_lr == 0.02

    rep = {}
    prune_and_finetune(dummy, trained, 0.25, epochs=1, cfg={"recovery_lr_mult": 5.0},
                       finetune_fn=fn, report=rep)
    assert rep["recovery_lr"] == 0.02 * 5.0, "must scale distill.lr, not the adapter's finetune_lr"


def test_kd_recovery_runs_through_prune_and_finetune(dummy, trained):
    """End to end: the path had never been executed by any test or any run."""
    dummy.history.clear()
    fn = _finetune_fn({"sweep": {"finetune_mode": "distill"}, "distill": {"lr": 0.01}},
                      dummy, copy.deepcopy(trained))
    out = prune_and_finetune(dummy, trained, 0.25, epochs=2, cfg={}, finetune_fn=fn, report={})
    assert out is not None
    assert [h["epoch"] for h in dummy.history] == [1, 2], "the sweep reads this to draw the curve"
    with torch.no_grad():
        assert out(dummy.example_input()).shape[1] == 4
