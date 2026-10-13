"""Regression tests for the pruning findings (see core/prune.py docstring)."""
import copy

import pytest
import torch
import torch.nn as nn

from core.prune import _widths, prune_model, recalibrate_bn


def _randomize_bn(model, seed=0):
    """Random BN statistics/affine so that removing (or mis-mapping) any channel changes the output."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            n = m.num_features
            m.running_mean.copy_(torch.randn(n, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(n, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(n, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(n, generator=g) * 0.1)
    return model.eval()


def masked_vs_pruned(model, x, ratio, ignored):
    """Zero exactly the channels the pruner removes (producer rows + BN affine) in a copy of the model, then
    prune the model for real: identical outputs <=> channels are removed consistently on producer and
    consumer layers (a mis-mapped index keeps shapes valid but changes the numbers)."""
    import torch_pruning as tp

    out_fns = {tp.prune_conv_out_channels, tp.prune_linear_out_channels, tp.prune_batchnorm_out_channels}
    if hasattr(tp, "prune_depthwise_conv_out_channels"):
        out_fns.add(tp.prune_depthwise_conv_out_channels)
    probe = copy.deepcopy(model).eval()
    names = {id(m): n for n, m in probe.named_modules()}
    ign = [probe.get_submodule(n) for n in ignored]
    pruner = tp.pruner.MetaPruner(probe, x, importance=tp.importance.GroupMagnitudeImportance(p=2),
                                  pruning_ratio=ratio, ignored_layers=ign)
    masks = []
    for group in pruner.step(interactive=True):   # inspect only, never prune the probe
        for dep, idxs in group:
            if dep.handler in out_fns:
                masks.append((names[id(dep.target.module)], list(idxs)))
    assert masks, "pruner selected nothing"
    masked = copy.deepcopy(model).eval()
    with torch.no_grad():
        for name, idxs in masks:
            mod = masked.get_submodule(name)
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.weight[idxs] = 0
                mod.bias[idxs] = 0
            else:
                mod.weight[idxs] = 0
                if mod.bias is not None:
                    mod.bias[idxs] = 0
    pruned = prune_model(model, x, ratio, [model.get_submodule(n) for n in ignored], round_to=None)
    with torch.no_grad():
        y_orig, y_masked, y_pruned = model(x), masked(x), pruned(x)
    scale = y_masked.abs().max().clamp_min(1e-12)
    return float((y_pruned - y_masked).abs().max() / scale), float((y_orig - y_masked).abs().max() / scale)


def _tiny():
    from dummy_adapter import TinyNet
    m = TinyNet()
    return _randomize_bn(m), torch.randn(2, 3, 16, 16), ["head"]


def _mbv2():
    import torchvision.models as tvm
    return _randomize_bn(tvm.mobilenet_v2(num_classes=10)), torch.randn(2, 3, 32, 32), ["classifier.1"]


def _resnet():
    import torchvision.models as tvm
    return _randomize_bn(tvm.resnet18(num_classes=10)), torch.randn(2, 3, 32, 32), ["fc"]


def _resnet50():
    """Bottleneck blocks: 1x1 -> 3x3 -> 1x1 with a downsample conv on the shortcut, so the residual add
    couples three producers. 25M parameters - the largest model in the matrix."""
    import torchvision.models as tvm
    return _randomize_bn(tvm.resnet50(num_classes=10)), torch.randn(2, 3, 64, 64), ["fc"]


def _yolo():
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel

    from adapters.ultralytics_compat import make_prunable
    m = DetectionModel("yolov8n.yaml", nc=80, verbose=False)
    m.model[-1].export = True
    make_prunable(m)
    return _randomize_bn(m), torch.randn(1, 3, 128, 128), ["model.22"]


@pytest.mark.parametrize("family", ["tiny", "mbv2", "resnet18", "resnet50", "yolov8n"])
def test_pruning_is_index_exact(family):
    torch.manual_seed(0)
    model, x, ignored = {"tiny": _tiny, "mbv2": _mbv2, "resnet18": _resnet,
                         "resnet50": _resnet50, "yolov8n": _yolo}[family]()
    diff, effect = masked_vs_pruned(model, x, 0.3, ignored)
    # a mis-mapped channel makes `diff` the same order as `effect`; exact bookkeeping leaves float noise
    assert effect > 1e-6, "masking changed nothing: the test could not detect a mapping error"
    assert diff < 0.01 * effect, f"pruned != masked (rel {diff:.2e} vs effect {effect:.2e}): channels mis-mapped"


def test_round_to_never_prunes_more_than_requested(dummy, trained):
    x = dummy.example_input()
    rep = {}
    p = prune_model(trained, x, 0.05, dummy.ignored_layers(trained), round_to=8, report=rep)
    assert rep["max_layer_cut"] <= 0.05 + 1e-9          # torch-pruning alone would cut 16 -> 8 (50%)
    assert _widths(p)["body.0"] == 16
    rep = {}
    prune_model(trained, x, 0.5, dummy.ignored_layers(trained), round_to=8, report=rep)
    assert 0 < rep["max_layer_cut"] <= 0.5 + 1e-9


def test_round_to_handles_grouped_convs(dummy):
    """round_to must not silently skip grouped convs (torch-pruning counts indices per channel group)."""
    class Grouped(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.bn = nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64)
            self.g, self.bn2 = nn.Conv2d(64, 64, 3, padding=1, groups=4), nn.BatchNorm2d(64)
            self.head = nn.Linear(64, 10)

        def forward(self, x):
            x = torch.relu(self.bn2(self.g(torch.relu(self.bn(self.a(x))))))
            return self.head(x.mean((2, 3)))

    torch.manual_seed(0)
    m = _randomize_bn(Grouped())
    x = torch.randn(2, 3, 16, 16)
    rep = {}
    p = prune_model(m, x, 0.25, [m.head], round_to=8, report=rep)
    assert p.g.out_channels == 48 and p.g.groups == 4      # not 64: the group was really pruned
    assert rep["max_layer_cut"] <= 0.25 + 1e-9
    with torch.no_grad():
        assert p(x).shape == m(x).shape


def test_recalibrate_bn_keeps_stats_when_it_gets_no_data(dummy, trained):
    p = prune_model(trained, dummy.example_input(), 0.3, dummy.ignored_layers(trained), round_to=None)
    before = p.body[1].running_mean.clone()
    assert recalibrate_bn(p, dummy.load_data()["calib"], "cpu", max_batches=0) == 0
    assert torch.equal(p.body[1].running_mean, before)     # never left with reset (0/1) statistics
    assert recalibrate_bn(p, [], "cpu", max_batches=8) == 0
    assert torch.equal(p.body[1].running_mean, before)


def test_recalibrate_bn_matches_batch_statistics(dummy, trained):
    x_all = torch.cat([b[0] for b in dummy.load_data()["calib"]])
    p = prune_model(trained, dummy.example_input(), 0.3, dummy.ignored_layers(trained), round_to=None)
    n = recalibrate_bn(p, dummy.load_data()["calib"], "cpu", max_batches=100)
    assert n == len(dummy.load_data()["calib"])
    bn, conv = p.body[1], p.body[0]
    with torch.no_grad():
        feats = conv(x_all)
    assert torch.allclose(bn.running_mean, feats.mean((0, 2, 3)), atol=1e-4)
    assert not bn.training and bn.momentum == 0.1  # restored


def test_attention_scale_consistent_after_pruning():
    timm = pytest.importorskip("timm")
    m = timm.create_model("deit_tiny_patch16_224", num_classes=10, img_size=32, patch_size=4).eval()
    x = torch.randn(2, 3, 32, 32)
    p = prune_model(m, x, 0.25, [m.get_classifier()], round_to=None)
    attn = p.blocks[0].attn
    assert attn.head_dim < 64 and abs(attn.scale - attn.head_dim ** -0.5) < 1e-12
    with torch.no_grad():
        fused = p(x)
        for blk in p.blocks:
            blk.attn.fused_attn = False
        manual = p(x)
    assert torch.allclose(fused, manual, atol=1e-4)


def test_unstructured_pruning_keeps_shapes_and_pays_off_only_after_compression(tmp_path, dummy):
    """prune.mode: unstructured zeroes weights instead of removing channels. Its entire benefit is
    file size AFTER compression - measured here: identical size_bytes, 4.25x smaller gzipped - so it
    must not be sold as a latency or RAM win, and the shapes must stay byte-identical."""
    from core.export import export_onnx
    from core.measure import _gz_size
    from core.prune import prune_and_finetune, prune_model, prune_unstructured

    model, x = dummy.build_model().eval(), dummy.example_input()
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}

    rep = {}
    sparse = prune_unstructured(model, 0.9, dummy.ignored_layers(model), report=rep)
    assert {n: tuple(p.shape) for n, p in sparse.named_parameters()} == shapes   # nothing removed
    assert sparse(x).shape == model(x).shape
    assert abs(rep["sparsity"] - 0.9) < 0.02 and rep["pruned_layers"] > 0
    # prune.remove() must have baked the mask in: a mask buffer would ride along into the export
    assert not any("mask" in n for n, _ in sparse.named_buffers())

    dense_f = export_onnx(model, x, tmp_path / "dense.onnx")
    sparse_f = export_onnx(sparse, x, tmp_path / "sparse.onnx")
    assert sparse_f.stat().st_size == dense_f.stat().st_size      # the honest part: no raw saving
    assert _gz_size([sparse_f]) < 0.5 * _gz_size([dense_f])       # the real part: compressed saving

    # structured, for contrast, really does shrink the tensors
    pruned = prune_model(model, x, 0.25, dummy.ignored_layers(model), round_to=None)
    assert {n: tuple(p.shape) for n, p in pruned.named_parameters()} != shapes

    rep = {}
    prune_and_finetune(dummy, model, 0.25, epochs=0, cfg={"mode": "unstructured"}, report=rep)
    assert rep["prune_mode"] == "unstructured"
    with pytest.raises(ValueError, match="unknown prune.mode"):
        prune_and_finetune(dummy, model, 0.25, epochs=0, cfg={"mode": "magnitude"})


def test_unstructured_sparsity_survives_finetuning(dummy, trained):
    """The first version baked the zeros in BEFORE fine-tuning, so every zeroed weight got a
    non-zero gradient and drifted off zero: asking for 50/80/90% left 0.11/0.25/2.36% actual zeros
    in the exported model - the whole compression benefit gone, silently. The original test missed
    it by using epochs=0, so this one must fine-tune for real."""
    import torch.nn.utils.prune as tprune

    from core.prune import prune_and_finetune

    rep = {}
    out = prune_and_finetune(dummy, trained, 0.8, epochs=2, cfg={"mode": "unstructured"}, report=rep)
    weights = [m.weight for m in out.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    tot = sum(w.numel() for w in weights)
    zeros = sum(int((w == 0).sum()) for w in weights)
    assert zeros / tot > 0.6, f"only {100 * zeros / tot:.1f}% zeros left: fine-tuning undid the pruning"
    assert rep["sparsity"] > 0.6
    # and the mask must be gone, or the export would carry weight_orig/weight_mask instead of weight
    assert not any(tprune.is_pruned(m) for m in out.modules())
    assert not any("mask" in n for n, _ in out.named_buffers())


def test_recovery_lr_mult_scales_only_the_pruning_finetune(dummy, monkeypatch):
    """An adapter's finetune_lr is tuned to not disturb a healthy model; repairing a pruned one needs
    more. recovery_lr_mult must reach adapter.finetune as an explicit lr, and 1.0 must change nothing."""
    from core.prune import prune_and_finetune

    seen = []
    monkeypatch.setitem(dummy.cfg, "finetune_lr", 2e-5)
    monkeypatch.setattr(dummy, "finetune", lambda m, ep, lr=None: seen.append(lr) or m)

    prune_and_finetune(dummy, dummy.build_model().eval(), 0.25, epochs=1, cfg={})
    assert seen == [None]                                   # default: the adapter's own value

    seen.clear()
    rep = {}
    prune_and_finetune(dummy, dummy.build_model().eval(), 0.25, epochs=1,
                       cfg={"recovery_lr_mult": 20.0}, report=rep)
    assert seen == [pytest.approx(4e-4)] and rep["recovery_lr"] == pytest.approx(4e-4)

    seen.clear()
    prune_and_finetune(dummy, dummy.build_model().eval(), 0.25, epochs=1, cfg={"recovery_lr_mult": 1.0})
    assert seen == [None]                                   # 1.0 is a no-op, not lr*1.0


def test_ratio_per_layer_overrides_the_uniform_ratio(dummy, trained):
    """A per-layer dict is how a computed allocation reaches torch-pruning. ratio 0 WITH a dict must
    prune only the named layers - the early `if ratio <= 0: return` used to swallow exactly that,
    which made the sensitivity probe report 0 usable layers."""
    from core.prune import prune_model

    x = dummy.example_input()
    w0 = _widths(trained)

    p = prune_model(trained, x, 0.25, dummy.ignored_layers(trained), round_to=None,
                    ratio_per_layer={"body.0": 0.5})
    w = _widths(p)
    assert w["body.0"] == w0["body.0"] // 2          # the named layer got its own ratio
    assert w["body.3"] == int(w0["body.3"] * 0.75)   # the rest still got the uniform one

    rep = {}
    only = prune_model(trained, x, 0.0, dummy.ignored_layers(trained), round_to=None,
                       ratio_per_layer={"body.0": 0.5}, report=rep)
    w = _widths(only)
    assert w["body.0"] == w0["body.0"] // 2
    assert w["body.3"] == w0["body.3"]               # ratio 0 elsewhere: untouched, not skipped
    assert rep["n_layer_ratios"] == 1
    assert sum(q.numel() for q in only.parameters()) < sum(q.numel() for q in trained.parameters())

    # an unknown name is a warning, not a crash
    prune_model(trained, x, 0.1, dummy.ignored_layers(trained), round_to=None,
                ratio_per_layer={"no.such.layer": 0.5})


def test_layer_sensitivity_is_label_free_and_ranks_layers(dummy, trained):
    """The probe prunes ONE layer at a time and scores the output distortion against the unpruned
    model - no labels, no task metric. It also has to measure what each cut really saves, because
    channel pruning propagates into the next layer's inputs."""
    from core.prune import layer_sensitivity

    rep = {}
    sens = layer_sensitivity(dummy, trained, {"probe_ratio": 0.5, "probe_batches": 1,
                                              "round_to": None}, rep)
    assert sens and rep["sensitivity_layers"] == len(sens)
    assert "head" not in sens                        # ignored_layers are never candidates
    for name, v in sens.items():
        assert -20.0 <= v["damage_db"] <= 60.0, (name, v)
        assert 0 < v["param_frac"] <= 1.0
        assert 0 <= v.get("mac_frac", 0) <= 1.0


def test_allocation_spends_the_budget_it_was_given(dummy, trained):
    """greedy and dp must both return a plan, only touch real layers, and report the cost axis they
    actually spent. The plans differ - that is the point of having both."""
    from core.prune import allocate_ratios, layer_sensitivity

    cfg0 = {"probe_ratio": 0.5, "probe_batches": 1, "round_to": None}
    sens = layer_sensitivity(dummy, trained, cfg0, {})
    names = set(dict(trained.named_modules()))

    plans = {}
    for method in ("greedy", "dp"):
        rep = {}
        alloc = allocate_ratios(dummy, trained, 0.2,
                                {**cfg0, "allocate": method, "sensitivity": sens,
                                 "alloc_cost": "params", "alloc_rounds": 3}, rep)
        assert set(alloc) <= names and all(0 < r <= 0.5 for r in alloc.values())
        assert rep["alloc_method"] == method and rep["alloc_cost"] == "params"
        assert rep["alloc_rounds"] >= 1 and rep["alloc_trace"]
        plans[method] = alloc

    # the cost axis is a real choice: parameters and MACs pick different layers on any real net
    rep = {}
    allocate_ratios(dummy, trained, 0.2, {**cfg0, "allocate": "dp", "sensitivity": sens,
                                          "alloc_cost": "macs", "alloc_rounds": 2}, rep)
    assert rep["alloc_cost"] == "macs"

    with pytest.raises(ValueError, match="alloc_cost"):
        allocate_ratios(dummy, trained, 0.2, {**cfg0, "sensitivity": sens, "alloc_cost": "flops"})

    # when the PROBE finds nothing usable, fall back to the uniform ratio instead of crashing.
    # (an empty "sensitivity" in cfg means "not cached, go measure", which is a different path)
    import core.prune as cp
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(cp, "layer_sensitivity", lambda *a, **k: {})
        assert cp.allocate_ratios(dummy, trained, 0.2, dict(cfg0)) == {}
    finally:
        monkeypatch.undo()


def test_probe_recalibrates_bn_before_scoring(dummy, trained, monkeypatch):
    """Without BN re-estimation the probe scores a model whose BatchNorm statistics belong to
    channels that no longer exist, which is not the damage pruning did. Measured on a detector: two
    layers probed at 0.008 and 0.034 whose real post-recalibration scores were 0.172 and 0.222, and
    the correlation with the truth went from +0.395 to +0.975 once BN was re-estimated."""
    import core.prune as cp

    calls = []
    real = cp.recalibrate_bn
    monkeypatch.setattr(cp, "recalibrate_bn",
                       lambda m, *a, **k: calls.append(1) or real(m, *a, **k))

    cp.layer_sensitivity(dummy, trained, {"probe_ratio": 0.5, "probe_batches": 1,
                                          "round_to": None}, {})
    assert calls, "the probe must re-estimate BN before scoring each candidate"

    calls.clear()
    cp.layer_sensitivity(dummy, trained, {"probe_ratio": 0.5, "probe_batches": 1, "round_to": None,
                                          "probe_bn_recalibrate": False}, {})
    assert not calls                      # and it stays switchable


def test_probe_can_rank_by_the_task_metric_instead_of_sqnr(dummy, trained):
    """SQNR tracks the metric on some tasks and not others - measured, +0.975 on a classifier's
    proxy but only +0.395 against a detector's mAP, because mAP turns on whether boxes clear an IoU
    threshold rather than on the output tensor's signal-to-noise. So the proxy is selectable."""
    from core.prune import allocate_ratios, layer_sensitivity

    base = {"probe_ratio": 0.5, "probe_batches": 1, "round_to": None}

    rep = {}
    s_sqnr = layer_sensitivity(dummy, trained, base, rep)
    assert rep["sensitivity_proxy"] == "sqnr"
    assert all("damage_db" in v and "quality" in v for v in s_sqnr.values())
    assert all(v["quality"] == v["damage_db"] for v in s_sqnr.values())

    rep = {}
    s_met = layer_sensitivity(dummy, trained, {**base, "probe_metric": "metric"}, rep)
    assert rep["sensitivity_proxy"] == "metric"
    assert s_met and all("quality" in v and "damage_db" not in v for v in s_met.values())
    assert all(0.0 <= v["quality"] <= 1.0 for v in s_met.values())   # it is a task score now

    # the allocator must work off either, without knowing which
    for sens in (s_sqnr, s_met):
        alloc = allocate_ratios(dummy, trained, 0.2,
                                {**base, "sensitivity": sens, "alloc_cost": "params",
                                 "alloc_rounds": 2})
        assert alloc and all(0 < r <= 0.5 for r in alloc.values())

    with pytest.raises(ValueError, match="probe_metric"):
        layer_sensitivity(dummy, trained, {**base, "probe_metric": "loss"}, {})


def test_an_allocated_plan_is_realised_exactly_as_it_was_measured(dummy, trained):
    """allocate_ratios measures its plan with the uniform ratio at 0 and drops zero-level layers from
    the dict, so 'leave this layer alone' is encoded as absence. prune_and_finetune then has to
    realise it the same way - passing the uniform ratio instead pruned every unlisted layer too:
    measured, a plan for 22.3% of parameters exported a model with 40.1% removed, and a shipped run
    planned for 19.7% removed 49.5%."""
    from core.prune import prune_and_finetune

    cfg = {"allocate": "greedy", "alloc_cost": "params", "probe_ratio": 0.25, "probe_batches": 1,
           "round_to": None, "alloc_rounds": 2}
    n0 = sum(p.numel() for p in trained.parameters())
    rep = {}
    out = prune_and_finetune(dummy, trained, 0.25, 0, cfg, report=rep)
    actual = 1 - sum(p.numel() for p in out.parameters()) / n0
    assert rep["alloc_frac"] > 0
    assert abs(actual - rep["alloc_frac"]) < 0.02, \
        f"plan promised {rep['alloc_frac']:.4f} but the exported model removed {actual:.4f}"

    # a user-written ratio_per_layer still LAYERS ON TOP of the uniform ratio - that is what the knob
    # means, and it is a different contract from an allocator plan
    w0 = _widths(trained)
    p = prune_and_finetune(dummy, trained, 0.25, 0, {"ratio_per_layer": {"body.0": 0.5},
                                                     "round_to": None})
    w = _widths(p)
    assert w["body.0"] == w0["body.0"] // 2
    assert w["body.3"] == int(w0["body.3"] * 0.75)     # unlisted layer still gets the uniform ratio


def test_bn_is_recalibrated_when_only_a_per_layer_plan_prunes(dummy, trained, monkeypatch):
    """`ratio > 0` was the test for 'did we prune', but a plan prunes with ratio 0 - so BN
    re-estimation and fine-tuning were both skipped, leaving statistics describing removed channels.
    That is the same defect that made the sensitivity probe wrong by 20x."""
    import core.prune as cp

    seen = []
    real = cp.recalibrate_bn
    monkeypatch.setattr(cp, "recalibrate_bn", lambda m, *a, **k: seen.append(1) or real(m, *a, **k))
    ft = []
    monkeypatch.setattr(dummy, "finetune", lambda m, ep, lr=None: ft.append(ep) or m)

    rep = {}
    cp.prune_and_finetune(dummy, trained, 0.0, 2, {"ratio_per_layer": {"body.0": 0.5},
                                                   "round_to": None}, report=rep)
    assert seen, "BN must be re-estimated when a per-layer plan pruned, even at ratio 0"
    assert ft == [2], "fine-tuning must run too"
    assert rep.get("bn_recal_batches") is not None


def test_unstructured_pruning_respects_ignored_layers(dummy, trained):
    """ids were taken from the caller's model but the targets came from the deepcopy, so the match
    never fired and layers the adapter asked to protect - the classifier head - were pruned anyway."""
    from core.prune import prune_unstructured

    head = dummy.ignored_layers(trained)
    assert head, "the dummy adapter must protect something for this test to mean anything"
    out = prune_unstructured(trained, 0.9, head, report={})
    names = {id(m): n for n, m in trained.named_modules()}
    for m in head:
        kept = out.get_submodule(names[id(m)])
        assert int((kept.weight == 0).sum()) == 0, "an ignored layer was zeroed anyway"


def test_unstructured_baseline_variant_does_not_crash(dummy, trained):
    """`alloc` was bound only inside the structured branch, but `did_prune = ratio > 0 or
    bool(alloc)` reads it on every path. `or` short-circuits whenever ratio > 0, so every
    unstructured test passed - until the ratio-0 BASELINE variant of an unstructured sweep ran and
    raised NameError before a single row was written."""
    from core.prune import prune_and_finetune

    rep = {}
    out = prune_and_finetune(dummy, trained, 0.0, epochs=0, cfg={"mode": "unstructured"}, report=rep)
    assert rep["prune_mode"] == "unstructured"
    assert out is not None
