"""Channel (structured) and magnitude (unstructured) pruning, then BN re-calibration + finetune.

`mode: structured` (default) physically removes channels, so the exported ONNX gets smaller AND
faster AND lighter on RAM. `mode: unstructured` zeroes individual weights: the tensor keeps its
shape, so ONNX Runtime on a plain CPU runs it at exactly the same speed and memory - the gain is
file size only, and only once the file is compressed (see size_bytes_gz). Both are available
because "compression" may be scored on file size; structured is the default because it is the only
one of the two that moves latency and peak RAM.

Lessons baked in (verified: masked-vs-pruned equivalence is bit-exact on every CNN family, so
the channel bookkeeping is right; the damage came from the method):
  * round_to only ever rounds the KEPT channel count UP. torch-pruning rounds it down, which
    turned "5%" into 50% on 16-channel layers (a depthwise CNN at 5%: accuracy 0.684 -> 0.117).
  * BatchNorm running statistics are re-estimated after pruning (label-free, calibration data):
    the same CNN pruned 10% without fine-tuning: accuracy 0.114 -> 0.792.
"""
from __future__ import annotations

import copy
import logging
import time
import types

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def _round_kept_up(self, n_pruned, current_channels, round_to):
    """Replacement for torch-pruning's BasePruner._round_to: keep ceil(kept / round_to) * round_to channels,
    so rounding can only prune LESS than requested (a layer smaller than round_to is left alone).

    Units: torch-pruning passes `n_pruned` in PER-channel-group units (len(dim_imp) ==
    current_channels // ch_groups) but `current_channels` as the layer's total width, so the real
    removal is n_pruned * ch_groups. `_cg_now` is recorded by _track_channel_groups(); with
    ch_groups == 1 this reduces exactly to kept = current_channels - n_pruned.
    """
    g = max(int(getattr(self, "_cg_now", 1) or 1), 1)
    kept = min(-(-(current_channels - n_pruned * g) // round_to) * round_to, current_channels)
    removed = max(current_channels - kept, 0)
    return (removed - removed % g) // g   # every channel group must keep the same size


def _track_channel_groups(pruner) -> None:
    """MetaPruner computes ch_groups while ranking but does not pass it to _round_to. Record it per
    (module, out/in-channel pruning) and note which module's width is being asked about."""
    cg, get_cg, DG = {}, pruner._get_channel_groups, pruner.DG

    def get_channel_groups(group):
        n = get_cg(group)
        dep = group[0].dep
        cg[(id(dep.target.module), DG.is_out_channel_pruning_fn(dep.handler))] = n
        return n

    pruner._get_channel_groups = get_channel_groups
    pruner._cg_now = 1
    for name, is_out in (("get_out_channels", True), ("get_in_channels", False)):
        fn = getattr(DG, name)

        def probe(module, _fn=fn, _is_out=is_out):
            pruner._cg_now = cg.get((id(module), _is_out), 1)
            return _fn(module)

        setattr(DG, name, probe)   # instance attribute: the pruner is discarded after prune_model


def _widths(model: nn.Module) -> dict:
    return {n: (m.out_channels if isinstance(m, nn.modules.conv._ConvNd) else m.out_features)
            for n, m in model.named_modules() if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear))}


def _importance(kind: str):
    import torch_pruning as tp

    imp = tp.importance
    group_mag = getattr(imp, "GroupMagnitudeImportance", None) or getattr(imp, "GroupNormImportance", None) \
        or imp.MagnitudeImportance
    if kind == "l1":
        return group_mag(p=1)
    if kind == "l2":
        return group_mag(p=2)
    if kind == "random":
        return imp.RandomImportance()
    if kind == "bn_scale":
        return imp.BNScaleImportance()
    raise ValueError(f"unknown importance {kind!r}")


def prune_model(model: nn.Module, example_input: torch.Tensor, ratio: float, ignored_layers=None,
                importance: str = "l2", round_to: int | None = 8, global_pruning: bool = False,
                iterative_steps: int = 1, inplace: bool = False, ratio_per_layer: dict | None = None,
                attention_mode: str = "head_dims", report: dict | None = None) -> nn.Module:
    """Remove `ratio` of the channels of every prunable layer group (per layer, unless global).

    Layer dependencies (residual adds, depthwise convs, concat) are resolved by torch-pruning's
    DependencyGraph, so the result is a valid, smaller dense model. Multi-head attention blocks
    (fused `qkv` Linear + `num_heads`) are detected automatically; attention_mode "head_dims"
    shrinks every head, "num_heads" removes whole heads. `report` (optional dict) receives
    params/MACs before and after and the largest per-layer fraction actually removed.

    `ratio_per_layer` maps a module NAME to its own ratio, overriding `ratio` for those layers. A
    uniform ratio treats every layer as equally redundant, which it is not; a single global
    importance threshold fixes that but trades compression away blindly (measured: it bought 0.34
    accuracy points and cost 1.21x of compression, with nothing controlling the exchange). An
    explicit allocation is how that trade gets decided on purpose - see allocate_ratios().
    """
    import torch_pruning as tp

    report = report if report is not None else {}
    if ratio <= 0 and not ratio_per_layer:
        # ratio 0 with a per-layer dict is the normal way to prune SOME layers only (the sensitivity
        # probe does exactly that), so the early return must not swallow it.
        return model if inplace else copy.deepcopy(model)
    m = model if inplace else copy.deepcopy(model)
    m = m.cpu().eval()
    x = example_input[:1].detach().cpu().float()
    ignored = list(ignored_layers or [])
    if not inplace:  # map ignored modules from the original model into the copy
        names = {id(mod): n for n, mod in model.named_modules()}
        ignored = [m.get_submodule(names[id(i)]) if id(i) in names else i for i in ignored]
    Pruner = getattr(tp.pruner, "MetaPruner", None) or tp.pruner.BasePruner
    kw = dict(importance=_importance(importance), iterative_steps=iterative_steps,
              ignored_layers=ignored, global_pruning=global_pruning)
    if round_to:
        kw["round_to"] = round_to
    unsupported = unsupported_attention(m)
    if unsupported:
        # e.g. attention with conv qkv + custom reshapes: torch-pruning can loop forever on these
        log.warning("not pruning %d attention block(s) of unsupported form: %s", len(unsupported),
                    sorted({type(u).__name__ for u in unsupported}))
        kw["ignored_layers"] = ignored + unsupported
    heads = attention_heads(m)
    if heads:  # multi-head attention: prune whole head dims consistently across q, k, v
        kw.update(num_heads=heads, prune_head_dims=attention_mode != "num_heads",
                  prune_num_heads=attention_mode == "num_heads", head_pruning_ratio=ratio)
        log.info("attention-aware pruning: %d attention blocks (%s)", len(heads), attention_mode)
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    widths0 = _widths(m)
    if ratio_per_layer:
        # torch-pruning keys this by MODULE, so resolve names against the copy we are pruning
        mods = dict(m.named_modules())
        missing = [n for n in ratio_per_layer if n not in mods]
        if missing:
            log.warning("ratio_per_layer names not in the model (ignored): %s", missing[:5])
        kw["pruning_ratio_dict"] = {mods[n]: float(r) for n, r in ratio_per_layer.items() if n in mods}
        report["n_layer_ratios"] = len(kw["pruning_ratio_dict"])
    try:
        pruner = Pruner(m, x, pruning_ratio=ratio, **kw)
    except TypeError:  # older torch-pruning used ch_sparsity
        kw.pop("pruning_ratio_dict", None)
        pruner = Pruner(m, x, ch_sparsity=ratio, **kw)
    if round_to:
        try:
            _track_channel_groups(pruner)
            pruner._round_to = types.MethodType(_round_kept_up, pruner)
        except Exception as e:  # unknown torch-pruning internals: no rounding beats over-pruning
            log.warning("cannot install the round_to guard (%s); pruning without round_to", e)
            pruner.round_to = None
    base_macs, base_params = tp.utils.count_ops_and_params(m, x)
    for _ in range(iterative_steps):
        pruner.step()
        sync_attention(m, pruner)
    macs, params = tp.utils.count_ops_and_params(m, x)
    widths1 = _widths(m)
    fracs = {n: 1 - widths1[n] / w for n, w in widths0.items() if n in widths1 and w}
    worst = max(fracs, key=fracs.get) if fracs else None
    log.info("pruned ratio=%.2f: params %.2fM -> %.2fM, MACs %.1fM -> %.1fM, largest layer cut %.0f%% (%s)",
             ratio, base_params / 1e6, params / 1e6, base_macs / 1e6, macs / 1e6,
             100 * fracs.get(worst, 0), worst)
    achieved = 1 - params / base_params if base_params else 0.0
    if achieved < 0.5 * ratio:  # e.g. rounding skipped whole groups: never let that pass silently
        log.warning("prune ratio=%.2f removed only %.1f%% of the parameters (round_to=%s): some layer "
                    "groups were skipped", ratio, 100 * achieved, round_to)
    if report is not None:
        report.update(params_before=int(base_params), params_after=int(params), macs_before=int(base_macs),
                      macs_after=int(macs), max_layer_cut=round(fracs.get(worst, 0.0), 4))
    for n, p in m.named_parameters():  # pruned layers get fresh Parameters; keep frozen ones frozen
        p.requires_grad_(n not in frozen)
    return m


@torch.no_grad()
def recalibrate_bn(model: nn.Module, loader, device: str | None = None, max_batches: int = 32) -> int:
    """Re-estimate BatchNorm running statistics on unlabelled data (cumulative average, no gradients).

    Pruning removes channels whose mean activation every downstream BN was calibrated against;
    fresh statistics recover much of the loss before (or instead of) fine-tuning.
    """
    bns = [mod for mod in model.modules() if isinstance(mod, nn.modules.batchnorm._BatchNorm)
           and mod.track_running_stats]
    if not bns or max_batches <= 0:
        return 0
    dev = device or next(model.parameters()).device
    model.to(dev).eval()
    momenta, n = {}, 0
    try:
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(dev, dtype=torch.float32)
            if not momenta:  # only reset once we know data is actually coming
                for bn in bns:
                    momenta[bn] = (bn.momentum, bn.running_mean.clone(), bn.running_var.clone(),
                                   bn.num_batches_tracked.clone())
                    bn.reset_running_stats()
                    bn.momentum = None  # cumulative moving average over all calibration batches
                    bn.train()
            try:
                model(x)
                n += 1
            except ValueError as e:  # e.g. a trailing batch with a single sample per channel
                log.debug("skipping BN calibration batch %d: %s", i, e)
    finally:
        for bn, (m, mean, var, cnt) in momenta.items():
            bn.momentum = m
            bn.eval()
            if n == 0:  # nothing was collected: never leave the model with reset (0/1) statistics
                bn.running_mean.copy_(mean)
                bn.running_var.copy_(var)
                bn.num_batches_tracked.copy_(cnt)
    if n == 0:
        log.warning("BN re-calibration got no usable batches; running statistics left unchanged")
    return n


def _attention_blocks(model: nn.Module):
    """Attention modules with a fused qkv projection and an int `num_heads` (timm/ViT style)."""
    for mod in model.modules():
        qkv = getattr(mod, "qkv", None)
        if isinstance(qkv, nn.Linear) and isinstance(getattr(mod, "num_heads", None), int):
            yield mod, qkv


def attention_heads(model: nn.Module) -> dict:
    return {qkv: mod.num_heads for mod, qkv in _attention_blocks(model)}


def unsupported_attention(model: nn.Module) -> list:
    """Modules that look like multi-head attention (`num_heads`) but not the supported fused-Linear-qkv
    form. They are excluded from pruning (their surroundings are still pruned)."""
    supported = {id(mod) for mod, _ in _attention_blocks(model)}
    return [mod for mod in model.modules()
            if isinstance(getattr(mod, "num_heads", None), int) and id(mod) not in supported
            and not isinstance(mod, nn.MultiheadAttention)]


def sync_attention(model: nn.Module, pruner) -> None:
    """After pruning, update num_heads / head_dim / attn_dim so the blocks' reshapes match the new
    qkv width (only attributes change, never forward(), so pickling and deepcopy keep working).

    The softmax scale follows the new head_dim (1/sqrt(head_dim)): that is what the fused
    scaled_dot_product_attention path (timm's default, and the ONNX export) computes anyway, so
    setting `scale` keeps the non-fused path identical to it.
    """
    for mod, qkv in _attention_blocks(model):
        mod.num_heads = int(getattr(pruner, "num_heads", {}).get(qkv, mod.num_heads))
        if hasattr(mod, "head_dim"):
            mod.head_dim = qkv.out_features // (3 * mod.num_heads)
            if isinstance(getattr(mod, "scale", None), float):
                mod.scale = mod.head_dim ** -0.5
        if hasattr(mod, "attn_dim"):
            mod.attn_dim = mod.num_heads * mod.head_dim


def prune_unstructured(model: nn.Module, ratio: float, ignored_layers=None, global_pruning: bool = True,
                       inplace: bool = False, keep_mask: bool = False,
                       report: dict | None = None) -> nn.Module:
    """Zero the `ratio` smallest-magnitude weights of every Conv/Linear, via torch.nn.utils.prune.

    Shapes do not change, so this buys nothing on latency or peak RAM on a CPU runtime without
    sparse kernels - it buys file size after compression, visible in `size_bytes_gz` rather than
    `size_bytes`.

    `keep_mask=True` leaves torch's re-parameterisation in place so the zeros SURVIVE fine-tuning;
    call `finalize_unstructured` afterwards to bake them in. Baking them in first and then
    fine-tuning silently undoes the pruning - every zeroed weight gets a non-zero gradient and
    drifts off zero. Measured: asking for 50/80/90% sparsity that way left 0.11/0.25/2.36% actual
    zeros in the exported model, i.e. no compression benefit at all.
    """
    import torch.nn.utils.prune as tprune

    report = report if report is not None else {}
    m = model if inplace else copy.deepcopy(model)
    m = m.cpu().eval()
    if ratio <= 0:
        return m
    # Map the ignored modules onto the COPY by name: id() from the caller's model never matches a
    # deepcopy, so this silently pruned layers the adapter asked to protect (the classifier head).
    names = {id(mod): n for n, mod in model.named_modules()}
    skip = {names[id(x)] for x in (ignored_layers or []) if id(x) in names}
    targets = [(mod, "weight") for n, mod in m.named_modules()
               if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)) and n not in skip]
    if not targets:
        log.warning("unstructured pruning found no Conv/Linear layer to prune")
        return m
    if global_pruning:      # one threshold across all layers: lets robust layers keep more weights
        tprune.global_unstructured(targets, pruning_method=tprune.L1Unstructured, amount=float(ratio))
    else:
        for mod, name in targets:
            tprune.l1_unstructured(mod, name, amount=float(ratio))
    if not keep_mask:
        for mod, name in targets:
            tprune.remove(mod, name)        # bake the mask in: no mask buffers in the exported graph
    total = sum(mod.weight.numel() for mod, _ in targets)
    zeros = sum(int((mod.weight == 0).sum()) for mod, _ in targets)
    report["sparsity"] = zeros / max(total, 1)
    report["pruned_layers"] = len(targets)
    log.info("unstructured pruning ratio=%.2f: %.1f%% of %d Conv/Linear weights are now zero "
             "(shapes unchanged: file size only, no latency or RAM change)%s",
             ratio, 100 * report["sparsity"], total, " [mask kept for fine-tuning]" if keep_mask else "")
    return m


def finalize_unstructured(model: nn.Module, report: dict | None = None) -> nn.Module:
    """Bake any torch.nn.utils.prune masks into the weights and drop the re-parameterisation.

    Must run AFTER fine-tuning (see prune_unstructured): while the mask is live the zeros are
    re-applied on every forward, which is what keeps them zero through gradient updates. Without
    this call the exported graph would carry weight_orig/weight_mask buffers instead of weights.
    """
    import torch.nn.utils.prune as tprune

    report = report if report is not None else {}
    done = 0
    for mod in model.modules():
        for name in ("weight", "bias"):
            if tprune.is_pruned(mod) and hasattr(mod, f"{name}_mask"):
                tprune.remove(mod, name)
                done += 1
    if done:
        tot = z = 0
        for mod in model.modules():
            if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                tot += mod.weight.numel()
                z += int((mod.weight == 0).sum())
        report["sparsity"] = z / max(tot, 1)
        log.info("baked in %d unstructured mask(s): %.1f%% of weights are zero after fine-tuning",
                 done, 100 * report["sparsity"])
    return model


def _out_sqnr_db(ref: list, cand: list) -> float:
    """Mean per-output-channel SQNR (dB) between two lists of output tensors. Label-free."""
    vals = []
    for a, b in zip(ref, cand):
        a, b = a.detach().float(), b.detach().float()
        if a.shape != b.shape:
            return -20.0        # shape changed: the probe is meaningless, treat as maximal damage
        ax = a.reshape(a.shape[0], a.shape[1], -1) if a.dim() > 2 else a.reshape(a.shape[0], -1, 1)
        bx = b.reshape(ax.shape)
        sig = (ax ** 2).mean(dim=(0, 2))
        noise = ((ax - bx) ** 2).mean(dim=(0, 2))
        keep = sig > sig.max() * 1e-6 if sig.numel() else sig.bool()
        if bool(keep.any()):
            vals.append(float((10 * torch.log10(sig[keep] / noise[keep].clamp_min(1e-20))).mean()))
    if not vals:
        return float("nan")
    return float(np.clip(np.mean(vals), -20, 60))


def _as_out_list(y):
    if isinstance(y, (list, tuple)):
        return [t for t in y if isinstance(t, torch.Tensor)]
    return [y]


def layer_sensitivity(adapter, model: nn.Module, cfg: dict | None = None,
                      report: dict | None = None) -> dict:
    """How much does pruning EACH layer alone hurt, and what does it save? Label-free.

    For every prunable layer: prune only that layer at `probe_ratio`, run the calibration batches,
    and compare the outputs against the unpruned model's - mean per-output-channel SQNR, the same
    proxy the quantization sensitivity search uses. Also records the parameter fraction that cut
    actually removed, which is NOT the layer's own size: channel pruning propagates into the next
    layer's inputs, so the saving has to be measured rather than assumed.

    Per layer: {"quality": higher is better, "param_frac", "mac_frac", and "damage_db" when SQNR
    was the proxy}.

    cfg probe_metric selects the proxy:
      "sqnr"   (default) output SQNR. Label-free, so it works on whatever is handed over, and cheap.
      "metric" adapter.evaluate() per layer. Costs one evaluation per layer, but measures the thing
               actually being optimised.

    CORRECTION, 2026-09-16. This docstring previously claimed SQNR does not track the task metric,
    citing a correlation of +0.395 on a detector. That number came from SIX HAND-PICKED layers and
    does not survive proper sampling. Re-measured with 16 layers drawn evenly across the probe's
    own score range (scripts/paper/sqnr_vs_metric.py):

        image-label task   top-1    n=16  Spearman +0.688  p=0.003
        object-box task    mean AP  n=16  Spearman +0.703  p=0.002

    SQNR ranks single-layer damage well on both tasks, and slightly BETTER on the detector. So
    "metric" is not a fix for a broken proxy; it is simply a more direct and far more expensive
    measurement. The open question it does not answer is why allocation built on this proxy still
    loses to uniform pruning (-2.72 vs -2.40 pt at ratio 0.25): per-layer damage predicts per-layer
    damage, but it may not be additive across layers.
    """
    cfg = cfg or {}
    report = report if report is not None else {}
    probe_ratio = float(cfg.get("probe_ratio", 0.25))
    n_batches = int(cfg.get("probe_batches", 2))
    proxy = str(cfg.get("probe_metric", "sqnr")).lower()
    if proxy not in ("sqnr", "metric"):
        raise ValueError(f"unknown prune.probe_metric {proxy!r} (sqnr | metric)")
    x = adapter.example_input()
    batches = []
    try:
        loader = adapter.bn_calibration_loader()
    except Exception:
        loader = None
    for _, b in zip(range(n_batches), loader or []):
        t = b[0] if isinstance(b, (list, tuple)) else b
        if isinstance(t, torch.Tensor):
            batches.append(t.detach().float())
    if not batches:
        batches = [x]

    # work on a copy: model.cpu().eval() would move the CALLER's model off the GPU and leave it in
    # eval mode, which silently slows every later epoch and changes BN/dropout behaviour
    was_training = model.training
    base = copy.deepcopy(model).cpu().eval()
    ref = None
    if proxy == "sqnr":
        with torch.no_grad():
            ref = [_as_out_list(base(b)) for b in batches]
    total = sum(p.numel() for p in base.parameters())
    base_macs = 0
    try:
        import torch_pruning as tp
        base_macs, _ = tp.utils.count_ops_and_params(base, x)
    except Exception:
        pass
    ign = adapter.ignored_layers(base) or []
    ignored = {id(m) for m in ign}
    names = [n for n, m in base.named_modules()
             if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear)) and id(m) not in ignored]

    t0, out = time.time(), {}
    for name in names:
        try:
            # ratio 0 everywhere, probe_ratio on this layer alone
            cand = prune_model(base, x, 0.0, ign, importance=cfg.get("importance", "l2"),
                               round_to=cfg.get("round_to", 8),
                               ratio_per_layer={name: probe_ratio}, report={})
            # Re-estimate BN before scoring. Without it the probe measures a model whose BatchNorm
            # statistics belong to channels that no longer exist, not the damage pruning did: the
            # metric probe read 0.008 and 0.034 on two layers whose real post-recalibration scores
            # were 0.172 and 0.222. It is label-free and a few batches, so it costs almost nothing
            # relative to being wrong by 20x.
            if cfg.get("probe_bn_recalibrate", True):
                recalibrate_bn(cand, adapter.bn_calibration_loader(), "cpu",
                               int(cfg.get("probe_bn_batches", 8)))
            got = None
            if proxy == "sqnr":
                with torch.no_grad():
                    got = [_as_out_list(cand(b)) for b in batches]
        except Exception as e:      # a layer torch-pruning cannot touch is simply not a candidate
            log.debug("sensitivity probe skipped %s: %s", name, e)
            continue
        saved = 1.0 - sum(p.numel() for p in cand.parameters()) / max(total, 1)
        if saved <= 0:              # rounding kept the layer whole: nothing to allocate here
            continue
        # MACs as well as parameters: they are NOT the same budget. Parameter-dense layers sit late,
        # where the spatial maps are small, so a parameter-optimal plan can barely move MACs -
        # measured, 23.6% of parameters for 9% of MACs. Parameters track file size, MACs track
        # latency, so the caller has to be able to choose (cfg alloc_cost).
        mac_saved = 0.0
        try:
            import torch_pruning as tp
            m1, _ = tp.utils.count_ops_and_params(cand, x)
            mac_saved = max(1.0 - m1 / max(base_macs, 1), 0.0)
        except Exception:
            pass
        if proxy == "metric":
            try:
                q = float(adapter.evaluate(cand)["score"])
            except Exception as e:
                log.debug("metric probe failed on %s: %s", name, e)
                continue
            out[name] = {"quality": q, "param_frac": saved, "mac_frac": mac_saved}
        else:
            dmg = float(np.mean([_out_sqnr_db(r, g) for r, g in zip(ref, got)]))
            out[name] = {"quality": dmg, "damage_db": dmg, "param_frac": saved,
                         "mac_frac": mac_saved}
    if was_training:
        model.train()
    report["sensitivity_layers"] = len(out)
    report["sensitivity_s"] = round(time.time() - t0, 1)
    report["sensitivity_proxy"] = proxy
    log.info("layer sensitivity: %d layer(s) probed at ratio %.2f in %.0fs (proxy: %s%s)",
             len(out), probe_ratio, time.time() - t0, proxy,
             ", label-free" if proxy == "sqnr" else ", needs labels")
    return out


def allocate_ratios(adapter, model: nn.Module, target: float, cfg: dict | None = None,
                    report: dict | None = None) -> dict:
    """Choose a per-layer ratio vector reaching `target` parameter reduction for the least damage.

    Two knapsack-style solvers over discretised per-layer levels, both driven by the label-free
    sensitivity probe:

      greedy  repeatedly hand the next level to whichever layer offers the most parameters saved per
              dB of damage. Cheap, and usually within noise of the optimum.
      dp      exact on the discretised problem: minimise summed damage subject to reaching the
              parameter budget, over levels x budget buckets.

    Neither is "better pruning" by itself. The point is that the accuracy/compression exchange gets
    DECIDED instead of falling out of one importance threshold - which was measured buying 0.34
    accuracy points while costing 1.21x of compression, with nothing controlling the trade.
    """
    cfg = cfg or {}
    report = report if report is not None else {}
    sens = cfg.get("sensitivity") or layer_sensitivity(adapter, model, cfg, report)
    if not sens:
        log.warning("no usable sensitivity data: falling back to the uniform ratio")
        return {}
    levels = [float(v) for v in cfg.get("alloc_levels", [0.0, 0.125, 0.25, 0.375, 0.5])]
    method = str(cfg.get("allocate", "greedy")).lower()
    probe = max(float(cfg.get("probe_ratio", 0.25)), 1e-9)

    # `quality` is higher-is-better whichever proxy produced it, so cost = how far below the best
    # layer it sits. That keeps the solvers identical for SQNR and for a task metric.
    def _q(v):
        return v.get("quality", v.get("damage_db", float("nan")))

    finite = [_q(v) for v in sens.values() if _q(v) == _q(v)]
    if not finite:
        return {}
    best_db = max(finite)
    cost_key = {"params": "param_frac", "macs": "mac_frac"}.get(
        str(cfg.get("alloc_cost", "params")).lower())
    if cost_key is None:
        raise ValueError(f"unknown prune.alloc_cost {cfg.get('alloc_cost')!r} (params | macs)")
    items = []
    for name, v in sens.items():
        db = _q(v)
        cost = v.get(cost_key, 0.0)
        if db != db or cost <= 0:
            continue
        hurt = max(best_db - db, 0.0)
        items.append((name, [(lv, cost * lv / probe, hurt * lv / probe) for lv in levels]))
    if not items:
        log.warning("no layer offers any %s saving: falling back to the uniform ratio", cost_key)
        return {}

    # Per-layer savings are NOT additive: pruning a layer's outputs already shrinks the next layer's
    # inputs, so summing the probe measurements overestimates (measured: a 25% target estimated at
    # 25.2-28.7% removed 18.2% for real). So allocate, prune, MEASURE, and if short, aim higher.
    solve = _alloc_dp if method == "dp" else _alloc_greedy
    kw = {"buckets": int(cfg.get("alloc_buckets", 200))} if method == "dp" else {}
    total = sum(p.numel() for p in model.parameters())
    x = adapter.example_input()
    base_macs = 0
    if cost_key == "mac_frac":
        try:
            import torch_pruning as tp
            base_macs, _ = tp.utils.count_ops_and_params(copy.deepcopy(model).cpu().eval(), x)
        except Exception:
            cost_key = "param_frac"         # cannot count MACs on this model: spend parameters
            log.warning("cannot count MACs on this model; allocating against parameters instead")
    ign = adapter.ignored_layers(model) or []
    rounds = max(int(cfg.get("alloc_rounds", 4)), 1)
    tol = float(cfg.get("alloc_tol", 0.01))
    aim, alloc, actual, history = target, {}, 0.0, []
    for it in range(rounds):
        cand = solve(items, min(aim, 0.95), **kw)
        cand = {n: r for n, r in cand.items() if r > 0}
        if not cand:
            break
        try:
            pruned = prune_model(model, x, 0.0, ign, importance=cfg.get("importance", "l2"),
                                 round_to=cfg.get("round_to", 8), ratio_per_layer=cand, report={})
            if cost_key == "mac_frac":
                import torch_pruning as tp
                m1, _ = tp.utils.count_ops_and_params(pruned, x)
                got = max(1.0 - m1 / max(base_macs, 1), 0.0)
            else:
                got = 1.0 - sum(p.numel() for p in pruned.parameters()) / max(total, 1)
        except Exception as e:
            log.warning("allocation round %d could not be realised (%s); keeping the previous plan", it, e)
            break
        history.append(round(got, 4))
        # keep the plan CLOSEST to the target, not the biggest: overshooting throws away accuracy
        # for compression nobody asked for (greedy reached 35% against a 25% target this way).
        if not alloc or abs(got - target) < abs(actual - target):
            alloc, actual = cand, got
        if abs(got - target) <= tol:
            break
        aim *= target / max(got, 1e-6)          # scale the ask by how far the real prune fell short
    cost_name = {"param_frac": "params", "mac_frac": "macs"}[cost_key]
    report.update({"alloc_method": method, "alloc_cost": cost_name,
                   # named for the axis it actually measured; alloc_param_frac was reporting a MAC
                   # fraction under a parameter-shaped name whenever alloc_cost was macs
                   "alloc_frac": round(actual, 4), "alloc_param_frac": round(actual, 4),
                   "alloc_target": target, "alloc_rounds": len(history),
                   "alloc_trace": ", ".join(f"{v:.3f}" for v in history),
                   "alloc_layers": int(sum(1 for r in alloc.values() if r > 0))})
    log.info("allocation (%s): %d/%d layer(s), %.1f%% of parameters actually removed "
             "(target %.1f%%, %d round(s): %s)", method, report["alloc_layers"], len(items),
             100 * actual, 100 * target, len(history), report["alloc_trace"])
    return alloc


def _alloc_greedy(items: list, target: float) -> dict:
    """Hand out levels in order of parameters-saved per dB of damage until the budget is reached."""
    alloc = {n: 0.0 for n, _ in items}
    cur = 0.0
    while cur < target:
        best, best_gain = None, -1.0
        for name, opts in items:
            here = alloc[name]
            nxt = next((o for o in opts if o[0] > here), None)
            if nxt is None:
                continue
            cur_s = next((s for lv, s, _ in opts if lv == here), 0.0)
            cur_h = next((h for lv, _, h in opts if lv == here), 0.0)
            d_s, d_h = nxt[1] - cur_s, nxt[2] - cur_h
            if d_s <= 0:
                continue
            gain = d_s / (d_h + 1e-6)       # parameters per dB; harmless layers go first
            if gain > best_gain:
                best, best_gain = (name, nxt[0], d_s), gain
        if best is None:
            break
        alloc[best[0]] = best[1]
        cur += best[2]
    return alloc


def _alloc_dp(items: list, target: float, buckets: int) -> dict:
    """Least total damage that still reaches the budget, exact on the discretised grid."""
    buckets = max(int(buckets), 1)
    step = target / buckets if target > 0 else 0.0
    INF = float("inf")
    dp = [INF] * (buckets + 1)
    dp[0] = 0.0
    choice = [{} for _ in range(buckets + 1)]
    for name, opts in items:
        nxt = [INF] * (buckets + 1)
        nxt_choice = [{} for _ in range(buckets + 1)]
        for b in range(buckets + 1):
            if dp[b] == INF:
                continue
            for lv, saved, hurt in opts:
                nb = min(buckets, b + int(round(saved / step))) if step > 0 else buckets
                if dp[b] + hurt < nxt[nb]:
                    nxt[nb] = dp[b] + hurt
                    nxt_choice[nb] = {**choice[b], name: lv}
        dp, choice = nxt, nxt_choice
    if dp[buckets] < INF:
        return choice[buckets]
    reachable = [b for b in range(buckets + 1) if dp[b] < INF]
    return choice[max(reachable)] if reachable else {}


def prune_and_finetune(adapter, model: nn.Module, ratio: float, epochs: int, cfg: dict | None = None,
                       finetune_fn=None, report: dict | None = None) -> nn.Module:
    """Prune a copy of `model`, re-calibrate BN (cfg bn_recalibrate, default on), then recover accuracy
    with adapter.finetune (or a custom fn, e.g. KD)."""
    cfg = cfg or {}
    report = report if report is not None else {}
    mode = cfg.get("mode", "structured")
    if mode not in ("structured", "unstructured"):
        raise ValueError(f"unknown prune.mode {mode!r} (structured | unstructured)")
    report["prune_mode"] = mode
    # bound before the branch: `did_prune` below reads it on every path, and short-circuiting on
    # `ratio > 0` hid that from every test until a ratio-0 (baseline) unstructured variant ran.
    alloc = None
    if mode == "unstructured":
        # keep the mask while fine-tuning runs, then bake it in below
        pruned = prune_unstructured(model, ratio, adapter.ignored_layers(model),
                                    global_pruning=cfg.get("global_pruning", True),
                                    keep_mask=epochs > 0, report=report)
    else:
        alloc = cfg.get("ratio_per_layer")
        planned = False
        # the probe costs one prune (and one evaluation, with probe_metric: metric) per layer, and
        # ratio 0 means "the baseline" - there is nothing to allocate
        if cfg.get("allocate") and not alloc and ratio > 0:
            alloc = allocate_ratios(adapter, model, ratio, cfg, report)
            planned = bool(alloc)
        # A PLAN is complete: a layer it left out was deliberately left whole, and allocate_ratios
        # measured it that way (uniform ratio 0). Realising it with the uniform `ratio` instead
        # prunes every unlisted layer as well - measured, a plan for 22.3% of parameters exported a
        # model with 40.1% removed, and a shipped run planned for 19.7% removed 49.5%. A dict the
        # USER wrote in ratio_per_layer still layers on top of `ratio`, which is what that knob means.
        eff_ratio = 0.0 if planned else ratio
        pruned = prune_model(model, adapter.example_input(), eff_ratio, adapter.ignored_layers(model),
                             importance=cfg.get("importance", "l2"), round_to=cfg.get("round_to", 8),
                             global_pruning=cfg.get("global_pruning", False),
                             iterative_steps=cfg.get("iterative_steps", 1), ratio_per_layer=alloc,
                             attention_mode=cfg.get("attention_mode", "head_dims"), report=report)
    # `ratio > 0` is the wrong test for "did we prune": a per-layer plan (or a user dict) prunes with
    # ratio 0, and skipping BN re-estimation there leaves statistics describing removed channels -
    # the exact defect that made the sensitivity probe wrong by 20x.
    did_prune = ratio > 0 or bool(alloc)
    if did_prune and cfg.get("bn_recalibrate", True):
        n = recalibrate_bn(pruned, adapter.bn_calibration_loader(), adapter.device,
                           int(cfg.get("bn_recal_batches", 32)))
        report["bn_recal_batches"] = n
    if did_prune and epochs > 0:
        fn = finetune_fn or adapter.finetune
        # Repairing a damaged model and stabilising a good one want opposite learning rates. An
        # adapter's finetune_lr is usually tuned for the second job - small enough not to disturb a
        # model that is already fine - and reusing it for the first leaves the pruned model unable
        # to move: in the study behind this (see "pruning recovery" in README.md) the loss
        # converged over the last epoch, 4.286 -> 4.282, at a metric far below the baseline. It was
        # not short of epochs. recovery_lr_mult scales the LR for this call only; 1.0 keeps the
        # adapter's own value.
        mult = float(cfg.get("recovery_lr_mult", 1.0) or 1.0)
        lr = None
        if mult != 1.0:
            # A KD recovery brings its own base LR (distill.lr, published as fn.base_lr); a plain
            # one uses the adapter's finetune_lr. Scaling the adapter's value on the KD path would
            # multiply a number that path never uses, and skipping KD entirely - which is what the
            # old `not finetune_fn` guard did - made the knob silently do nothing there.
            base = getattr(fn, "base_lr", None) if finetune_fn else adapter.cfg.get("finetune_lr")
            if base:
                lr = float(base) * mult
                report["recovery_lr"] = lr
                log.info("recovery LR %.2g (%.3gx the adapter's finetune_lr %.2g)", lr, mult, float(base))
        pruned = fn(pruned, epochs) if lr is None else fn(pruned, epochs, lr)
    if mode == "unstructured":
        finalize_unstructured(pruned, report)      # no-op when no mask was kept
    return pruned.eval()
