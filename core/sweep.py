"""Parameter sweep: prune ratio x precision -> prune, fine-tune, export, quantize, measure.

One failing combination is logged in the `error` column and the sweep continues.
Rows are appended to sweep.csv as soon as they are measured (crash-safe, resumable).
"""
from __future__ import annotations

import copy
import gc
import json
import logging
import random
import re
import time
import traceback
from pathlib import Path

import shutil

import pandas as pd
import torch

from . import provenance
from .baseline import save_model
from .config import set_seed
from .distill import distill
from .export import export_onnx
from .measure import env_info, host_peak_rss_mb, measure
from .model_io import capabilities, convert_tflite_to_onnx
from .pareto import pareto_mask, plot_pareto
from .prune import prune_and_finetune
from .qat import qat_finetune
from .quantize import quantize_onnx
from .score import candidates, score_table

log = logging.getLogger(__name__)


def _precisions(sw: dict) -> list[dict]:
    """Normalise precision entries: "fp32" | "int8" | "int8_dynamic" | {name, method, keep_fp32, ...}."""
    out = []
    for p in sw.get("precisions", ["fp32", "int8"]):
        if isinstance(p, str):
            p = {"name": p}
        p = dict(p)
        p.setdefault("method", {"fp32": "none", "int8": "static", "int8_dynamic": "dynamic"}.get(p["name"], "static"))
        out.append(p)
    return out


def ok_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that were actually measured (no error, not skipped)."""
    ok = df["error"].isna() if "error" in df else pd.Series(True, index=df.index)
    if "skipped" in df:
        ok &= df["skipped"].isna()
    return ok


def variant_name(ratio: float, prec: str) -> str:
    return f"p{round(ratio * 100):02d}_{prec}"


def _finetune_fn(cfg: dict, adapter, teacher):
    if cfg.get("sweep", {}).get("finetune_mode", "plain") != "distill":
        return None
    kd = cfg.get("distill", {})

    every = int(getattr(adapter, "cfg", {}).get("eval_every", 0) or 0)
    ev = (lambda m: adapter.evaluate(m)["score"]) if every else None

    def fn(model, epochs, lr=None):
        # `lr` exists so this matches adapter.finetune's signature; recovery_lr_mult scales the KD
        # learning rate the same way it scales a plain one instead of being silently ignored.
        return distill(teacher, model, adapter, epochs, temperature=kd.get("temperature", 4.0),
                       alpha=kd.get("alpha", 0.7), lr=lr if lr is not None else kd.get("lr", 0.01),
                       optimizer=kd.get("optimizer", "sgd"),
                       max_steps_per_epoch=getattr(adapter, "cfg", {}).get("max_steps_per_epoch"),
                       history=getattr(adapter, "history", None), eval_fn=ev, eval_every=every)
    fn.base_lr = float(kd.get("lr", 0.01))   # what recovery_lr_mult scales on this path
    return fn


def _source_fp32(baseline, fmt: str, models_dir: Path, cfg: dict, adapter, fp32_name: str) -> Path | None:
    """Deployable, not-yet-quantized file for a non-prunable baseline (TorchScript / ONNX / TFLite)."""
    ex = cfg.get("export", {})
    if fmt in ("torchscript", "torch_export"):
        return export_onnx(baseline, adapter.example_input(), models_dir / f"{fp32_name}.onnx",
                           input_names=adapter.input_names(), output_names=adapter.output_names(),
                           opset=ex.get("opset", 17), atol=ex.get("atol", 1e-3), rtol=ex.get("rtol", 1e-3))
    src = Path(baseline)
    dst = models_dir / f"{fp32_name}{src.suffix}"
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return dst


def run_sweep(cfg: dict, adapter, baseline, out_dir: str | Path, resume: bool = False,
              prefix: str = "", finalize_result: bool = True) -> pd.DataFrame:
    """baseline: nn.Module (full flow), TorchScript, or a path to .onnx / .tflite.

    Non-prunable baselines skip prune ratios > 0 (rows marked `skipped`, not `error`).
    TFLite: measured as given; INT8 variants go through ONNX only if tf2onnx is installed.
    """
    target = cfg.get("target_runtime", "onnx")
    if target != "onnx":
        # Reserved: a TFLite deployment target needs the TFLite converter as quantizer and LiteRT
        # for every measurement, so the whole pipeline switches together. Not built until the
        # organisers confirm the required submission format.
        raise NotImplementedError(f"target_runtime={target!r} is reserved; only 'onnx' is implemented")
    out_dir = Path(out_dir)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"
    sw, ex, qcfg = cfg.get("sweep", {}), cfg.get("export", {}), cfg.get("quantize", {})
    precs = _precisions(sw)
    ratios = [float(r) for r in sw.get("prune_ratios", [0.0])]
    epochs = int(sw.get("finetune_epochs", 1))
    caps = capabilities(baseline)
    fmt = caps["format"]
    log.info("baseline format: %s (prune=%s)", fmt, caps["prune"])

    rows: list[dict] = []
    if resume and csv_path.exists():
        prev = pd.read_csv(csv_path)
        keep = prev["error"].isna() if "error" in prev else pd.Series(True, index=prev.index)
        if "skipped" in prev:
            keep &= prev["skipped"].isna()
        # Only this variant's own rows. run_sweep_grid calls this once per adapter_variant against
        # the SAME sweep.csv and concatenates the results, so loading every row here duplicated the
        # whole file once per variant - and the duplication compounded on each further resume.
        # Variant rows are distinguished by the prefix their names carry.
        if prefix:
            keep &= prev["variant"].astype(str).str.startswith(prefix)
        elif "_adapter_variant" not in cfg:
            pass                      # single-variant sweep: every row belongs to it
        rows = prev[keep].to_dict("records")
    done = {r["variant"] for r in rows}

    def flush():
        """Write this call's rows WITHOUT dropping rows another call already put in the file.

        run_sweep_grid calls run_sweep once per adapter variant and once per student, all against
        the same sweep.csv, and only concatenates at the very end. Overwriting here meant the file
        on disk held just the variant being measured: a crash during the fourth student threw away
        the first three, and `resume` then re-ran them because it could not see them either.
        """
        df = pd.DataFrame(rows)
        if csv_path.exists():
            try:
                prev = pd.read_csv(csv_path)
                keep = prev[~prev["variant"].astype(str).isin(set(df["variant"]) if len(df) else set())]
                if prefix:          # also drop this prefix's stale rows, e.g. an errored retry
                    keep = keep[~keep["variant"].astype(str).str.startswith(prefix)]
                df = pd.concat([keep, df], ignore_index=True)
            except Exception as e:  # a corrupt or partial file must not stop the sweep
                log.debug("could not merge previous %s: %s", csv_path.name, e)
        df.to_csv(csv_path, index=False)

    def vname(ratio, p):
        return f"{prefix}{variant_name(ratio, p)}"

    measure_order: dict = {}    # variant name -> position in the randomised measurement sequence

    def base_row(ratio, p):
        row = {"variant": vname(ratio, p["name"]), "prune_ratio": ratio, "precision": p["name"],
               "source_format": fmt}
        # Recorded so the shuffle below is auditable. An unrecorded randomisation trades one kind of
        # irreproducibility for another, and this column is what lets a reader check that a
        # suspicious result is not simply the variant that always ran last.
        row["measure_order"] = measure_order.get(row["variant"])
        row.update(adapter_variant)
        return row

    adapter_variant = dict(cfg.get("_adapter_variant") or {})   # e.g. {"img_size": 256}: a sweep dimension
    data = adapter.load_data()
    for ratio in ratios:
        todo = [p for p in precs if vname(ratio, p["name"]) not in done]
        # Randomise measurement order. Without this, a variant's position in time is perfectly
        # determined by its position in the config, so thermal drift and cache state are confounded
        # with the variant itself - measured on this repo as a 7.30% "difference" at 3.9 sigma
        # between two plan-identical models, which collapsed to -0.72% under randomised order.
        # Seeded from the config seed so the sequence is still exactly reproducible.
        # int seed: random.Random rejects a tuple, and a tuple is what I first wrote here -
        # it passed ast.parse and a measure.py smoke test and then failed every sweep in the
        # batch. Mixing the ratio in keeps each ratio's order independent but reproducible.
        random.Random(int(cfg.get("seed", 0)) * 100003 + round(ratio * 1000)).shuffle(todo)
        measure_order.update({vname(ratio, p["name"]): i for i, p in enumerate(todo)})
        if not todo:
            continue
        if ratio > 0 and not caps["prune"]:
            for p in todo:
                rows.append({**base_row(ratio, p), "skipped": f"pruning needs a PyTorch nn.Module (got {fmt})"})
            flush()
            continue
        set_seed(cfg.get("seed", 0))
        t0 = time.time()
        model, onnx_src, prune_rep = None, None, {}
        if getattr(adapter, "history", None) is not None:
            adapter.history.clear()      # per-variant curve, not cumulative across the grid
        try:
            if fmt == "torch":
                model = prune_and_finetune(adapter, baseline, ratio, epochs, cfg.get("prune"),
                                           finetune_fn=_finetune_fn(cfg, adapter, baseline), report=prune_rep)
                if ratio > 0:
                    save_model(copy.deepcopy(model), models_dir / f"{prefix}p{round(ratio * 100):02d}.pt")
                fp32 = export_onnx(model, adapter.example_input(), models_dir / f"{vname(ratio, 'fp32')}.onnx",
                                   input_names=adapter.input_names(), output_names=adapter.output_names(),
                                   opset=ex.get("opset", 17), atol=ex.get("atol", 1e-3), rtol=ex.get("rtol", 1e-3))
            else:
                model = baseline if fmt in ("torchscript", "torch_export") else None
                fp32 = _source_fp32(baseline, fmt, models_dir, cfg, adapter, vname(ratio, "fp32"))
            onnx_src = fp32 if fp32.suffix == ".onnx" else None
            prune_s = time.time() - t0
        except Exception as e:
            log.error("prune/export failed for ratio %.2f: %s", ratio, e)
            for p in todo:
                rows.append({**base_row(ratio, p), "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-1500:]})
            flush()
            continue

        for p in todo:
            name = vname(ratio, p["name"])
            log.info("=== %s ===", name)
            t1 = time.time()
            qinfo = {}
            try:
                src_fp32, src_onnx = fp32, onnx_src
                if p.get("qat") and fmt == "torch":
                    # quantization-aware fine-tune of a copy, then export it as this variant's FP32 source
                    qat_cfg = {**cfg.get("qat", {}), **(p["qat"] if isinstance(p["qat"], dict) else {})}
                    qmodel = qat_finetune(adapter, copy.deepcopy(model), int(qat_cfg.get("epochs", 1)), qat_cfg)
                    src_fp32 = export_onnx(qmodel, adapter.example_input(), models_dir / f"{name}_fp32.onnx",
                                           input_names=adapter.input_names(),
                                           output_names=adapter.output_names(), opset=ex.get("opset", 17),
                                           atol=ex.get("atol", 1e-3), rtol=ex.get("rtol", 1e-3))
                    src_onnx = src_fp32
                    qinfo["qat_epochs"] = int(qat_cfg.get("epochs", 1))
                if p["method"] == "none":
                    path = src_fp32
                else:
                    if src_onnx is None:  # TFLite source: quantize through ONNX when possible
                        try:
                            src_onnx = onnx_src = convert_tflite_to_onnx(
                                fp32, models_dir / f"{fp32.stem}_from_tflite.onnx")
                        except RuntimeError as e:
                            rows.append({**base_row(ratio, p), "skipped": str(e)})
                            flush()
                            continue
                    q = {**qcfg, **{k: v for k, v in p.items() if k not in ("name",)}}
                    path = quantize_onnx(src_onnx, models_dir / f"{name}.onnx", data.get("calib"), q, info=qinfo)
                quant_s = time.time() - t1
                row = measure(adapter, path, model, name, cfg.get("measure"))
                row.update({**base_row(ratio, p), "finetune_epochs": epochs if ratio else 0,
                            "prune_finetune_s": round(prune_s, 1), "quantize_s": round(quant_s, 1)})
                row.update({k: v for k, v in prune_rep.items()
                            if k in ("max_layer_cut", "bn_recal_batches", "sparsity", "prune_mode",
                                     "recovery_lr", "alloc_method", "alloc_cost", "alloc_layers",
                                     "alloc_frac", "alloc_param_frac", "alloc_trace",
                                     "sensitivity_layers", "sensitivity_s", "sensitivity_proxy")})
                # str(x), not x: a diagnostic reporting structured values (calib_auto_tried) must not
                # be able to fail the variant inside ", ".join()
                row.update({k: (", ".join(str(x) for x in v) if isinstance(v, list) else v)
                            for k, v in qinfo.items()})
                # what the TOOLCHAIN cost to produce this variant, as opposed to what the variant
                # costs to run: on a RAM-limited machine this is the number that decides feasibility
                row["host_peak_rss_mb"] = round(host_peak_rss_mb(), 1)
                row["train_epochs_run"] = len(getattr(adapter, "history", []) or [])
                _save_curve(out_dir, name, getattr(adapter, "history", None))
            except Exception as e:
                log.error("variant %s failed: %s", name, e)
                row = {**base_row(ratio, p), "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-1500:]}
            rows.append(row)
            flush()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    for col in ("error", "skipped"):
        if col not in df:
            df[col] = pd.NA
    return finalize(df, cfg, out_dir, adapter) if finalize_result else df


def student_tag(spec: dict) -> str:
    """Filename-safe identifier for one `sweep.students` entry: {arch: net_s, width_mult: 0.5} -> net_s_w050."""
    name = str(spec.get("arch") or "student")
    w = spec.get("width_mult")
    return name if w is None else f"{name}_w{round(float(w) * 100):03d}"


def build_student(cfg: dict, adapter, teacher, spec: dict, models_dir: Path, resume: bool = False,
                  name: str | None = None):
    """A smaller architecture, distilled from the baseline. Cached on disk so a resumed sweep does
    not retrain it — KD is minutes per student and the sweep around it is restartable.

    `name` is the caller's de-duplicated tag: two specs can share a student_tag, and the cache file
    has to keep them apart or the second would load the first."""
    from .baseline import load_model

    kd = {**cfg.get("distill", {}), **spec}
    path = models_dir / f"{name or student_tag(spec)}_student.pt"
    if resume and path.exists():
        log.info("reusing cached student %s", path.name)
        return load_model(path)

    # Everything that is not a KD knob describes the architecture (arch, width_mult, pretrained, ...);
    # build_student takes **_ so an adapter that does not know a key simply ignores it.
    student = adapter.build_student(**{k: v for k, v in spec.items()
                                       if k not in ("epochs", "temperature", "alpha", "lr", "optimizer")})
    epochs = int(kd.get("epochs", 5))
    log.info("distilling %s for %d epochs", student_tag(spec), epochs)
    student = distill(teacher, student, adapter, epochs, temperature=kd.get("temperature", 4.0),
                      alpha=kd.get("alpha", 0.7), lr=kd.get("lr", 0.01),
                      optimizer=kd.get("optimizer", "sgd"),
                      max_steps_per_epoch=getattr(adapter, "cfg", {}).get("max_steps_per_epoch"))
    save_model(student, path)
    return student


def run_sweep_grid(cfg: dict, make_adapter, make_baseline, out_dir: str | Path, resume: bool = False):
    """Run the prune x precision grid once per `sweep.adapter_variants` entry (e.g. input resolutions)
    and once per `sweep.students` entry (a smaller architecture distilled from the baseline).

    Each entry gets its own rows, prefixed in the shared sweep.csv ("s256_p00_int8",
    "net_s_w050_p00_int8"). Students and the teacher therefore land in ONE table, which is
    what makes them comparable: `score.normalize: minmax` spans every row, so a student's
    compression is measured against the teacher rather than against itself. Scoring a student
    sweep on its own would report every variant as 1x.

    Without either key this is a single plain sweep.
    """
    from .config import deep_merge

    sw = cfg.get("sweep", {})
    variants = sw.get("adapter_variants") or [{}]
    students = sw.get("students") or []
    models_dir = Path(out_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    # flush() now accumulates across the calls below instead of overwriting, so a fresh run has to
    # clear the file first or last run's rows would be adopted as if they were this run's.
    csv_path = Path(out_dir) / "sweep.csv"
    if not resume and csv_path.exists():
        csv_path.unlink()

    frames, adapter = [], None
    for av in variants:
        cfg_v = deep_merge(cfg, {"adapter_args": av, "_adapter_variant": av}) if av else cfg
        tag = "_".join(f"{str(v).replace('.', '')}" for v in av.values()) + "_" if av else ""
        if av:
            log.info("=== adapter variant %s (prefix %r) ===", av, tag)
        adapter = make_adapter(cfg_v)
        teacher = make_baseline(adapter, cfg_v)
        frames.append(run_sweep(cfg_v, adapter, teacher, out_dir, resume=resume,
                                prefix=tag, finalize_result=False))

        if students and not isinstance(teacher, torch.nn.Module):
            # Distillation needs a trainable teacher; an ONNX/TFLite baseline cannot provide one.
            log.warning("sweep.students ignored: the baseline is %s, not an nn.Module",
                        type(teacher).__name__)
            continue
        # Two specs that differ only in a key student_tag does not read (anything but arch and
        # width_mult) would otherwise share a tag, and the second would silently overwrite the
        # first's rows and files. Disambiguate rather than teach the tag every possible key.
        tags, seen = [], {}
        for spec in students:
            t = student_tag(spec)
            seen[t] = seen.get(t, 0) + 1
            tags.append(t if seen[t] == 1 else f"{t}{seen[t]}")

        for spec, s_name in zip(students, tags):
            s_tag = f"{tag}{s_name}_"
            log.info("=== student %s (prefix %r) ===", s_name, s_tag)
            try:
                student = build_student(cfg_v, adapter, teacher, spec, models_dir, resume, name=s_name)
            except Exception as e:      # one unbuildable architecture must not end the sweep
                log.warning("student %s failed: %s", student_tag(spec), e)
                continue
            frames.append(run_sweep(cfg_v, adapter, student, out_dir, resume=resume,
                                    prefix=s_tag, finalize_result=False))

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    df.to_csv(Path(out_dir) / "sweep.csv", index=False)
    return finalize(df, cfg, Path(out_dir), adapter)


def _save_curve(out_dir: Path, variant: str, history) -> None:
    """One CSV per variant under curves/: epoch, loss, lr, seconds and the metric where evaluated.

    Both columns are needed to read a run: in the pruning study the loss kept falling (0.5747 ->
    0.5355 over 18 extra epochs) while the metric gap barely moved (-3.18 -> -2.83 pt). Loss alone
    says "keep training"; the metric says the budget is spent. Task-agnostic - a list of dicts.
    """
    if not history:
        return
    d = out_dir / "curves"
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", variant)
    pd.DataFrame(history).to_csv(d / f"{safe}.csv", index=False)


def plot_curves(out_dir: Path) -> list[Path]:
    """One PNG per run: loss (left axis) and the task metric (right) against epoch, per variant."""
    d = out_dir / "curves"
    files = sorted(d.glob("*.csv")) if d.is_dir() else []
    if not files:
        return []
    from .pareto import _pyplot

    plt = _pyplot()
    if plt is None:
        return []

    fig, ax = plt.subplots(figsize=(9, 5))
    ax2, any_metric = ax.twinx(), False
    for i, f in enumerate(files):
        c = pd.read_csv(f)
        if "loss" not in c or c.empty:
            continue
        col = f"C{i % 10}"
        ax.plot(c["epoch"], c["loss"], color=col, label=f.stem)
        if "metric" in c and c["metric"].notna().any():
            m = c[c["metric"].notna()]
            ax2.plot(m["epoch"], m["metric"], color=col, ls="--", marker="o", ms=3)
            any_metric = True
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (solid)")
    ax2.set_ylabel("task metric (dashed)" if any_metric else "task metric - set adapter_args.eval_every")
    ax.set_title(f"{out_dir.name}: training curves")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out_dir / "curves.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return [png]


def finalize(df: pd.DataFrame, cfg: dict, out_dir: Path, adapter=None) -> pd.DataFrame:
    """Score, mark the Pareto front, write sweep.csv / summary.json, Pareto PNGs and training curves."""
    ok = ok_mask(df)
    df = score_table(df, cfg.get("score"))
    xs = cfg.get("sweep", {}).get("pareto_x", ["size_bytes", "latency_p50_ms", "peak_rss_mb"])
    elig = (df["eligible"] & ok) if "eligible" in df else ok   # a front of variants nobody may pick is useless
    for x in xs:
        if x in df:
            m = pd.Series(False, index=df.index)
            m[elig] = pareto_mask(df.loc[elig, x].values, df.loc[elig, "accuracy"].values)
            df[f"pareto_{x}"] = m
    df.to_csv(out_dir / "sweep.csv", index=False)
    plots = []
    for x in xs:
        p = plot_pareto(df[ok], x, out_dir / f"pareto_{x}.png", title=f"{cfg.get('task', '')}: accuracy vs {x}")
        if p:
            plots.append(str(p))
    plots += [str(x) for x in plot_curves(out_dir)]      # training curves: loss AND metric per epoch
    n_skipped = int(df["skipped"].notna().sum()) if "skipped" in df else 0
    shortlist = candidates(df, xs)
    keep = ["variant", "accuracy", "metric_name", "size_bytes", "latency_p50_ms", "peak_rss_mb", "score", "why"]
    eligible = df["eligible"] if "eligible" in df else ok
    summary = {**provenance(), "n_variants": int(len(df)), "n_failed": int((~ok).sum()) - n_skipped, "n_skipped": n_skipped,
               "n_ineligible": int((ok & ~eligible).sum()), "plots": plots,
               # ranking uses eligible rows only: a variant below the accuracy constraint is not an answer
               "best_by_score": (df.loc[df.loc[eligible, "score"].idxmax(), "variant"]
                                 if eligible.any() and df.loc[eligible, "score"].notna().any() else None),
               "candidates": shortlist[[c for c in keep if c in shortlist]].to_dict("records"),
               "ineligible": df.loc[ok & ~eligible, ["variant", "accuracy", "ineligible_reason"]].to_dict("records"),
               "adapter": adapter.describe() if adapter else None, "env": env_info(),
               "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return df
