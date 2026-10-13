"""One command: compress a model and write a submittable deliverable folder.

  python scripts/compress.py --task classification                      # sweep, then pick and package
  python scripts/compress.py --config configs/detection.yaml --reuse    # reuse an existing sweep
  python scripts/compress.py --task classification --model given.pt --max-acc-drop 0.01

Picks the highest-scoring variant that PASSES the accuracy constraint (score.max_accuracy_drop), never one
that is merely small and fast, and writes deliverables/<run>/:
  best.onnx        the model to submit
  best_fp32.onnx   the same compression stage before INT8 (convert from this if another format is needed)
  best_model.pt    the pruned PyTorch model, when pruning produced one
  REPORT.md        what was chosen, why, numbers vs the baseline, and the caveats
  metadata.json    machine-readable version of the same
  candidates.csv   the shortlist considered; sweep.csv the full grid
"""
import json
import logging
import shutil
from pathlib import Path

from _common import ROOT, base_parser, resolve_config  # noqa: F401  (sets sys.path)

import pandas as pd

from core import AUTHOR, provenance
from core.baseline import get_baseline
from core.export import stamp_onnx
from core.config import build_adapter, results_dir, run_name, set_seed, setup_env
from core.measure import env_info, format_table
from core.score import candidates, score_table
from core.sweep import finalize, ok_mask, run_sweep_grid

log = logging.getLogger(__name__)


def _nt(row) -> str:
    """"4 threads" / "1 thread" for whichever count the un-suffixed columns in this row came from."""
    n = row.get("primary_threads")
    n = int(n) if n is not None and n == n and n else 1      # n == n rejects NaN (an errored variant)
    return f"{n} thread{'s' if n != 1 else ''}"


def fmt_size(n) -> str:
    return f"{n / 2**20:.2f} MB" if n >= 2**20 else f"{n / 1024:.1f} KB"


def _preprocessing(adapter) -> dict:
    """Never let a broken preprocessing() kill the packaging step - an unusable deliverable is bad,
    no deliverable is worse."""
    try:
        return dict(adapter.preprocessing() or {})
    except Exception as e:
        log.warning("adapter.preprocessing() failed, deliverable will not document it: %s", e)
        return {}


def _input_signature(adapter) -> dict:
    try:
        x = adapter.example_input()
        return {"name": (adapter.input_names() or ["input"])[0], "shape": list(x[:1].shape)}
    except Exception as e:
        log.debug("input signature unavailable: %s", e)
        return {}


def _pipeline_section(cfg: dict, base, best) -> list:
    """What was actually done to get from the baseline to this file.

    The submission guidelines ask the report to summarise the compression pipeline, and a judge
    rebuilding it needs the steps that were taken rather than the ones that were available.
    """
    steps, q = [], cfg.get("quantize", {})
    variant, kd = str(best["variant"]), cfg.get("distill", {})
    student = variant.rsplit("p", 1)[0].rstrip("_") if "p" in variant else ""
    if student:
        steps.append(f"**Distillation** - the submitted network is `{student}`, not the baseline "
                     f"architecture: {int(kd.get('epochs', 0))} epochs of knowledge distillation "
                     f"(temperature {kd.get('temperature', 4.0)}, alpha {kd.get('alpha', 0.7)}) with the "
                     f"baseline as teacher. This is what reduces the parameter count; quantization "
                     f"cannot.")
    ratio = float(best.get("prune_ratio") or 0)
    if ratio:
        p = cfg.get("prune", {})
        steps.append(f"**Structured pruning** - {100 * ratio:.0f}% of channels removed "
                     f"({p.get('importance', 'l2')} importance, "
                     f"{'global' if p.get('global_pruning') else 'per-layer'} threshold), then "
                     f"{'BN recalibration and ' if p.get('bn_recalibrate', True) else ''}"
                     f"{int(best.get('finetune_epochs') or 0)} fine-tuning epochs. Channels are removed, "
                     f"not masked, so the tensors themselves get smaller.")
    if str(best.get("precision", "")).startswith("int8"):
        steps.append(f"**INT8 quantization** - {q.get('method', 'static')}, "
                     f"{'per-channel' if q.get('per_channel', True) else 'per-tensor'} weights, "
                     f"{q.get('activation_type', 'uint8')} activations / {q.get('weight_type', 'int8')} "
                     f"weights, calibrated with "
                     f"`{best.get('calibrate_method') or q.get('calibrate_method', 'auto')}`.")
    if not steps:
        return []
    return ["## Compression pipeline", ""] + [f"{i}. {s}" for i, s in enumerate(steps, 1)] + [""]


def _aihub_section(cfg: dict, base, best) -> list:
    """The hosted-device profile and why that device. The packet asks for the rationale because the
    target board is not in the catalogue, so the choice is a judgement that has to be defended."""
    if pd.isna(best.get("aihub_latency_ms")):
        return []
    L = ["## Target device profile (Qualcomm AI Hub)", "",
         f"Profiled on **{best.get('aihub_device')}**, CPU only.", "",
         "| | baseline | chosen |", "|---|---|---|",
         f"| AI Hub latency | {base.get('aihub_latency_ms', float('nan')):.2f} ms | "
         f"**{best['aihub_latency_ms']:.2f} ms** |"]
    if pd.notna(best.get("aihub_peak_memory_mb")):
        L.append(f"| AI Hub peak memory | {base.get('aihub_peak_memory_mb', float('nan')):.1f} MB | "
                 f"**{best['aihub_peak_memory_mb']:.1f} MB** |")
    note = cfg.get("aihub", {}).get("device_rationale")
    L += ["", note or
          "The competition's target board is not in the AI Hub catalogue, so this is the nearest "
          "available low-power profile; see `configs/` for the preference list and the reasoning.", ""]
    return L


def write_report(out: Path, cfg: dict, adapter, base, best, short, rejected, files: dict) -> Path:
    metric = best["metric_name"]
    drop = float(base["accuracy"]) - float(best["accuracy"])
    desc = adapter.describe()
    L = [f"# {run_name(cfg)} - compression deliverable", "",
         f"**Submit `{files['best'].name}`** (variant `{best['variant']}`).", "",
         "| | baseline | chosen | change |", "|---|---|---|---|",
         f"| variant | {base['variant']} | **{best['variant']}** | |",
         f"| {metric} | {base['accuracy']:.4f} | **{best['accuracy']:.4f}** | {-drop:+.4f} "
         f"({100 * float(best['accuracy']) / max(float(base['accuracy']), 1e-9):.1f}% retained) |",
         f"| size | {fmt_size(base['size_bytes'])} | **{fmt_size(best['size_bytes'])}** | "
         f"{base['size_bytes'] / max(best['size_bytes'], 1):.2f}x smaller |",
         ]
    # Parameter count is half of what compression is judged on, and the half quantization cannot
    # move: an INT8 copy is a quarter of the file with every weight still there. Reporting only
    # the file would overstate the compression of anything that was merely quantized.
    if base.get("params") and best.get("params"):
        bp, cp = float(base["params"]), float(best["params"])
        L.append(f"| parameters | {bp / 1e6:.2f} M | **{cp / 1e6:.2f} M** | {bp / max(cp, 1):.2f}x fewer |")
    # --quick and any custom profile may measure only one thread count, so neither column is assumed
    for t in (1, 4):
        col = f"latency_p50_ms_t{t}"
        if col in best and pd.notna(best.get(col)) and pd.notna(base.get(col)):
            L.append(f"| latency p50, {t} thread{'s' if t > 1 else ''} | {base[col]:.3f} ms | "
                     f"{best[col]:.3f} ms | {base[col] / max(best[col], 1e-9):.2f}x |")
    # The source travels with the number: a figure modelled from CPU TDP is not a measurement and
    # must not be read as one.
    def _energy(row):
        if pd.isna(row.get("energy_j")):
            return "not measured"
        src = row.get("energy_source") or ""
        return f"{row['energy_j']:.4g} J" + (f" ({src})" if src else "")

    L += [f"| peak RSS, {_nt(best)} | {base['peak_rss_mb']:.2f} MB | {best['peak_rss_mb']:.2f} MB | |",
          f"| MACs | {base['macs'] / 1e6:.1f} M | {best['macs'] / 1e6:.1f} M | |",
          f"| energy / inference | {_energy(base)} | {_energy(best)} | |", ""]
    L += _pipeline_section(cfg, base, best)
    L += _aihub_section(cfg, base, best)
    sc = cfg.get("score", {})
    limit = (f"at most {100 * float(sc['max_accuracy_drop_rel']):.0f}% below the baseline, relatively"
             if sc.get("max_accuracy_drop_rel") is not None else
             f"at most {sc.get('max_accuracy_drop')} absolute {metric} points below the baseline")
    L += ["## Why this one", "",
          f"- Highest total score among variants that pass the accuracy constraint ({limit}). "
          f"A weighted score alone would rank small-and-fast variants that lost far more accuracy."]
    if len(rejected):
        L.append("- Rejected despite scoring at least as high: "
                 + ", ".join(f"`{r['variant']}` ({metric} {r['accuracy']:.4f})" for _, r in rejected.iterrows()))
    if pd.notna(best.get("int8_sqnr_db")):
        L.append(f"- INT8 health check: {best['int8_sqnr_db']:.1f} dB output SQNR vs FP32 over "
                 f"{int(best.get('int8_quantized_nodes') or 0)} quantized ops (below ~15 dB would mean a "
                 f"damaged conversion).")
    L += ["", "## Other candidates considered", "",
          # label from the data: the un-suffixed column follows measure.primary_threads (4 with the
          # shipped configs, 1 under --quick), so a hardcoded "1T" here silently mislabelled it
          f"| variant | {metric} | size | latency {_nt(best)} | why shortlisted |", "|---|---|---|---|---|"]
    for _, r in short.iterrows():
        L.append(f"| {r['variant']} | {r['accuracy']:.4f} | {fmt_size(r['size_bytes'])} | "
                 f"{r['latency_p50_ms']:.3f} ms | {r.get('why', '')} |")
    pre, sig = _preprocessing(adapter), _input_signature(adapter)
    L += ["", "## Input and preprocessing", ""]
    if sig:
        L.append(f"- input `{sig['name']}`, shape `{sig['shape']}` (batch is dynamic), "
                 f"`{pre.get('dtype', 'float32')}`, layout `{pre.get('layout', '?')}`"
                 + (f", colour `{pre['color']}`" if pre.get("color") else ""))
    if pre.get("steps"):
        L += ["", "Apply these in order to a raw sample before calling the model:", ""]
        L += [f"{i}. {t}" for i, t in enumerate(pre["steps"], 1)]
        if pre.get("output_note"):
            L += ["", f"Output: {pre['output_note']}."]
    else:
        L += ["", "**This adapter declared no preprocessing.** The reported metric cannot be "
                  "reproduced from the model file alone - fill in TaskAdapter.preprocessing()."]
    env = env_info()
    L += ["", "## How to read these numbers", "",
          f"- {metric} was evaluated on the exported file itself, on "
          f"{desc.get('n_val') or desc.get('n_test') or '?'} samples ({desc.get('dataset', desc.get('task'))}).",
          f"- Latency / CPU / RAM come from {env['cpu']} with ONNX Runtime {env['onnxruntime']}, pinned to "
          f"performance cores, median of repeated runs: RELATIVE values for comparing variants, not "
          f"target-hardware figures.",
          "- Speed-up depends on the thread count; both 1-thread and 4-thread values are given above.",
          "- Energy is empty unless a real measurement source is configured: it is never estimated.", "",
          "## Reproduce", "", "```bash",
          f"python scripts/sweep.py --config configs/{Path(cfg['_config_path']).name}",
          f"python scripts/compress.py --config configs/{Path(cfg['_config_path']).name} --reuse",
          "```", "", "---", "", f"Produced by edge-ai-toolkit - author {AUTHOR}.", ""]
    p = out / "REPORT.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--model", help="model file to compress (.pt/.onnx/.tflite); default: config/cached baseline")
    ap.add_argument("--arch", help="'pkg.module:factory' for a state_dict .pt")
    ap.add_argument("--reuse", action="store_true", help="reuse results/<run>/sweep.csv instead of sweeping")
    ap.add_argument("--max-acc-drop", type=float, help="override score.max_accuracy_drop for this run")
    ap.add_argument("--out", help="deliverable dir (default deliverables/<run>)")
    args = ap.parse_args()

    setup_env()
    cfg = resolve_config(args)
    if args.max_acc_drop is not None:
        cfg.setdefault("score", {})["max_accuracy_drop"] = args.max_acc_drop
    set_seed(cfg.get("seed", 0))
    adapter = build_adapter(cfg, args.device)
    res = results_dir(run_name(cfg), args.quick)

    if args.reuse and (res / "sweep.csv").exists():
        df = finalize(pd.read_csv(res / "sweep.csv"), cfg, res, adapter)
    else:
        model_cfg = {**cfg.get("model", {}), **({"arch": args.arch} if args.arch else {})}
        # run_sweep_grid, not run_sweep: the plain form ignores sweep.adapter_variants, so the
        # one-command deliverable path silently dropped whole sweep dimensions (input resolution)
        # that scripts/sweep.py does explore. It also finalizes, so no second call here.
        df = run_sweep_grid(cfg, lambda c: build_adapter(c, args.device),
                            lambda ad, c: get_baseline(ad, args.model, model_cfg=model_cfg), res)
    if "eligible" not in df:
        df = score_table(df, cfg.get("score"))

    ok = df[ok_mask(df)]
    elig = ok[ok["eligible"]]
    if elig.empty:
        raise SystemExit("no variant passes the accuracy constraint: raise score.max_accuracy_drop "
                         "(--max-acc-drop) or fine-tune longer")
    best = elig.loc[elig["score"].idxmax()]
    base = ok.iloc[0]
    short = candidates(df, cfg.get("sweep", {}).get("pareto_x", []))
    rejected = ok[(~ok["eligible"]) & (ok["score"] >= best["score"])]

    out = Path(args.out) if args.out else ROOT / "deliverables" / res.name
    out.mkdir(parents=True, exist_ok=True)
    files = {"best": out / f"best{Path(best['model_file']).suffix}"}
    shutil.copyfile(best["model_file"], files["best"])
    # match the variant PREFIX as well: with sweep.adapter_variants several rows share
    # (prune_ratio, precision) and differ only by the prefix (e.g. "640_p00_fp32" vs "320_p00_fp32"),
    # so the pair alone could ship a different resolution's FP32 file as "the same model before INT8"
    pref = str(best["variant"]).rsplit("p", 1)[0] if "p" in str(best["variant"]) else ""
    fp32 = ok[(ok["prune_ratio"] == best["prune_ratio"]) & (ok["precision"] == "fp32")
              & ok["variant"].astype(str).str.startswith(pref)]
    if len(fp32) and Path(fp32.iloc[0]["model_file"]).exists():
        files["fp32"] = out / "best_fp32.onnx"
        shutil.copyfile(fp32.iloc[0]["model_file"], files["fp32"])
    pt = res / "models" / f"{pref}p{round(float(best['prune_ratio']) * 100):02d}.pt"
    if pt.exists():
        files["torch"] = out / "best_model.pt"
        shutil.copyfile(pt, files["torch"])
    short.to_csv(out / "candidates.csv", index=False)
    shutil.copyfile(res / "sweep.csv", out / "sweep.csv")
    for f in files.values():   # the deliverable is what leaves the building: stamp it at packaging time
        if f.suffix == ".onnx":
            stamp_onnx(f)
    report = write_report(out, cfg, adapter, base, best, short, rejected, files)
    meta = {**provenance(), "chosen": best["variant"], "max_accuracy_drop": cfg.get("score", {}).get("max_accuracy_drop"),
            "metrics": {k: (None if pd.isna(best.get(k)) else best.get(k))
                        for k in ("metric_name", "accuracy", "size_bytes", "macs", "latency_p50_ms_t1",
                                  "latency_p50_ms_t4", "peak_rss_mb", "cpu_ms_per_inf", "energy_j",
                                  "int8_sqnr_db")},
            "baseline": {"variant": base["variant"], "accuracy": float(base["accuracy"]),
                         "size_bytes": int(base["size_bytes"])},
            "rejected_higher_scoring": list(rejected["variant"]),
            "files": {k: v.name for k, v in files.items()}, "env": env_info(), "adapter": adapter.describe(),
            # Without this the artifact is not reproducible outside this repo - see
            # TaskAdapter.preprocessing(). Empty dict means the adapter declared nothing.
            "preprocessing": _preprocessing(adapter), "input": _input_signature(adapter)}
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print()
    print(format_table(short.to_dict("records"),
                       ["variant", "accuracy", "size_bytes", "latency_p50_ms", "peak_rss_mb", "score", "why"]))
    if len(rejected):
        print(f"\nrejected despite a higher score (accuracy constraint): {', '.join(rejected['variant'])}")
    print(f"\nchosen: {best['variant']} - {best['metric_name']} {best['accuracy']:.4f} "
          f"({float(best['accuracy']) - float(base['accuracy']):+.4f} vs baseline), "
          f"{fmt_size(best['size_bytes'])}, {best['latency_p50_ms_t1']:.2f} ms 1T")
    print(f"deliverable: {out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")
    print(f"\n{report.name} is the one-page summary to hand in with the model.")


if __name__ == "__main__":
    main()
