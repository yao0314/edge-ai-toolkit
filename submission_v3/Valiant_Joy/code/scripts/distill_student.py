"""Distil the baseline into smaller architectures, then quantize, and report both halves of the
rubric's compression criterion.

The rubric (Software Track packet, section 10.2) bands compression on the reduction in parameter
count AND on-disk file size, 10x in both for the top band. Quantization moves only the second:
INT8 makes every weight a quarter of the size and leaves the count untouched, so a quantized copy
of the baseline is a 4x file and a 1x parameter count. Cutting the count means removing channels
or using a smaller architecture — this script does the second, with the baseline as the teacher,
and is the missing half of `scripts/compress.py` (which prunes and quantizes a given model but
never builds a student, although every adapter has `build_student`).

  python scripts/distill_student.py --task classification
  python scripts/distill_student.py --config configs/classification.yaml --students mobilenet_v2@0.5,mobilenet_v2@0.35
  python scripts/distill_student.py --task classification --quick      # smoke test, minutes

Writes results/<run>_students/: students.csv, the ONNX files, and a printed table whose last two
columns are the rubric's two ratios and the accuracy band each student lands in.
"""
import json
import logging
import time
from pathlib import Path

from _common import ROOT, base_parser, resolve_config  # noqa: F401  (sets sys.path)

import pandas as pd
import torch

from core import provenance
from core.baseline import get_baseline, save_model
from core.config import build_adapter, results_dir, run_name, set_seed, setup_env
from core.distill import distill
from core.export import export_onnx
from core.measure import count_params
from core.quantize import quantize_onnx

log = logging.getLogger(__name__)

# The rubric's bands, so the table can say which one a row lands in instead of leaving the reader
# to divide. Keys are the floor of each band.
COMPRESSION_BANDS = ((10, "Excellent 18-20"), (5, "Good 13-17"), (2, "Fair 7-12"), (0, "Poor 0-6"))
ACCURACY_BANDS = ((0.97, "Excellent 22-25"), (0.90, "Good 16-21"), (0.75, "Fair 9-15"), (0, "Poor 0-8"))


def band(value: float, bands) -> str:
    for floor, label in bands:
        if value >= floor:
            return label
    return bands[-1][1]


def parse_students(spec: str | None, cfg: dict) -> list[dict]:
    """"mobilenet_v2@0.5,shufflenet_v2_x0_5" -> [{"arch": ..., "width_mult": ...}, ...].

    Without --students, fall back to the config's `distill.students` (a list) or `distill.student`
    (one entry), so a config can pin the ladder it wants.
    """
    if spec:
        out = []
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            arch, _, w = item.partition("@")
            out.append({"arch": arch, **({"width_mult": float(w)} if w else {})})
        return out
    kd = cfg.get("distill", {})
    return list(kd.get("students") or [kd.get("student", {"width_mult": 0.5})])


def evaluate(adapter, model) -> float:
    return float(adapter.evaluate(model)["score"])


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--model", help="teacher file (.pt); default: config/cached baseline")
    ap.add_argument("--arch", help="'pkg.module:factory' for a state_dict .pt teacher")
    ap.add_argument("--students", help="comma-separated arch[@width], e.g. mobilenet_v2@0.5,mobilenet_v2@0.35")
    ap.add_argument("--epochs", type=int, help="KD epochs per student (default: distill.epochs)")
    ap.add_argument("--no-int8", action="store_true", help="skip quantization (FP32 students only)")
    args = ap.parse_args()

    setup_env()
    cfg = resolve_config(args)
    set_seed(cfg.get("seed", 0))
    adapter = build_adapter(cfg, args.device)
    out = results_dir(run_name(cfg) + "_students", args.quick)
    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)

    kd = cfg.get("distill", {})
    epochs = args.epochs if args.epochs is not None else int(kd.get("epochs", 5))
    ex = cfg.get("export", {})
    data = adapter.load_data()
    x = adapter.example_input()

    teacher = get_baseline(adapter, args.model,
                           model_cfg={**cfg.get("model", {}), **({"arch": args.arch} if args.arch else {})})
    t_acc, t_params = evaluate(adapter, teacher), count_params(teacher)
    t_onnx = export_onnx(teacher, x, models / "teacher_fp32.onnx", input_names=adapter.input_names(),
                         output_names=adapter.output_names(), opset=ex.get("opset", 17),
                         atol=ex.get("atol", 1e-3), rtol=ex.get("rtol", 1e-3))
    t_size = Path(t_onnx).stat().st_size
    log.info("teacher: %s %.4f, %.2f M params, %.1f MB", adapter.metric_name() if hasattr(adapter, "metric_name")
             else "score", t_acc, t_params / 1e6, t_size / 2**20)

    rows = [{"variant": "teacher_fp32", "arch": "(baseline)", "precision": "fp32", "kd_epochs": 0,
             "accuracy": t_acc, "params": t_params, "size_bytes": t_size,
             "param_ratio": 1.0, "size_ratio": 1.0, "acc_rel": 1.0, "train_s": 0.0}]

    for spec in parse_students(args.students, cfg):
        tag = spec.get("arch", "student") + (f"@{spec['width_mult']}" if "width_mult" in spec else "")
        name = tag.replace("@", "_w").replace(".", "")
        log.info("=== student %s: %d KD epochs ===", tag, epochs)
        t0 = time.time()
        try:
            student = adapter.build_student(**spec)
        except Exception as e:
            log.warning("cannot build %s: %s", tag, e)
            continue
        student = distill(teacher, student, adapter, epochs, temperature=kd.get("temperature", 4.0),
                          alpha=kd.get("alpha", 0.7), lr=kd.get("lr", 0.01),
                          optimizer=kd.get("optimizer", "sgd"))
        train_s = time.time() - t0
        save_model(student, models / f"{name}.pt")

        s_params = count_params(student)
        fp32 = export_onnx(student, x, models / f"{name}_fp32.onnx", input_names=adapter.input_names(),
                           output_names=adapter.output_names(), opset=ex.get("opset", 17),
                           atol=ex.get("atol", 1e-3), rtol=ex.get("rtol", 1e-3))
        made = [("fp32", fp32)]
        if not args.no_int8:
            qinfo: dict = {}
            try:
                made.append(("int8", quantize_onnx(fp32, models / f"{name}_int8.onnx", data.get("calib"),
                                                   cfg.get("quantize"), info=qinfo)))
            except Exception as e:
                log.warning("quantizing %s failed: %s", tag, e)

        s_acc = evaluate(adapter, student)   # the ONNX path is measured by scripts/measure.py; this
        for prec, path in made:              # is the torch metric, identical for fp32 and int8 weights
            size = Path(path).stat().st_size
            rows.append({"variant": f"{name}_{prec}", "arch": tag, "precision": prec,
                         "kd_epochs": epochs, "accuracy": s_acc, "params": s_params,
                         "size_bytes": size, "param_ratio": t_params / max(s_params, 1),
                         "size_ratio": t_size / max(size, 1), "acc_rel": s_acc / max(t_acc, 1e-9),
                         "train_s": round(train_s, 1)})

    df = pd.DataFrame(rows)
    # The criterion is "10x in BOTH", so the weaker half is the one that scores.
    df["compression_x"] = df[["param_ratio", "size_ratio"]].min(axis=1)
    df["compression_band"] = df["compression_x"].map(lambda v: band(v, COMPRESSION_BANDS))
    df["accuracy_band"] = df["acc_rel"].map(lambda v: band(v, ACCURACY_BANDS))
    df.to_csv(out / "students.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(
        {**provenance(), "teacher": {"accuracy": t_acc, "params": t_params, "size_bytes": t_size},
         "kd_epochs": epochs, "students": [r["variant"] for r in rows[1:]]}, indent=2), encoding="utf-8")

    show = df[["variant", "accuracy", "acc_rel", "params", "param_ratio", "size_ratio",
               "compression_x", "compression_band", "accuracy_band"]].copy()
    show["params"] = (show["params"] / 1e6).round(2)
    for c in ("acc_rel", "param_ratio", "size_ratio", "compression_x"):
        show[c] = show[c].round(2)
    print("\n" + show.to_string(index=False))
    passing = df[(df["acc_rel"] >= 0.97) & (df["compression_x"] >= 10)]
    print(f"\n{len(passing)} of {len(df) - 1} students clear both top bands "
          f"(>=10x compression and >=97% of baseline accuracy).")
    if len(passing):
        best = passing.sort_values("compression_x", ascending=False).iloc[0]
        print(f"smallest that clears both: {best['variant']} "
              f"({best['compression_x']:.1f}x, {100 * best['acc_rel']:.1f}% of baseline)")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
