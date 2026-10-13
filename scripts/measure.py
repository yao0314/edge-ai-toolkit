"""Measure a model and print a metrics table.

  # our own baseline for a task (trains + caches it the first time)
  python scripts/measure.py --task classification [--quick]

  # a model we were handed: .pt (module / checkpoint / TorchScript / state_dict+--arch), .onnx, .tflite
  python scripts/measure.py --task classification --model given.onnx          # with accuracy
  python scripts/measure.py --model given.tflite --input-shape 1,224,224,3    # resources only, no adapter
"""
import json
import math

from _common import ROOT, base_parser  # noqa: F401  (sets sys.path)

import pandas as pd
import torch

from core import provenance
from core.baseline import get_baseline
from core.config import (build_adapter, config_for_task, load_config, results_dir, run_name, set_seed,
                         setup_env)
from core.export import export_onnx
from core.measure import env_info, format_table, measure
from core.model_io import load_model, model_format


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--model", "--weights", "--onnx", dest="model",
                    help="model file: .pt/.pth (module, checkpoint, TorchScript, state_dict), .onnx, .tflite")
    ap.add_argument("--arch", help="'pkg.module:factory' building the architecture for a state_dict")
    ap.add_argument("--input-shape", help="e.g. 1,3,224,224 - needed when no --task/--config is given")
    ap.add_argument("--retrain", action="store_true", help="ignore cached baseline weights")
    ap.add_argument("--name", default=None, help="variant name in the table")
    args = ap.parse_args()
    setup_env()

    has_task = bool(args.task or args.config)
    if has_task:
        cfg = load_config(args.config or config_for_task(args.task), quick=args.quick)
        set_seed(cfg.get("seed", 0))
        adapter = build_adapter(cfg, args.device)
        task = run_name(cfg)
    else:
        if not (args.model and args.input_shape):
            ap.error("without --task/--config give both --model and --input-shape (resources-only mode)")
        cfg, adapter, task = {}, None, "adhoc"
    out = results_dir(task, args.quick)
    x = (torch.randn(*[int(s) for s in args.input_shape.split(",")]) if args.input_shape
         else adapter.example_input())
    mcfg = cfg.get("measure")

    model_cfg = {**cfg.get("model", {}), **({"arch": args.arch} if args.arch else {})}
    if adapter is not None:
        model = get_baseline(adapter, args.model, args.retrain, model_cfg=model_cfg)
    else:
        model = load_model(args.model, model_cfg.get("arch"), model_cfg.get("arch_kwargs"))
    fmt = model_format(model)
    name = args.name or ("baseline_fp32" if not args.model else f"given_{fmt}")

    if fmt in ("torch", "torchscript", "torch_export"):
        ex = cfg.get("export", {})
        path = export_onnx(model, x, out / "models" / f"{name}.onnx",
                           input_names=adapter.input_names() if adapter else ["input"],
                           output_names=adapter.output_names() if adapter else ["output"],
                           opset=ex.get("opset", 17), atol=ex.get("atol", 1e-3), rtol=ex.get("rtol", 1e-3))
        row = measure(adapter, path, model, name, mcfg, example_input=x)
    else:
        row = measure(adapter, model, None, name, mcfg, example_input=x)
    row.update({"source_format": fmt, "precision": "as_given" if args.model else "fp32", "prune_ratio": 0.0})
    rows = [row]

    print()
    print(format_table(rows))
    if adapter is None:
        print("\n(resources only: no adapter, so accuracy is empty and the input was random data)")
    if any(isinstance(r.get("macs"), float) and math.isnan(r["macs"]) for r in rows):
        print("(MACs/params unavailable for this format)")
    stem = "baseline" if not args.model else f"measure_{name}"
    pd.DataFrame(rows).to_csv(out / f"{stem}.csv", index=False)
    with open(out / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump({**provenance(), "rows": rows, "adapter": adapter.describe() if adapter else None,
                   "env": env_info(),
                   "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}, f, indent=2, default=str)
    print(f"\nsaved {out / (stem + '.csv')}")


if __name__ == "__main__":
    main()
