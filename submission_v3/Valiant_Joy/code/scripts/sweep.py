"""Sweep prune ratios x precisions, write results/<task>/sweep.csv + Pareto PNGs.

  python scripts/sweep.py --config configs/classification.yaml [--quick] [--resume]
  python scripts/sweep.py --config configs/x.yaml --model given.onnx     # ONNX only: no pruning
"""
from pathlib import Path

from _common import base_parser, resolve_config  # noqa: F401  (sets sys.path)

from core.baseline import get_baseline
from core.config import build_adapter, results_dir, run_name, set_seed, setup_env
from core.measure import format_table
from core.score import candidates
from core.sweep import ok_mask, run_sweep_grid


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--model", "--weights", dest="model",
                    help="baseline file we were handed: .pt (module/checkpoint/state_dict/TorchScript), "
                         ".onnx or .tflite. Default: config `model.path`, else cached/trained baseline")
    ap.add_argument("--arch", help="'pkg.module:factory' for a state_dict .pt")
    ap.add_argument("--resume", action="store_true", help="skip variants already in sweep.csv")
    ap.add_argument("--out", help="output dir (default results/<task>[_quick])")
    args = ap.parse_args()

    setup_env()
    cfg = resolve_config(args)
    set_seed(cfg.get("seed", 0))
    adapter = build_adapter(cfg, args.device)
    out = results_dir(run_name(cfg), args.quick) if not args.out else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model_cfg = {**cfg.get("model", {}), **({"arch": args.arch} if args.arch else {})}
    df = run_sweep_grid(cfg, lambda c: build_adapter(c, args.device),
                        lambda ad, c: get_baseline(ad, args.model, model_cfg=model_cfg), out,
                        resume=args.resume)

    ok = df[ok_mask(df)]
    print()
    print(format_table(ok.to_dict("records"), ["variant", "accuracy", "size_bytes", "macs", "latency_p50_ms",
                                               "peak_rss_mb", "cpu_ms_per_inf", "energy_j", "score", "eligible"]))
    bad_acc = ok[~ok["eligible"]] if "eligible" in ok else ok.iloc[0:0]
    if len(bad_acc):
        print(f"\n{len(bad_acc)} variant(s) excluded by the accuracy constraint (score.max_accuracy_drop):")
        for _, r in bad_acc.iterrows():
            print(f"  {r['variant']}: {r['ineligible_reason']}")
    short = candidates(df, cfg.get("sweep", {}).get("pareto_x", []))
    if len(short):
        print("\nCandidates to choose from (best scores + Pareto fronts) - pick one, do not assume the top row:")
        print(format_table(short.to_dict("records"),
                           ["variant", "accuracy", "size_bytes", "latency_p50_ms", "peak_rss_mb", "score", "why"]))
    for col, title in (("skipped", "skipped"), ("error", "failed")):
        bad = df[df[col].notna()]
        if len(bad):
            print(f"\n{len(bad)} variant(s) {title}:")
            for _, r in bad.iterrows():
                print(f"  {r['variant']}: {r[col]}")
    print(f"\nsaved {out / 'sweep.csv'} and Pareto plots in {out}")


if __name__ == "__main__":
    main()
