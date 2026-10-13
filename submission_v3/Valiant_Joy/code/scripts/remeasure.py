"""Re-measure runtime metrics (latency / CPU / RAM / energy) of an existing sweep, without retraining.

Use it after changing measurement settings, or on another machine (e.g. the competition
environment or a real board) with the same results folder copied over. Accuracy, params
and MACs are kept from the CSV; scores and Pareto plots are recomputed.

  python scripts/remeasure.py --config configs/classification.yaml [--csv results/classification/sweep.csv]
"""
import re
from pathlib import Path

from _common import base_parser, resolve_config  # noqa: F401

import pandas as pd

from core.config import build_adapter, results_dir, run_name, setup_env
from core.energy import measure_energy
from core.measure import DEFAULT_MEASURE_CFG, format_table, primary_thread_count, runtime_metrics
from core.sweep import finalize, ok_mask


def _set(df, i, k, v):
    """df.loc[i, k] = v, widening the column dtype when needed (pandas 3 refuses silent upcasts)."""
    if k in df and isinstance(v, str) and df[k].dtype != object:
        df[k] = df[k].astype(object)
    df.loc[i, k] = v


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--csv", help="sweep csv (default results/<task>[_quick]/sweep.csv)")
    ap.add_argument("--rescore-only", action="store_true",
                    help="do not re-measure: re-derive the primary-thread columns from the existing "
                         "per-thread columns and re-score (use after changing measure.primary_threads "
                         "or the score settings)")
    args = ap.parse_args()
    setup_env()
    cfg = resolve_config(args)
    adapter = build_adapter(cfg, args.device)
    csv = Path(args.csv) if args.csv else results_dir(run_name(cfg), args.quick) / "sweep.csv"
    df = pd.read_csv(csv)
    x = adapter.example_input()
    mcfg = cfg.get("measure", {})
    stale = [c for c in df.columns if c.startswith(("latency_", "cpu_", "peak_rss", "rss_steady", "n_runs", "cpus_"))]

    ok = ok_mask(df)
    if args.rescore_only:
        # The CSV - not the config - is the source of truth for what was measured here: nothing is
        # re-run, so asking for a thread count the file does not contain used to copy nothing while
        # still stamping primary_threads with it, relabelling 1-thread numbers as 4-thread.
        measured = sorted({int(m.group(1)) for c in df.columns
                           if (m := re.fullmatch(r"latency_p50_ms_t(\d+)", c))})
        if not measured:
            raise SystemExit(f"{csv} has no latency_p50_ms_t<N> column to re-score from")
        p = primary_thread_count({**DEFAULT_MEASURE_CFG, **mcfg, "threads": measured})
        for k in ("latency_p50_ms", "latency_p90_ms", "cpu_util_pct", "cpu_ms_per_inf", "peak_rss_mb"):
            if f"{k}_t{p}" in df:
                df[k] = df[f"{k}_t{p}"]
        df["primary_threads"] = p
        print(f"re-scoring from the t{p} columns (measured: {measured}; no new measurements)")
        df = finalize(df, cfg, csv.parent, adapter)
        print()
        print(format_table(df[ok_mask(df)].to_dict("records"),
                           ["variant", "accuracy", "latency_p50_ms", "peak_rss_mb", "score", "eligible"]))
        print(f"updated {csv}")
        return
    for i in df.index[ok]:
        path = df.loc[i, "model_file"] if "model_file" in df else df.loc[i, "onnx"]
        print(f"re-measuring {df.loc[i, 'variant']} ...", flush=True)
        df.loc[i, stale] = float("nan")
        for k, v in runtime_metrics(path, x, mcfg).items():
            _set(df, i, k, v)
        e, src = measure_energy(path, df.loc[i, "variant"], None, mcfg.get("energy"))
        _set(df, i, "energy_j", e)
        _set(df, i, "energy_source", src)
    df = finalize(df, cfg, csv.parent, adapter)
    print()
    print(format_table(df[ok_mask(df)].to_dict("records"),
                       ["variant", "accuracy", "size_bytes", "latency_p50_ms", "peak_rss_mb", "cpu_ms_per_inf", "score"]))
    print(f"\nupdated {csv}")


if __name__ == "__main__":
    main()
