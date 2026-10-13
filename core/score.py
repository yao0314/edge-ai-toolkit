"""Configurable overall score.

Weights come from the published rubric (Software Track pre-competition packet, section 10):
accuracy 25, compression 20, resource efficiency 20, energy 20, out of the 85 points a machine
can score. The remaining 15 (documentation 10, presentation 5) are judged from the report.

Each group is the mean of its normalised columns unless it sets `agg: min`, which takes the worst
one instead. Compression uses `min` because the rubric's bands require the reduction in parameter
count AND in on-disk size: a variant that shrinks the file 40x while leaving the parameter count
untouched (plain INT8, or magnitude pruning, which keeps tensor SHAPES) scores on the parameter
count, not the file. Normalised values are in [0, 1], 1 = best. Groups whose columns are all
missing (e.g. energy_j without a meter) are dropped and the remaining weights renormalised —
reported in `score_note`.

A weighted score alone is NOT a model selector: with equal weights, accuracy is only 25% of it, so a
variant that lost 20 accuracy points can outrank everything because it is small and fast. Therefore
`max_accuracy_drop` / `min_accuracy` are applied first as a hard constraint (`eligible` column), and
`candidates()` returns a shortlist (top scores + Pareto fronts) instead of one winner.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_SCORE_CFG = {
    "normalize": "minmax",
    # HARD CONSTRAINT, applied before ranking: a variant that loses more accuracy than this (absolute,
    # vs. the FP32 baseline) is ineligible whatever its score. Without it, equal weights put accuracy at
    # 25% and happily rank a variant that lost 20 points first, because it is small and fast.
    "max_accuracy_drop": None,       # absolute metric points below the baseline
    "max_accuracy_drop_rel": 0.03,   # OR a fraction of the baseline. The rubric's top accuracy band
                                     # (22-25 of 25) is "retains at least 97% of baseline accuracy",
                                     # so 0.03 relative is the real constraint; the absolute form is
                                     # left off because 0.02 absolute is 2% of a 0.92 top-1 but ~5%
                                     # of a 0.41 mAP. When both are set the stricter one wins.
    "min_accuracy": None,            # optional absolute floor (e.g. 0.90 top-1 / 0.35 mAP)
    # Rubric section 10, renormalised over the 85 machine-scored points.
    "weights": {"accuracy": 25 / 85, "compression": 20 / 85, "resource": 20 / 85, "energy": 20 / 85},
    "groups": {
        "accuracy": {"cols": ["accuracy"], "higher_is_better": True},
        # Both halves of the rubric's compression criterion, aggregated with `min` so the weaker one
        # decides. size_bytes is the raw file; add "size_bytes_gz" (or swap to it) if compression is
        # ever judged on a compressed artifact: magnitude-pruned weights keep their SHAPE, so
        # size_bytes cannot see them at all - two variants differed by 1 byte raw while the gzipped
        # sizes were 5.01x vs 3.73x, and the scorer therefore always preferred the unpruned one.
        "compression": {"cols": ["size_bytes", "params"], "agg": "min"},
        "resource": {"cols": ["peak_rss_mb", "cpu_ms_per_inf", "latency_p50_ms"]},
        "energy": {"cols": ["energy_j"]},
    },
    "baseline_variant": None,  # for normalize=baseline; default: first row with prune_ratio 0 & fp32
}


def _norm(col: pd.Series, higher: bool, method: str, base: float | None) -> pd.Series:
    col = col.astype(float)
    if method == "baseline":
        if base is None or not np.isfinite(base) or base == 0:
            return pd.Series(np.nan, index=col.index)
        # ratio vs baseline: >1 better than baseline. Not bounded to [0, 1].
        return col / base if higher else base / col
    lo, hi = col.min(skipna=True), col.max(skipna=True)
    if not np.isfinite(lo) or hi == lo:
        return pd.Series(np.where(col.notna(), 1.0, np.nan), index=col.index)
    return (col - lo) / (hi - lo) if higher else (hi - col) / (hi - lo)


def baseline_row(out: pd.DataFrame, ok: pd.Series, cfg: dict):
    """The reference variant: config `baseline_variant`, else the unpruned FP32 row, else the first row."""
    if cfg.get("baseline_variant"):
        hit = out[out["variant"] == cfg["baseline_variant"]]
        if len(hit):
            return hit.iloc[0]
    cand = out[ok & (out.get("prune_ratio", 0) == 0) & (out.get("precision", "fp32") == "fp32")]
    if not len(cand):
        cand = out[ok]
    return cand.iloc[0] if len(cand) else None


def eligibility(out: pd.DataFrame, ok: pd.Series, cfg: dict) -> tuple[pd.Series, pd.Series]:
    """Hard accuracy constraints, applied before any ranking. Returns (eligible, reason)."""
    eligible = ok.copy()
    reason = pd.Series([None] * len(out), index=out.index, dtype=object)
    base = baseline_row(out, ok, cfg)
    drop, floor = cfg.get("max_accuracy_drop"), cfg.get("min_accuracy")
    rel = cfg.get("max_accuracy_drop_rel")
    if rel is not None and base is not None and np.isfinite(base["accuracy"]):
        rel_abs = float(rel) * float(base["accuracy"])
        drop = rel_abs if drop is None else min(float(drop), rel_abs)
    for i in out.index[ok]:
        acc = out.at[i, "accuracy"]
        if not np.isfinite(acc):
            eligible[i], reason[i] = False, "no accuracy"
            continue
        if floor is not None and acc < float(floor):
            eligible[i], reason[i] = False, f"accuracy {acc:.4f} < min_accuracy {float(floor):.4f}"
        elif drop is not None and base is not None and np.isfinite(base["accuracy"]) \
                and acc < float(base["accuracy"]) - float(drop):
            lost = float(base["accuracy"]) - acc
            eligible[i], reason[i] = False, (f"accuracy {acc:.4f} is {lost:.4f} below {base['variant']} "
                                             f"({100 * lost / max(float(base['accuracy']), 1e-9):.1f}% relative; "
                                             f"budget {float(drop):.4f})")
    return eligible, reason


def score_table(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add score_<group>, `score`, and `eligible`/`ineligible_reason`. Rows with an `error` are ignored.

    `score` is computed for every measured row, but ranking and "best" must use `eligible` only:
    a variant below the accuracy constraint is never a valid answer, however good its other metrics.
    """
    cfg = {**DEFAULT_SCORE_CFG, **(cfg or {})}
    out = df.copy()
    ok = out["error"].isna() if "error" in out else pd.Series(True, index=out.index)
    if "skipped" in out:
        ok &= out["skipped"].isna()
    base_row = baseline_row(out, ok, cfg) if cfg["normalize"] == "baseline" else None

    used, notes = {}, []
    for g, spec in cfg["groups"].items():
        w = float(cfg["weights"].get(g, 0))
        cols = [c for c in spec["cols"] if c in out and out.loc[ok, c].notna().any()]
        if not cols:
            if w:
                notes.append(f"{g}: no data, weight {w} dropped")
            continue
        parts = []
        for c in cols:
            base = float(base_row[c]) if base_row is not None else None
            parts.append(_norm(out.loc[ok, c], spec.get("higher_is_better", False), cfg["normalize"], base))
        stacked = pd.concat(parts, axis=1)
        out.loc[ok, f"score_{g}"] = stacked.min(axis=1) if spec.get("agg") == "min" else stacked.mean(axis=1)
        if w:
            used[g] = w
    total = sum(used.values())
    out["score"] = np.nan
    if total > 0:
        out.loc[ok, "score"] = sum(out.loc[ok, f"score_{g}"] * (w / total) for g, w in used.items())
    eligible, reason = eligibility(out, ok, cfg)
    out["eligible"], out["ineligible_reason"] = eligible, reason
    n_out = int((ok & ~eligible).sum())
    if n_out:
        notes.append(f"{n_out} variant(s) excluded by the accuracy constraint")
        log.warning("%d variant(s) fail the accuracy constraint and are excluded from ranking: %s", n_out,
                    list(out.loc[ok & ~eligible, "variant"]))
    out["score_note"] = "; ".join(notes + ["official formula unknown - weights from config"])
    return out


def rank(df: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Eligible rows, best score first."""
    d = df[df["eligible"]] if "eligible" in df else df
    return d.sort_values("score", ascending=False).head(top)


def candidates(df: pd.DataFrame, pareto_cols: list[str] | None = None, top: int = 3) -> pd.DataFrame:
    """Shortlist for a human to choose from: the best-scoring eligible variants plus every eligible
    variant on a Pareto front (best accuracy for its size / latency / RAM). Never a single answer."""
    d = df[df["eligible"]] if "eligible" in df else df
    if d.empty:
        return d
    keep = set(rank(d, top).index)
    for col in pareto_cols or []:
        flag = f"pareto_{col}"
        if flag in d:
            keep |= set(d.index[d[flag].fillna(False).astype(bool)])
    out = d.loc[sorted(keep)].copy()
    out["why"] = ["; ".join(
        ([f"top-{top} score"] if i in set(rank(d, top).index) else [])
        + [f"Pareto {c}" for c in (pareto_cols or []) if f"pareto_{c}" in d and bool(d.at[i, f"pareto_{c}"])])
        for i in out.index]
    return out.sort_values("score", ascending=False)
