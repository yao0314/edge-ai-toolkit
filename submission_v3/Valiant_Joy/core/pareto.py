"""Pareto front (minimise cost x, maximise quality y) and plotting."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LABELS = {
    "size_bytes": "model size (KB)", "latency_p50_ms": "latency p50 (ms, relative)",
    "peak_rss_mb": "peak RSS above empty session (MB)", "macs": "MACs (M)", "params": "params (M)",
    "cpu_ms_per_inf": "CPU time / inference (ms)", "energy_j": "energy (J / inference)",
}
SCALE = {"size_bytes": 1 / 1024, "macs": 1e-6, "params": 1e-6}


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True for points not dominated by any other (lower-or-equal x AND higher-or-equal y, one strict)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    mask = np.zeros(len(x), bool)
    for i in np.where(keep)[0]:
        dom = keep & (x <= x[i]) & (y >= y[i]) & ((x < x[i]) | (y > y[i]))
        mask[i] = not dom.any()
    return mask


def _sd_column(df: pd.DataFrame, x: str) -> str | None:
    """Spread of repeated measurements for column x, matched to the thread count x itself holds.

    x is the un-suffixed column, which follows measure.primary_threads (4 in the shipped configs).
    Taking the first *_sd_t<N> in sort order gave the 1-thread spread, so the error bars belonged to
    a different measurement than the points they were drawn on.
    """
    hits = [c for c in df.columns if re.fullmatch(rf"{re.escape(x)}_sd_t\d+", c) and df[c].notna().any()]
    if not hits:
        return None
    if "primary_threads" in df and df["primary_threads"].notna().any():
        want = f"{x}_sd_t{int(df['primary_threads'].dropna().iloc[0])}"
        if want in hits:
            return want
    return sorted(hits)[0]


def _pyplot():
    """matplotlib.pyplot, or None when it is not installed.

    Plots are a presentation extra; sweep.csv, summary.json and the deliverable folder are the
    actual output. A hard import here aborted finalize() AFTER sweep.csv had been written, so a
    completed sweep produced no summary.json and scripts/compress.py then refused to package it -
    the whole run lost to a missing optional plotting library.
    """
    try:
        import matplotlib
    except ImportError:
        log.warning("matplotlib is not installed: skipping plots (results and summary are unaffected)")
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_pareto(df: pd.DataFrame, x: str, out_path: str | Path, y: str = "accuracy", title: str = "",
                label_col: str = "variant") -> Path | None:
    """Scatter + Pareto front. Variants that fail the accuracy constraint (`eligible` False) are drawn
    greyed out and never join the front: they are not choices. Error bars show the measurement spread
    over repeated runs (measure.repeats) when it was recorded."""
    plt = _pyplot()
    if plt is None:
        return None

    d = df[df[x].notna() & df[y].notna()] if x in df and y in df else df.iloc[0:0]
    if d.empty:
        return None
    scale = SCALE.get(x, 1)
    xs, ys = d[x].astype(float) * scale, d[y].astype(float)
    elig = d["eligible"].fillna(True).astype(bool).values if "eligible" in d else np.ones(len(d), bool)
    sd_col = _sd_column(d, x)
    xerr = (d[sd_col].astype(float) * scale).fillna(0).values if sd_col else None

    front = np.zeros(len(d), bool)
    if elig.any():
        front[elig] = pareto_mask(xs.values[elig], ys.values[elig])

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=130)
    if xerr is not None:
        ax.errorbar(xs, ys, xerr=xerr, fmt="none", ecolor="0.6", elinewidth=0.9, capsize=2, zorder=1)
    prec = d["precision"] if "precision" in d else pd.Series("model", index=d.index)
    markers = {"fp32": "o", "int8": "s", "int8_dynamic": "D"}
    for p in sorted(prec.unique()):
        sel = ((prec == p).values) & elig
        if sel.any():
            ax.scatter(xs[sel], ys[sel], s=46, marker=markers.get(p, "^"), label=p, zorder=3,
                       edgecolor="white", linewidth=0.6)
    if (~elig).any():
        ax.scatter(xs[~elig], ys[~elig], s=40, marker="x", color="0.65", zorder=2,
                   label="below accuracy constraint")
    order = np.argsort(xs.values[front])
    if front.any():
        ax.plot(xs.values[front][order], ys.values[front][order], "--", color="0.35", lw=1.2,
                label="Pareto front (eligible)", zorder=2, drawstyle="steps-post")
    for xi, yi, name, f, e in zip(xs, ys, d[label_col], front, elig):
        ax.annotate(str(name), (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=7,
                    color="0.1" if f else ("0.45" if e else "0.7"))
    metric = d["metric_name"].iloc[0] if "metric_name" in d else y
    ax.set_xlabel(LABELS.get(x, x) + (f"  (error bars: +-1 sd of {sd_col.split('_sd_')[-1]} runs)" if sd_col else ""))
    ax.set_ylabel(f"{metric} (higher is better)")
    ax.set_title(title or f"{metric} vs {LABELS.get(x, x)}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
