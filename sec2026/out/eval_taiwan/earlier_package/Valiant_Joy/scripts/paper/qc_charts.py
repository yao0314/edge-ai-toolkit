"""Quality-control charts for the measurement runs, so a result can be read at a glance.

Five chart types, each with one job:
  dots     - one row per model: the measured ratio with its decision band. Answers "who moved?"
  pareto   - counts by reason, sorted, with the cumulative share. Answers "what dominates?"
  control  - a value over time with centre line and limits. Answers "was the machine steady?"
  heatmap  - files x release transitions. Answers "where is the drift?"
  strips   - every file's change, one row per transition. Answers "which release moved things?"

    python scripts/paper/qc_charts.py a2      # the A2 third-party scan
    python scripts/paper/qc_charts.py b       # the drift map (reads whatever batches exist)

The b driver is deliberately NOT run while the map is being measured: rendering costs a couple of
seconds of CPU, and the protocol wants a quiet machine.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "qc"
EXPB = ROOT / "results" / "expB"
FLOOR = 7.0                       # the study's registered latency floor, in percent
DELIVERY_WARN, DELIVERY_SUSPECT = 0.90, 0.95
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d6d6d6"
BLUE, ORANGE, GREEN = "#2a6fb8", "#d1600f", "#1f8a6d"
DIV = LinearSegmentedColormap.from_list("drift", [GREEN, "#eceff0", ORANGE])

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.dpi": 130, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def _minus(s: str) -> str:
    return s.replace("-", "−")


def _title(ax, title: str, sub: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=16)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8.6, color=MUTED, va="bottom")


def dots(df: pd.DataFrame, title: str, sub: str, path: Path) -> None:
    """One row per model: percent change with the +/-FLOOR decision band."""
    df = df.sort_values("pct").reset_index(drop=True)
    h = max(2.4, 0.26 * len(df) + 1.3)
    fig, ax = plt.subplots(figsize=(7.2, h))
    ax.axvspan(-FLOOR, FLOOR, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
    for i, r in df.iterrows():
        beyond = abs(r.pct) >= FLOOR and r.disjoint
        colour = (ORANGE if r.pct > 0 else GREEN) if beyond else "#9aa0a6"
        if r.get("control"):
            colour = INK
        ax.plot([0, r.pct], [i, i], color=colour, lw=1.2, alpha=0.5, zorder=2)
        ax.plot(r.pct, i, "o", ms=6.5, color=colour, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
        if beyond or r.get("control"):
            ax.annotate(_minus(f"{r.pct:+.1f}%"), (r.pct, i), xytext=(9 if r.pct >= 0 else -9, 0),
                        textcoords="offset points", va="center",
                        ha="left" if r.pct >= 0 else "right", fontsize=8.5, color=colour,
                        fontweight="bold")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.label, fontsize=8.2)
    ax.set_xlabel("change in four-thread latency, 1.23.2 to 1.24.4 (%)")
    ax.set_ylim(-0.8, len(df) - 0.2)
    lim = max(12, float(np.abs(df.pct).max()) * 1.35)
    ax.set_xlim(-lim, lim)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0, len(df) - 0.45, f"grey band: the {FLOOR:.0f}% decision floor", ha="center",
            va="bottom", fontsize=8, color=MUTED)
    _title(ax, title, sub)
    fig.savefig(path)
    plt.close(fig)


def pareto(counts: dict[str, int], title: str, sub: str, path: Path, xlabel: str) -> None:
    s = pd.Series(counts).sort_values(ascending=False)
    cum = s.cumsum() / s.sum() * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.bar(range(len(s)), s.values, color=BLUE, width=0.62, zorder=3)
    for i, v in enumerate(s.values):
        ax.annotate(str(v), (i, v), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.index, fontsize=8.2, rotation=0)
    ax.set_ylabel("models")
    ax.set_ylim(0, s.max() * 1.25)
    ax2 = ax.twinx()
    ax2.plot(range(len(s)), cum.values, color=ORANGE, marker="o", ms=4.5, lw=1.4, zorder=4)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("cumulative share (%)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.grid(False)
    for i, v in enumerate(cum.values):
        ax2.annotate(f"{v:.0f}%", (i, v), xytext=(0, 6), textcoords="offset points", ha="center",
                     fontsize=8, color=ORANGE)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel(xlabel)
    _title(ax, title, sub)
    fig.savefig(path)
    plt.close(fig)


def _ichart(ax, values: list[float], labels: list[str], ylabel: str) -> dict:
    """Individuals chart into an existing axis: centre line, +/-3 sigma from the mean moving range."""
    v = np.asarray(values, dtype=float)
    cl = float(v.mean())
    mr = float(np.abs(np.diff(v)).mean()) if len(v) > 1 else 0.0
    sigma = mr / 1.128 if mr else (float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    ucl, lcl = cl + 3 * sigma, cl - 3 * sigma
    ax.axhspan(lcl, ucl, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.axhline(cl, color=MUTED, lw=1.0, zorder=1)
    out = (v > ucl) | (v < lcl)
    ax.plot(range(len(v)), v, "-", color=BLUE, lw=1.2, zorder=2)
    ax.plot(np.arange(len(v))[~out], v[~out], "o", ms=5, color=BLUE, zorder=3)
    if out.any():
        ax.plot(np.arange(len(v))[out], v[out], "o", ms=7, color=ORANGE, zorder=4)
    ax.annotate(f"mean {cl:.3f}", (len(v) - 1, cl), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=MUTED)
    ax.annotate(_minus(f"+3σ {ucl:.3f}"), (0, ucl), xytext=(0, 3), textcoords="offset points",
                fontsize=8, color=MUTED)
    ax.annotate(f"−3σ {lcl:.3f}", (0, lcl), xytext=(0, -11), textcoords="offset points",
                fontsize=8, color=MUTED)
    ax.set_xticks(range(len(v)))
    ax.set_xticklabels(labels, fontsize=7.6, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return {"cl": cl, "ucl": ucl, "lcl": lcl, "sigma": sigma, "n_out": int(out.sum()),
            "spread_pct": float((v.max() / v.min() - 1) * 100) if v.min() > 0 else float("nan")}


def control(values: list[float], labels: list[str], title: str, sub: str, path: Path,
            ylabel: str) -> dict:
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    stats = _ichart(ax, values, labels, ylabel)
    _title(ax, title, sub)
    fig.savefig(path)
    plt.close(fig)
    return stats


def heatmap(m: pd.DataFrame, title: str, sub: str, path: Path, cbar_label: str) -> None:
    """Rows x columns of percent change, diverging from a neutral zero."""
    lim = max(FLOOR * 1.5, float(np.nanmax(np.abs(m.values))) if m.size else FLOOR)
    fig, ax = plt.subplots(figsize=(max(6.4, 0.86 * m.shape[1] + 3.2),
                                    max(2.6, 0.32 * m.shape[0] + 1.8)))
    im = ax.imshow(m.values, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels([_minus(c) for c in m.columns], fontsize=7.4, rotation=45, ha="right")
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(m.index, fontsize=7.8)
    ax.set_xticks(np.arange(-0.5, m.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "·", ha="center", va="center", fontsize=9, color=MUTED)
            elif abs(v) >= FLOOR:
                ax.text(j, i, _minus(f"{v:+.0f}"), ha="center", va="center", fontsize=7.4,
                        color="white" if abs(v) > lim * 0.62 else INK, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label(cbar_label, fontsize=8.4, color=MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.6, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    _title(ax, title, sub)
    fig.savefig(path)
    plt.close(fig)


def strips(groups: list[tuple[str, np.ndarray]], title: str, sub: str, path: Path,
           xlabel: str) -> None:
    """One row per group, every member as a dot, with the +/-FLOOR decision band."""
    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.42 * len(groups) + 1.4)))
    ax.axvspan(-FLOOR, FLOOR, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
    rng = np.random.default_rng(7)               # jitter only, never the data
    for i, (_, vals) in enumerate(groups):
        v = np.asarray(vals, dtype=float)
        v = v[~np.isnan(v)]
        if not len(v):
            continue
        y = i + rng.uniform(-0.16, 0.16, len(v))
        beyond = np.abs(v) >= FLOOR
        ax.plot(v[~beyond], y[~beyond], "o", ms=4.6, color="#9aa0a6", alpha=0.85, zorder=2,
                markeredgecolor="white", markeredgewidth=0.6)
        for k in np.where(beyond)[0]:
            ax.plot(v[k], y[k], "o", ms=6.4, color=ORANGE if v[k] > 0 else GREEN, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.7)
        ax.plot(np.median(v), i, "|", ms=15, color=INK, mew=1.6, zorder=4)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([_minus(g) for g, _ in groups], fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _title(ax, title, sub)
    fig.savefig(path)
    plt.close(fig)


def a2() -> None:
    csv = ROOT / "results" / "paper" / "a2_thirdparty_exploratory.csv"
    d = pd.read_csv(csv)
    man = json.loads((ROOT / "results" / "expA2" / "manifest.json").read_text(encoding="utf-8"))
    meta = {r["variant"]: r for r in (man.get("measured_set") or [])}
    rows = []
    for (fam, var), g in d.groupby(["family", "variant"]):
        g4 = g[g["threads"] == 4]
        a = g4[g4["ort"] == "1.23.2"]["p50_ms"]
        b_ = g4[g4["ort"] == "1.24.4"]["p50_ms"]
        if a.empty or b_.empty:
            continue
        m = meta.get(var)
        name = (m.get("architecture_family") if m else None) or {"mobilevit": "MobileViT-XXS (ours)",
                                                                 "deit": "DeiT-tiny (ours)",
                                                                 "cls": "MobileNetV2 (ours)"}.get(fam, fam)
        rows.append({"label": name, "pct": (b_.median() / a.median() - 1) * 100,
                     "disjoint": bool(b_.min() > a.max() or b_.max() < a.min()),
                     "control": var == "p00_int8"})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    dots(df, "Experiment A2: what one ONNX Runtime release did to unchanged files",
         "18 third-party static INT8 models (grey) and our three controls (black), five sessions per cell",
         OUT / "a2_dots.png")

    pool = json.loads((ROOT / "results" / "expA2" / "pool.json").read_text(encoding="utf-8"))
    by_rule = pool["counts"]["by_first_failing_rule"]
    nice = {"A1": "A1 parse", "A2": "A2 input\nshape/dtype", "A3": "A3 not\nstatic 8-bit",
            "A5": "A5 dynamic\ncontrol flow", "A7": "A7 over\n60 MB", "A8": "A8 screening\n> 60 ms",
            "A9": "A9 family or\npublisher cap", "L4 (gated repository)": "L4 gated\nrepository"}
    pareto({nice.get(k, k): v for k, v in by_rule.items()},
           "Why 28 of 46 candidates never reached the measurement",
           "first failing admission rule, one count per candidate",
           OUT / "a2_pareto.png", "admission rule")

    screens = [(r["variant"], r.get("screen_ms")) for r in (man.get("measured_set") or [])
               if r.get("screen_ms")]
    control([s for _, s in screens], [v for v, _ in screens],
            "Screening pass: latency of each admitted model on 1.24.4",
            "individuals chart; the band is +/-3 sigma from the mean moving range",
            OUT / "a2_screen_control.png", "screening latency (ms)")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("a2_*.png"))))


# ------------------------------------------------------------------ item B: the release-drift map

def _cells(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (file, threads, release): the median of that cell's sessions."""
    if "status" in d.columns:
        d = d[d["status"].fillna("OK").eq("OK")]
    d = d[pd.to_numeric(d["p50_ms"], errors="coerce").notna()]
    g = (d.groupby(["family", "variant", "threads", "ort"], as_index=False)
          .agg(p50=("p50_ms", "median"), n=("p50_ms", "size"),
               delivery=("core_delivery", "median"), rss=("peak_rss_mb", "max")))
    g["file"] = g["family"] + "/" + g["variant"]
    return g


def _load_b() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    plan = json.loads((EXPB / "batch_plan.json").read_text(encoding="utf-8"))
    releases = [str(r) for r in plan["releases"]]
    batch_files = sorted((EXPB / "batches").glob("b[0-9][0-9]_*.csv"))
    frames = [pd.read_csv(p) for p in batch_files]
    d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    anchors = []
    for p in sorted((EXPB / "batches").glob("anchor_*.csv")):
        m = re.match(r"anchor_(.+)_b(\d+)\.csv$", p.name)
        a = pd.read_csv(p)
        a["tag"], a["boundary"] = m.group(1), int(m.group(2))
        anchors.append(a)
    an = pd.concat(anchors, ignore_index=True) if anchors else pd.DataFrame()
    return d, an, releases, [p.name for p in batch_files]


def b() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d, an, releases, done = _load_b()
    print(f"batches on disk: {len(done)} -> {', '.join(done) or 'none yet'}")

    # 1. the machine itself: did the anchors hold still while the map was being measured?
    if not an.empty:
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0))
        for ax, tag in zip(axes, sorted(an["tag"].unique())):
            g = (an[an["tag"] == tag].groupby("boundary", as_index=False)
                 .agg(p50=("p50_ms", "median")))
            st = _ichart(ax, g["p50"].tolist(), [f"b{int(x):02d}" for x in g["boundary"]],
                         "median latency (ms)")
            ax.text(0.995, 0.06, f"{tag}: spread {st['spread_pct']:.1f}%, "
                                 f"{st['n_out']} outside the limits",
                    transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
        _title(axes[0], "Was the machine steady? The two anchors at every batch boundary",
               "same file, same release (1.30.0), three sessions each; drift here is the machine, not a release")
        axes[-1].set_xlabel("batch boundary")
        fig.tight_layout()
        fig.savefig(OUT / "b_anchor_control.png")
        plt.close(fig)

    if d.empty:
        print("no batch CSV yet: anchor chart only")
        return

    c = _cells(d)
    c4 = c[c["threads"] == 4]
    present = [r for r in releases if r in set(c4["ort"])]
    trans = [(present[i], present[i + 1]) for i in range(len(present) - 1)]

    # 2. where is the drift? files x consecutive release transitions
    rows, cols = {}, [f"{a}->{b_}" for a, b_ in trans]
    for f, g in c4.groupby("file"):
        by = g.set_index("ort")["p50"]
        rows[f] = [(by[b_] / by[a] - 1) * 100 if a in by and b_ in by else np.nan
                   for a, b_ in trans]
    m = pd.DataFrame(rows, index=cols).T.sort_index()
    heatmap(m, "Release drift map: what each ONNX Runtime step did to an unchanged file",
            f"{m.shape[0]} files x {m.shape[1]} consecutive releases, four threads, five sessions per "
            f"cell; cells beyond the {FLOOR:.0f}% floor are labelled",
            OUT / "b_drift_heatmap.png", "change in latency (%)")

    # 3. which release moved things? every file's change, one row per transition
    strips([(c_, m[c_].values) for c_ in cols],
           "Which release step moved anything",
           f"one dot per file; the bar is the median; grey band is the {FLOOR:.0f}% decision floor",
           OUT / "b_transition_strips.png", "change in four-thread latency (%)")

    # 4. is the measurement itself sound? core delivery per release
    groups = [(r, c4[c4["ort"] == r]["delivery"].values) for r in present]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axhspan(DELIVERY_SUSPECT, 1.02, color=GRID, alpha=0.5, lw=0, zorder=0)
    rng = np.random.default_rng(11)
    for i, (_, vals) in enumerate(groups):
        v = np.asarray(vals, dtype=float)
        v = v[~np.isnan(v)]
        if not len(v):
            continue
        ax.plot(i + rng.uniform(-0.16, 0.16, len(v)), v, "o", ms=4.2,
                color=BLUE if v.min() >= DELIVERY_SUSPECT else ORANGE, alpha=0.8,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    ax.axhline(DELIVERY_WARN, color=ORANGE, lw=1.0, ls="--", zorder=2)
    ax.annotate("toolkit warns below 0.90", (0, DELIVERY_WARN), xytext=(2, -11),
                textcoords="offset points", fontsize=8, color=ORANGE)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g for g, _ in groups], fontsize=7.8, rotation=45, ha="right")
    ax.set_ylabel("core delivery")
    ax.set_xlabel("ONNX Runtime release")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _title(ax, "Did the four threads actually run? Core delivery per release",
           "cpu_ms per inference / (threads x latency); the grey band is the healthy region")
    fig.savefig(OUT / "b_core_delivery.png")
    plt.close(fig)

    beyond = int((np.abs(m.values) >= FLOOR).sum())
    print(f"cells beyond the {FLOOR:.0f}% floor: {beyond} of {int(np.isfinite(m.values).sum())}")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("b_*.png"))))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "a2"
    if which == "a2":
        a2()
    elif which == "b":
        b()
    else:
        raise SystemExit(f"nothing to draw for {which!r} yet")
