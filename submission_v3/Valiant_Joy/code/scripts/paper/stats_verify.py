"""Independent recompute of the statistics the paper will quote (reviser, 2026-09-19)."""
import glob, re, itertools
from collections import Counter
from functools import lru_cache
from math import comb
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
R = Path("C:/Users/User/Desktop/SEC/edge-ai-toolkit/results")
bis = pd.read_csv(R / "paper/ort_version_bisect.csv")
spin = pd.read_csv(R / "paper/spin_version.csv"); spin["spin"] = spin["spin"].astype(str)
cov = pd.read_csv(R / "paper/coverage_mobilevit.csv")
dcs = pd.read_csv(R / "paper/dynconv_scale.csv")
rng = np.random.default_rng(7)


def sess(df, var, t, rel):
    return df[(df.variant == var) & (df.threads == t) & (df.ort == rel)].p50_ms.values


def three(var, t, rel):
    return [sess(bis[bis.family == "mobilevit"], var, t, rel),
            sess(spin[spin.spin == "1"], var, t, rel),
            sess(cov, var, t, rel)]


@lru_cache(None)
def c(m, n, u):
    if u < 0:
        return 0
    if m == 0 or n == 0:
        return 1 if u == 0 else 0
    return c(m - 1, n, u - n) + c(m, n - 1, u)


def hl_exact(x, y, alpha=0.05):
    d = np.sort((np.log(y)[None, :] - np.log(x)[:, None]).ravel())
    m, n = len(x), len(y)
    tot = comb(m + n, m)
    cdf = np.cumsum([c(m, n, u) / tot for u in range(m * n + 1)])
    k = max(k for k in range(1, m * n + 1) if cdf[k - 1] <= alpha / 2)
    return np.exp(np.median(d)), np.exp(d[k - 1]), np.exp(d[m * n - k])


print("## 1. pooled 1.24 step, 4T")
for lab, var, runs in [("static", "p00_int8", 3), ("fp32", "p00_fp32", 2), ("dyn", "p00_int8_dyn_mm", 2),
                       ("fp32 all3", "p00_fp32", 3), ("dyn all3", "p00_int8_dyn_mm", 3)]:
    a = three(var, 4, "1.23.2")[:runs]; b = three(var, 4, "1.24.4")[:runs]
    X, Y = np.concatenate(a), np.concatenate(b)
    p = stats.mannwhitneyu(Y, X, alternative="two-sided", method="exact").pvalue
    hl, lo, hi = hl_exact(X, Y)
    bs = []
    for _ in range(20000):
        xa = np.concatenate([rng.choice(s, len(s)) for s in a]); yb = np.concatenate([rng.choice(s, len(s)) for s in b])
        bs.append(np.median(yb) / np.median(xa))
    print(f"{lab:10s} n={len(X)}v{len(Y)} X {X.min():.3f}-{X.max():.3f} Y {Y.min():.3f}-{Y.max():.3f} "
          f"ratio_med {np.median(Y)/np.median(X):.3f} MW p={p:.2g} HL {hl:.3f} [{lo:.3f},{hi:.3f}] "
          f"boot [{np.percentile(bs, 2.5):.3f},{np.percentile(bs, 97.5):.3f}]")

print("## 2a. DeiT dyn faster than static per release")
d = bis[bis.family == "deit"]
for t in (1, 4):
    m = d[d.threads == t].groupby(["ort", "variant"]).p50_ms.median().unstack()
    r = m["p00_int8"] / m["p00_int8_dyn_mm"]
    k = int((r > 1).sum()); n = len(r)
    print(t, f"{k}/{n}", f"{r.min():.3f}-{r.max():.3f}", "sign p=%.2g" % stats.binomtest(k, n, 0.5).pvalue)

print("## 2b. 31 pairs -> settings")
rows = []
for f in sorted(glob.glob(str(R / "*" / "sweep.csv"))):
    run = Path(f).parent.name; s = pd.read_csv(f)
    if "latency_p50_ms_t1" not in s or "latency_p50_ms_t4" not in s or "params" not in s:
        continue
    model = {49249: "AE", 2236682: "MobileNetV2", 23528522: "ResNet-50", 3157200: "YOLOv8n"}.get(int(s.params.iloc[0]))
    if model is None:
        continue
    lat = {str(r.variant): r for r in s.itertuples()}
    for v, r in lat.items():
        if not re.fullmatch(r"(\d+_)?p00_int8", v):
            continue
        fp = lat[v.replace("_int8", "_fp32")]
        keep = (fp.latency_p50_ms_t4 / r.latency_p50_ms_t4) / (fp.latency_p50_ms_t1 / r.latency_p50_ms_t1)
        res = v.split("_")[0] if v[0].isdigit() else ("224" if run.startswith("stress") else ("140" if model == "AE" else "32"))
        rows.append(dict(run=run, setting=f"{model}@{res}", keep=keep))
P = pd.DataFrame(rows)
g = P.groupby("setting").keep.median()
print(len(P), "pairs", P.run.nunique(), "runs", (P.keep < 1).sum(), "below; median %.3f" % P.keep.median())
print(g.round(3).to_dict(), "settings below 1: %d/%d, sign p=%.3g" % ((g < 1).sum(), len(g),
      stats.binomtest(int((g < 1).sum()), len(g), 0.5).pvalue))
print("without AE: pairs", (P.setting != "AE@140").sum(), "range keep %.3f-%.3f" % (P[P.setting != "AE@140"].keep.min(), P[P.setting != "AE@140"].keep.max()))

print("## 2c. dyn-conv slower than static in each setting")
sett = {}
for run, pfs, name in [("stress_resnet50_dyn", [""], "ResNet-50@224"), ("classification_dyn", [""], "MobileNetV2@32"),
                       ("detection_dyn", ["192_", "256_", "320_"], "YOLOv8n")]:
    s = pd.read_csv(R / run / "sweep.csv"); lat = {str(r.variant): r for r in s.itertuples()}
    for pf in pfs:
        nm = name + ("@" + pf[:-1] if pf else "")
        for t in (1, 4):
            sett.setdefault(nm, []).append(getattr(lat[pf + "p00_int8_dyn_conv"], f"latency_p50_ms_t{t}")
                                           / getattr(lat[pf + "p00_int8"], f"latency_p50_ms_t{t}"))
for fam in dcs.family.unique():
    q = dcs[dcs.family == fam]
    nm = {"mnv2_32": "MobileNetV2@32", "mnv2_224": "MobileNetV2@224", "y192": "YOLOv8n@192",
          "y320": "YOLOv8n@320", "y640": "YOLOv8n@640"}.get(fam, fam)
    for t in (1, 4):
        med = lambda suf: q[(q.threads == t) & q.variant.str.endswith(suf)].p50_ms.median()
        sett.setdefault(nm, []).append(med("p00_int8_dyn_conv") / med("p00_int8"))
print({k: [round(x, 1) for x in v] for k, v in sett.items()})
print("settings", len(sett), "all slower:", all(min(v) > 1 for v in sett.values()),
      "sign p=%.3g" % stats.binomtest(len(sett), len(sett), 0.5).pvalue)

print("## 4. 7% floor basis")
s = pd.read_csv(R / "classification_mobilevit/sweep.csv"); lat = {str(r.variant): r for r in s.itertuples()}
for t in (1, 4):
    a, b = getattr(lat["p00_int8"], f"latency_p50_ms_t{t}"), getattr(lat["p00_int8_qat"], f"latency_p50_ms_t{t}")
    print(t, f"{a:.3f} {b:.3f} gap {abs(b/a-1)*100:.2f}%")


def null_gap(df, key):
    res = []
    for _, cc in df.groupby(key):
        if len(cc) != 3:
            continue
        lg = np.log(cc.p50_ms.values); res.extend(lg - lg.mean())
    res = np.array(res) * np.sqrt(3 / 2)
    N = 200000
    a = np.median(rng.choice(res, (N, 3)), axis=1); b = np.median(rng.choice(res, (N, 3)), axis=1)
    gap = np.abs(np.exp(b - a) - 1)
    return (np.percentile(gap, 95), np.percentile(gap, 99), (gap > 0.07).mean(), (gap <= 0.07).mean())


kb = ["ort", "family", "variant", "threads"]
print("sweep p95 p99 P>7% pct-of-7%", ["%.4f" % x for x in null_gap(bis, kb)])
print("coverage", ["%.4f" % x for x in null_gap(cov, kb)])
print("dynconv", ["%.4f" % x for x in null_gap(dcs, kb)])
sw = bis.groupby(kb).p50_ms.agg(lambda x: (x.max() - x.min()) / x.median())
print("sweep (max-min)/median: median %.4f max %.4f cells %d" % (sw.median(), sw.max(), len(sw)))

print("## coverage matched static (static_mmw) vs dyn_mm, stratified exact")
cells = []
for (rel, t), cc in cov.groupby(["ort", "threads"]):
    x = cc[cc.variant == "p00_int8_dyn_mm"].p50_ms.values; y = cc[cc.variant == "p00_int8_static_mmw"].p50_ms.values
    cells.append((x, y))
    print(rel, t, "ratio %.3f" % (np.median(y) / np.median(x)), "disjoint" if y.min() > x.max() else "")
obs = 0; dist = Counter({0.0: 1})
for x, y in cells:
    allv = np.concatenate([x, y]); r = stats.rankdata(allv); obs += r[len(x):].sum()
    cnt = Counter(sum(r[list(ix)]) for ix in itertools.combinations(range(len(allv)), len(y)))
    nd = Counter()
    for a_, pa in dist.items():
        for b_, pb in cnt.items():
            nd[a_ + b_] += pa * pb
    dist = nd
T = sum(dist.values()); mean = sum(k * v for k, v in dist.items()) / T
p2 = sum(v for k, v in dist.items() if abs(k - mean) >= abs(obs - mean) - 1e-9) / T
print("stratified exact two-sided p = %.2g" % p2)

print("## spin-off dC/dW with session bootstrap")
so = spin[(spin.spin == "0") & (spin.variant == "p00_int8") & (spin.threads == 4)]
A = so[so.ort == "1.23.2"][["p50_ms", "cpu_ms_per_inf"]].values; B = so[so.ort == "1.24.4"][["p50_ms", "cpu_ms_per_inf"]].values
med = lambda M: np.median(M, axis=0)
dW = med(B)[0] - med(A)[0]; dC = med(B)[1] - med(A)[1]
print("wall %.3f->%.3f cpu %.3f->%.3f dC/dW %.2f" % (med(A)[0], med(B)[0], med(A)[1], med(B)[1], dC / dW))
bs = []
for _ in range(20000):
    a_ = A[rng.integers(0, 3, 3)]; b_ = B[rng.integers(0, 3, 3)]
    w = med(b_)[0] - med(a_)[0]
    if w > 0:
        bs.append((med(b_)[1] - med(a_)[1]) / w)
print("boot 95%% [%.2f, %.2f], share >2.5: %.2f" % (np.percentile(bs, 2.5), np.percentile(bs, 97.5), np.mean(np.array(bs) > 2.5)))
