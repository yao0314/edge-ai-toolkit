# Experiment B: release drift of unchanged ONNX files across eleven ONNX Runtime releases

**Registered protocol. Written 2026-09-23 by an independent designer, before any Experiment B
timing exists. Revised 2026-09-23, same day, in answer to a pre-mortem by a separate critic agent;
still before any Experiment B timing exists. No measurement was taken while this file was written
or revised — the only machine work done for the revision was reading stored CSVs, reading model
files with `onnx.load`, and taking SHA-256 of files already on disk.**

Item B of the follow-up study. The question is not "is the new release faster". It is: when the
runtime under a file changes and **the file does not**, how often does its speed move at all, by how
much, in which direction, does the move survive later releases, and is the quantized path more
fragile than the float path of the same weights.

The reason the mean is the wrong summary is already on record in this project's own data. Pooling
every `(model, variant, threads)` cell that exists at the 1.23.2 → 1.24.4 transition across the three
stored CSVs — 88 cells from `results/paper/ort_version_bisect.csv`,
`results/paper/ort_version_scope.csv` and `results/paper/a2_thirdparty_exploratory.csv` — gives
(recomputed this session):

| statistic | all 88 cells | the 84 cells that are not `ae` |
|---|---|---|
| median change | **−0.34 %** | −0.28 % |
| mean change | −1.68 % | +0.33 % |
| interquartile range | −2.60 % … +1.04 % | −2.04 % … +1.13 % |
| minimum | **−51.45 %** (`ae` static INT8, 1 thread) | −28.19 % (`y256` static INT8, 4 threads) |
| maximum | **+46.35 %** (`mv256` static INT8, 4 threads) | +46.35 % |
| total range | **97.8 percentage points** | **74.5 percentage points** |
| cells beyond the 7 % floor | 14 / 88 (15.9 %) | 10 / 84 (11.9 %) |
| cells beyond 20 % | 9 / 88 (10.2 %) | 5 / 84 (6.0 %) |

The second column is there because of pre-mortem item R9: the sub-millisecond `ae` model is the
minimum, and the dispersion headline must not rest on it. It does not — the range is 74.5 points
without it. **Both columns are computed, both are published, and the numeric prediction of §6 P3 is
registered on the conservative one.**

Two caveats on that pilot table, stated because it is quoted as motivation: it counts the 64 px
MobileViT static file three times (once from each CSV — `bisect/mobilevit`, `scope/mv64`,
`a2/mobilevit` control), and it pools cells measured in three different runs. It motivates the
experiment; it is not a result, and nothing in §6 is computed this way.

`CLAUDE.md` already records the same fact in the project's own words: "1.23.2 -> 1.24.4 made a static
hybrid file 31% slower at four threads and a static YOLOv8n file 28% faster". **B measures the
dispersion, on purpose, and fixes in advance what counts as a move.**

---

## 0. Delta index — what this registration fixes

| § | fixed here |
|---|---|
| 1 | the two arms, and the one thing only Arm CONTRAST can answer |
| 2 | the release ladder, which pairs are primary, and why two of them are not single-release steps |
| 3 | the exact grid of both arms, the NOT-APPLICABLE set, the clustered denominators; what already exists and why none of it enters the map |
| 4 | the per-cell quantity, MOVED / UNRESOLVED / STILL, spikes, the meaning of overlap |
| 5 | sessions, order, seeds, quiet machine, what makes a session invalid (no outlier rule), the replication stage, the two anchors |
| 6 | six headline statistics, each with a numeric prediction and a refutation pattern |
| 7 | the release-notes check, blinded with decoys so it cannot become hindsight |
| 8 | validity conditions, what voids the whole map, abort criteria |
| 9 | per-stage time budget for one night, and the pre-flight recomputation that decides the cuts |
| 10 | the sentence the paper may write under each outcome, including the null |
| 11 | artefacts, paths, seeds, and the five harness changes that are permitted |
| 12 | what is barred, and the one bounded mechanism annex that is not |
| 13 | **answers to the pre-mortem, item by item** |

---

## 1. The two arms

### 1.1 Arm BREADTH — 18 third-party static INT8 files, eleven releases

Files we did not make, quantized by toolchains we did not run. These answer "how often does a
published INT8 artefact move when the runtime under it changes", which is the question a deployer
actually has. They are the 18 models admitted under `results/paper/a2_design_registered.md` and
described in `results/expA2/manifest.json` (`counts.measured = 18`, `architecture_families_matrix_rows
= 15`, `publishers = 9`, `export_tooling_values_A11 = 11`). All 18 are present on disk under
`D:\sec_models` in 8 input-shape directories (1 + 2 + 1 + 9 + 1 + 1 + 2 + 1 = 18 `.onnx` files).

**The analysis unit of this arm is the architecture family, n = 15, not the file, n = 18.** That is
not a new decision; it is A2's own registered decision, reused rather than re-argued:
`results/expA2/analysis_unit_note.md:12-13` — "The A2 pool has **18 admitted files in 15
architecture families**. Three families hold two files each: efficientnet (a04, c01), mobilenet_v2
(a02, c02), resnet (a03, c05)" — and `:15-17`, "Primary unit: the architecture family, n = 15 …
both are reported per file underneath." §3.1 gives the denominators this produces.

### 1.2 Arm CONTRAST — our six models as FP32 / static INT8 / dynamic INT8 of the same weights

This is the arm the third-party files structurally cannot support, and the registration for A2 says
so in as many words: `results/paper/a2_design_registered.md:288-289` — "Third-party quantized files
have no FP32 or dynamic sibling from the same quantizer, so that clause cannot be applied here."
Without a float twin, "the INT8 path is fragile" is unfalsifiable: any move could be the model, the
shape, or the release. With a twin built from the same weights, the same shape and the same run, the
float sibling is the control.

Six models have a complete FP32 / static-INT8 / dynamic-INT8 trio on disk, verified by SHA-256 in
this session:

| family | directory | shape | variants | sizes (MB) | dynamic kind |
|---|---|---|---|---|---|
| `mv64` | `results/mobilevit_r64/models` | 3,64,64 | `p00_fp32` / `p00_int8` / `p00_int8_dyn_mm` | 4.01 / 1.80 / 2.70 | MatMul |
| `mv256` | `results/mobilevit_r256/models` | 3,256,256 | `p00_fp32` / `p00_int8` / `p00_int8_dyn_mm` | 4.01 / 1.80 / 2.70 | MatMul |
| `deit224` | `results/deit_224/models` | 3,224,224 | `p00_fp32` / `p00_int8` / `p00_int8_dyn_mm` | 22.12 / 6.16 / 6.42 | MatMul |
| `r50` | `results/stress_resnet50_dyn/models` | 3,224,224 | `p00_fp32` / `p00_int8` / `p00_int8_dyn_conv` | 94.03 / 24.04 / 23.69 | **Conv** |
| `mnv2` | `results/classification_dyn/models` | 3,32,32 | `p00_fp32` / `p00_int8` / `p00_int8_dyn_conv` | 8.92 / 2.62 / 2.42 | **Conv** |
| `y256` | `results/detection_dyn/models` | 3,256,256 | `256_p00_fp32` / `256_p00_int8` / `256_p00_int8_dyn_conv` | 12.69 / 3.53 / 3.37 | **Conv** |

The `dynamic kind` column is load-bearing from here on: the three **Conv** variants do not run on any
release before 1.24.4 (§3.2), and that is fixed into the grid rather than discovered at 02:00.

The three `_dyn` directories are used for `r50`, `mnv2` and `y256` because each one holds the **whole
trio**, and its FP32 and static files are byte-identical to the copies in the non-`_dyn` directory
(SHA-256 verified this session: `19b355585c64` / `41a987f1bc46` for `r50`, `c0980c89d299` /
`ad770b998718` for `mnv2`, `1aa3b72e152c` / `c4a6e37c4a14` for `y256`). Using one directory per model
puts all three variants of a model inside one shuffled batch, which §5.2 requires.

Input shapes were read from the files themselves this session (`onnx.load`, no runtime): `mv64`
`[batch,3,64,64]`, `mv256` `[batch,3,256,256]`, `deit224` `[batch,3,224,224]`, `r50`
`[batch,3,224,224]`, `mnv2` `[batch,3,32,32]`, `y256` `[batch,3,256,256]`; opset 17 for all six.

### 1.3 Deliberately excluded, with the reason

* **`ae` (1-D autoencoder, `results/anomaly/models`)** has `p00_fp32` and `p00_int8` only — there is
  no dynamic sibling on disk, and building one is barred this round. It therefore cannot enter the
  trio grid. This is a loss: `ae` produced the **largest** moves in the stored data (−38.7 % FP32 1T,
  −51.5 % static 1T, −44.1 % / −41.3 % at 4T from 1.23.2 to 1.24.4, all disjoint,
  `results/paper/ort_version_scope.csv`). `ae` is measured as a **two-variant appendix** (FP32 +
  static, same grid) and reported separately; it contributes to §6 P1/P2/P5 and to persistence,
  **not** to the paired FP32-vs-dynamic test, and in P3 it falls inside the sub-millisecond class
  that is reported separately (§6 P3, and R9 in §13).
* **DeiT-tiny at 32 px** (`results/classification_deit/models`, the `deit` family of
  `ort_version_bisect.csv`) is a different file from `deit224` (SHA-256 `4e3cf32eadffc1cb` vs
  `deffb7d51b6a0331`). It is not added; six trios plus `ae` is the registered set.
* **Third-party FP32 twins** do not exist and are not sought.

---

## 2. The release ladder and the release pairs

### 2.1 The eleven releases

`results/expB/envs.json` records eleven environments under `D:\sec_envs`, each reporting its ORT
version and numpy 2.4.6: **1.16.3, 1.20.1, 1.21.1, 1.22.1, 1.23.2, 1.24.4, 1.25.1, 1.26.0, 1.27.0,
1.28.0, 1.30.0**. All eleven interpreters report **Python 3.11.9** (checked this session with
`python -V`; no ORT import).

**The eleven runtimes are eleven distinct binaries, and this was checked by hash, not by version
string** (pre-mortem R5). SHA-256 (first 12 hex) of the three shipped binaries in
`Lib/site-packages/onnxruntime/capi/`, taken this session:

| release | `onnxruntime_pybind11_state.pyd` | `onnxruntime.dll` | `onnxruntime_providers_shared.dll` |
|---|---|---|---|
| 1.16.3 | `11545c116906` | *absent (monolithic in this release)* | `fd04e715c9f4` |
| 1.20.1 | `83f4061d6bc0` | `2f92a249542d` | `ff0a090202ab` |
| 1.21.1 | `e6ca84c822e2` | `a93e75cd85a2` | `c80df011efa5` |
| 1.22.1 | `cf57f10c1d27` | `b7d046825412` | `121329b5f87d` |
| 1.23.2 | `967a5a759076` | `93457c4f1790` | `056da4c9eec7` |
| 1.24.4 | `fbd869b4b1ff` | `ef720fc44a4e` | `f123ce892247` |
| 1.25.1 | `fae1c8be8b8d` | `bb656592046f` | `c56c986e157e` |
| 1.26.0 | `28a78fe15545` | `b7dfcb4dea88` | `669fa0fbb553` |
| 1.27.0 | `071e3be75111` | `7b9ef55ecaa6` | `4cc699327866` |
| 1.28.0 | `8fb176cc2c96` | `14e186627f10` | `d9a39bc6b34e` |
| 1.30.0 | `06b712319c15` | `44741724166a` | `fabb99add45a` |

All eleven triples are distinct. **Three install trees are duplicates of a release already on the
ladder and are excluded by hash, not by name:** the project `.venv` is byte-identical to
`D:\sec_envs\ort1300` in all three binaries, and `.envs\ort1232` / `.envs\ort1244` are byte-identical
to `D:\sec_envs\ort1232` / `D:\sec_envs\ort1244`. See V1 in §8.

### 2.2 Both pair families are analysed, and they answer different questions

**Adjacent pairs (primary).** The question as posed — "when a release changes, how often does an
unchanged file move" — has the adjacent transition as its natural unit. There are **10** adjacent
transitions on this ladder.

**Baseline-anchored (secondary).** Every release against **1.23.2** as a fixed baseline, 10
comparisons per series. This is the form the project already uses
(`results/paper/FINDINGS.md:2540`: "Change of median latency vs 1.23.2 (%)"), it is what a deployer
sitting on an old pin actually faces, and it is the only way to read cumulative drift and reversion
off one row.

Both are computed from the same sessions; neither costs extra machine time.

### 2.3 Two of the ten adjacent pairs are not single-release steps — stated now, not later

The ladder skips upstream releases in two places. **1.16.3 → 1.20.1** spans four minor versions and
**1.28.0 → 1.30.0** spans two. The other eight are single-minor steps:

| # | transition | kind |
|---|---|---|
| 1 | 1.16.3 → 1.20.1 | **GAPPED** (4 minors) |
| 2 | 1.20.1 → 1.21.1 | single |
| 3 | 1.21.1 → 1.22.1 | single |
| 4 | 1.22.1 → 1.23.2 | single |
| 5 | 1.23.2 → 1.24.4 | single |
| 6 | 1.24.4 → 1.25.1 | single |
| 7 | 1.25.1 → 1.26.0 | single |
| 8 | 1.26.0 → 1.27.0 | single |
| 9 | 1.27.0 → 1.28.0 | single |
| 10 | 1.28.0 → 1.30.0 | **GAPPED** (2 minors) |

**The primary denominator for every "how often" statistic in §6 is the eight single-minor
transitions.** The two gapped transitions are reported separately and never pooled into the primary
share, because a move across them cannot be attributed to one release. The distinction is not
cosmetic: in `results/paper/ort_version_bisect.csv` the eight single-minor transitions moved
**17/96 = 17.7 %** of cells while the two gapped ones moved **14/24 = 58.3 %** (recomputed this
session).

---

## 3. The grid, the NOT-APPLICABLE set, and the denominators

### 3.0 NOT-APPLICABLE: `*_dyn_conv` does not exist below 1.24.4

`results/paper/FINDINGS.md:2571-2572`: "On 1.23.2 the dynamic-Conv files do not run at all:
NOT_IMPLEMENTED, 'Could not find an implementation for ConvInteger(10)'." Three of the six CONTRAST
models carry exactly that variant (`r50`, `mnv2`, `y256`, §1.2). The receipt is in the stored data:
`results/paper/ort_version_scope.csv` has **462 rows where the full grid is 480**, and the 18 missing
rows are exactly `mnv2/r50/y256 × *_dyn_conv × {1,4} threads × 1.23.2 × 3 reps` — enumerated this
session, no other cell is missing and every present cell has exactly 3 reps.

**Registered, before the first session:**

> `p00_int8_dyn_conv` and `256_p00_int8_dyn_conv` are **NOT-APPLICABLE** on every release below
> 1.24.4 — 1.16.3, 1.20.1, 1.21.1, 1.22.1, 1.23.2 — unless the §9 stage-0 load probe proves
> otherwise. They exist on the six releases 1.24.4 … 1.30.0, giving those series 5 adjacent
> transitions of which 4 are single-minor.

The four older releases have never been tested with that variant, so the *fact* is settled by the
stage-0 probe (§9 stage 0, V8) and not by assumption: one `InferenceSession` construction per (file,
release), 38 files × 11 releases = **418 constructions**, no timing, written to
`results/expB/load_probe.json` and hash-stamped **before any timing session runs**. Whatever the
probe finds is the NOT-APPLICABLE set; the denominators below are recomputed from it mechanically by
the analysis script, and the version in this file is the expected case. A cell in the NOT-APPLICABLE
set is never measured, never counted in a denominator, and is listed by name in the paper.

The same probe settles the other load hazard (pre-mortem R10). Read from the files this session with
`onnx.load`: **`b05` is IR version 10, opset 20** — the only file on the ladder above IR 9 / opset 19
— and five files carry `com.microsoft` contrib operators in the graph (`c04` QLinearAdd; `b15`
QLinearGlobalAveragePool; `b16` QLinearAveragePool + QLinearConcat; `a06` QLinearAdd + QLinearConcat;
`b18` QLinearAdd + QLinearConcat + QLinearLeakyRelu). `b05` on 1.16.3 is the expected casualty. It is
handled the same way: probe, then NOT-APPLICABLE, then out of the denominator with its name printed.

### 3.1 Arm BREADTH

18 files × **1 variant** (each file is its own variant; they have no siblings) × **2 thread counts**
(1 and 4) × **11 releases** × **5 sessions** = **1 980 sessions**, 36 measured cells per release.

Thread counts: 1 and 4, as in every version run this project has done
(`results/paper/FINDINGS.md:2169`). Four threads is where the registered step rule lives
(`results/paper/a_design_registered.md:300`); one thread is the control that separates a kernel
change (moves at both) from a thread-pool change (moves at four only).

**Denominators (pre-mortem R4):**

| unit | adjacent | single-minor | clusters |
|---|---|---|---|
| **architecture family × threads × transition — PRIMARY** | 15 × 2 × 10 = **300** | 15 × 2 × 8 = **240** | 15 families |
| file × threads × transition — reported underneath | 18 × 2 × 10 = 360 | 18 × 2 × 8 = 288 | — |

A family is MOVED at `(threads, transition)` if **any** of its files is MOVED there, UNRESOLVED if
none moved and any is UNRESOLVED, STILL otherwise — the disjunction rule of
`results/expA2/analysis_unit_note.md:16-17`. The three two-file families are `efficientnet`
(a04, c01), `mobilenet_v2` (a02, c02), `resnet` (a03, c05); the other twelve hold one file each.

### 3.2 Arm CONTRAST

| block | models | variants | releases | sessions |
|---|---|---|---|---|
| MatMul-dynamic trios | `mv64`, `mv256`, `deit224` | 3 | 11 | 3 × 3 × 2 × 11 × 5 = **990** |
| Conv models, float + static | `r50`, `mnv2`, `y256` | 2 | 11 | 3 × 2 × 2 × 11 × 5 = **660** |
| Conv models, dynamic | `r50`, `mnv2`, `y256` | 1 | **6** (1.24.4 … 1.30.0) | 3 × 1 × 2 × 6 × 5 = **180** |
| | | | | **1 830** |

Plus the `ae` appendix: 1 model × 2 variants × 2 threads × 11 releases × 5 = **220 sessions**.

**Denominators (pre-mortem R1 and R4):**

| unit | adjacent | single-minor | clusters |
|---|---|---|---|
| pair-cell `(model, variant, threads, transition)` | 180 + 120 + 30 = **330** | 144 + 96 + 24 = **264** | 12 `(model, threads)` / 6 `model` |
| matched trio `(model, threads, transition)` | 60 + 30 = **90** | 48 + 24 = **72** | 12 / 6 |
| `ae` appendix pair-cells | 40 | 32 | 1 |

The old figures in this registration's first version — 288 single-minor pair-cells and 96 trios —
counted `*_dyn_conv` cells on releases where the variant cannot run. **264 and 72 are the
registered denominators**, and the analysis script recomputes both from
`results/expB/load_probe.json` rather than hard-coding them.

### 3.3 Map total

**1 830 (CONTRAST) + 1 980 (BREADTH) + 220 (`ae`) = 4 030 map sessions**, plus ~202 anchor sessions
(§5.6) and the replication stage (§5.5). Single-minor pair-cells: **264 + 288 (files) + 32 = 584**,
or **264 + 240 (families) + 32 = 536** on the primary units.

### 3.4 Which cells already exist — and why **none of them enters the map**

This is the one place where this design departs from the brief, and the departure is the design's
main load-bearing decision. It is made on the study's own recorded evidence.

**What exists that overlaps the grid:**

| CSV | overlapping configuration | sessions/cell |
|---|---|---|
| `results/paper/a2_thirdparty_exploratory.csv` | all 18 BREADTH models × 2 threads × **2 releases** (1.23.2, 1.24.4) = 72 of the arm's 396 cells | **5** |
| `results/paper/ort_version_bisect.csv` | `mv64` (byte-identical file: `p00_fp32` = `8bba9618966c5739`, `p00_int8` = `911de4ee32579660`, `p00_int8_dyn_mm` = `9c0025026d71ead8`, same in `results/classification_mobilevit/models` and `results/mobilevit_r64/models`) × 3 variants × 2 threads × **all 11 releases** = 66 of the arm's cells | **3** |
| `results/paper/ort_version_scope.csv` | all 6 CONTRAST models × up to 3 variants × 2 threads × **4 releases** (1.23.2, 1.24.4, 1.25.1, 1.30.0), plus `ae` | **3** |

**They are not reused as map cells, for two independent reasons, either of which is sufficient.**

**(a) Cross-run comparison manufactures moves above the floor.** The project measured this:
`results/paper/FINDINGS.md:2894-2897` — byte-identical files, matched by SHA-256, show between-run
latency ratios of **1.004–1.174** across 17 of 18 file groups, and the entry concludes "**A harness
ratio carries up to about 10% between-run spread beyond the within-run floor.**" A 17.4 % between-run
ratio on an unchanged file is more than twice the 7 % floor. Comparing a 1.23.2 median measured on
2026-09-23 with a 1.25.1 median measured tonight would put the run difference inside the release
difference with no way to separate them. The whole point of the harness's global shuffle
(`scripts/paper/ort_version_bisect.py:190-193`: "Shuffle across EVERYTHING, so variant, thread count
and version are all decorrelated from position in time") is defeated the moment version is
confounded with run.

**(b) Three sessions cannot support the disjoint-range rule.** With `n = m` sessions per side, the
exact two-sided probability that one side's range lies entirely below the other's, under
exchangeability, is `2 / C(2n, n)`:

| sessions per side | exact two-sided p |
|---|---|
| 3 | 2/20 = **0.100** |
| 4 | 2/70 = 0.029 |
| **5** | 2/252 = **0.0079** |
| 7 | 2/3432 = 0.00058 |

`results/paper/FINDINGS.md:3499` already uses the n=5 value — "the exact two-sided Mann-Whitney p is
2/252 = 0.008 for this cell alone". The bisect and scope runs used 3 sessions
(`results/paper/FINDINGS.md:2169`), so a "disjoint" mark in those CSVs carries **p = 0.10**. In a
536-cell map that is 54 expected chance moves. Unusable.

**What the existing CSVs are used for instead — both uses are registered now:**

1. **Pilot.** Every numeric prediction in §6 is derived from them, and from nothing measured after
   this file is written.
2. **Run-to-run replicate.** After the map is complete, the 1.23.2 and 1.24.4 columns of Arm BREADTH
   are compared cell-by-cell against `a2_thirdparty_exploratory.csv` (same files, same thread counts,
   same 5-session protocol, different run and different night). That comparison is a **direct
   measurement of the between-run term** for 72 cells — the quantity `FINDINGS.md:2894-2897`
   estimates at up to ~10 %. It is reported as a validity figure, never as drift.

**Nothing in `a2_thirdparty_exploratory.csv`, `ort_version_bisect.csv` or `ort_version_scope.csv` is
re-measured, overwritten, moved or edited.** B writes new CSVs at new paths (§11).

---

## 4. The primary quantity and the rule for "this cell moved"

### 4.1 Per-cell quantity

A **cell** is `(arm, model, variant, threads, release)`. Its primary quantity is

> **`m` = the median of the cell's five session `p50_ms` values.**

`p50_ms` is the child's median per-call latency over its measurement window
(`scripts/paper/ort_version_bisect.py:73-85`: 1.5 s warm-up, then a window of `max(2.0 s, 40
iterations)`, `p50 = np.percentile(lat, 50)`). Between-session variance is the dominant term, so the
five repeats are five **processes**, never more iterations inside one
(`scripts/paper/ort_version_bisect.py:24-26`).

**A cell is analysed at exactly five sessions or not at all** (pre-mortem R2). The analysis asserts
`status = OK` on exactly five rows for every `(arm, model, variant, threads, release)`. Anything else
— four, three, six — makes the cell **VOID**, and every pair-cell touching it VOID. A cell is never
analysed at reduced `n`, because the exact level of the disjointness test is a function of `n`
(0.0079 at 5, 0.029 at 4, 0.100 at 3) and a map that silently mixes them has no chance budget. The
harness change (d) of §11.2 writes a row for every attempted session, so this is computable from the
artefact instead of from scrollback.

A **pair-cell** is `(arm, model, variant, threads, release_a → release_b)`. Its quantity is

> **`Δ = 100 × (m_b / m_a − 1)` percent**, and the two session ranges `[min, max]` over five sessions.

Absolute medians in ms are recorded and reported alongside every ratio. Ratios alone hide the case
where a 40 % move is 0.2 ms.

### 4.2 MOVED / UNRESOLVED / STILL

Taken unchanged in form from `results/paper/a_design_registered.md:298-306` (§5.1 "This model
stepped, in exactly these words"), generalised from one transition to any transition and from
one-sided to two-sided, because B must count files that got **faster** as well as slower:

* **MOVED** iff `|Δ| ≥ 7 %` **AND** the two cells' five-session ranges are **disjoint**
  (`max` of one side's five `p50_ms` < `min` of the other side's five).
* **STILL** iff `|Δ| < 7 %`. (Range overlap is not consulted; a sub-floor difference is not a move
  however clean it looks. `c56_56x3/c03` at 1 thread in `a2_thirdparty_exploratory.csv` is exactly
  this case: −5.96 % with disjoint ranges — **STILL**.)
* **UNRESOLVED** iff `|Δ| ≥ 7 %` **but the ranges overlap**.

Where the floor is written: **7 %** is the study's between-session latency floor —
`results/paper/FINDINGS.md:629` ("quote 7% as the latency floor"),
`results/paper/FINDINGS.md:1958` ("Noise floors used by every rule: latency 7% between sessions"),
`FLOOR_REL = 0.07` at `results/paper/e2_design_registered.md:370`. Its history — it was 2 %, the
challengers' plan-identical re-runs forced it to 7 % — is at `results/paper/FINDINGS.md:588-629`. The
disjoint-range form is registered at `results/paper/a_design_registered.md:300-302` and reused at
`results/paper/a2_design_registered.md:286-287`. **B does not change the floor, in either
direction.**

**Deviation from Experiment 26, stated in advance.** `results/paper/FINDINGS.md:2171-2173` defines a
step as additionally requiring persistence at later releases *and* quiet siblings. B **separates**
those: persistence is §4.4 and is *measured and reported*, not folded into the move definition; the
sibling condition is the object of study in Arm CONTRAST (§6 P4) and would be circular if it were
also part of the definition. Arm BREADTH has no siblings at all
(`results/paper/a2_design_registered.md:288-289`).

### 4.3 What an overlapping pair means — and why it is not "no move"

UNRESOLVED means exactly this: **the median difference is larger than the floor, and five sessions a
side were not enough to separate the two distributions.** It is evidence of neither stability nor
drift. It is never silently folded into either.

Every "share that moved" in §6 is therefore reported as a **bracket**, both ends stated:

> lower bound = `MOVED / N`   upper bound = `(MOVED + UNRESOLVED) / N`

with `N` = all pair-cells in the denominator (MOVED + UNRESOLVED + STILL). If the bracket is wider
than 10 percentage points on the primary statistic, the headline is the bracket, not a point value.

UNRESOLVED pair-cells are the input to the replication stage (§5.5) together with MOVED ones.

### 4.4 Persistence

For a MOVED pair-cell at transition `r_i → r_{i+1}` with sign `s = sign(Δ)`, and for every later
release `r_j` on the ladder:

* **PERSISTENT** — for all `r_j` after `r_{i+1}`: `sign(m_j/m_i − 1) = s` and `|m_j/m_i − 1| ≥ 7 %`.
  (The file never comes back.)
* **REVERTED** — there is an `r_k` with `|m_k/m_i − 1| < 7 %`, and it stays inside the floor through
  1.30.0.
* **MIXED** — anything else; the exact sequence is printed rather than labelled.

Reported as three counts plus the **survival curve**: of the MOVED pair-cells, the fraction still
outside the floor 1, 2, 3, … releases later. A MOVED pair-cell at transition 9 or 10 has too few
later releases to classify and is reported as **NOT-ASSESSABLE**, with its count given.

### 4.5 A spike is one event, not two pair-cells (pre-mortem R4-i)

Eleven releases of one `(model, variant, threads)` **series** are one trajectory: its 8 single-minor
Δs are built from 9 shared medians, so a single aberrant release median *necessarily* produces two
MOVED pair-cells of opposite sign. Registered, before measurement:

> Two consecutive MOVED pair-cells within one series, at `r_{i-1}→r_i` and `r_i→r_{i+1}`, with
> **opposite signs**, are one **SPIKE** at `r_i`, counted as **one** level-shift event.
> All other MOVED pair-cells are one **LEVEL SHIFT** each.

Every §6 share is published twice: as the pair-cell share (the unit the question is posed in) and as
**distinct level-shift events per series** (the unit that is not double-counted). If the two differ
by more than 20 % relative, the paper leads with the event count. Spikes are also listed by name,
because a spike at `r_i` is itself a finding — one release that is an outlier in both directions.

---

## 5. Sessions, ordering, seeds, preconditions, invalid sessions

### 5.1 Sessions per cell: five

Fixed by §3.4(b): five is the smallest number for which the disjoint-range rule carries an exact
two-sided p of 0.0079, matching `results/paper/FINDINGS.md:3499` and the five sessions
`results/paper/a_design_registered.md:298` already assumes. Four would give 0.029; three gives 0.10.

**The false-move budget, pre-registered.** Under exchangeability, expected chance MOVED cells are
`N × 0.0079`:

| denominator | N | expected chance MOVED |
|---|---|---|
| CONTRAST single-minor pair-cells | 264 | **2.09** |
| BREADTH single-minor file pair-cells | 288 | **2.27** |
| BREADTH single-minor **family** pair-cells | 240 | **2.27** (a two-file family has chance `1−(1−0.0079)² = 0.0157`) |
| `ae` appendix | 32 | 0.25 |
| **map total, single-minor** | **584 files / 536 families** | **4.6 / 4.2** |
| map total, all ten transitions | 730 / 670 | 5.8 / 5.3 |

Every share reported in §6 is stated next to this expectation. The `|Δ| ≥ 20 %` subset is effectively
free of it: a chance-disjoint pair must *also* clear 20 %, and the project's resampled null for the
quieter sweep puts `P(gap > 7 %)` at 0.07 % (`results/paper/FINDINGS.md:2886-2890`).

**Sessions are never reduced below five.** Every abort path in §8 cuts cells, never sessions.

### 5.2 Ordering, batching, seeds

* **One harness invocation per model-family batch**, output to a file, never to a pipe — the A2
  pattern (`scripts/paper/a2_exploratory_run.py:8`, `:47-57`, `:144`). A pipe nobody drains is what
  hung an earlier attempt.
* **Inside each batch, all applicable releases, both thread counts, all variants and all five repeats
  are shuffled together** under a logged seed (`scripts/paper/ort_version_bisect.py:190-193`). This
  is what makes every release-to-release comparison a within-batch comparison, and thermal drift a
  variance term instead of a bias.
* **No cell may span two invocations** (V7, §8).
* **P-core pinning** as the harness already does it (`pick_cpus(t, "auto")`,
  `scripts/paper/ort_version_bisect.py:203`), recorded per row in `pinned_cpus`.
* **Spin left at the runtime default** (`--spin` empty). Spin is a separate factor with its own
  registered run (`results/paper/spin_version.csv`); mixing it in would double the grid and confound
  the map. What makes this safe rather than an unmeasured confound is V9 in §8 and R11 in §13.

**The two arms are interleaved, so arm is not confounded with time** (pre-mortem R6). Batches run in
this fixed order, heaviest first within each arm, alternating arms. Base seed 71; batch *k* in this
order uses seed 71 + *k*.

| # | arm | batch | families arg | variants | seed | est. |
|---|---|---|---|---|---|---|
| 1 | CONTRAST | r50 | `r50=results/stress_resnet50_dyn/models:3,224,224` | `p00_fp32,p00_int8,p00_int8_dyn_conv` | 72 | 54.7 min |
| 2 | BREADTH | c3_224x224 | `c3_224x224=D:/sec_models/c3_224x224:3,224,224` | a01,a02,a03,b02,b03,b15,b16,c01,c05 | 73 | 69.4 min |
| 3 | CONTRAST | y256 | `y256=results/detection_dyn/models:3,256,256` | `256_p00_fp32,256_p00_int8,256_p00_int8_dyn_conv` | 74 | 23.6 min |
| 4 | BREADTH | c224_224x3 | `c224_224x3=D:/sec_models/c224_224x3:224,224,3` | a04,c02 | 75 | 14.7 min |
| 5 | CONTRAST | mv64 | `mv64=results/mobilevit_r64/models:3,64,64` | `p00_fp32,p00_int8,p00_int8_dyn_mm` | 76 | 22.1 min |
| 6 | BREADTH | c3_320x320 | `c3_320x320=D:/sec_models/c3_320x320:3,320,320` | b01,b18 | 77 | 14.7 min |
| 7 | CONTRAST | mv256 | `mv256=results/mobilevit_r256/models:3,256,256` | `p00_fp32,p00_int8,p00_int8_dyn_mm` | 78 | 22.1 min |
| 8 | BREADTH | c3_299x299 | `c3_299x299=D:/sec_models/c3_299x299:3,299,299` | b05 | 79 | 9.1 min |
| 9 | CONTRAST | deit224 | `deit224=results/deit_224/models:3,224,224` | `p00_fp32,p00_int8,p00_int8_dyn_mm` | 80 | 22.1 min |
| 10 | BREADTH | c1_64x64 | `c1_64x64=D:/sec_models/c1_64x64:1,64,64` | a05 | 81 | 7.4 min |
| 11 | CONTRAST | mnv2 | `mnv2=results/classification_dyn/models:3,32,32` | `p00_fp32,p00_int8,p00_int8_dyn_conv` | 82 | 18.8 min |
| 12 | BREADTH | c3_112x112 | `c3_112x112=D:/sec_models/c3_112x112:3,112,112` | c04 | 83 | 7.4 min |
| 13 | appendix | ae | `ae=results/anomaly/models:1,140` | `p00_fp32,p00_int8` | 84 | 14.7 min |
| 14 | BREADTH | c3_240x320 | `c3_240x320=D:/sec_models/c3_240x320:3,240,320` | a06 | 85 | 7.4 min |
| 15 | BREADTH | c56_56x3 | `c56_56x3=D:/sec_models/c56_56x3:56,56,3` | c03 | 86 | 7.4 min |

The eight BREADTH batches are exactly the `families_arg` of `results/expA2/manifest.json` minus the
three control families (`counts.harness_families = 8`). On these estimates the **time-weighted centre
of CONTRAST's sessions is 156.7 min into the night and BREADTH's is 159.3 min** — 2.6 minutes apart
out of 315.6, against ~186 minutes apart under a CONTRAST-then-BREADTH order. That is the whole point
of the interleave, and the actual figure is recomputed from `t_utc` after the run and published.

Replication-stage seed **91**; release-notes decoy sampling seed **97**; cluster-bootstrap seed
**101**. A2's seeds 63/64/65 (`results/expA2/manifest.json`, `harness` block) are not reused.

### 5.3 Quiet-machine precondition

Per batch, before its first session, in the VA4 form registered at
`results/paper/a_design_registered.md:400-401` and `results/paper/e2_design_registered.md:421`:

> AC power; no WSL distro running; machine-wide `Win32_PerfFormattedData_PerfOS_Processor` `_Total`
> **≤ 15 %** averaged over 5 s, achieved within 10 one-minute checks.

`LoadPercentage` and `other_load_pct` are not used and must not appear — both are on record as false
alarms (`CLAUDE.md`, "Measurement discipline"). Each check is appended to
`results/expB/precondition.json` with its UTC timestamp, sample count, mean and max. A batch whose
precondition was not recorded is invalid (§5.4 iii).

### 5.4 What makes a **session** invalid — the complete list

Only these three, decided before the run:

1. The child subprocess returned non-zero or hit the 900 s timeout
   (`scripts/paper/ort_version_bisect.py:204-208`).
2. The interpreter reported an ORT version other than the one expected for that environment
   (resolution is by **reported** version, `scripts/paper/ort_version_bisect.py:124-131`, and by
   binary hash, V1).
3. The batch's quiet-machine record is missing.

**Every attempted session is written to the CSV, valid or not** (pre-mortem R2). Harness change (d),
§11.2: a failed or skipped session gets a row with `p50_ms` empty and a `status` field
(`OK` / `NONZERO_EXIT` / `TIMEOUT` / `MISSING_FILE` / `VERSION_MISMATCH`). The old harness printed the
failure to stdout and executed `continue` (`scripts/paper/ort_version_bisect.py:206-208`), and skipped
a missing file with no message at all (`:201-202`), so a short cell was invisible in the artefact.

**There is no outlier rule, and this is deliberate.** The study has never had one and B does not
introduce one. A session is **never** dropped for being slow, fast, or far from its cell's median.
The reason is structural: the disjoint-range rule is precisely the statistic that one stray session
*should* be able to veto. Removing the stray to "clean up" the cell would convert a 0.0079 test into
an unbounded one. A cell whose five sessions are widely spread is reported as wide, and lands in
UNRESOLVED if its medians differ — which is the correct answer.

Core delivery (`cpu_ms_per_inf / (threads × p50)`,
`scripts/paper/ort_version_bisect.py:26-29`, `:88`) is **recorded and reported, not gated**. This
follows the scope run (`results/paper/FINDINGS.md:2534`: "462 cells measured, 78 below the gate (all
reported ungated"). Historical below-0.95 rates in the three stored CSVs, computed this session:
bisect **18.9 %** (75/396), scope **16.9 %** (78/462), A2 **25.2 %** (106/420). A gate at 0.95 would
discard a quarter of the map. The per-batch below-gate rate is a validity figure (V6, §8), and every
published table states the arm's below-gate share.

### 5.5 Replication stage — an independent run, never a pooled one

After all fifteen batches complete, every **MOVED** and every **UNRESOLVED** pair-cell is re-measured
at **five fresh sessions per side, inside one shuffled invocation per model family, seed 91**.

The first version of this registration added two sessions per side and recomputed the verdict on
seven — five from the map invocation and two from an invocation hours later. That is precisely the
cross-run comparison §3.4(a) rejects, and it is biased in one direction: a between-run offset widens
*both* sides' ranges without moving either median, so pooled ranges stop being disjoint and MOVED
flows to WEAKENED. Registered instead, in the stratified form the project already has on record
(`results/paper/FINDINGS.md:2882-2884`: "Stratified by run, all four 3-v-3 strata are disjoint: exact
p = (1/20)^4"):

* The replication run is its own **stratum**. Its verdict is computed from its own five-a-side
  sessions alone, by the §4.2 rule, with its own exact level of 0.0079.
* **The map's verdict is never revised, and the two runs' sessions are never pooled into one range.**
* Per pair-cell, the published label is the pair of verdicts:
  **REPLICATED** (MOVED in both runs, same sign; joint exact level `0.0079² = 6.2 × 10⁻⁵`),
  **NOT-REPLICATED** (MOVED in the map, not in the replication run),
  **RESOLVED-MOVED** / **RESOLVED-STILL** for pair-cells that were UNRESOLVED in the map,
  **NOT-TESTED** if the stage was truncated by the cap below.
* The difference between the two runs' Δ on the same pair-cell is itself published: it is a second,
  independent measurement of the between-run term of `FINDINGS.md:2894-2897`, on cells chosen for
  being large.

**Cap and fixed priority, so the stage cannot overrun the night.** Pair-cells are processed in
descending `|Δ|`, ties broken by `(arm, model, variant, threads, transition index)` ascending. Before
the stage starts, its per-cell cost is computed from the map's own medians with the §9 cost model;
the list is cut at the last pair-cell whose cumulative cost stays under **75 minutes**, and everything
below the cut is published as NOT-TESTED with its `|Δ|`. No judgement is exercised at 04:00: the order
and the cap are fixed here.

Predicted cost at the §6 P1 point predictions: ~100 pair-cells over ~150 distinct cells × 5 sessions
≈ 750 sessions ≈ 55 min.

### 5.6 The anchors — two of them, one small and one heavy

Two fixed configurations are measured once every **40 map sessions**, in every batch, for the whole
night, always on the single chosen 1.30.0 interpreter:

| anchor | cell | pilot median | what it can see |
|---|---|---|---|
| **A-small** | `mnv2` `p00_int8`, 4 threads | 0.4396 ms (1.30.0, scope) | per-call overhead drift; a 2.0 s window is ~4 500 calls |
| **A-heavy** | `r50` `p00_fp32`, 4 threads | 24.497 ms (median over releases, scope) | sustained four-core throughput; a 2.0 s window is ~82 full-width iterations |

The first version of this registration had only A-small. A 0.44 ms probe measures the fixed cost of a
call; it is not obviously able to see a frequency or thermal change that would move a 1 434 ms cell
(pre-mortem R7). A-heavy is the answer, and it is nearly free: both anchors cost `0.52 + 1.5 + 2.0 =
4.02 s` per session because both windows are floor-limited at 2.0 s. ~101 injections × 2 anchors ×
4.02 s ≈ **13.5 minutes** for the night.

Nothing about either anchor changes across the night, so any movement in them is the machine. They
are the only instruments that can tell a genuine release effect from a window of machine state.
**V4 is evaluated on both, and fires if either moves.** Anchor rows go to their own CSV and never
into the map.

---

## 6. The headline statistics — fixed in advance, each with a prediction and a refutation

All six are computed on the **eight single-minor transitions** unless stated. Gapped transitions are
reported separately and excluded from the primary numbers. Every share is printed next to the chance
expectation of §5.1, and next to its distinct-level-shift count (§4.5).

---

### P1 — the share of cells that moved at all

**Statistic.** Per arm: `MOVED / N` and `(MOVED + UNRESOLVED) / N`, with `N` the single-minor
denominators of §3.1 and §3.2 — **BREADTH `N = 240` family pair-cells (primary), 288 file pair-cells
(underneath); CONTRAST `N = 264`** — and the same split by thread count and by variant class
(FP32 / static INT8 / dynamic INT8).

**Interval.** Clopper–Pearson assumes independent pair-cells and they are not (§4.5, §3.1), so the
published interval is a **cluster bootstrap**: 2 000 resamples with replacement over the arm's
clusters — 15 architecture families for BREADTH, 6 models for CONTRAST — seed 101, percentile
interval. `n_clusters` is printed beside every share. Clopper–Pearson is reported only as a labelled
"independence-assuming" secondary.

**Arm comparison.** "BREADTH is lower than CONTRAST" is claimed only if the two cluster-bootstrap
intervals **do not overlap**. Otherwise the paper states the two shares and says the difference is
not established at this sample size.

**Pilot.** `ort_version_bisect.csv` (our own two families, 3 sessions, so flagging is *easier* than
in B): **17/96 = 17.7 %** of single-minor pair-cells moved — FP32 12.5 %, static INT8 18.8 %, dynamic
INT8 21.9 %. `ort_version_scope.csv` (3 sessions, 3 transitions): 18 of 114 available pair-cells
= 15.8 %. `a2_thirdparty_exploratory.csv` (18 third-party files, **5 sessions**, the single
transition 1.23.2 → 1.24.4): **0 of 36** moved; the largest was −5.96 %, below the floor.

**PRE-REGISTERED PREDICTION.**
*BREADTH:* **2 %–12 %** of the 240 family pair-cells, point prediction **6 % (≈ 14 cells)**. The
reasoning is explicit: third-party files showed 0/36 at the one transition that moved our own files
most, but that is a single transition and eight are being added.
*CONTRAST:* **10 %–25 %** of 264, point **16 % (≈ 42 cells)**, from the bisect's 17.7 % discounted
for the harder 5-session disjointness test.
*Both:* BREADTH's share is **lower** than CONTRAST's, with non-overlapping cluster intervals.

**REFUTES THE FRAMING.** BREADTH ≤ **1.5 %** (≤ 4 cells, inside the 2.3 expected by chance at
240 × 0.0079) **and** CONTRAST ≤ 3 % → almost nothing moves anywhere; "release drift" is not a
phenomenon of released artefacts and the study's own steps were a property of two hand-built files.
This is the **null outcome** and it is publishable (§10).

---

### P2 — the share that moved by more than 20 %

**Statistic.** `|Δ| ≥ 20 %` among MOVED pair-cells, as a share of `N`, reported with absolute ms
alongside so a large ratio on a 0.4 ms model is visible as such, and split by the sub-millisecond
class of P3.

**Pilot.** Bisect single-minor: **3/96 = 3.1 %**. Pooled 1.23.2 → 1.24.4 across all three CSVs: 9 of
88 cells (10.2 %), all of them our own files (`ae`, `mv64`, `mv256`, `y256`, `mobilevit`); 5 of 84
(6.0 %) when `ae` is set aside.

**PRE-REGISTERED PREDICTION.** BREADTH **0–2 of 240 (0 %–0.8 %)**, point **1**. CONTRAST **4–18 of
264 (1.5 %–6.8 %)**, point **8 (3.0 %)**.

**REFUTES THE FRAMING.** BREADTH ≥ 5 % beyond 20 % → published third-party INT8 artefacts are *more*
volatile than our own, A2's 0/36 was an artefact of measuring one transition, and the paper must lead
with third-party volatility rather than with our own step.

---

### P3 — dispersion inside a single release

**Statistic.** For each transition, over all map cells measured at it: median `Δ`, mean `Δ`, IQR, min,
max, **range = max − min**, and the count beyond the floor. The headline is the pair
(**median**, **range**) per transition.

Cells per transition, from §3: transitions 1–5 have 30 CONTRAST + 36 BREADTH + 2 `ae` = **68**;
transitions 6–10 have 36 + 36 + 2 = **74** (the `*_dyn_conv` series exist from 1.24.4).

**Reported twice, and the prediction is on the conservative half** (pre-mortem R9). A pair-cell is
**sub-millisecond** if `max(m_a, m_b) < 1.0 ms`. Eight CONTRAST/appendix cells per transition are
(all four `ae` cells, `mnv2 p00_fp32` at both thread counts, `mnv2 p00_int8` at both); **no
third-party file is sub-millisecond** — the smallest is `b15` at 4 threads, 1.454 ms. The project has
already ruled on what those cells mean: `results/paper/FINDINGS.md:2577` — the `ae` model "got ~40-50%
faster in every arm at 1.24.4: a per-call overhead change, visible only because the model is that
small; **its ratios are indicative only**."

> **P3-all** — every cell.
> **P3-core** — sub-millisecond cells excluded. **The numeric prediction below is registered on
> P3-core.**

**Pilot (1.23.2 → 1.24.4, three CSVs, computed this session).** All 88 cells: median **−0.34 %**,
range **97.8 pt**. The 84 non-`ae` cells: median **−0.28 %**, range **74.5 pt**. Removing the
sub-millisecond model costs 23 points of range and changes nothing qualitative.

**PRE-REGISTERED PREDICTION (on P3-core).** At least **one** of the eight single-minor transitions
shows **range ≥ 40 pt while its median |Δ| ≤ 3 %** — a release that is "flat on average" and moves
individual files by tens of percent in both directions. Predicted count of such transitions: **1–3**,
point **2**. Predicted median |Δ| across all eight transitions: **≤ 4 %** for seven of eight.

**REFUTES THE FRAMING.** Every transition has **range ≤ 15 pt** on P3-core → the mean *is* a fair
summary of a release, a vendor's "X % faster" transfers to individual files, and the premise of item
B is wrong.

---

### P4 — is the quantized path more fragile than the float path

**Design.** Arm CONTRAST only. A **matched trio** is `(model, threads, transition)` where all three
variants exist: 48 (MatMul models, 8 single-minor transitions) + 24 (Conv models, 4 single-minor
transitions from 1.24.4) = **72 single-minor trios**, each with an FP32, a static-INT8 and a
dynamic-INT8 verdict on the *same* weights, the *same* shape and the *same* shuffled batch.

**72 trios are not 72 independent observations** (pre-mortem R4): they are 12 `(model, threads)`
series, or 6 models, each contributing up to 8 correlated transitions. The primary test is therefore
at the cluster level, and it is exact.

**Primary test — exact cluster sign-flip (randomisation) test.** For each cluster `c` and each of its
single-minor transitions, score `+1` if static MOVED and FP32 did not, `−1` if FP32 MOVED and static
did not, `0` otherwise; `d_c` is the cluster's sum. The statistic is `S = Σ_c d_c`. Under the null
that the variant label is exchangeable *at the cluster level*, each `d_c` may flip sign
independently, so the null distribution is the complete enumeration of `2^K` sign vectors
(`K = 12` clusters, 4 096 vectors — exact, no seed). Two-sided
`p = #{ |Σ ε_c d_c| ≥ |S| } / 2^K`. Run twice: FP32 vs static, FP32 vs dynamic.
**Conservative secondary at `K = 6` clusters (model only, both thread counts inside the cluster).**
A direction is claimed only if the 12-cluster and the 6-cluster tests agree in sign.

**Magnitude test — exact cluster sign test on `|ln(m_b/m_a)|`.** Per cluster, the median over its
transitions of `|ln r_static| − |ln r_fp32|`; exact two-sided sign test over the 12 clusters
(minimum attainable `p = 2/4096`). The trio-level Wilcoxon signed-rank on all 72 trios is reported as
a labelled independence-assuming secondary, as is the trio-level exact McNemar and the pooled 2×2
Fisher.

**Interval.** Cluster bootstrap over the 6 models, 2 000 resamples, seed 101, on the discordance
ratio (INT8-only-moved / FP32-only-moved), with `n_clusters` printed beside it.

**Pilot (bisect, 4 clusters, 32 trios, computed this session).** FP32 vs static: 6 discordant — 4
static-only, 2 FP32-only; per cluster `d_c` = `mobilevit t1 −1`, `mobilevit t4 0`, `deit t1 0`,
`deit t4 +3`, so `S = +2` with **2 non-zero clusters of 4**. FP32 vs dynamic: 3 discordant, all
dynamic-only, `S = +3`, 1 non-zero cluster. Median `|ln|` difference (static − FP32) per cluster:
−0.0025, −0.0053, +0.0321, +0.0177.

**Power, computed and registered rather than hoped for.** A cluster test over `K` clusters of which
`k` are non-zero cannot go below `p = 2/2^k`: it needs `k ≥ 6` to reach 0.05 (`k = 5` gives 0.0625).
Scaling the pilot's 2-of-4 to 12 clusters predicts **4–9 non-zero clusters**.

**PRE-REGISTERED PREDICTION.** 12–25 discordant trios; INT8-side share **60 %–75 %**; 4–9 non-zero
clusters; and therefore **the cluster sign-flip test is predicted NOT to reach p < 0.05** (predicted
p in the range 0.03–0.50). The prediction is registered in that form on purpose: the honest headline
will be *direction, the per-cluster table, and a cluster-bootstrap interval on the discordance ratio*,
not a significance claim, and saying so now prevents the interval from being quietly replaced by a
p-value if it happens to land under 0.05. The **magnitude** sign test is predicted to land in
0.02–0.25, with static INT8's `|ln|` larger than its FP32 twin's in **8–11 of 12 clusters**.

**REFUTES THE FRAMING.** FP32-only-moved **≥** INT8-only-moved (`S ≤ 0`) in both the 12- and the
6-cluster test → the quantized path is **not** more fragile; drift is a property of the release, not
of the precision, and the second half of item B's premise is dead. Also refuting, more subtly: if the
discordance ratio is ~1.0 but *both* are high, drift is real and precision-blind — a different, still
publishable, finding.

**What stays inconclusive.** With 6 models on one CPU, a true INT8:FP32 fragility ratio below about
2:1 cannot be separated from 1:1 at this sample size. That is stated in the paper rather than hidden
in a p-value.

---

### P5 — does a move persist

**Statistic.** Of the MOVED pair-cells with at least two later releases on the ladder: counts of
PERSISTENT / REVERTED / MIXED (§4.4), plus the survival curve at 1, 2, 3 … releases later, split by
arm and by variant class, and with SPIKEs (§4.5) listed separately — a spike is by construction not
persistent.

**Pilot.** One case is on record and registered: the MobileViT-XXS 64 px static file's step at
1.24.4 persisted — `results/paper/FINDINGS.md:2171-2172` built persistence at 1.25.1 and 1.30.0 into
the Experiment 26 rule, and `ort_version_scope.csv` shows `mv64` static 4T at +25.3 % (1.24.4) then
−0.7 % and +5.3 % across the next two, i.e. it stays up.

**PRE-REGISTERED PREDICTION.** **≥ 50 %** of assessable MOVED pair-cells are PERSISTENT to the *next*
release; **≤ 35 %** are still outside the floor at 1.30.0.

**REFUTES THE FRAMING.** **< 20 %** persist even one release → a "move" is not a property of a
release at all; the map is a noise map and must be published as a measurement-methodology result
(§10), with the anchors and the run-to-run replicate as its evidence.

---

### P6 — does a MOVED verdict reproduce in an independent run (new)

**Statistic.** Of the MOVED pair-cells re-measured in the §5.5 replication stage: the share
**REPLICATED**, overall and split by `|Δ|` band (7–20 %, ≥ 20 %), by arm and by variant class; plus
the distribution of `Δ_map − Δ_replication` on those cells, which is a direct measurement of the
between-run term on large-effect cells.

**Pilot.** `results/paper/FINDINGS.md:2880-2884`: the 1.24 step of the byte-identical 64 px MobileViT
static file reproduced in **four independent runs** (1.311x, 1.297x, 1.355x, 1.253x), with all four
3-v-3 strata disjoint. That is the behaviour of a real move. `FINDINGS.md:2894-2897` gives the other
side: byte-identical files carry up to ~10 % between-run spread, which is what a marginal 7–10 % move
has to survive.

**PRE-REGISTERED PREDICTION.** **≥ 80 %** of MOVED pair-cells with `|Δ| ≥ 20 %` replicate;
**≥ 45 %** of all MOVED pair-cells replicate; and the replication rate is **higher** in the ≥ 20 %
band than in the 7–20 % band.

**REFUTES THE FRAMING.** **< 25 %** of MOVED pair-cells replicate → the map's MOVED verdicts are not
reproducible; P1's share is then published **as an upper bound only**, labelled as such in every
table, and the paper's finding becomes a measurement-methodology one (§10 outcome 9).

---

## 7. The release-notes check — blinded, with decoys, so it cannot become hindsight

The failure mode is specific and easy: read the notes after seeing which cells moved, find a
plausible sentence, and declare the move "documented". The procedure below makes that impossible by
construction, at the cost of one extra agent and about twenty minutes.

**Step 1 — retrieve, before analysing.** For each of the eleven releases, the published release body
at `https://github.com/microsoft/onnxruntime/releases/tag/v<version>` is saved verbatim to
`results/expB/notes/<version>.md`. For each file: retrieval UTC, the URL, and the SHA-256 go into
`results/expB/notes_manifest.json`. **Nothing is interpreted at retrieval time.** (Retrieval happens
in the run round; no network access is used in this design round.)

**Step 2 — seal the results first.** When both arms and the replication stage are complete, the
analyst writes `results/expB/moves_sealed.json` — every pair-cell with `Δ`, both session ranges, its
MOVED / UNRESOLVED / STILL verdict and its replication label — and logs its SHA-256. The result is
frozen before any note is read against it.

**Step 3 — build the blind set with decoys.** Under seed **97**: take every MOVED pair-cell, and draw
an equal number of **STILL** pair-cells from the *same transition*, matched on thread count where
possible. Strip each item to `(release pair, model id, variant class, thread count)` — **no
direction, no magnitude, no verdict, no ordering that reveals which is which**. Write
`results/expB/notes_blind_set.json` and log its SHA-256.

**Step 4 — an independent agent classifies.** A separate agent, which has not seen either CSV or
`moves_sealed.json`, receives only the saved notes and the blind set. For each item it returns one of:

* **DOCUMENTED** — the newer release's notes name a change to a code path this item's model
  plausibly exercises (a named operator or operator family, the CPU execution provider, the
  quantization / QDQ path, a named fusion, the thread pool, or a kernel/MLAS dependency change),
  **and** the direction claimed in the notes is not contradicted by the item's configuration. It must
  quote the sentence, with the release and line.
* **NOT-DOCUMENTED** — the notes contain nothing that reaches this item.
* **AMBIGUOUS** — only a generic claim ("performance improvements", "various bug fixes") with no
  named path. **A generic claim is never DOCUMENTED.**

Output → `results/expB/notes_blind_verdicts.json`, SHA-256 logged, **before** unblinding.

**Step 5 — unblind and report a 2×2.** DOCUMENTED rate among MOVED cells vs among decoys, with a
two-sided Fisher exact test on the four exact counts, and — because items from one model are not
independent — a cluster permutation over models/families, 2 000 draws, seed 101, reported beside it.

**Who decides, and what the author may do.** The blinded agent's verdict is final. The author may not
overturn it, upgrade an AMBIGUOUS to a DOCUMENTED, or re-run the classification with a different
prompt. A disagreement is recorded as a dated note in `results/expB/notes_blind_verdicts.json`
alongside the original verdict, which is never edited. The order — retrieve, seal, blind, classify,
unblind — is itself the record, and every step is hash-stamped so the order is checkable afterwards
by anyone.

**The null here is a result.** If the DOCUMENTED rate is the same for moved cells and decoys, the
notes carry no information about which files moved, and the paper says exactly that (§10).

---

## 8. Validity conditions, what voids the map, abort criteria

### 8.1 Validity conditions

* **V1 — eleven distinct runtime binaries.** Each environment must report its expected ORT version
  and numpy 2.4.6 (`results/expB/envs.json`), Python 3.11.9, **and match the SHA-256 triple recorded
  in §2.1**. Resolution is by reported version *and* by hash. Version strings cannot tell two builds
  apart, and this ladder has three duplicate install trees: the project `.venv` is byte-identical to
  `D:\sec_envs\ort1300`, and `.envs\ort1232` / `.envs\ort1244` to their `D:\sec_envs` twins.
  `interpreters()` globs `ort*` under `--scratch` and **always** appends the project `.venv`
  (`scripts/paper/ort_version_bisect.py:128-132`), so with `--scratch D:\sec_envs` it yields twelve
  interpreters and 1.30.0 would silently receive double the sessions under two `env_dir` values.
  Harness change (b) drops any interpreter whose binary triple is already claimed. Exactly one
  interpreter per release is used; its absolute path and its triple go into
  `results/expB/manifest.json`. If two environments that should differ have the **same** triple, the
  map is VOID for that pair of releases — they are one runtime.
* **V2 — byte identity of every model.** SHA-256 of all 18 + 20 `.onnx` files recorded before the
  first session and again after the last. Any change voids that model everywhere. The premise of the
  whole study is that **the file did not change**; the quantizer is held fixed and every runtime is
  fed the same file (`scripts/paper/ort_version_bisect.py:18-21`).
* **V3 — quiet machine**, §5.3, per batch, recorded.
* **V4 — anchor stability, on both anchors.** For each anchor, `A_first` = median of the first third
  of its sessions, `A_last` = median of the last third. If **either** anchor has
  `|A_last/A_first − 1| ≥ 7 %`, the machine drifted over the night: the map is **VOID for
  cross-transition magnitude comparisons**, and only within-transition contrasts survive — which
  includes P4, whose trios are all within one transition. Labelled as such wherever published. The
  two anchors are also reported against each other: A-heavy moving while A-small does not is a
  frequency/thermal signature and is stated as one.
* **V5 — uniform-shift label, not an exclusion.** For each transition, if **≥ 80 %** of its cells
  move the same sign **and** the median `|Δ|` for that transition is **≥ 7 %**, the transition is
  labelled **UNIFORM-SHIFT**. It stays in P1, P2 and P4; it is additionally reported on its own with
  three pieces of evidence: (i) the per-batch breakdown — a genuine release effect must appear
  independently in most of the fifteen batches, which are fifteen separate invocations in fifteen
  different time windows; (ii) both anchors' behaviour in the same windows; (iii) the replication
  stage's verdicts for its cells. **A transition is moved out of the primary numbers only if V4 has
  fired**, i.e. only when the anchors say the machine itself drifted.
  The first version of this registration excluded UNIFORM-SHIFT transitions from P1/P2/P4 on the
  grounds that a near-uniform move is "at least as consistent with machine state as with two
  releases". That justification is contradicted by the design's own shuffle: inside a batch, release
  is decorrelated from position in time (`scripts/paper/ort_version_bisect.py:190-193`), so a window
  of bad machine state raises the variance of *every* release in that batch and cannot produce a
  shift aligned to one transition. Excluding uniform shifts would have deleted the largest true
  release effects and made P3's "dispersion, not mean" headline partly definitional. The pattern that
  prompted the rule is real and is worth reporting: in `results/paper/ort_version_bisect.csv` **all 12
  series fall** at 1.28.0 → 1.30.0 (−1.5 % to −26.2 %, median −12.5 %) and **11 of 12 rise** at
  1.26.0 → 1.27.0 (−0.3 % to +22.5 %, median +9.2 %). Under this registration both are labelled,
  published and counted.
* **V6 — core delivery reported, not gated** (§5.4). If a batch's below-0.95 rate exceeds **45 %**
  (roughly double the worst historical rate of 25.2 %), the batch is re-run once; a second failure
  voids that batch.
* **V7 — no cell spans two invocations.** The replication stage is a separate run and is analysed as
  a separate stratum (§5.5); its sessions are never pooled with the map's. Grounds:
  `results/paper/FINDINGS.md:2894-2897`, between-run ratios of byte-identical files reaching 1.174.
* **V8 — the NOT-APPLICABLE set is fixed by the stage-0 load probe** (§3.0, §9 stage 0), hash-stamped
  before the first timing session. A `(file, release)` pair that fails to construct an
  `InferenceSession` is NOT-APPLICABLE: never measured, never in a denominator, listed by name. If
  the probe's NOT-APPLICABLE set differs from the expected one in §3.0, the denominators are
  recomputed from the probe and the difference is stated in the paper.
* **V9 — the spin default is observed, not assumed** (pre-mortem R11). With ORT's default spinning,
  idle intra-op threads burn CPU, so `core_delivery ≈ 1` at four threads; with spinning off they
  sleep and it collapses. Measured, this session, in `results/paper/spin_version.csv`: median
  `core_delivery` **1.015** (range 0.976–1.114) at `spin=1`, **0.488** (0.377–0.633) at `spin=0`.
  The diagnostic is therefore already in every row the harness writes. Registered rule: for each
  release, the median 4-thread `core_delivery` over the map is published; if any release's median
  falls below **0.80**, that release is flagged **SPIN-DEFAULT-CHANGED** and every transition
  touching it is reported separately, since a spin-policy change is worth up to 1.5× and would swamp
  any release effect. On the stored evidence this is not expected to fire: across all eleven releases
  in `ort_version_bisect.csv` the 4-thread median `core_delivery` lies between **0.975 and 1.022**.

### 8.2 What voids the whole map

Any one of:

* V1 fails for **≥ 3** of the eleven releases;
* V2 fails for **≥ 2** models;
* either anchor drifts by **≥ 15 %** (V4 at more than double its threshold);
* fewer than **8** of the eleven releases complete **both** arms;
* a model file's SHA-256 differs between the pre-run and post-run record in a way that cannot be
  traced to a recorded action.

A voided map is reported as voided, with both anchor series and the precondition log published. It is
not partially rescued.

### 8.3 Abort criteria and the cuts, chosen now rather than at 03:00

* **The cuts are a pre-flight decision, not a 03:00 one** (pre-mortem R6 and R13). At the end of
  stage 0 the run script recomputes the night's estimate from (i) the measured session-construction
  cost per `(file, release)` that the load probe records and (ii) the pilot `p50` table of §9. If the
  recomputed estimate for map + anchors exceeds **6.0 h**, the cuts below are applied **before batch
  1 starts**, in this order and no other, until the estimate is under 6.0 h:
  1. Arm BREADTH drops **1 thread**, keeping 4 threads (the thread count the step rule is written at,
     `results/paper/a_design_registered.md:300`): −990 sessions, ≈ −1.15 h. Applied to the whole arm
     or not at all, so the arm stays internally homogeneous.
  2. Arm BREADTH drops releases **1.21.1 and 1.22.1**, keeping both ladder ends and every transition
     with known activity.
  3. The `ae` appendix is dropped (−220 sessions, −0.25 h).
* **Hard stop.** If the wall clock at a **batch boundary** exceeds **8.0 h** from the start of batch
  1, stop. Batches are whole models across all releases and no cell spans one, so an incomplete night
  loses named models, not releases, and the map stays internally comparable. The unstarted batches are
  listed in the paper.
* **Quiet machine.** If a batch boundary fails ten one-minute checks three times in one night: stop.
  The night resumes at the next unstarted batch on a later night — with the caveat that cross-batch
  comparison is not made (V7), which the grid already respects since every comparison is within a
  model.
* **Never** cut sessions below five (§5.1), never drop a model to make a pattern cleaner, never
  extend the night past the hard stop to "just finish".
* **Session timeout.** A session exceeding the 900 s child timeout
  (`scripts/paper/ort_version_bisect.py:204-205`) is recorded with `status = TIMEOUT` and not retried
  inside the batch. Any cell without exactly five `status = OK` rows is VOID (§4.1).
* **Thermal / process hygiene.** After every batch, check for leftover `python` processes and stop
  own processes if the machine overloads (`CLAUDE.md`, "Working rules"). No background process
  outlives the run.

---

## 9. Time budget — one night

Per-session cost is calibrated from the A2 run, not guessed:
`t ≈ 0.52 s + 1.5 s (warm-up) + max(2.0 s, 40 × p50)`. The **0.52 s** fixed overhead is the
session-weighted mean residual after fitting that model to A2's eleven batches against their logged
wall clocks (`results/expA2/exploratory_run.log:7-17`), recomputed this session per batch:

| batch | sessions | wall (s) | model (s) | residual per session | largest file in batch |
|---|---|---|---|---|---|
| c1_64x64 | 20 | 78 | 70.0 | 0.40 | 18.41 MB |
| c224_224x3 | 40 | 162 | 140.0 | 0.55 | 23.33 MB |
| c3_112x112 | 20 | 78 | 70.0 | 0.40 | 1.30 MB |
| c3_224x224 | 180 | 756 | 663.6 | 0.51 | 24.56 MB |
| c3_240x320 | 20 | 78 | 70.0 | 0.40 | 0.44 MB |
| c3_299x299 | 20 | 102 | 89.2 | 0.64 | 23.71 MB |
| c3_320x320 | 40 | 162 | 140.0 | 0.55 | 6.83 MB |
| c56_56x3 | 20 | 78 | 70.0 | 0.40 | 0.11 MB |
| cls / deit / mobilevit | 60 | 246 | 210.0 | 0.40–0.70 | < 5 MB |

**The residual does not track file size** over the 0.11–24.56 MB range actually measured (0.40 s at
0.11 MB and at 0.44 MB; 0.51 s at 24.56 MB), so the constant is a constant, not a per-megabyte term
(pre-mortem R13). The largest CONTRAST file is `r50 p00_fp32` at **94.03 MB**, 3.8× beyond that range,
which is why stage 0 **measures** the construction cost per `(file, release)` rather than
extrapolating it, and why §8.3's cuts are decided from that measurement before batch 1.

| stage | content | sessions | estimate | budget |
|---|---|---|---|---|
| 0 | pre-flight: V1 hashes, V2 hashes, **V8 load probe (418 session constructions, wall clock recorded, no timing measurement)**, notes retrieval, precondition, budget recomputation | 0 | 30 min | 45 min |
| 1 | the fifteen interleaved batches (§5.2): CONTRAST 1 830 + `ae` 220 + BREADTH 1 980 | 4 030 | **5.26 h** | 6.25 h |
| — | anchors, both, interleaved every 40 sessions | ~202 | 14 min | 20 min |
| 2 | replication stage at 5 fresh sessions a side, capped | ≤ 750 | 55 min | 75 min |
| 3 | analysis, seal, blind set, mechanism annex — **no timing** | 0 | 45 min | 60 min |
| | **total** | **~4 980** | **7.7 h** | **9.6 h** |

**Per arm, the measurement estimate that the cuts are judged against:**

| arm | sessions | estimate | note |
|---|---|---|---|
| **CONTRAST** | 1 830 | **2.72 h** | 3.43 h / 1 980 sessions in the worst case that `*_dyn_conv` loads on the five older releases too |
| `ae` appendix | 220 | 0.25 h | |
| **BREADTH** | 1 980 | **2.29 h** | 1.15 h if cut 1 fires |
| anchors | 202 | 0.23 h | |
| replication | ≤ 750 | ≤ 1.25 h | hard cap, fixed priority (§5.5) |

Per-batch estimates are in the §5.2 table. The heaviest CONTRAST batch is `r50` at 54.7 min and the
heaviest BREADTH batch is `c3_224x224` (nine files) at 69.4 min.

One cost concentration, stated so it is not discovered at 02:00: `r50 p00_int8_dyn_conv` at 1 thread
has a scope-run median of **1 434.5 ms**, so one session costs ~59 s. Because the variant is
NOT-APPLICABLE below 1.24.4, it is 30 sessions and **~30 min**, not 55 sessions and ~54 min — 9 % of
the night for one cell instead of 16 %. It is kept: dynamic INT8 on convolutions is the project's
registered worst case (`CLAUDE.md`, "Quantization defaults"), its drift is part of the question, and
cutting sessions is barred. `y256 256_p00_int8_dyn_conv` at 1 thread (263 ms, ~12.6 s per session,
6.3 min) is the second largest.

---

## 10. What the paper may write under each outcome — one sentence each

Written now, so no sentence is composed after seeing the numbers. `k`, `n` and percentages are filled
from the results; nothing else changes.

1. **Drift is common and INT8-biased** (P1 CONTRAST ≥ 10 %, P4 `S > 0` in both cluster tests with the
   magnitude sign test at p < 0.05) — "Across eleven ONNX Runtime releases and *n* unchanged files,
   *k* % of (file, thread count, adjacent-release) cells changed median latency by at least 7 % with
   disjoint session ranges, and the quantized siblings of the same weights moved in *k* cells where
   their float twin did not, against *j* the other way (*c* of 12 series clusters agreeing)."
2. **Drift is common and precision-blind** (P1 high, discordance ratio ≈ 1) — "Release-to-release
   latency drift of unchanged files is common (*k* % of cells) but not a quantization effect: float
   and INT8 siblings of the same weights moved at indistinguishable rates (*k* vs *j* discordant
   cells, exact cluster p = *p*)."
3. **Third-party artefacts are stable, ours are not** (BREADTH ≤ 3 %, CONTRAST ≥ 10 %, non-overlapping
   cluster intervals) — "Published third-party INT8 artefacts were near-immune to release change
   (*k*/240 architecture-family cells), while files from our own export path moved in *k*/264, which
   locates the fragility in the export toolchain rather than in the runtime."
4. **The mean hides the map** (P3-core fires: some transition with range ≥ 40 pt and median |Δ| ≤ 3 %)
   — "ONNX Runtime *x.y* moved the median unchanged file by *k* % while moving individual files across
   a *r*-point range, so a single release-level speed-up figure does not transfer to any particular
   model."
5. **Moves do not last** (P5 < 20 % persistent) — "Most changes that cleared the floor at one release
   did not survive the next, so a single-release comparison overstates what a deployer will keep."
6. **Release notes do not predict the map** (§7 Fisher and cluster permutation both p ≥ 0.05) — "In a
   blinded classification with matched decoys, the published release notes flagged moved and unmoved
   files at indistinguishable rates (*k*/*n* vs *j*/*m*, exact p = *p*), so the notes cannot be used
   to decide whether to re-measure."
7. **NULL — almost nothing moved anywhere** (P1 BREADTH ≤ 1.5 % and CONTRAST ≤ 3 %) — "Across
   *N* (file, thread count, adjacent-release) cells spanning eleven releases, only *k* cells changed
   by more than the study's 7 % floor with disjoint session ranges — within the *e* expected by
   chance at this test's exact level — so unchanged ONNX files are, on this CPU, stable across
   releases, and the large steps this project reported earlier are confined to *the named files*."
8. **VOID** (§8.2) — "The release map is not reported: *condition* failed; both anchor series and the
   precondition log are published in its place."
9. **The verdicts do not reproduce** (P6 < 25 %) — "Of *k* cells that met the study's move rule in one
   night, only *j* met it again in an independent run of the same protocol, so the share reported here
   is an upper bound and the disjoint-range rule at five sessions is not by itself sufficient
   evidence of a release effect."

Outcomes 1–6 and 9 are not exclusive; the paper may write more than one, in the order listed.

---

## 11. Artefacts, paths, seeds, and the five permitted harness changes

### 11.1 Written by Experiment B

| artefact | path |
|---|---|
| this registration | `results/paper/b_design_registered.md` |
| Arm CONTRAST rows (+ `ae`) | `results/paper/b_drift_contrast.csv` |
| Arm BREADTH rows | `results/paper/b_drift_breadth.csv` |
| anchor rows (both anchors) | `results/paper/b_drift_anchor.csv` |
| replication-stage rows | `results/paper/b_drift_replicate.csv` |
| mechanism annex (§12), no timing | `results/paper/b_mechanism_annex.csv` |
| model + interpreter manifest, binary hashes, seeds, pre/post SHA-256 | `results/expB/manifest.json` |
| stage-0 load probe: LOADS / FAILS + construction wall clock per (file, release) | `results/expB/load_probe.json` |
| quiet-machine record, per batch | `results/expB/precondition.json` |
| sealed verdicts | `results/expB/moves_sealed.json` |
| release notes, verbatim | `results/expB/notes/<version>.md` |
| notes retrieval manifest (URL, UTC, SHA-256) | `results/expB/notes_manifest.json` |
| blind set / blind verdicts | `results/expB/notes_blind_set.json`, `results/expB/notes_blind_verdicts.json` |
| run log | `results/expB/b_run.log` |

**Existing files that are read and never written:** `results/expB/envs.json`,
`results/expA2/manifest.json`, `results/expA2/analysis_unit_note.md`,
`results/paper/a2_thirdparty_exploratory.csv`, `results/paper/ort_version_bisect.csv`,
`results/paper/ort_version_scope.csv`, `results/paper/spin_version.csv`,
`results/paper/FINDINGS.md`, `results/paper/a_design_registered.md`,
`results/paper/a2_design_registered.md`, `results/paper/e2_design_registered.md`.

### 11.2 The only harness changes permitted, named here so they cannot be invented later

The run round works on a **copy** at `scripts/paper/b_drift_run.py` plus a copy of the bisect harness;
`scripts/paper/ort_version_bisect.py` itself is not edited. Five changes, and no others:

**(a)** Two columns added to every row: `t_utc` (ISO-8601 UTC at session start) and `order_index`
(position in the shuffled list). The current CSV has neither
(`env_dir,ort,family,variant,threads,rep,spin,batch,pinned_cpus,p50_ms,iters,cpu_ms_per_inf,core_delivery`),
so machine drift cannot be regressed against time after the fact. V4, V5 and §5.2's arm-centre figure
all need it.
**(b)** `interpreters()` must drop any interpreter whose binary SHA-256 triple is already claimed by
another entry, and must record the triple per row's `env_dir` (V1). Dropping the `project` entry by
name is not sufficient: the duplicate is a property of the bytes.
**(c)** An `--anchor` option injecting the **two** §5.6 anchor cells every N sessions into their own
CSV.
**(d)** A row for **every attempted session**, with a new `status` column
(`OK` / `NONZERO_EXIT` / `TIMEOUT` / `MISSING_FILE` / `VERSION_MISMATCH`) and `p50_ms` left empty on
failure, replacing the silent `continue` at `scripts/paper/ort_version_bisect.py:201-202` and
`:206-208`. §4.1's "exactly five OK rows" rule is computed from this.
**(e)** A `--not-applicable results/expB/load_probe.json` option that removes NOT-APPLICABLE
`(file, release)` pairs from the cell list **before** the shuffle, so the grid, the seed's ordering
and the denominators all agree (V8).

**The child benchmark body is unchanged** (`scripts/paper/ort_version_bisect.py:52-89`): affinity,
`intra_op_num_threads`, sequential execution, `ORT_ENABLE_ALL`, 1.5 s warm-up, `max(2.0 s, 40
iterations)` window, `p50`, `cpu_ms_per_inf`, `core_delivery`. Changing it would break comparability
with the three pilot CSVs that §3.4 relies on.

### 11.3 Seeds

Batch seeds 72–86 (§5.2), replication 91, decoy sampling 97, cluster bootstrap 101. The cluster
sign-flip test of P4 is a complete enumeration and uses no seed. Every seed is echoed into
`results/expB/b_run.log` and `results/expB/manifest.json` at run time. A2's 63/64/65 are not reused.

---

## 12. Barred in Experiment B

* Re-quantizing, re-exporting or rebuilding **any** model. The premise is an unchanged file.
* Changing the **7 %** floor, in either direction, for any cell or any statistic.
* Any outlier rule, trimmed mean, or dropping a session for being extreme (§5.4).
* Fewer than five sessions per cell, anywhere, for any reason (§5.1, §8.3); analysing a cell that has
  other than exactly five `OK` sessions (§4.1).
* Promoting rows from `a2_thirdparty_exploratory.csv`, `ort_version_bisect.csv` or
  `ort_version_scope.csv` into the map, or comparing a map cell against one of them as though the two
  were the same run (§3.4, V7); pooling the map run's sessions with the replication run's (§5.5).
* Editing any existing registration or CSV. B writes only the paths in §11.1.
* Reading a release note against a specific result before `moves_sealed.json` exists (§7).
* Quoting `screen_ms` as a latency, a baseline or a speed-up
  (`results/expA2/manifest.json`, `screen_ms_is_not_a_result`; `results/paper/a_design_registered.md:235`).
  The stage-0 construction wall clock is under the same bar: it sizes the budget and is never a result.
* Reporting a ratio without the absolute milliseconds beside it (§4.1).
* Claiming a move is "documented" on the strength of a generic release-note line (§7 step 4).
* Reporting any share in §6 without the chance expectation of §5.1 and the `n_clusters` printed next
  to it.
* Counting 18 files as 18 independent observations of a release effect (§3.1), or 72 trios as 72
  independent pairs (§6 P4).

### 12.1 The one thing that is *not* barred: a bounded mechanism annex

Naming a **cause** for any move stays barred: B measures *that* files moved and whether the notes
predicted it, and why a particular kernel changed may not be asserted from it. But the project already
owns a no-timing instrument that answers a narrower, factual question, and refusing to run it would
leave the write-up a bare benchmark table (pre-mortem R12). `scripts/paper/optgraph_version_diff.py`
dumps ORT's **optimized** graph under each runtime and diffs it — the right object, since this
project's own rule is that the file's graph is not the executed graph. Its decision rule is already
registered and has already fired once: `results/paper/FINDINGS.md:1972-1975` — "**1.24 rule fires:
'identical optimized graph; the change is in kernels or threading'.** The static file's executed graph
is the same node for node at 1.23.2 and 1.24.4, yet it runs 31 % slower at 4 threads."

Registered now, so the selection cannot be made after seeing what would be convenient:

* **After** `moves_sealed.json` and **after** the blinded notes verdicts are sealed (§7), take the
  **12** REPLICATED MOVED pair-cells with the largest `|Δ|`, at most **2 per model or architecture
  family**, ties broken by `(arm, model, variant, threads, transition index)` ascending.
* For each, dump the optimized graph under both releases of the pair and compare the ordered node
  list (domain, op type, in/out counts, attributes), as Experiment 23a did.
* Each pair gets exactly one of two labels — **GRAPH-IDENTICAL** or **GRAPH-DIFFERS** — with the op
  deltas listed. Nothing else may be written: GRAPH-IDENTICAL licenses only the sentence "the
  executed graph did not change, so the difference is in the kernels or the threading";
  GRAPH-DIFFERS licenses only "the executed graph changed by *these* nodes".
* No timing. ~24 session constructions, ≈ 5 minutes, inside stage 3.

---

## 13. Answers to the pre-mortem

Thirteen items were raised. **Nine are fixed in the protocol, three are rejected on stored evidence
with the check that would catch them anyway kept in, and one is fixed in part with the residual risk
named.** Each entry says what changed and, where something cannot be fixed, which outcome it leaves
inconclusive.

### R1 — `*_dyn_conv` does not run below 1.24.4, and the grid counted it anyway — **FIXED**

Confirmed, and confirmed harder than stated: enumerating `ort_version_scope.csv` this session shows
462 rows against a 480-row grid, and the 18 missing rows are exactly
`mnv2/r50/y256 × *_dyn_conv × {1,4} × 1.23.2 × 3 reps`, with every other cell at exactly 3 reps.
§3.0 now declares the NOT-APPLICABLE set; §3.2 carries the corrected denominators (**264**
single-minor pair-cells, **72** trios, not 288 and 96); §5.1's chance budget is computed on them;
§9 stage 0 adds the 418-construction load probe (V8) that settles the four untested older releases
before any timing session; harness change (e) removes those cells before the shuffle so no time is
spent on them and the seed's ordering is consistent. The budget *drops* by 0.71 h as a result.

### R2 — cells that quietly finish with fewer than five sessions — **FIXED**

Confirmed at `scripts/paper/ort_version_bisect.py:201-202` (missing file: silent `continue`) and
`:206-208` (failure: printed to stdout, `continue`, no row). Two changes: §4.1 makes "exactly five
`status = OK` rows" a precondition of analysis and VOIDs anything else — never analysed at reduced
`n` — and §11.2 change (d) writes a row for every attempted session with a `status` column, so a short
cell is visible in the artefact rather than in scrollback. §8.3's timeout rule now points at the same
computable condition.

### R3 — the confirmation stage broke the rule the design is built on — **FIXED, by replacing the stage**

Confirmed: five sessions from the map invocation plus two from an invocation hours later is exactly
the cross-run comparison §3.4(a) rejects, and the bias is one-way — a between-run offset widens both
ranges without moving either median, so pooled ranges stop being disjoint and MOVED drifts to
WEAKENED. §5.5 is rewritten: the replication stage is **five fresh sessions a side in its own
shuffled invocation**, analysed as its own stratum with its own exact level, the map's verdict is
never revised, and the two runs are never pooled. The stratified form is the project's own
(`FINDINGS.md:2882-2884`). This also turns the stage into a result rather than an audit: P6 is new,
it predicts ≥ 80 % replication above 20 % and ≥ 45 % overall, and < 25 % demotes P1's share to an
upper bound (§10 outcome 9). Cost is 750 sessions against the old design's ~300, paid for by R1's
saving and capped at 75 minutes with a fixed priority order.

### R4 — 576 pair-cells are not 576 observations — **FIXED**

Both halves confirmed, one of them with a correction. (i) Series dependence: §4.5 defines SPIKE and
publishes distinct level-shift events beside the pair-cell share. (ii) File dependence: A2 already
registered the unit against itself (`results/expA2/analysis_unit_note.md:12-17`), so §3.1 makes the
**architecture family (n = 15)** BREADTH's primary denominator with files underneath, and P1's
interval becomes a cluster bootstrap over families with `n_clusters` printed. P4's primary becomes an
**exact cluster sign-flip test** over the 12 `(model, threads)` clusters, complete enumeration of
2^12, with a conservative 6-model version that must agree in sign, and a cluster-level exact sign test
on magnitudes; McNemar, Wilcoxon and Fisher survive as labelled independence-assuming secondaries.
**Correction to the pre-mortem:** no architecture family holds nine files — the largest holds two
(efficientnet a04/c01, mobilenet_v2 a02/c02, resnet a03/c05). Nine is the file count of the
`c3_224x224` *input-shape directory*, which is a batch, not a family.
**What this leaves inconclusive:** with 6 models and 15 families, the cluster tests cannot resolve
small effects. §6 P4 registers that in advance — the cluster test is *predicted* not to reach
p < 0.05, and the headline is the direction plus an interval, not a significance claim.

### R5 — the independent variable was the only thing not hashed — **FIXED**

Confirmed, and verified independently this session: `D:\sec_envs\ort1300` and the project `.venv` are
byte-identical in all three shipped binaries (`pyd` `06b712319c15`, `onnxruntime.dll` `44741724166a`,
`providers_shared` `fabb99add45a`), and `.envs\ort1232` / `.envs\ort1244` are byte-identical to their
`D:\sec_envs` twins. §2.1 now carries the full SHA-256 table for all eleven releases — all eleven
triples distinct — V1 resolves interpreters by hash as well as by version string and voids a pair of
releases that share a triple, and harness change (b) drops duplicates by bytes rather than by the
name `project`. A by-product worth recording: 1.16.3 ships no separate `onnxruntime.dll`.

### R6 — arm order was time order — **FIXED**

Confirmed for the old order: CONTRAST, then `ae`, then BREADTH put the two arms' time-weighted
centres ~186 minutes apart in a ~316-minute night, so P1's central comparison was confounded with
machine state. §5.2 interleaves the fifteen batches, heaviest first within each arm, alternating
arms; on the §9 estimates the centres are **156.7 min (CONTRAST) and 159.3 min (BREADTH)**, 2.6
minutes apart, and the realised figure is recomputed from `t_utc` and published. The interleave broke
the old mid-night cut rule, which needed BREADTH to be entirely in the future, so §8.3 moves the cuts
to a **pre-flight** decision computed from stage 0 and replaces the mid-night checkpoint with a
batch-boundary hard stop — which is better anyway, since it removes the 03:00 judgement call the
brief forbids.

### R7 — the anchor could not see the drift it certifies — **FIXED**

Confirmed as a design gap: a 0.44 ms probe measures per-call overhead, and V4 was using it to certify
magnitude comparisons on cells up to 1 434 ms. §5.6 adds **A-heavy** (`r50 p00_fp32`, 4 threads,
24.5 ms — a 2.0 s window of full four-core work) beside **A-small** (`mnv2 p00_int8`, 4 threads,
0.4396 ms on 1.30.0). Both windows are floor-limited at 2.0 s, so both cost 4.02 s and the pair costs
~13.5 min for the night. V4 fires if **either** moves ≥ 7 %, and the two are reported against each
other, since A-heavy moving alone is a frequency/thermal signature.
*(Small correction: 0.433 ms is the anchor cell's 1.23.2 median; its 1.30.0 median — the release the
anchor actually runs on — is 0.4396 ms.)*

### R8 — V5 deleted true uniform release effects — **FIXED**

Accepted, and the critic's reasoning is the design's own: inside a batch the shuffle decorrelates
release from position in time (`ort_version_bisect.py:190-193`), so a window of bad machine state
raises every release's variance in that batch and **cannot** produce a shift aligned to one
transition. The old V5 would therefore have deleted the largest genuine release effects and made
P3's "dispersion, not mean" headline partly definitional. V5 is now a **label, not an exclusion**:
UNIFORM-SHIFT transitions stay in P1, P2 and P4 and are additionally reported with a per-batch
breakdown (a real release effect must reappear in most of fifteen independent invocations), both
anchors' behaviour in the same windows, and their replication verdicts. Only V4 — the anchors saying
the machine drifted — can move a transition out of the primary numbers.

### R9 — `ae` drives the dispersion headline — **PARTLY CORRECT, FIXED BY SPLITTING**

Checked rather than assumed. `ae` is indeed the pilot minimum (−51.45 %), and the project already
rules its ratios "indicative only" (`FINDINGS.md:2577`). But removing it leaves the pooled pilot range
at **74.5 points** (min −28.19 % `y256` static 4T, max +46.35 % `mv256` static 4T), so the ≥ 40-point
prediction does not depend on it. §6 P3 now reports **P3-all** and **P3-core** (sub-millisecond
pair-cells excluded, defined as `max(m_a, m_b) < 1.0 ms`), and **the numeric prediction is registered
on P3-core**, the conservative half. Eight CONTRAST/appendix cells per transition are sub-millisecond;
no third-party file is (the smallest is 1.454 ms), so BREADTH is unaffected.

### R10 — contrib ops and new opsets on old releases — **FIXED (by the same probe as R1)**

Confirmed and quantified this session by reading the files: **`b05` is IR version 10 / opset 20** —
the only file above IR 9 — and five files carry `com.microsoft` operators in the graph (`c04`, `b15`,
`b16`, `a06`, `b18`, listed in §3.0). Rather than guess which release drops which, V8's stage-0 load
probe decides it: 418 `InferenceSession` constructions, no timing, hash-stamped, and whatever fails is
NOT-APPLICABLE — out of the denominator, named in the paper. `b05` on 1.16.3 is the expected casualty,
which costs the 1.16.3 column one of 18 files, not the column.

### R11 — the spin default is unmeasured — **REJECTED ON STORED EVIDENCE, DIAGNOSTIC KEPT**

The confound would be large if it were real — `results/paper/spin_version.csv` shows spin-off is
1.33–1.77× slower at four threads — but the data already answer it, twice. (i) The observable is
already in every row: median `core_delivery` at `spin=1` is **1.015** (range 0.976–1.114) and at
`spin=0` is **0.488** (0.377–0.633), so the column separates the two policies cleanly; across **all
eleven releases** in `ort_version_bisect.csv` the 4-thread median `core_delivery` lies between
**0.975 and 1.022**, i.e. spinning is on by default on every release of this ladder. (ii) The project
has already tested the step under both policies: `FINDINGS.md:1998-2007`, "Rule: dW (spin off) =
+23.7 % ≥ 7 % ⇒ **not the spin policy**." The item is nonetheless kept as **V9**, a published
per-release 4-thread `core_delivery` median with a 0.80 trigger that flags a release
SPIN-DEFAULT-CHANGED and reports its transitions separately. Cost: zero — the column already exists.

### R12 — "just a benchmark table" — **ACCEPTED IN PART; a bounded annex is added**

Naming a cause remains barred. But the project owns `optgraph_version_diff.py`, a no-timing
instrument whose decision rule is already registered and has already fired
(`FINDINGS.md:1972-1975`), and it answers a factual question — did the executed graph change —
rather than a causal one. §12.1 adds it under a selection rule fixed in advance (the 12 largest
REPLICATED moves, ≤ 2 per family, run **after** the notes verdicts are sealed so it cannot contaminate
§7), two permitted labels, two permitted sentences, ~24 session constructions and ~5 minutes.

### R13 — the wall-clock constant was fitted on small files — **PARTLY REJECTED, RESIDUAL FIXED BY MEASUREMENT**

The premise is partly wrong and the risk is real but smaller than claimed. A2's largest measured file
is `a03` at **24.56 MB**, not 11.7 MB, and the per-session residual recomputed per batch this session
(§9 table) does **not** grow with file size across 0.11–24.56 MB: 0.40 s at 0.11 MB and 0.44 MB,
0.51 s at 24.56 MB, 0.64 s on a 23.71 MB batch whose latency is the largest in the pool. So the
constant is a constant over the measured range. `r50 p00_fp32` at **94.03 MB** is 3.8× beyond it, so
the fix is to measure rather than extrapolate: stage 0's load probe records the construction wall
clock per `(file, release)`, and §8.3 recomputes the night's estimate from it and applies the cuts
**before batch 1**. If `r50`'s construction turns out to cost, say, 3 s instead of 0.5 s, that is
`3 models-worth of variants × 2 threads × 11 releases × 5 × 2.5 s ≈ 11 min` on the whole night — and
it is spent knowingly, before the night starts, instead of discovered at 04:00.

### The pre-mortem's own arithmetic, corrected where it matters

| claim | status |
|---|---|
| "24 of 96 P4 trios, 24 of 288 CONTRAST cells" lost to R1 | correct; the corrected denominators are 72 and 264 |
| 462 rows vs a 480-row grid, the 18 missing being the `dyn_conv`/1.23.2 cells | correct, enumerated this session |
| `D:\sec_envs\ort1300` byte-identical to the project `.venv` | correct, and true of all three binaries |
| "whichever architecture family happens to hold nine files" | wrong: the largest family holds two files; nine is a shape directory |
| "94 MB vs 11.7 MB" | A2's largest file is 24.56 MB; the residual does not scale with size over that range |
| "0.433 ms probe" | 0.433 ms is the anchor's 1.23.2 median; on the release it runs on it is 0.4396 ms |
| `FINDINGS.md:2572` for the ConvInteger quote | the sentence begins at `:2571` |
| "the average is not the interesting quantity — the dispersion is" | correct, and unchanged: 97.8 points of range at a −0.34 % median, 74.5 without the sub-millisecond model |

---

*Registered 2026-09-23; revised 2026-09-23 in answer to the pre-mortem, before any Experiment B
measurement. Grid, rules, predictions, refutation patterns, validity conditions, abort criteria and
outcome sentences are fixed as of this file. Any later change is a new registration with its own
date, not an edit to this one.*
