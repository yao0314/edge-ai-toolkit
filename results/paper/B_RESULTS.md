# Experiment B — release drift of unchanged ONNX files

**Measured 2026-09-23 17:37 to 23:20 Taipei, 5.71 h, fifteen batches, none failed.**
Protocol: `results/paper/b_design_registered.md` (sha256 `5bb5d73d14f5`), registered before the first
timing row. Harness: `scripts/paper/ort_version_bisect.py` (sha256 `170a66325b07`), unedited during
the run. Runner: `scripts/paper/b_run.py`. Analysis: `scripts/paper/b_analyse.py`.

The question: **when the runtime is upgraded and the model file is not touched at all, how often does
the file change speed?**

---

## 1. What was measured

| | |
|---|---|
| files | 38 (18 third-party static INT8, our six models in their variants, one appendix model) |
| releases delivering data | **10** — 1.20.1, 1.21.1, 1.22.1, 1.23.2, 1.24.4, 1.25.1, 1.26.0, 1.27.0, 1.28.0, 1.30.0 |
| thread counts | 1 and 4, pinned to four P-cores |
| sessions per cell | 5 fresh processes |
| cells with exactly five OK sessions | **736** (3,680 sessions) |
| primary pair-cells (eight single-minor transitions) | **584** |

The registered denominators are met exactly: **584** at the file level and **536** on the primary
units (264 CONTRAST + 240 BREADTH families + 32 appendix), as written in §3.3 before measurement.

**ONNX Runtime 1.16.3 produced no data at all** (76 cells). Its compiled extension is built against
numpy 1.x and every session that reaches an inference fails under the numpy 2.4.6 that the rebuilt
environments carry (the six `*_dyn_conv` cells and `b05`'s two cells fail earlier, when the session
is created). The
stage-0 probe did not catch it because the probe constructs a session and never runs an inference.
This costs only the `1.16.3 → 1.20.1` transition, which §2.3 had already classified as GAPPED and
excluded from the primary denominator, so **the primary analysis is untouched**. **[corrected]** A
further 24 cells are NOT-APPLICABLE (`*_dyn_conv` has no kernel below 1.24.4; 30 in all, 6 of them
among the 76 at 1.16.3), and the 2 unclassified cells are `b05`'s two 1.16.3 cells: 100 unusable
cells in all (836 − 736). *Replaces:* "A further 30 cells are NOT-APPLICABLE … and 2 are
unclassified", which counted 8 cells twice.

## 2. The primary result

A pair-cell is one file at one thread count across one release step. It is **MOVED** when the median
moves by at least 7 % *and* the two sides' five-session ranges do not overlap; **UNRESOLVED** when it
clears the floor but the ranges overlap; **STILL** otherwise. Every share is reported as a bracket,
lower bound `MOVED/N` and upper bound `(MOVED+UNRESOLVED)/N`.

| unit | MOVED | UNRESOLVED | STILL | N | share |
|---|---|---|---|---|---|
| **single-minor pair-cells (primary)** | **59** | 32 | 493 | **584** | **10.1 % – 15.6 %** |
| BREADTH, by architecture family | 25 | 10 | 205 | 240 | 10.4 % – 14.6 % |
| CONTRAST | 29 | 21 | 214 | 264 | 11.0 % – 18.9 % |
| appendix `ae` | 5 | 1 | 26 | 32 | 15.6 % – 18.8 % |
| gapped transitions (reported apart) | 19 | 4 | 53 | 76 | 25.0 % – 30.3 % |
| against the 1.23.2 baseline (cumulative) | 147 | 36 | 447 | 630 | 23.3 % – 29.0 % |
| single-minor, FP32 files | 11 | 11 | 90 | 112 | 9.8 % – 19.6 % |
| single-minor, static INT8 files | 37 | 16 | 347 | 400 | 9.3 % – 13.3 % |
| single-minor, dynamic INT8 files | 11 | 5 | 56 | 72 | 15.3 % – 22.2 % |

**[corrected]** **In about one single-minor pair-cell in eight, an unchanged file moves past the
floor — FP32 files as often as INT8 ones.** Across the whole ladder, against a fixed 1.23.2 baseline,
it is about one in four. *Replaces:* "about one in eight single-release upgrades" (the unit is the
pair-cell, file × thread count × step) without the split by variant, added 2026-09-24.

## 3. The paper's two cases, reproduced in fresh environments

Both were measured again from scratch: rebuilt interpreters on another disk, randomised order, five
fresh processes per cell, on a different night.

### 3.1 MobileViT-XXS, static INT8, 1.23.2 → 1.24.4

| variant of the same weights | 4 threads | 1 thread |
|---|---|---|
| **64 px static INT8** | **+40.3 %** (1.648 → 2.311 ms) **MOVED** | +3.3 % STILL |
| **256 px static INT8** | **+43.9 %** (9.627 → 13.850 ms) **MOVED** | — |
| 64 px FP32 | −1.1 % STILL | +4.5 % STILL |
| 64 px dynamic-MatMul INT8 | −10.2 % (faster) | +3.3 % STILL |

The paper reported +31 % at one size. Two sizes now carry it, at +40.3 % and +43.9 %, with the
control structure intact: the same weights exported to FP32 do not move, the dynamically quantized
sibling does not move, and one thread does not move. The regression belongs to the static INT8 path
at four threads and nowhere else.

**It never comes back.** The 64 px series after the step: 2.311, 2.321, 2.333, 2.394, 2.348, 2.267 ms
through 1.30.0 — every later transition inside ±3.5 %, and **+37.6 % against 1.23.2 at the newest
release measured**.

### 3.2 YOLOv8n, static INT8, 1.23.2 → 1.24.4

| | before | after | change |
|---|---|---|---|
| 4 threads | 6.537 ms | 4.896 ms | **−25.1 %** |
| 1 thread | 10.362 ms | 8.225 ms | **−20.6 %** |

The paper reported −28 %. It moves at **both** thread counts, which separates it from the MobileViT
case: a change in the kernels themselves, not in how work is handed to the thread pool.

Independent consistency check: the same file on 1.24.4 reads 4.896 ms here and 4.914 ms in the older
`results/detection/sweep.csv` — two unrelated runs agreeing to 0.4 %.

**This costs nothing.** The file is byte-identical across the step, so size and accuracy are
unchanged; for reference the quantization step itself (FP32 → INT8, at 256 px) costs 0.0064 mAP50-95
and buys 3.59× on file size.

## 4. What the third-party files add

The regression is not a property of our models.

| file | transition | 1 thread | 4 threads | reading |
|---|---|---|---|---|
| `c3_299x299/b05` | 1.21.1 → 1.22.1 | **+96.4 %** (104.4 → 205.0 ms) | **+81.1 %** | one release, one file |
| `c3_112x112/c04` | 1.26.0 → 1.27.0 | **+80.3 %** (7.6 → 13.8 ms) | **+80.0 %** | kernel |
| `c3_224x224/a01` | 1.21.1 → 1.22.1 | −0.8 % | **+70.5 %** | thread pool |
| `c3_224x224/b16` | 1.20.1 → 1.21.1 | **+52.2 %** (20.8 → 31.7 ms) | +34.9 % | kernel |
| `c3_224x224/c01` | 1.22.1 → 1.23.2 | +0.4 % | **−43.1 %** | thread pool |
| `c224_224x3/c02` | 1.22.1 → 1.23.2 | — | **−41.3 %** | thread pool |
| `ae/p00_int8` | 1.23.2 → 1.24.4 | **−51.1 %** (0.165 → 0.081 ms) | — | |

**`b05` is one release breaking one file, in both directions.** Its 299 px trajectory at one thread:
201.4 → 104.4 (−48.2 %) → 205.0 (+96.4 %) → 104.1 (−49.2 %). 1.22.1 is an outlier for this file and
1.23.2 puts it back. Under §4.5 this is **one spike event**, not two moves; all four spikes in the map
belong to this file.

**Two signatures, separable because both thread counts were measured.** A file that moves at one and
four threads points at the kernels (`b16`, `b03`, `c04`); a file that moves only at four threads
points at the thread pool (`a01`, `c01`, `a03`, and both MobileViTs). The study's original finding is
of the second kind.

**Two releases stand out.** 1.26.0 → 1.27.0 slows many files (+80, +18, +17, +13, +13, +8, +7);
1.28.0 → 1.30.0 speeds many up (−16, −15, −15, −14, −11, −8).

## 5. Does a move persist?

Of the MOVED pair-cells with at least two later releases: **PERSISTENT 21, REVERTED 21, MIXED 16,
NOT-ASSESSABLE 20**. Roughly a third of assessable moves never come back inside the floor — including
the MobileViT regression, which is still there at 1.30.0.

## 6. How good is the measurement

**The anchors.** Two fixed files were re-measured on one release at all sixteen batch boundaries.
No point falls outside ±3σ on either. The limits are wide, and honestly so: the heavy anchor's centre
line is 25.085 ms with limits at 21.6 and 28.6 ms, a total spread of 18.4 % across the night; the
light anchor 10.9 %. The machine was not still, but it was in control.

**Where the noise lives.** Within a cell, over five fresh processes, the median spread is **3.5 % at
one thread and 9.0 % at four** (166 cells each, quiet batches). Four-thread measurement is 2.6× as
noisy as one-thread on the same machine and the same files. **The study's 7 % floor is a four-thread
floor**, and that is why every "moves only at four threads" finding is hard-won.

Seven cells out of 424 have one session 2–6× slower than its four siblings; in six of them the
session's core delivery collapses with it, so the threads were waiting rather than computing. The
median resists them and the range records them, which is what the no-outlier-rule is for: those cells
land in UNRESOLVED and cost resolution, not correctness.

## 6b. Independent replication (§5.5 and P6)

**Run 2026-09-24, 10:22 to 12:58, 2115 rows.** Every adjacent MOVED and UNRESOLVED pair-cell of the
map — 114 of them — re-measured in an independent run: five fresh sessions a side, one shuffled
invocation per model, seed 91, its own stratum, **never pooled with the map, and the map's verdicts
never revised**. One deviation, recorded before launch in `results/expB/replication_note.md`: the time
cap was raised from 75 to 240 minutes; the priority order is unchanged, so the registered 75-minute
stage is a prefix of this run and is reported separately. Analysis: `scripts/paper/b_replicate_analyse.py`
→ `results/expB/analysis/b_replication.json`.

| | replicated | P6 prediction |
|---|---|---|
| all MOVED | **69 / 78 = 88.5 %** | ≥ 45 % |
| MOVED with \|Δ\| ≥ 20 % | **28 / 28 = 100 %** | ≥ 80 % |
| MOVED with 7 ≤ \|Δ\| < 20 % | 41 / 50 = 82.0 % | lower than the large band |
| primary denominator (single-minor MOVED) | **52 / 59 = 88.1 %** | — |
| registered 75-min prefix only | 37 / 41 = 90.2 % (21 / 21 in the large band) | — |

**All three pre-registered predictions hold, and the refutation (< 25 %) does not fire.** Every move of
20 % or more reproduced. **[corrected]** The nine that did not are all 7.1–12.8 % moves and all kept
their sign; one still cleared the floor with overlapping ranges (UNRESOLVED), the other eight came back
at 0.1–6.5 %. *Replaces:* "most of them the same sign and just under the floor the second time". Of the 36 UNRESOLVED cells, 5 resolve as MOVED and 31 as STILL.
The between-run difference on MOVED cells is 1.7 pp at the median, 5.2 pp at p90, 10.7 pp at most.

The replication also settles the two files where A2 and this map disagreed, both in A2's favour:
`a03` reads +0.0 % (A2), −8.0 % (map), **−0.1 %** (replication); `b02` reads −2.9 %, +7.8 %,
**−3.5 %**. And the 256 px one-thread cell of MobileViT reads +5.2 % in the replication against +7.1 %
in the map — consistently positive, mostly below the floor.

The ten FAILED replication rows are `y256`'s dynamic-conv variant on 1.23.2, where that kernel does not
exist; the model-level release union included 1.23.2 for other cells, and no pair-cell uses them.

## 7. What this does not claim

* ~~One run.~~ **Resolved by §6b**: all 114 moved and unresolved pair-cells were re-measured
  independently, and 88.5 % of the MOVED ones — 100 % of the large ones — reproduced.
* **Near the floor a single run is unstable.** Two third-party files measured in both the A2 scan and
  this map land on opposite sides of the 7 % floor: `a03` (+0.0 % then −8.0 %) and `b02` (−2.9 % then
  +7.8 %). Both gaps are inside the between-run spread the registration already documents (up to
  1.174). Any sentence of the form "all 18 third-party models stayed inside the floor" belongs to the
  A2 run and is not a property of the files.
* **1.16.3 is absent**, so the map is ten releases and nine adjacent transitions.
* **Two cells' failure cause is listed as OTHER** in the analysis output; the batch CSV records it:
  `b05`'s file has ONNX IR version 10, newer than 1.16.3 supports.
* **Absolute millisecond values** from batches measured while the machine was busier carry a few per
  cent of inflation; the anchors bound it at roughly 16 % on a heavy model and under 4 % on a light
  one. Ratios within a batch are unaffected, because the release order is shuffled inside the batch
  and a common-mode slowdown cancels.

## 8. Where everything is

| | |
|---|---|
| per-batch measurements | `results/expB/batches/b01…b15_*.csv` |
| anchors, sixteen boundaries | `results/expB/batches/anchor_A-{small,heavy}_b*.csv` |
| a second measurement of batch 1 (the V6 re-run; the map uses the first) | `results/expB/batches/r50_repeat/` |
| verdicts, spikes, persistence | `results/expB/analysis/b_analysis.json` |
| run log, batch plan, load probe | `results/expB/b_run.log`, `batch_plan.json`, `load_probe.json` |
| figures | `results/qc/b_drift_heatmap.png`, `b_transition_strips.png`, `b_anchor_control.png`, `b_core_delivery.png` |
