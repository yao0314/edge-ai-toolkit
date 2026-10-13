# Twelve experiments — purpose, method, rigour

For the TAAI 2026 submission. Written 2026-09-16.

Grouped by what they are *for*. Group A builds the mechanism (why INT8 sometimes loses);
group C rules out that the mechanism is a configuration artifact; D removes the scope limitation;
E defends the paper against the two attacks that can actually sink it.

Ordered within each group by dependency, not by cost.

Notation: **falsifies** = the outcome that would make us retract the claim. Every experiment has
one. If it does not, it is not an experiment.

---

## Group A — the mechanism

### 1. Q/DQ boundary density (static, no runs)

**Purpose.** Turn "INT8 is sometimes slower" into a rule a reader can apply to their own graph
without measuring anything. The hypothesised mechanism is not *how many* ops were quantized but
how many times the graph **crosses** between int8 and fp32 regions, because each crossing costs a
quantize/dequantize round-trip that does no useful arithmetic.

**Method.** Parse every exported `.onnx` in `results/*/models/`. Per variant compute:
- count of `QuantizeLinear` / `DequantizeLinear` nodes
- count of **transitions**: consecutive Q→DQ pairs with no compute node between them (a pure
  round-trip), and DQ→compute→Q sequences (a genuine int8 island)
- the set of op types left in fp32
- normalise by MACs, by node count, and by layer count — three candidate denominators

**Rigour.**
- The metric must be *transitions*, not node counts. A model whose Q/DQ sit only at the input and
  output has two nodes and near-zero boundary density; one that alternates has the same node count
  and a high density. Reporting node counts would repeat the correlation we already have (416 vs
  74) rather than improve on it.
- Pre-register the denominator before looking at the correlation, and report all three. Choosing
  the denominator after seeing which correlates is the classic way to manufacture a result.
- **The 87 variants are not independent.** Variants within a family share an architecture, so a
  correlation pooled across families can be driven entirely by family means while being absent or
  reversed within every family — Simpson's paradox. Report within-family and between-family
  separately, and state n for each.
- **Falsifies:** density fails to correlate with measured speedup *within* at least two families.
  A between-family-only correlation is a statement about architectures, not about boundaries, and
  must be reported as such.

### 2. Per-node profiling (ORT profiler)

**Purpose.** Attribution rather than correlation. Measure the fraction of wall-clock that Q/DQ
nodes actually consume, and how much falls back to fp32 kernels. This is the ground truth that
experiment 1's static density is a proxy *for*; the result of the pair is the correlation between
them.

**Method.** A dedicated run — never folded into the measurement harness. `SessionOptions
.enable_profiling = True`, same core pinning, warmup then N measured iterations, parse the emitted
JSON, aggregate `dur` by `op_type` and by node name.

**Rigour.**
- **Separate run, mandatory.** Profiling adds per-node instrumentation overhead; folding it into
  the harness would contaminate every headline latency in the paper.
- **1 thread only.** Under threading, per-node durations overlap and their sum exceeds wall-clock,
  so "Q/DQ is X% of runtime" is not a well-defined quantity at 4T. State this explicitly rather
  than reporting a percentage that cannot be reproduced.
- Report the profiler's own overhead: wall-clock for the same model with profiling on and off. The
  reader needs to know the distortion.
- Confound: ORT may fuse or place Q/DQ differently when profiling is enabled. Cross-check the node
  count in the profile against experiment 1's static count; if they differ, the profile is
  describing a different graph and must say so.
- **Falsifies:** Q/DQ time share is similar between fast and slow variants. Then boundary crossing
  is not the mechanism and we must look at kernel fallback instead.

### 3. Thread sweep

**Purpose.** The most novel finding currently rests on two points. Six of the slow cases are
*faster* at 1 thread (1.04–1.68×) and slower at 4 — quantization appears to destroy parallel
scaling, not single-thread speed. Two points cannot distinguish a gentle slope from a cliff, and
the memory-bandwidth mechanism predicts a specific shape.

**Method.** threads ∈ {1, 2, 4, 6} with pinning restricted to P-cores; {8, 12} as a **separate
condition**. Same models, `repeats` > 1 so every point carries an SD.

**Rigour.**
- This machine is heterogeneous: i7-13620H, 6 P-cores + 4 E-cores, 16 logical. `pick_cpus` takes
  P-cores first, so **1–6 threads are comparable and 8–12 are not** — at 8 the run spills onto
  E-cores or SMT siblings. Putting 8 on the same curve as 4 would confound parallel scaling with
  core heterogeneity. Report them as two conditions and name the physical cores used, from
  `cpu_topology`.
- Report **absolute** latencies alongside ratios. Both fp32 and int8 become bandwidth-bound as
  threads grow, so a converging *ratio* can occur for reasons that have nothing to do with
  quantization; the absolute curves show which.
- The crossover claim requires the two curves to be separable beyond their SDs at the crossing
  point. If they are not, report the crossover as an interval, not a number.
- **Falsifies:** the speedup-vs-threads curve is flat within SD. Then the 4T slowdowns are noise
  and this finding disappears.

### 4. RAM growth attribution

**Purpose.** Several slow INT8 variants use *more* peak RAM than fp32 (MobileViT p50: 2.07×). That
is unexplained, and a reviewer will ask. Candidates: fp32 and int8 buffers coexisting across a
boundary, larger intermediate tensors, or allocator arena behaviour.

**Method.** Compare measured `peak_rss_mb` against static quantities: total weight bytes, largest
intermediate tensor from shape inference, and the number of live boundary tensors from experiment
1. Control run with `enable_cpu_mem_arena = False`.

**Rigour.**
- RSS includes ORT's allocator arena, which deliberately over-allocates. Without the arena-off
  control, "INT8 uses more memory" may be "the arena grew differently", which is a different
  claim. The control separates them.
- Our empty-session baseline subtraction already removes the runtime's own footprint — say so, and
  report `peak_rss_total_mb` as well so the reader can see both.
- Peak RSS is a lifetime high-water mark, so a transient during session initialisation counts
  toward it. Report `rss_steady_mb` alongside; a gap between them localises the growth to setup.
- **Falsifies:** with the arena disabled the difference vanishes. Then it is an allocator artifact
  and should be reported as one.

---

## Group C — is the mechanism a configuration artifact?

These three exist because ONNX Runtime's own documentation names configuration traps that produce
exactly the symptom we report. If we cannot rule them out, the finding is unciteable.

### 5. S8S8 vs U8S8

**Purpose.** `core/quantize.py:62` carries this comment:

> `"activation_type": "uint8",  # U8S8: fast path on x86 (S8S8 QDQ was slower than FP32 there); fine on ARM`

Both clauses are unsupported. The parenthetical asserts a measurement we have no data for, and
"fine on ARM" describes hardware we have never run on. Either substantiate or retract.

**Method.** Re-quantize the same fp32 ONNX files with both activation/weight type pairs and measure
identically.

**Rigour.**
- Same calibration set, same seed, same thread pinning, same repeats; the type pair is the only
  difference.
- Report **accuracy as well as latency**. The type pair affects numerical range, so a speed
  comparison alone is incomplete.
- This is a defensive experiment: either outcome is publishable as a configuration-sensitivity
  note, and the honest version is that we shipped a default we had not measured.

### 6. QDQ vs QOperator

**Purpose.** ORT's documentation states *"S8S8 with QOperator will be slow on x86-64 CPUs and
should be avoided in general"*. We always use QDQ, so we currently cannot claim our slowdowns are
not a format artifact — we have simply never been in the trap.

**Method.** A 2×2 with experiment 5: {QDQ, QOperator} × {U8S8, S8S8}.

**Rigour.** The 2×2 is the entire point. Varying format alone leaves the type pair as a confound
and vice versa; ORT's documented trap is a specific *combination*, and only the full factorial
separates them.

### 7. reduce_range

**Purpose.** ORT recommends `reduce_range=True` on hardware without VNNI to avoid accumulator
saturation. We set `False` and never tested it. Verified on this machine: AVX2 yes, **AVX512F no**,
so no AVX512-VNNI; AVX-VNNI presence could not be confirmed with the available tools.

**Method.** Toggle `reduce_range`, measure accuracy and latency across families.

**Rigour.**
- This is primarily an **accuracy** control, not a speed one. If `reduce_range` materially changes
  accuracy, then every accuracy number in the paper is configuration-dependent in a way we have
  not disclosed — which would matter more than the latency question.
- Report the CPU's confirmed feature set and state plainly that AVX-VNNI could not be confirmed.
  Do not claim "no VNNI"; claim "no AVX512-VNNI, AVX-VNNI unconfirmed".

---

## Group D — scope

### 8. ARM validation

**Purpose.** x86-only is the paper's largest framing limitation and the only one a single
experiment can remove. The question is binary and interesting either way: does the architecture
dichotomy (some families lose under INT8) hold on ARM?

**Method.** Oracle Cloud Ampere A1, always-free tier. Offline install from the existing vendored
aarch64 bundle. Minimum viable scope: MobileViT and ResNet50 at p00, 1T and 4T.

**Rigour.**
- **The platform choice is itself a rigour decision.** AWS `t4g` free tier and GitHub Actions ARM
  runners are burstable or shared; latency published from a throttled instance would undermine the
  measurement-rigour argument the whole paper rests on. Ampere A1 OCPUs are dedicated cores.
- Run the competing-load detector and report `other_load_pct` per measurement. If it exceeds the
  threshold, discard and re-run — on a cloud VM this is not optional.
- Report the aarch64 feature set (`dotprod`, `i8mm` in `/proc/cpuinfo`). That is the ARM analogue
  of the VNNI question and it determines whether the result generalises to other ARM parts or is
  specific to Neoverse N1.
- Either outcome is a result: holds → the scope widens and "edge" is defensible in the title;
  does not hold → the finding is x86-specific, which is a narrower but more interesting claim about
  kernel coverage.

---

## Group E — defending the paper

These two do not produce findings. They decide whether the findings we already have survive review.
They are listed last and should be run first.

### 9. Measurement variance

**Purpose.** The attack that can end the paper: *"6 of 8 settings flip"* is worth nothing if the
flips sit inside measurement noise. We have latency SDs from `repeats`. **We have no accuracy
variance at all.**

**Method.** Re-evaluate accuracy under resampling: different evaluation-subset draws and seeds, to
obtain an SD per variant. Then test whether each argmax flip exceeds the combined accuracy and
compression uncertainty.

**Rigour.**
- **Run this before writing the claim, not after.** If the flips are inside the noise, the thesis
  changes and everything downstream changes with it.
- The detection runs used `n_val = 64`. mAP on 64 images carries large variance, and two of the
  eight settings are detection. This must be quantified, and if it is large, those settings either
  get a larger evaluation set or are reported with the uncertainty shown.
- Pre-register the rule: a flip is reported as a flip only if the compression difference exceeds
  the combined SD. Flips inside the interval are reported as ties.
- **Falsifies:** the flips are not separable from noise. Then the honest paper is the fallback
  version — a measurement contribution with no argmax claim.

### 10. Finetuning-schedule control

**Purpose.** The second attack. Wang et al. (arXiv:2301.05219) revived 2017 L1-norm pruning to
match modern methods purely by improving the fine-tuning LR schedule. Our five strategies all ran
20 epochs with warmup 0.05 and `recovery_lr_mult` left at default — so "pruning earns nothing at a
2.0-point budget" is currently attributable to our recovery protocol rather than to pruning.

**Method.** Take the **best** strategy (global pruning, −2.34 pt) at ratio 0.25 and sweep
`recovery_lr_mult` ∈ {1, 2, 5, 10, 20} crossed with a longer epoch budget. Report the full curve.

**Rigour.**
- Sweep the **best** strategy, not the worst. Showing that the strongest candidate cannot be
  rescued is the strong form of the claim; showing it for the weakest proves nothing.
- Report the curve rather than the endpoint, so saturation is visible and the reader can see we did
  not simply stop early. Existing circumstantial support: ResNet50 12→30 epochs bought 0.35 points,
  and on the detector `recovery_lr_mult` ×20 with warmup moved the metric 0.1755 → 0.1728. Neither
  is a control on these five strategies.
- **Falsifies, and this is the good outcome:** some schedule brings global pruning under 2.0
  points. Then the headline claim becomes "pruning pays only under a tuned recovery schedule that
  the literature does not report", which is a better paper than the one we are currently writing.

---

## Dependencies

```
9  ──────────────────────────────► must finish before the argmax claim is written
10 ──────────────────────────────► must finish before the pruning claim is written

1 ──► 2        density is the predictor; profiling is the target; the pair is the result
5 ──► 6        the 2x2 factorial only works run together
3              independent, but only interpretable with 1 and 2 in hand
4              independent
7              independent
8              last: it is the widest claim and should rest on a settled mechanism
```

Experiments 9 and 10 are gates, not contributions. If 9 fails the paper becomes the fallback
version; if 10 fails the paper becomes a different and better one. Both outcomes are fine. Writing
the paper before running them is not.

---

## Group F — added 2026-09-16

### 11. Does a label-free output-fidelity proxy predict the task metric? (cross-task)

**Purpose.** A single-model observation (SQNR vs mAP, r = +0.395, n = 6) is not worth reporting on
its own — at that n the confidence interval spans from negative to strongly positive and excludes
no hypothesis. It becomes worth reporting if it is the middle of a **gradient**, because then the
failure has a structure a practitioner can anticipate rather than being one unlucky result.

The mechanism: SQNR measures signal-to-noise on the output TENSOR. Task metrics sit at varying
distances from that tensor.

| metric | how it is derived from the output tensor | distance |
|---|---|---|
| top-1 | argmax | near — only the index of the maximum matters |
| AUC | rank all samples by score | middle — absolute values are irrelevant, order is not |
| mAP50-95 | box coords through an IoU threshold, confidence ranking, averaged over 10 thresholds | far — a step function sits in the definition |

**Hypothesis.** Output-fidelity proxies cannot predict metrics whose definition contains a
**threshold** or a **ranking**. A small uniform shift barely moves SQNR but can push a box across
an IoU threshold; a small reordering of confidences barely moves SQNR but reorders the whole
precision-recall curve.

**Prediction, therefore falsifiable:** the proxy's rank-correlation with the measured metric should
decrease monotonically across top-1 → AUC → mAP.

**Method.** `scripts/paper/sqnr_vs_metric.py` on three configs. Per layer: prune that layer alone
at 0.5, re-estimate BN (label-free, so the stale-statistics effect of experiment/finding
`bn_recalibration` is not a confound), evaluate the real metric. 16 layers per task.

**Rigour.**
- **Layers must be sampled evenly across the probe's own score range**, never the best or worst 16.
  Sampling the extremes inflates any correlation. `spread()` in the script does this.
- **Report Spearman as the primary statistic.** The hypothesis is about *ranking* ability — can the
  proxy order the layers correctly — so rank correlation is the right test. Pearson is secondary.
- **n = 16 per task does not support a significance claim.** What the design supports is a
  comparison of three coefficients' direction and magnitude. The paper must say "gradient", not
  "significant correlation".
- Three tasks is three, not thirty. Even if the gradient appears, the claim is bounded to these
  three metric shapes on these three models.
- **Falsifies, and this is a live risk:** the proxy predicts equally poorly on classification. Then
  there is no gradient, the mechanism hypothesis is dead, and the whole thing goes to the appendix
  as a single negative observation — or is cut.

**Consequence if it holds.** Most allocation methods rank layers by a label-free output-fidelity
proxy, because labels are expensive. If such proxies fail structurally on thresholded/ranked
metrics, those methods are unsound for detection, tracking and retrieval — which are exactly the
edge workloads. That connects this experiment to the paper's spine: what gets measured is not what
matters.

### 12. Is pruning damage additive across layers?

**Purpose.** Replaces experiment 11, which died productively. Single-layer sensitivity predicts
single-layer damage well (Spearman +0.688 classification, +0.703 detection, both n = 16). Yet DP
knapsack allocation built on exactly that signal loses to uniform pruning (−2.72 vs −2.40 pt at
ratio 0.25). If the signal is good and the optimiser is exact, the failure must be in the
**objective**: the assumption that total damage is the sum of per-layer damages.

**Method.** Take the DP allocator's chosen plan for ratio 0.25 (recoverable from the run's
`prune_report`, or re-derive it with the same config and seed). Then compare:
- **predicted** damage: the sum of the measured single-layer damages for each layer at its allotted
  ratio, from the `sqnr_*.csv` and additional single-layer measurements where the allotted ratio
  differs from 0.5
- **measured** damage: the accuracy of the model with the whole plan applied, which we already have

**Rigour.**
- Single-layer damages must be measured at the ratio the plan actually assigns, not at a fixed 0.5.
  Comparing a plan of mixed ratios against damages measured at one ratio would test nothing.
- BN re-calibration must be applied identically in both, or the stale-statistics effect (measured
  separately: 0.1000 → 0.6522 at ratio 0.25) will dominate and mask the additivity question.
- Report the **sign** of the gap, not only its size. Sub-additive (the whole is less damaging than
  the sum) and super-additive point at different mechanisms and have opposite practical advice.
- A control that costs nothing: uniform pruning is also a plan, so the same prediction-vs-measurement
  comparison applies to it. If additivity fails equally for both, non-additivity cannot explain why
  the allocator specifically loses.
- **Falsifies:** predicted and measured damage agree within measurement error for both plans. Then
  additivity holds, and the allocator's loss must come from somewhere else — the `round_to`
  quantisation of channel counts, or the 5-level discretisation of the allocation grid.
