# Findings log

Outcome of each experiment in `EXPERIMENT_PLAN.md`, recorded as it lands — including the ones that
failed. A negative result that is not written down gets re-derived, and worse, a hypothesis that
was falsified but not recorded tends to creep back into the writing.

Every entry states the pre-registered falsification condition and whether it fired.

---

## Experiment 1 — Q/DQ boundary density · 2026-09-16 · **HYPOTHESIS FALSIFIED**

**Script.** `scripts/paper/qdq_boundary_density.py` → `results/paper/qdq_density.csv`
174 ONNX files parsed, 95 INT8, 81 with both Q/DQ nodes and measured latency.

**Pre-registered falsification:** *"density fails to correlate with measured speedup within at
least two families."* **It fired.**

| metric | pooled (n=81) | within MobileViT (n=9) | within detection (n=6) |
|---|---|---|---|
| `boundary_per_gmac` | rho = −0.951, p < 0.0001 | **+0.354** (sign reversed) | −0.598, p = 0.21 |
| `productive_dq` | +0.181, p = 0.11 | −0.455, p = 0.22 | n/a (constant) |
| `rt_per_compute` | −0.128, p = 0.26 | +0.297, p = 0.44 | n/a (constant) |

**Why the pooled correlation is spurious.** `boundary_per_gmac` divides by MACs, and MACs span
589,824 (DeiT) to ~4 GMac (detection) across families. The metric is therefore dominated by
1/MACs, so rho = −0.951 restates "larger models benefit more from quantization" — known, trivial,
and nothing to do with boundary crossings. This is a textbook spurious-denominator / Simpson's
paradox result, and only the pre-registered within-family control caught it.

**A second reason the continuous hypothesis cannot work.** In `classification_unstructured`,
`detection` and `detection_prune` the within-family correlation is undefined because the density is
**constant** across variants. Static INT8 produces the same graph structure regardless of prune
ratio, so density is a property of (architecture, quantization method), not of the variant. It can
only ever explain between-family variation — and that is confounded by model size.

**What survived, and it is cleaner than the original hypothesis.** The *categorical* contrast
between quantization methods, measured within a single architecture:

| family | method | n | 1T speedup | 4T speedup |
|---|---|---|---|---|
| DeiT-tiny | dynamic (0 Q/DQ nodes) | 3 | 1.79 (1.51–2.05) | **1.27 (1.14–1.41)** |
| DeiT-tiny | static QDQ | 3 | 1.38 (1.04–1.68) | **0.87 (0.78–0.97)** |
| MobileViT | dynamic (0 Q/DQ nodes) | 3 | 1.04 (1.02–1.06) | **1.08 (1.03–1.12)** |
| MobileViT | static QDQ | 9 | 0.82 (0.68–0.94) | **0.67 (0.63–0.72)** |

The 4T ranges do not overlap in either family. Static falls below 1.0 in both; dynamic stays above
1.0 in both. The structural difference is not a correlation but a fact about the graph: MobileViT
static INT8 carries 416 QuantizeLinear and 599 DequantizeLinear nodes against 90 compute ops, of
which only 213 dequantizes (36%) lead to arithmetic; the dynamic variant carries zero of either.
ResNet50 and MobileNetV2 static reach 87–92% productive dequantizes.

**Limitation that must be fixed before this can be a rule.** Only two families have both methods,
because `int8_dyn_mm` was hand-written into the two transformer configs and nowhere else. The claim
is currently "on transformers, prefer dynamic", with n = 2 architectures. Extending `dyn_mm` to the
CNN families (ResNet50, MobileNetV2, YOLOv8n) would establish whether the advantage is
architecture-specific — which is the version a reader can act on. Cost: one config line per family
plus a quantize-and-measure pass, no training.

**Writing consequence.** Do not report a density correlation. Report the categorical contrast, the
graph statistics as the mechanism, and the n = 2 limitation plainly.

---

## Experiment 11 — proxy vs task metric, cross-task · 2026-09-16 · **HYPOTHESIS FALSIFIED, AND IT REFUTED AN EARLIER RESULT**

**Script.** `scripts/paper/sqnr_vs_metric.py` → `results/paper/sqnr_{classification,anomaly,detection}.csv`

Hypothesis: a label-free output-fidelity proxy loses predictive power as the task metric moves
further from the output tensor (argmax → ranking → thresholded ranking).

| task | metric | shape | n | Spearman rho | p |
|---|---|---|---|---|---|
| classification | top-1 | argmax | 16 | +0.688 | 0.003 |
| anomaly | AUC | pure ranking | **5** | +0.500 | 0.391 |
| detection | mAP50-95 | threshold + ranking | 16 | **+0.703** | **0.002** |

**No gradient. Detection is slightly HIGHER than classification.** The proxy ranks single-layer
damage well on the metric furthest from the output tensor. The mechanism hypothesis is dead.

**And it refutes a result this project had been citing all along.** The prior claim — "output SQNR
correlates only +0.395 with mAP damage, so SQNR-driven allocation cannot work on a detector" — came
from **six hand-picked layers**. Re-measured with 16 layers drawn evenly across the probe's own
score range, the same setup gives +0.703, p = 0.002. The original number was a sampling artifact.

Consequences, all acted on:
- Experiment 11 is **removed from the paper**. Not as "evidence too thin" but as refuted.
- `core/prune.py`'s `layer_sensitivity` docstring encoded +0.395 as a lesson. Corrected in place
  with both new numbers and the reason the old one was wrong.
- The anomaly point is unusable regardless: the 1D autoencoder has only 5 prunable layers.

**The question this opens is better than the one it closed.** If the proxy ranks single-layer damage
at rho ≈ 0.7, why does allocation built on it still lose to uniform pruning?

```
uniform L2    -2.40 pt
DP knapsack   -2.72 pt      built on a proxy that predicts single-layer damage well
```

The proxy-is-bad explanation is gone. The remaining candidate is that **damage is not additive**:
per-layer sensitivity predicts per-layer damage, but the sum over a plan does not predict the damage
of executing that plan, because removing layer A changes layer B's sensitivity. That would
invalidate the assumption underneath every evaluate-each-layer-then-knapsack method, not just ours.

Recorded as experiment 12.

---

## Experiment 9 — measurement variance · 2026-09-16 · **GATE FIRED. TWO CLAIMS RETRACTED.**

**Script.** `scripts/paper/measurement_variance.py` → `results/paper/variance_classification.csv`
6 models x 8 independent 80% draws of the CIFAR-10 validation set.

```
accuracy SD   0.135 pt (p00) .. 0.291 pt (p50)    median 0.180 pt
```

SD grows as accuracy falls, so the noisiest measurements are exactly the heavily-pruned variants
the claims depend on. Two independent measurements differ with combined SD ~= sqrt(2) x 0.20 =
**0.28 pt**.

### Retraction 1 — the five-strategy ordering

| strategy | drop | separable from the next? |
|---|---|---|
| global pruning | −2.34 | vs uniform: gap **0.15 pt** — NO |
| uniform L2 | −2.49 | vs DP: gap **0.23 pt** — NO |
| DP knapsack | −2.72 | vs BN-scale: gap 0.54 — marginal |
| BN-scale | −3.26 | vs BN-scale+global: gap 1.45 — yes |
| BN-scale + global | −4.71 | |

The top three are **not separable**. So "DP knapsack loses to uniform" and "the most complex method
loses to the simplest" — both asserted repeatedly in conversation before this was measured — are
**not supported**. What survives: BN-scale importance is measurably worse than L2, and combining it
with global pruning is worse still. Everything else is a tie.

This also removes experiment 12's motivation: it was framed as explaining why DP loses. It does not
lose. The additivity question stands on its own merits but is no longer a diagnosis.

### Retraction 2 — the argmax flip, and with it the paper's main thesis

The separability check needs at least two eligible variants to compare. Counting them:

```
22 of 48 (setting x budget) cells have <= 1 eligible variant
```

Where there is no competition, "the winner changed" only means a different single variant became
eligible. Restricted to cells with >= 2 eligible variants:

| setting | budgets with competition | distinct winners | what actually changes |
|---|---|---|---|
| classification | 1 | 1 | nothing |
| classification_deit | 6 | 3 | prune ratio, same method (`*_int8_dyn_mm`) |
| classification_mobilevit | 6 | 3 | **method** (qat / search_pct / qat) |
| anomaly | 6 | 3 | prune ratio, same method (`pXX_int8`) |
| detection | 5 | 2 | input resolution, same method |
| detection_640 | 1 | 1 | nothing |
| stress_imagenette | 0 | — | never had competition |
| stress_resnet50 | 1 | 1 | nothing |

So **4 of 8, not 6 of 8**, and of those four, three flip only the RATIO or RESOLUTION inside a
single method. That is a Pareto front read along the other axis, which is precisely the duality
attack the literature review predicted:

> "On a monotone accuracy-vs-compression Pareto curve, sweeping the ratio and sweeping the budget
> are the same curve read along different axes."

**Genuine cross-method flips: n = 1 setting (MobileViT).** That does not support a thesis.

### Consequence

The main claim ("we sweep the budget and show the best-compression variant changes") is dead. The
fallback identified during the literature review is what remains, and it is unaffected because it
reports measurements rather than a methodological claim:

> First accuracy-budget-indexed, wall-clock-measured compression benchmark across 7 architecture
> families and 87 variants on ONNX Runtime CPU, including non-classification workloads; confirms
> Kuzmin et al. (2023) on CPU-deployed small models; quantifies a failure mode absent from
> FLOPs-based benchmarks (26% of INT8 variants slower than fp32, concentrated in transformer and
> 1D architectures).

The two findings that survived today support exactly that version:
- static-QDQ vs dynamic quantization, within-architecture, non-overlapping ranges, with a
  mechanism grounded in ONNX Runtime's own op registries (26 op types static vs 7 dynamic)
- quantization destroying parallel scaling: 1T speedup 1.04-1.68x turning into 4T slowdown, with
  no counter-example in the other direction

### Note on method

All three of today's falsifications came from the same place: **a sampling or eligibility control
that was written down before the result was looked at.** Pooled n=81 became a spurious denominator;
n=6 hand-picked became n=16 evenly sampled; 6-of-8 became 4-of-8 once "eligible variants >= 2" was
enforced. None of these would have been caught by re-reading the analysis.

---

## Experiment 10 — finetuning schedule control · 2026-09-16 · **GATE FIRED. THE PRUNING CLAIM IS AN ARTIFACT OF OUR OWN SCHEDULE.**

**Script.** `scripts/paper/finetune_schedule_control.py` → `results/paper/finetune_schedule.csv`
Strongest strategy only (global pruning, ratio 0.25, 20 epochs, warmup 0.05), sweeping
`recovery_lr_mult`. Same seed per cell, so initialisation and data order are held fixed.

```
lr_mult   1.0   top1 0.9027   -2.08 pt   over the 2.0-point budget
lr_mult   2.0   top1 0.9059   -1.76 pt   FITS
lr_mult   5.0   top1 0.9124   -1.11 pt   FITS
lr_mult  10.0   top1 0.9125   -1.10 pt   FITS
lr_mult  20.0   top1 0.9142   -0.93 pt   FITS
```

**Pre-registered falsification:** *"some schedule brings global pruning under 2.0 points."* It
fired at the very first step away from the default.

The curve is monotone and saturating (2.08 → 1.76 → 1.11 → 1.10 → 0.93), so this is a real trend
rather than one lucky cell, and the 1.15-point span is four times the combined measurement SD
(0.28 pt from experiment 9). The knob is `prune.recovery_lr_mult`, which exists in this codebase
precisely for this purpose and which every run so far left at its default.

### What this retracts

"21 of 23 run winners prune zero", "pruning earns nothing inside a 2.0-point budget", and the
memory note built on them are **artifacts of an untuned recovery schedule**, not properties of
pruning. This is exactly the failure Wang et al. (arXiv:2301.05219) documented when they revived
2017 L1-norm pruning to match modern methods purely by improving the fine-tuning schedule — and it
is the attack the literature review flagged as "the hole I would attack if I were reviewing".

### What it opens, which is better than what it closed

At ratio 0.25 the pruned INT8 model is 1,603,509 bytes against the 8,919,751-byte fp32 baseline:
**5.56x**, against 3.40x for quantize-only. If the tuned schedule holds after quantization, pruning
does not merely become admissible — it buys 1.6x more compression than the paper currently reports
as the ceiling.

### Required follow-up before any of this is written down

1. **Quantize the tuned model and re-measure accuracy.** The 0.9142 above is the pruned FP32 model.
   INT8 on top costs something; the whole claim depends on the pair, not the pruned model alone.
2. **Repeat on the other strategies.** Only global pruning was swept. If a tuned schedule rescues
   uniform and DP knapsack too, then the strategy comparison has to be redone at each strategy's
   own best schedule — comparing methods at one arbitrary shared schedule is the very error Wang
   et al. describe.
3. **Check the baseline is not also improvable.** `recovery_lr_mult` only applies after pruning, so
   the unpruned model did not get the same treatment. A fair comparison may need the fp32 and
   quantize-only baselines re-tuned as well, or the gain is partly an artifact of tuning one arm.
4. **Sweep beyond 20.** The curve has flattened but not turned; the optimum may be past the grid.

Item 3 is the one that could take the finding away again, and it must be done before writing.

---

## Data-hygiene incident — competing load, 03:04–03:18 · 2026-09-16

Batch 1 (experiment 1b, static vs dynamic quantization on CNNs) and five independent challenger
agents ran latency measurements on the same machine at the same time. The challengers were spawned
to *design* attacks on the INT8 performance claim; they went straight to executing them, several
concurrently, while `sweep.py` was measuring.

The evidence is in the challengers' own numbers:

    dynamic_matmul       opt=all  t= 4     2.289 ms  (sd 2.441)

An SD larger than the measurement. `core/measure.py` warns on competing load (`other_load_pct`) but
a warning is ignorable, and it was ignored — by both sides, because neither knew about the other.

**Discarded:** `results/classification_dyn`, `results/stress_resnet50_dyn`, `results/detection_dyn`
moved to `results/_quarantine_contaminated_0316/` rather than deleted. Their ACCURACY and SIZE
columns are deterministic and therefore still valid; a re-run that disagrees on accuracy would be a
real signal. The LATENCY columns are unusable.

**Also found, and worse:** an orphaned `scripts/paper/run_remaining.sh` from 02:47 was still alive,
sitting in its `idle()` wait loop. It was waiting for the python-process count to drop to 1 — and
would then have launched a *second complete copy* of the batch, silently, on top of the first. Two
drivers, both convinced they had the machine to themselves. Killed. The idle-wait was written to
prevent exactly this and instead made the duplicate patient enough to be invisible.

**What this costs the paper:** nothing yet — the affected numbers were caught before they were
written down, which is the whole point of measuring SD in the first place. Experiment 9's variance
work is what made the contamination legible; without a known 0.28 pt noise floor, `sd 2.441` would
have looked like a large-but-plausible spread.

**Standing rule added to memory:** challengers design, and the numbers are produced afterwards by
one process, sequentially. The independence requirement is on the DESIGN, and that is untouched.

---

## Experiment 12 — the parallel-scaling deficit, properly measured · 2026-09-16 · **CLAIM SURVIVES, BUT NARROWER AND MUCH STRONGER**

Not a planned experiment. It came out of fixing a pairing bug in `thread_sweep.py`, which was about
to compare every INT8 variant against the *first* fp32 model in its group — so `p25_int8` would have
been measured against `p00_fp32` and the cost of pruning charged to quantization. Fixing that made
it obvious the same bug was in how the finding itself had been framed.

### The old framing was the weakest possible view of the effect

Finding B was carried as: *"6 variants are faster at 1 thread and slower at 4, with no case in the
opposite direction."* Six out of 71 INT8 variants is 8%. It reads as a handful of oddities, and the
"no counter-examples" argument rests on 6 observations drawn from 3 runs.

But **a sign flip is not the effect.** A flip happens only when INT8's single-thread advantage is
small enough for the scaling deficit to overturn it. Counting flips measures how many models
happened to sit near the crossover — not how general the deficit is.

> **Corrected by experiment 14.** I called the flip count "the weakest view of the effect". It is
> worse than weak: `sp4 = sp1 * eff_ratio` is an exact identity (verified to 6.7e-16 over all 71
> pairs), so "no case in the opposite direction" is *entailed* by the eff_ratio result it was being
> offered as independent support for. Only 2 of 71 pairs could have produced a reverse flip. And
> three of the six flips sit 2.1-2.9% below unity, inside a 3.9-7.3% null floor. Delete the flip
> table; it contains no information the eff_ratio column does not already carry. The effect is the scaling
ratio itself, `latency(1T) / latency(4T)`, and that is available on *every* pair:

    71 sibling-matched pairs, 23 runs
    mean fp32 scaling  1.991x        mean INT8 scaling  1.448x
    INT8 scales worse in 66/71 pairs; 23/23 runs have a negative mean
    per-run Wilcoxon p = 2.4e-07

23 of 23 runs, no exceptions. The same data that supported "6 cases" supports this.

### And then the control nearly took it away

The obvious attack: **INT8 models are faster, and a faster model has less work per thread, so of
course it scales worse.** If that is the whole story, quantization explains nothing.

Tested on the fp32 arms alone — unquantized, so any latency/scaling relation there is the pure
"less work scales worse" effect:

    fp32 models only, n=17:  1T latency vs 1->4T scaling   Spearman rho = +0.958, p = 1.45e-09

The confound is not a quibble. It is nearly deterministic: slower models genuinely parallelise
better. So the raw deficit above is substantially the speedup being counted twice.

Controlling for it (fit `scaling ~ ln(latency)` on the fp32 models, then ask what an *unquantized*
model as fast as this INT8 model would be expected to reach):

    residual = INT8 scaling - expected scaling at its own 1T speed
    mean residual -0.508, negative in 63/71 pairs and 19/23 runs
    per-run Wilcoxon p = 0.00025,  sign test p = 0.0026

It survives — but 23/23 becomes 19/23, and the four exceptions are not noise:

    classification_mobilevit  -1.062        stress_resnet50_prune     +0.319
    classification_deit       -0.983        stress_resnet50           +0.327
    detection                 -0.689        stress_resnet50_prune30   +0.331
    detection_640             -0.651        stress_imagenette         +0.479
    (plain classification runs cluster tightly at -0.34 to -0.39)

### What the claim actually is now

Not *"quantization destroys parallel scaling"* — too broad, and the speed confound eats most of it.
The defensible claim is narrower and says more:

> **Quantization imposes a parallel-scaling penalty beyond what its own speedup predicts. The
> penalty is concentrated in transformers (MobileViT −1.06, DeiT −0.98) and detection (−0.65 to
> −0.69); plain CNN classification shows a small consistent penalty (−0.34 to −0.39); and ResNet50
> and Imagenette show the opposite sign — their INT8 models scale BETTER than an fp32 model of the
> same speed.**

This is worth more than the universal version. A universal effect invites "so it's an ORT
implementation detail"; an effect that is large on transformers, moderate on detection, absent on
ResNet50 and *reversed* on Imagenette demands a mechanism — and it hands experiment 2 a
pre-registered prediction instead of a fishing licence.

### Pre-registered falsification for experiment 2 (node profiling), written before it runs

Per-run Q/DQ share of node time must track the residual above. Specifically:

- MobileViT and DeiT INT8 must show the **highest** Q/DQ share of the runs profiled.
- stress_resnet50 must show a **markedly lower** Q/DQ share than MobileViT/DeiT.
- **If ResNet50's Q/DQ share is comparable to MobileViT's, the mechanism is refuted** and the
  residual pattern needs another explanation (memory bandwidth, thread-pool granularity, or the
  op mix) before any of this is written down.

### Caveats that must ship with it

1. The pairs come from existing sweeps measured at different times. The comparison is *within* a
   sweep — sibling and INT8 measured back to back — so ambient load hits both arms, but this is not
   a purpose-built measurement. Experiment 3's thread sweep is, and it is in the running batch.
2. Only 1T and 4T. Two points cannot tell a gentle slope from a cliff; the thread sweep adds 2 and 6
   inside the P-core class and 8/12 as a separate spilled condition.
3. `n=17` fp32 models for the control fit. The fit is a one-parameter correction on a very strong
   relation, but it is still a fit, and a reviewer may reasonably want the control repeated within
   architecture rather than pooled.

Reproduce: `python scripts/paper/scaling_deficit.py` -> `results/paper/scaling_deficit.csv`

---

## Experiment 13 — finding A decomposed · 2026-09-16 · **CLAIM HOLDS AND IS ALSO TWO DIFFERENT CLAIMS**

Finding A was: *dynamic INT8 beats static QDQ within an architecture, with non-overlapping 4-thread
ranges.* It reproduces — 6 of 6 dynamic variants beat their static stablemate at 4 threads, 1.45x on
DeiT and 1.56x on MobileViT. Those two numbers look like one finding. They are not.

Every dynamic variant compared against its own fp32 and static INT8 stablemates at the same prune
ratio, on speed **and size and accuracy together**:

    classification_deit  (n=3)
      speed   dynamic 1.27x fp32    static 0.87x fp32    dynamic 1.45x static
      size    dynamic keeps 29% of fp32 bytes,  static keeps 31%
      acc     dynamic -0.05 pt      static -0.35 pt

    classification_mobilevit  (n=3)
      speed   dynamic 1.08x fp32    static 0.69x fp32    dynamic 1.56x static
      size    dynamic keeps 74% of fp32 bytes,  static keeps 63%
      acc     dynamic -0.02 pt      static -8.33 pt   (-15.99 pt at ratio 0.50)

### Two mechanisms, and only one of them is a win

**DeiT — a genuine win.** Dynamic is 1.27x faster than *fp32*, compresses to 29% (static manages
31%), and costs 0.05 pt. It beats static on every axis at once. DeiT is attention-heavy, so
`op_types_to_quantize: [MatMul, Gemm]` reaches most of the weight bytes.

**MobileViT — not a win, a retreat.** Dynamic is 1.08x fp32: within noise of not quantizing at all.
It keeps **74%** of the fp32 bytes, *worse compression than static's 63%*. It "wins" only because
static is harmful here — 0.69x speed and −8.33 pt accuracy. MobileViT is conv-dominated, so
MatMul-only quantization declines to touch most of the network. What the 1.56x measures is **the
damage static does**, not a benefit dynamic provides.

Reporting "dynamic beats static by 1.45–1.56x" without the size column would be the efficiency
misnomer Dehghani et al. (arXiv:2110.12894) name directly: a single cost metric chosen so the
method looks good. A reviewer opens the size column and the MobileViT half of the claim inverts.

### What both surviving findings now say together

Experiment 12 (scaling) and experiment 13 (static/dynamic) converge on the same partition, from
independent measurements:

| architecture | static INT8 speed vs fp32 | scaling residual | static INT8 accuracy |
|---|---|---|---|
| MobileViT | 0.69x | −1.06 | −8.33 pt |
| DeiT | 0.87x | −0.98 | −0.35 pt |
| detection | — | −0.65 to −0.69 | — |
| classification (plain CNN) | — | −0.34 to −0.39 | — |
| ResNet50 / Imagenette | — | **+0.32 to +0.48** | — |

**ORT static QDQ INT8 fails on transformers and conv-heavy mobile nets and is fine on plain CNNs.**
That is one thesis with two independent lines of evidence, and it is falsifiable: stage 1b is
measuring three CNN families right now, and if static INT8 *also* underperforms there, the partition
is wrong and both findings collapse into "static QDQ is just slow", which is a much weaker paper.

### Pre-registered, before stage 1b reports

- Plain CNNs (classification, ResNet50, detection): static INT8 should be **faster than fp32** at
  4 threads, and dynamic should gain **little or nothing** over static.
- If dynamic beats static on CNNs by a margin comparable to the transformers' 1.45–1.56x, the
  architecture partition is refuted.

### Caveat that must ship with it

n=2 architectures, both transformers, 3 prune ratios each — and ratios within a run are not
independent observations. Stage 1b exists to take this to n=5. Until it reports, this is a
decomposition of a narrow result, not a general claim.

Reproduce: `python scripts/paper/static_vs_dynamic.py` -> `results/paper/static_vs_dynamic.csv`

---

## Experiment 14 — independent challenger triage · 2026-09-16 · **BOTH HEADLINE CLAIMS KILLED AS WORDED; ONE SMALLER FINDING SURVIVES EACH**

Five challenger agents, designed and run independently (see
`memory/adversarial-tests-need-a-separate-agent.md`), attacking the two INT8 performance claims from
different angles: runtime internals, confounds, triviality, generality, statistics. 6 agents,
854,942 tokens, 199 tool uses, 34 minutes. Their verdict: *"Very exposed… plan to retract-and-replace
rather than defend."*

Every kill below was **re-verified here** before being accepted. Three of them need no machine time
at all, which is the most uncomfortable part: they were findable from a text editor at any point in
the last three days.

### Claim A — the mechanism is not merely unsupported, it is about the wrong object

**A1. The Q/DQ counts describe a graph that never executes.** Every number in the mechanism
sentence — 416 Q / 599 DQ / 90 compute ops / "36% productive" — is parsed from the `.onnx` file.
ORT's `QDQSelectorActionTransformer` deletes ~90% of those nodes before the first inference. Three
challengers independently dumped `optimized_model_filepath`:

    MobileViT  416/599 -> 40/97        DeiT  442/593 -> 63/102        ResNet50  74/182 -> 1/1

**And recounted on the executed graph the variable points the wrong way**: the FAST family (DeiT,
1.38x) executes *more* surviving Q/DQ (165) than the slow one (MobileViT, 137), and under our own
"compute" definition DeiT is 25% productive against MobileViT's 36%. Surviving boundaries are ~15%
of INT8 kernel time — zeroing all of them still leaves INT8 slower than fp32.

This also retro-explains experiment 1. The Simpson's-paradox reversal there was not a sampling
quirk: we were correlating latency against a quantity that does not exist at run time.

**A2. The 26-vs-7 registry contrast was never measured.** Verified in
`configs/classification_mobilevit.yaml:23` and `classification_deit.yaml:23`: the dynamic arm is
hand-written `{method: dynamic, op_types_to_quantize: [MatMul, Gemm]}` — and `Gemm` is not in
`IntegerOpsRegistry` at all. So the experiment compares **26 op types against two hand-picked ones**,
with calibration mode perfectly confounded with op coverage. The registry-size story was a narrative
bolted onto an experiment that varied something else. Two lines of config.

**A3. The penalty tracks model SIZE, not architectural heterogeneity.** Topology is identical across
p00/p25/p50 — same graph, fewer channels — yet the static speedup moves monotonically:

    MobileViT 1T  0.942 -> 0.831 -> 0.681        DeiT 1T  1.684 -> 1.407 -> 1.045

Reproduced here exactly. The "n=9" cell is three models measured three times, and the printed range
"(0.68–0.94)" is that dose–response curve presented as if it were uncertainty.

**What survives A, and the replacement mechanism.** MobileViT static QDQ really is slower than fp32,
robustly, across every configuration anyone tried (0.61–0.93). The mechanism visible in the same
dumps is **fusion blocking**, not boundary density: static QDQ prevents the QuickGelu/Gelu/
SkipLayerNorm fusions that fp32 and dynamic both receive — DeiT static runs `Erf 12 + Div 24 +
Sqrt 24 + QLinearMul 48` where fp32 runs `BiasGelu 15` — and substitutes QLinearAdd/Mul/Sigmoid for
cheap fp32 elementwise ops. About a third of the current scope, and **it makes the opposite
prediction**: restricting `op_types_to_quantize` to Conv+MatMul strictly *increases* boundary count
and should make the model *faster*. That is a clean, cheap, falsifying test.

### Claim B — dead as worded, three ways, none needing machine time

**B1. "No case in the opposite direction" is an arithmetic identity, not corroboration.** Verified
here to 6.7e-16 over all 71 pairs:

    sp4 == sp1 * eff_ratio          where eff_ratio = scale_int8 / scale_fp32

A flip requires `1 < sp1 < 1/eff_ratio`; a *reverse* flip requires `sp1 < 1` AND `eff_ratio > 1`.
Only **2 of 71 pairs** could even have produced one. The absence was forced by the eff_ratio result
it was being offered as independent evidence for.

**B2. Half the flip table is ties.** Three of the six sit 2.1–2.9% below unity against a measured
single-session null floor of 3.9–7.3% — a floor the challengers established with a plan-identical
pair (`p00_int8` vs `p00_int8_qat`: same 1851 nodes, same 416 Q/599 DQ, different weights only) that
the harness reports as 7.30% apart at 3.9 sigma, collapsing to −0.72% under randomised order.

    deit p50 22.1%   anomaly p50 19.0%   deit p25 14.0%  |  deit p00 2.9%   anomaly p25 2.5%   classification p25 2.1%

**B3. The framing is not invariant to flags unrelated to quantization.** Toggling `allow_spinning`
flipped 2 of 3 tested crossovers, in *opposite* directions; pinning to 4 vs 6 CPUs swings the DeiT
fp32 4-thread baseline by 1.86x with no model change. And MobileViT fp32 — which supplies nine of
the ten "slow at both" cases — has a fitted parallel region of 0.011 ms out of 1.453 ms. **There is
no parallel scaling there to destroy.**

**What survives B — and it is the strongest result in the project.** `eff_ratio` median 0.727, below
1 in **66 of 71** paired variants, Wilcoxon p = 5.9e-13, and it **survives `log(lat_t1)` as a
covariate** (is_int8 coefficient −0.306, t = −7.15, ΔR² = 0.105). The small-model confound was
tested and did not kill it.

This is the same result as experiment 12, derived independently, with the same confound control and
the same 66/71. Two separate derivations, one of them by agents with no access to my reasoning,
agreeing on the number *and* on the control. Lead with this; delete the six-case flip table.

### A fourth kill, aimed at the instrument rather than the claim

`other_load_pct` — the competing-load alarm this project has trusted all night, including for
tonight's contamination audit — **is not measuring competing load.** Verified here over 232
measurements: it fires on 141 of them (61%) while correlating with actual core delivery at
**Spearman −0.057, p = 0.39**. Indistinguishable from zero.

The instrument that does work is computable from columns already recorded:

    core_delivery = cpu_ms_per_inf / (threads * latency_p50_ms)      1.0 = every requested core delivered
    median 0.981 at 1 thread, 0.995 at 4; below 0.95 in only 14/129 four-thread measurements

So the existing corpus is in better shape than the alarm suggested — but for three days the project
has been reading a gauge wired to nothing. `core/measure.py` gets `core_delivery` as the gate, after
the running batch finishes (editing it mid-batch would split the CSV schema across stages).

Related: `core/sweep.py` runs `for ratio: for precision:` with no shuffle and no cooldown, so
**measurement order is perfectly nested inside variant** — thermal drift and cache state are
confounded with the thing being measured. That is a design defect in the harness, not in any one
result, and it is why the challengers' randomised-order re-run collapsed a 7.30% "difference"
between two plan-identical models to −0.72%.

### What this costs, and what it is worth

Both headline sentences have to be withdrawn. What replaces them is smaller, but it is *earned*:
one robust statistical result with its confound controlled and independently reproduced, and one
mechanism (fusion blocking) that makes a falsifiable prediction in the opposite direction from the
one we published.

The uncomfortable arithmetic: three of these kills cost zero machine time. The overnight batch was
never the bottleneck — reading our own config file was.

### Still genuinely open, in the challengers' priority order

1. **Is it one ORT release?** Their pilot (at 24–52% background load, so not quotable) found
   MobileViT static is *not* slow on 1.16.3 (1.006 at 4T) or 1.20.1 (1.056), and that DeiT dynamic is
   *slower* than fp32 at 4T on 1.20.1 (0.912) — contradicting the second half of claim A. If that
   holds on a clean run, the real finding is a **1.28 → 1.30 MobileViT regression**: more useful,
   more publishable, and a different paper. This is the top priority once the machine frees up.
2. Is it batch size 1? 3. Is it 65 tokens (DeiT sequence length)? 4. Is it op coverage rather than
   calibration mode — the confound in A2, which a three-arm run separates cleanly.

---

## Audit — the noise floor is about 3x too optimistic · 2026-09-16

`core/measure.py` runs `repeats` probe processes **back to back** inside one measurement, so the
`*_sd` columns are a WITHIN-condition spread: same thermal state, same cache state, same session
of the machine's life. Measured over 270 cells that have a recorded SD:

    t=1  median relative SD 1.10%   p90 2.63%
    t=4  median relative SD 1.79%   p90 4.61%

The challengers measured the BETWEEN-session floor by re-running plan-identical models in fresh
sessions at randomised order: **3.9-7.3%**. Roughly 3x larger. Their sharpest demonstration is a
pair the repo already contains - `p00_int8` vs `p00_int8_qat`, same 1851 nodes, same 416 Q / 599 DQ,
differing only in weights - which the harness reports as **7.30% apart at 3.9 sigma**, collapsing to
**-0.72%** once measurement order is randomised.

**So the bar every latency claim in this project has been judged against is too low.** What that
does and does not change:

| claim | effect size | survives a 7% floor? |
|---|---|---|
| dynamic vs static, transformers | 45-56% | yes, comfortably |
| ResNet50 static INT8 vs fp32 | 148% (2.48x) | yes |
| dynamic Conv on a CNN | -1750% (0.054x) | yes |
| scaling-deficit residual | ~25-35% in scaling units | yes |
| three of the six "sign flips" | 2.1-2.9% | **no - they were always ties** |

Nothing load-bearing moves. But the floor has to be restated as 7%, not 2%, and the three flips
were never observations. Experiment 9's 0.28 pt figure is an ACCURACY spread and is unaffected by
this - accuracy is deterministic given the model and the eval set; it is the latency floor that was
wrong.

**One thing the challengers were wrong about, and it matters for the fix list.** Their protocol note
says "a single process leaks ORT thread pools between sessions and every number from a shared-process
sweep tonight was garbage". True of their own ad-hoc scripts; **not** true of this harness -
`core/measure.py:216` spawns an independent `_run_probe` subprocess per repeat per thread count, and
has since the measurement rewrite. Worth stating because the temptation after a triage this
comprehensive is to accept every item, and accepting this one would mean rewriting something that is
already correct.

**Fix, queued with the other harness work:** interleave repeats across variants instead of running
them consecutively, so the recorded SD is a between-session spread rather than a within-session one.
Until then, quote 7% as the latency floor and treat the `*_sd` columns as a lower bound.

---

## Experiment 18 — dynamic Conv on two CNNs · 2026-09-16 · **MY OWN PRE-REGISTERED PREDICTION FAILED, AND THE REPLACEMENT IS BETTER**

Stage 1b measured the dynamic `[Conv, MatMul]` arm on two CNNs:

| run | arch | fp32 1T | dyn 1T | speed | accuracy | size |
|---|---|---|---|---|---|---|
| classification | mobilenet_v2, 32px | 0.944 ms | 15.86 ms | **0.060x** | **−11.07 pt** | 27% |
| stress_resnet50 | resnet50 | 94.74 ms | 1401.23 ms | **0.068x** | **−0.46 pt** | 25% |

### The speed prediction I registered was wrong

`dynamic_quant_cost_model.py` pre-registered: *"the dynamic penalty should rise monotonically with
activation volume per quantized op."* ResNet50 has vastly more activation volume than a 32px
MobileNetV2, and its penalty is **slightly smaller** (0.068 vs 0.060). Flat, not rising. **The
prediction is falsified.**

The mechanism is fine; my derivation from it was not. `DynamicQuantizeLinear` does O(N) work over an
N-element activation, while the convolution it feeds does O(N·K). The *ratio* is therefore set by
kernel size and fusion, not by N — so a roughly **scale-invariant ~15x penalty** is what the
mechanism actually predicts, and I confused absolute overhead with relative overhead when writing
the prediction down. The corrected prediction is that the penalty stays near-constant across
resolutions; detection (320px) is the third point and is running now. If detection lands far from
15x, the mechanism is in trouble rather than the arithmetic.

This is the first prediction this project has registered and then failed, and it failed the useful
way: the number that broke it also fixed the reasoning.

### The accuracy split is the real finding, and it unifies the whole project

−11.07 pt on MobileNetV2 against −0.46 pt on ResNet50, from the *same* quantization method. Nothing
about activation volume explains a 24x difference in damage. The architectures do:

| model | blocks | static INT8 | dynamic Conv |
|---|---|---|---|
| MobileNetV2 | depthwise separable | −0.14 pt | **−11.07 pt** |
| MobileViT-XXS | MV2 inverted residual + attention | **−4.62 pt** (−15.99 at ratio 0.50) | — |
| ResNet50 | plain conv | −0.10 pt | −0.46 pt |
| DeiT | attention / MatMul | −0.28 pt | −0.01 pt |

**Static and dynamic fail on the same structures for the same reason, and the difference between
them is calibration.** `configs/classification_mobilevit.yaml:14` already recorded the cause
without anyone connecting it to the dynamic arm: the SiLU inverted-residual blocks have *"channels
up to ±90 next to channels within ±1, and one per-tensor activation scale erases the small ones"*.

- **Static** quantization calibrates offline, and this project's `calibrate_method: auto` selects
  **percentile 99.99**, which clips the ±90 outliers and preserves the small channels. That is why
  MobileNetV2 static is fine (−0.14 pt) and why MobileViT went from 0.468 to 0.894 on the pruned
  ResNet50 when percentile replaced minmax.
- **Dynamic** quantization has no calibration. It takes the **exact min/max** of the actual
  activation tensor at run time — which is maximally sensitive to exactly those outliers. It is
  minmax calibration, recomputed per inference, with no percentile option available.

So dynamic quantization is not "static without the calibration step". It is **static pinned to the
worst calibration method**, and it fails precisely where minmax fails: heterogeneous per-channel
activation ranges, i.e. depthwise separable convolutions.

### Why this matters more than the thing it replaced

It makes one statement cover every accuracy failure recorded in this project, it names the structure
responsible (depthwise separable / inverted residual), and it predicts where INT8 will fail on a
model nobody has measured yet — which is the property a paper needs and a table of speedups does not
have.

### Pre-registered, before the remaining runs report

1. **Detection** (320px, a conv detector) should show a dynamic speed penalty near 15x. Far from it
   refutes the scale-invariance derivation above.
2. Detection's dynamic **accuracy** damage should be small if its backbone is plain-conv, and large
   if it is a MobileNet-family backbone. This is checkable from the config before the number lands,
   and is therefore a real prediction rather than a retrofit — `configs/detection.yaml` must be read
   and the expectation written down *now*.
3. On MobileNetV2, forcing static quantization to **minmax** should reproduce a large part of the
   dynamic arm's −11.07 pt. If it does not, the calibration explanation is wrong and the difference
   is something else about the `ConvInteger` path.

Item 3 is the decisive one: it is cheap, it isolates calibration from everything else, and it can
falsify the unification in a single run.

### Registered at 04:15, while detection was still measuring

`configs/detection.yaml` is **YOLOv8n** (`weights/detection/yolov8n.pt`, 320px, COCO-500 val).
YOLOv8n's backbone is CSPDarknet-style Conv-BN-SiLU with C2f blocks — **standard convolutions, no
depthwise separable layers.** Under the unification above that makes it structurally like ResNet50,
not like MobileNetV2. So:

1. **Speed:** dynamic penalty near 15x, since the derivation says the ratio is scale-invariant.
2. **Accuracy:** damage closer to ResNet50's −0.46 pt than to MobileNetV2's −11.07 pt.

But one hedge has to be stated as part of the prediction rather than added afterwards if the number
disappoints. Detection has a second, *independent* known vulnerability this codebase already warns
about: the head concatenates box coordinates (0–320 px) with class scores (0–1), and
`concat_range_mismatch` in `core/quantize.py` exists precisely because one shared scale across that
32x range destroys the smaller tensor. Dynamic quantization cannot use percentile clipping to
mitigate it. So the honest registered interval is:

> **mAP damage worse than ResNet50's −0.46 pt because of the head, but well short of MobileNetV2's
> −11.07 pt.** If detection lands at or beyond −11 pt, the depthwise explanation does not survive as
> the *primary* cause and range heterogeneity from any source becomes the better statement. If it
> lands near −0.5 pt, the head risk is smaller than the codebase assumes and that warning deserves
> re-examination too.

Both outcomes are informative, which is the test being worth running.

---

## Synthesis — two independent failure modes, with different structural predictors · 2026-09-16

The anomaly run settles something the narrative had been blurring. A 1-D convolutional autoencoder,
static INT8, three prune ratios:

    p00  1T 1.863x  4T 1.066x   accuracy -0.02 pt
    p25  1T 1.525x  4T 0.975x   accuracy +0.04 pt
    p50  1T 1.306x  4T 0.810x   accuracy +0.01 pt

Accuracy is untouched — INT8 is free here — and yet the scaling deficit is among the clearest in the
corpus (1.86x collapsing to 1.07x, and below unity at higher ratios). **Accuracy damage and the
scaling deficit are not the same phenomenon and do not co-occur.**

|  | accuracy intact | accuracy damaged |
|---|---|---|
| **scales normally** | ResNet50, MobileNetV2 (static) | — |
| **scaling deficit** | anomaly 1-D AE, DeiT | MobileViT (static), MobileNetV2 (dynamic Conv) |

Each axis has its own structural predictor, and neither predicts the other:

**Axis 1 — accuracy.** Per-channel *activation-range heterogeneity*, which is a property of
depthwise separable / inverted-residual blocks, interacting with the calibration method. Channels
spanning ±90 beside channels within ±1 cannot share one per-tensor scale. Percentile calibration
clips and survives; exact min/max — which is what dynamic quantization does, per inference —
does not. Everything with plain convolutions, attention, or Linear layers is fine: ResNet50 −0.10,
anomaly −0.02, DeiT −0.28, MobileNetV2 static −0.14.

**Axis 2 — speed.** The parallel-scaling deficit, which survives a control for absolute speed in
19 of 23 runs and is architecture-specific in a pattern that does *not* line up with axis 1: largest
on MobileViT and DeiT, moderate on detection, small on plain CNN classification, and reversed on
ResNet50 and Imagenette. The candidate mechanism is fusion blocking (experiment 16 tests it against
its own opposite prediction).

**Why the separation matters more than either finding alone.** A practitioner asking "will INT8 hurt
me?" is asking two questions with two different answers, and the structural features that answer
them are visible *before* any measurement: look for depthwise separable blocks to predict accuracy
risk, and look at op mix and fusion opportunities to predict speed risk. That is a usable result.
It also explains why the single-claim framings kept collapsing under challenge — they were averaging
two independent effects into one sentence, and every counter-example a challenger found was simply a
model sitting in the off-diagonal cell.

---

## Experiment 2 — node profiling · 2026-09-16 · **PREDICTION HOLDS AT 50:1, AND FUSION BLOCKING IS TWICE THE COST OF THE MECHANISM WE PUBLISHED**

The pre-registered condition, written in experiment 12 before this ran: *MobileViT and DeiT INT8
must show the highest Q/DQ share; ResNet50 markedly lower; **if ResNet50 is comparable to
MobileViT, the mechanism is refuted.*** One thread, separate from the headline harness, warmup
iterations dropped.

    classification_mobilevit   Q/DQ 15.4% of node time
    classification_deit        Q/DQ 14.3%
    stress_resnet50            Q/DQ  0.3%        <- 95.2% of its time is QLinearConv

**A 50:1 gap, in the predicted direction, tracking the scaling residuals exactly** (MobileViT −1.06,
DeiT −0.98, ResNet50 +0.33). ResNet50's INT8 graph is almost pure fused int8 arithmetic; there is
nothing for boundaries to cost.

### But the bigger number is the one nobody was looking at

Profiling each model's static arm against its *dynamic* sibling makes the fusion-blocking mechanism
quantitative for the first time:

| | static QDQ | dynamic MatMul |
|---|---|---|
| **DeiT** fused activation/norm ops | LayerNormalization 4.1% only — **Gelu is gone** | Gelu 8.6%, LayerNormalization 5.7%, SkipLayerNormalization 1.2% |
| **DeiT** decomposed elementwise | QLinearAdd 12.1%, QLinearMul 8.0%, Erf 4.6%, Div 2.7%, Sqrt 1.0% → **28.5%** | Sqrt 1.5%, Div 0.9% → **2.4%** |
| **MobileViT** fused | LayerNormalization 3.4% only — **QuickGelu is gone** | QuickGelu 7.9%, LayerNormalization 4.6%, SkipLayerNormalization 1.6% |
| **MobileViT** decomposed elementwise | QLinearAdd 10.3%, QLinearMul 9.8%, QLinearSigmoid 5.3%, Sqrt 1.3%, Div 0.8% → **27.5%** | Sqrt 2.0%, Div 1.2% → **3.2%** |

Static QDQ spends **27–28% of node time on decomposed elementwise work its dynamic sibling does not
do at all**, because in the dynamic graph those ops fuse back into Gelu / QuickGelu /
SkipLayerNormalization. The Gelu node is simply absent from both static profiles.

**So fusion blocking costs roughly twice what the boundary overhead costs — 27.5% against 15.4% on
MobileViT — and the boundary story is the one this project published.** Together they account for
~43% of MobileViT static's node time as overhead the fp32 and dynamic paths never pay, which is
enough to explain 0.72× against fp32 without anything else.

### Why this finishes the architecture partition

A CNN has no Gelu and no LayerNorm to fuse, so there is nothing for static QDQ to break. ResNet50
runs 95.2% QLinearConv and pays 0.3% in boundaries. That is the whole partition in one sentence:

> **ORT static QDQ INT8 is safe where there is nothing to fuse, and costly exactly where the fp32
> graph depends on activation/normalisation fusion — which is what distinguishes transformers and
> conv-heavy mobile nets from plain CNNs.**

### What this does and does not establish

This is **observational**. It shows the fused ops are present in one graph and absent in the other,
and it sizes the difference. It does not establish direction: an alternative reading is that
whatever makes static slow also happens to prevent fusion. Experiment 16 is the causal test, and it
is still the right one to run — restrict `op_types_to_quantize` to Conv+MatMul, which *increases*
boundary count while *restoring* fusion opportunities. Boundary density predicts slower; fusion
blocking predicts faster. Nothing here pre-empts that.

Note also that the challengers found fusion blocking qualitatively and asserted the boundaries were
"only ~15% of int8 kernel time". That figure reproduces here to the decimal — 15.4% — from an
independent measurement, which is worth stating because it is the second time tonight their number
and ours have landed on the same value from different directions.

### Correction to the detection prediction, registered at 04:15

Detection (YOLOv8n, three resolutions) came back:

    penalty   192px 12.6x   256px 14.5x   320px 13.0x
    mAP       192px -0.67   256px -0.75   320px +0.01

**Speed, prediction version 2 — HOLDS.** Five measurements now span MobileNetV2 at 32px (16.8×),
ResNet50 (14.8×) and YOLOv8n at three resolutions (12.6–14.5×). All inside 12.6–16.8×. Scale
invariance is the right reading, and version 1 (penalty rises with volume) remains correctly
falsified.

**Accuracy — one bound was wrong.** The registered interval was "worse than ResNet50's −0.46 pt
because of the head, but well short of −11.07". Two of three resolutions landed inside it; 320px
came in at **+0.01 pt**, better than fp32 and outside the lower bound. So the hedge I added about
the head — box coordinates at 0–320 px sharing a scale with 0–1 scores — **was unnecessary**. That
was the explicitly registered alternative outcome: *"if it lands near −0.5 pt, the head risk is
smaller than the codebase assumes and that warning deserves re-examination too."* It does.
`concat_range_mismatch` in `core/quantize.py` is calibrated on an assumption that this run does not
support, and the warning threshold deserves a second look rather than continued trust.

The depthwise hypothesis is unaffected and strengthened: YOLOv8n is plain-conv, and its dynamic
accuracy damage is ≤0.75 pt at every resolution — like ResNet50's −0.46, nothing like MobileNetV2's
−11.07.

### Detection static INT8, for the record

Remarkably stable across resolutions, and another plain-conv architecture where static INT8 simply
works: **2.09–2.25× at 1 thread, 1.27–1.32× at 4, −0.61 to −0.69 pt mAP, 28% of fp32 bytes.** The
1T→4T fall from ~2.2× to ~1.3× is the scaling deficit again, plainly visible in a fourth
architecture family.

One curiosity worth a line: the dynamic arm *improves* from 1T to 4T at every resolution
(0.069→0.080, 0.077→0.085, 0.079→0.093). Whatever else `DynamicQuantizeLinear` is, it parallelises
better than the fp32 graph it replaces — consistent with it being a large, embarrassingly parallel
reduction.

---

## Correction — I over-claimed the replacement gauge · 2026-09-16 04:40

Earlier tonight I retired `other_load_pct` (fires on 61% of measurements, correlates −0.057 with
real core delivery) and called `cpu_ms_per_inf / (threads × latency_p50_ms)` **"the instrument that
works"**. That was published in the report and queued as a `core_delivery_warn: 0.90` gate in
`core/measure.py`. The thread sweep shows I validated only half its range:

    t= 1  pinned 1 cpu  [0]                        median 0.942
    t= 2  pinned 2 cpus [0,2]                      median 0.991
    t= 4  pinned 4 cpus [0,2,4,6]                  median 0.983
    t= 6  pinned 6 cpus [0,2,4,6,8,10]             median 0.991
    t= 8  pinned 8 cpus [0,2,4,6,8,10,1,3]         median 2.434   <- impossible as a fraction
    t=12  pinned 12 cpus [...]                     median 1.231   <- impossible as a fraction

A *delivery fraction* cannot exceed 1. Two mechanisms produce values above it, and both were
foreseeable: ORT's thread pool **spin-waits** by default (`spinning: True`), burning CPU while idle;
and `cpu_ms_per_inf` is total CPU over the window divided by run count — a **mean** — while
`latency_p50_ms` is a **median**, so the ratio breaks whenever the latency distribution has a tail,
which is exactly what happens once threads spill past the 6 physical P-cores.

**So the metric is interpretable as a delivery fraction only when the thread count is at or below
the physical core count and the latency distribution is tight.** Outside that it is a CPU-burn
ratio, which is still diagnostic but is a different quantity with a different reading.

What this does and does not change:

- The criticism of `other_load_pct` stands unchanged. It was validated at ρ = −0.057, p = 0.39 over
  232 measurements and it fails on its own terms.
- The audit of the existing corpus stands: every measurement in it is at 1 or 4 threads, inside the
  valid range, and the medians there (0.981, 0.995) are sound.
- **The queued `core/measure.py` gate is wrong as drafted** and must not ship at
  `core_delivery_warn: 0.90` applied unconditionally. It has to be skipped, or read differently,
  when `threads > physical cores`.
- `scripts/paper/ort_version_bisect.py` uses the same gate at 1 and 4 threads only, so it is inside
  the valid range and needs no change.

Worth saying plainly: I spent the night documenting a gauge nobody had validated, then proposed a
replacement and validated it over exactly the range where the old data happened to live. The thread
sweep caught it within two hours only because it was the first experiment to measure outside that
range. The general lesson is not "check your instruments" — it is that an instrument validated only
where you already have data is not validated.

---

## A candidate mechanism for axis 2, and a sharper test for experiment 16 · 2026-09-16 04:55

The node profiles and the scaling residuals were derived independently. Put side by side, the static
INT8 arm's *memory-bound* work tracks the scaling deficit:

    run                        Q/DQ   elementwise   combined   scaling residual
    classification_mobilevit  15.4%       28.0%       43.4%        -1.062
    classification_deit       14.3%       28.5%       42.8%        -0.983
    stress_resnet50            0.3%        3.8%        4.1%        +0.327

(elementwise = QLinearAdd/QLinearMul/QLinearSigmoid/Erf/Div/Sqrt/Pow/ReduceMean — the ops static QDQ
substitutes for fused Gelu/QuickGelu/SkipLayerNormalization.)

**n = 3.** This is an ordering check, not a correlation, and MobileViT and DeiT are nearly tied on
both axes so really it is a two-point contrast. Stated as such deliberately: the last time this
project let three points carry a mechanism it was counting nodes that never executed.

### The hypothesis it suggests

Fusion blocking does not merely add work — it changes the *kind* of work. A fused `Gelu` or
`SkipLayerNormalization` is arithmetic-dense: many FLOPs per byte moved. The `QLinearAdd` /
`QLinearMul` / `Erf` / `Div` chain that replaces it is memory-bound: few FLOPs per byte, so it
saturates bandwidth rather than cores. **Memory-bound kernels both cost more wall-clock and scale
worse with threads** — which would make axis 1's mechanism (fusion blocking) and axis 2's symptom
(the parallel-scaling deficit) the same thing seen twice, rather than two coincidences on the same
architectures.

That would be a tidier result than the two-axis split recorded above. It should be resisted until
tested, because tidiness is exactly what has been wrong here three times tonight.

### Sharper pre-registration for experiment 16, before it runs

The fusion-blocking test already predicts that restricting `op_types_to_quantize` to `[Conv, MatMul]`
makes the model **faster** despite raising the boundary count. The hypothesis above adds a second,
independent prediction from the same run, at no extra cost:

> It should also make the model **scale better** — the 1T→6T parallel efficiency of
> `int8_static_cm` should sit above plain `int8`, not merely its absolute latency.

Three outcomes, all informative:

- **Both improve** → fusion blocking explains axis 1 and axis 2 together, and the two-axis framing
  collapses into one mechanism with two symptoms.
- **Latency improves, scaling flat** → the mechanisms really are separate; keep the two-axis
  framing, and axis 2 still has no mechanism.
- **Neither improves** → fusion blocking is refuted as a cause and the profile above was
  correlational after all. Publish the effect without a mechanism rather than reach for a third.

`thread_sweep.py` already measures the required curve, so this needs the fusion-blocking sweep to be
followed by one thread sweep on its output and nothing more.

---

## Experiment 21 — the scaling deficit in the right currency · 2026-09-16 05:05 · **STRONGER, AND IT CORRECTS A PUBLISHED CLAIM**

The thread sweeps allow Amdahl's law to be fitted rather than efficiency tables to be eyeballed:

    speedup(n) = 1 / ((1 - p) + p/n)

`p` is the fraction of the work that is parallel. It is **dimensionless and already normalised for
model size**, which means the whole log-latency confound control from experiment 12 becomes
unnecessary — the quantity that control was correcting for is built into the metric.

### MobileViT has no parallel region at all, and the challengers were right about it

Fitted on the 4-point same-core-class curve:

    classification_mobilevit   p00_fp32          p = -0.019      serial 1.428 of 1.402 ms
                               p00_int8          p = -0.405
                               p00_int8_qat_fp32 p = -0.015
    (every variant negative; fp32 baseline_fp32 p = 0.078)

**Negative.** Not merely small — Amdahl cannot describe the curve, because adding threads adds only
overhead. The challengers said *"MobileViT fp32 has a fitted parallel region of 0.011 ms out of
1.453 ms — there is no parallel scaling there to destroy."* This is worse than their reading and
confirms their point outright. **Every MobileViT scaling number in this project is a difference
between two quantities that do not exist**, and MobileViT supplied 12 of the 71 pairs and the
largest residual (−1.06) in the headline result.

### DeiT is the real case, and the mechanism shows up in the serial term

    DeiT p00     fp32           p = 0.752   1T 11.188 ms   serial 2.774 ms
                 int8 (static)  p = 0.416   1T  7.047 ms   serial 4.116 ms
                 int8_dyn_mm    p = 0.528   1T  5.811 ms   serial 2.745 ms

Static INT8 makes the model **faster overall** (11.19 → 7.05 ms at one thread) while making its
**serial component 48% larger** (2.77 → 4.12 ms). Dynamic quantization makes it faster still
(5.81 ms) and leaves the serial component **untouched** (2.745 vs 2.774 ms — within noise).

That is the fusion mechanism stated in the currency that matters: *static QDQ converts parallel work
into serial work; dynamic quantization does not.* The memory-bound `QLinearAdd/QLinearMul/Erf/Div`
chains that replace fused `Gelu` are exactly the kind of work that does not parallelise, and the
serial-millisecond column measures them directly.

### The corpus re-analysed, with degenerate runs excluded

With 1T and 4T only, `p` is exactly determined, so every pair in the corpus can be re-expressed.
Two runs fall below p = 0.25 on their **fp32** arm — MobileViT (−0.156) and anomaly (0.085):

    all 23 runs              66/71 pairs worse, 23/23 runs
    21 runs with p >= 0.25   54/55 pairs worse, 21/21 runs, mean dp -0.254,
                             per-run Wilcoxon p = 9.5e-07

**Four of the five original counter-examples lived in the two excluded runs.** The exceptions
concentrated precisely where the metric was meaningless, which is what a principled exclusion looks
like as opposed to a convenient one.

On which — the exclusion has to defend itself, because post-hoc filtering is the error this log is
full of. Three things make it legitimate, and they were all true before the numbers were seen:

1. The criterion came from an **independent party** — the challengers argued MobileViT had no
   parallel region hours before this was fitted, from their own measurement.
2. It is applied to the **fp32 control arm only**. The treatment arm cannot influence which runs are
   kept, so the filter cannot be tuned toward the result.
3. It is a **physical criterion** ("is there parallel work here at all"), not an outcome criterion.

The unfiltered numbers are reported above and in `results/paper/scaling_deficit.csv` regardless.

### What this corrects in the published report

Experiment 12 concluded, and the report states, that ResNet50 and Imagenette show the **opposite
sign** — INT8 scaling *better* than an fp32 model of the same speed (+0.32 to +0.48). In the Amdahl
framing they do not:

    stress_resnet50      p(fp32) 1.003   dp -0.101
    stress_resnet50_prune30      0.995   dp -0.092
    stress_resnet50_prune        0.980   dp -0.081
    stress_imagenette            0.925   dp -0.079

They lose parallel fraction like everything else — **just far less** (−0.08 to −0.10, against −0.25
to −0.35 for DeiT and CNN classification and −0.20 to −0.29 for detection). The positive residuals
were an artifact of the log-latency regression used to control the speed confound; the Amdahl metric
needs no such control and does not produce them.

So the corrected statement is a **gradient, not a sign flip**: every architecture with real parallel
work loses parallel fraction under static INT8, and the size of the loss tracks how much of the fp32
graph depended on fusion. That is a better result than the one it replaces — a monotone
architecture-ordered effect rather than an unexplained reversal — and the report must be updated to
stop claiming a reversal that the better metric does not show.

### Honest limits

- The 4-point fits carry 1.8–12.1% worst-case residuals against a 7% measurement floor, so the fits
  are at the edge of what the data resolves. The 2-point corpus estimates have no residual check at
  all — they are exact by construction and therefore inherit the full measurement error.
- `dp < 0` is algebraically the same statement as "INT8 scales worse"; this is **not independent
  corroboration** of experiment 12, it is the same observation in better units. What is new is the
  principled exclusion, the serial-millisecond decomposition, and the correction to the sign claim.
- ResNet50's thread sweep is still running. Its 4-point fit will either confirm the 2-point estimate
  above (p ≈ 1.0, dp ≈ −0.10) or contradict it, and it is the single most useful check remaining.

### Correction to experiment 21, within the hour, before any of it was published

Two things in the entry above are wrong and are retracted here.

**1. The 2-point serial decomposition is unusable.** With only 1T and 4T, `p` is exactly determined
and therefore inherits the full measurement error — and near p → 1 the map from speedup to serial
time is violently non-linear:

    ResNet50 fp32, S(4) = 3.951 (measured)  -> p = 0.9959, serial 0.392 ms of 94.74
                   S(4) = 3.860 (-2.3%)     -> p = 0.9879, serial 1.145 ms
                   S(4) = 3.800 (-3.8%)     -> p = 0.9825, serial 1.662 ms

A 2.3% shift — well inside the 7% noise floor — moves the serial estimate by 4x. The two methods
disagree wildly on the same model: the 2-point estimate says ResNet50 static INT8 **adds 2.705 ms**
of serial work; the 4-point thread-sweep fit, which has a residual check (4.2%), says it **removes
0.149 ms**. The corpus-wide serial table computed from 2-point estimates is an artifact and must
not be used or quoted.

**2. "dp negative in 21/21 runs" overstates the effect, and the reason is arithmetic.** `p` is a
fraction. INT8 exists to accelerate the arithmetic — the parallel part — so even with serial time
completely unchanged the serial *fraction* rises and dp goes negative. ResNet50 is the clean
demonstration: serial 3.269 -> 3.120 ms, i.e. **no added serial work at all**, and yet dp = −0.069.
A negative dp is therefore the *expected* signature of a quantizer doing its job, not evidence of
damage.

### What actually survives, stated conservatively

Only the three 4-point thread-sweep fits can support a serial-time claim, and one of them is
degenerate. That leaves **n = 2 architectures** — but they are the two extremes of the partition and
the contrast is large and clean:

    DeiT      fp32 serial 2.774 ms -> static 4.116 ms   (+1.342 ms, +48%)
                                   -> dynamic 2.745 ms  (-0.029 ms, within noise)
    ResNet50  fp32 serial 3.269 ms -> static 3.120 ms   (-0.149 ms, none added)
    MobileViT p < 0 for every variant - no parallel region, unusable

> **Static QDQ adds absolute serial work on the architecture whose fp32 graph depends on fusion
> (+48% on DeiT) and none on the plain CNN (ResNet50). Dynamic quantization, which preserves the
> fusions, adds none even on DeiT.**

That is the sharpest true statement available tonight, and it rests on two architectures plus the
node profiles that explain them (28.5% of DeiT static's node time in decomposed elementwise chains
against 3.8% for ResNet50).

The raw observation from experiment 12 — INT8 scales worse 1T->4T in 66 of 71 pairs — stands
unchanged; it is a direct measurement with no model behind it. What does not stand is any
decomposition of it into serial and parallel parts outside the three runs that have a real curve.

**Queued as a consequence:** thread sweeps are cheap relative to what they settle, and every claim
of this kind needs four points rather than two. `detection`, `anomaly`, `classification` and
`stress_imagenette` should get one before the paper quotes any serial decomposition beyond DeiT and
ResNet50.

---

## Experiments 5+6+7 — the quantization-configuration factorial · 2026-09-16 05:07

Eight cells, calibration pinned (auto resolved to **minmax**, recorded in the CSV so it cannot
confound the axes), same timing settings as the headline measurements.

    tag                  format     act    rr   top-1 drop   SQNR    bytes      t1      t4
    qdq_u8i8_rr0         qdq        uint8  no     -0.13    27.3 dB  2,621,368  1.567x  1.090x   <- shipped default
    qdq_u8i8_rr1         qdq        uint8  yes    -0.10    22.9 dB  2,621,368  1.558x  1.100x
    qdq_i8i8_rr0         qdq        int8   no     -0.08    27.3 dB  2,621,813  0.849x  0.474x
    qdq_i8i8_rr1         qdq        int8   yes    -0.12    22.8 dB  2,621,813  0.861x  0.478x
    qoperator_u8i8_rr0   qoperator  uint8  no     -0.13    27.3 dB  2,436,268  1.584x  1.128x
    qoperator_u8i8_rr1   qoperator  uint8  yes    -0.10    22.9 dB  2,436,268  1.586x  1.120x
    qoperator_i8i8_rr0   qoperator  int8   no     -0.11    27.3 dB  2,436,713  0.098x  0.073x
    qoperator_i8i8_rr1   qoperator  int8   yes    -0.12    22.7 dB  2,436,713  0.095x  0.076x

### 1. The unsupported comment in the codebase is now substantiated

`core/quantize.py:62` has carried `# U8S8: fast path on x86 (S8S8 QDQ was slower than FP32 there)`
asserting a measurement nobody had recorded. Measured: **S8S8 in QDQ format is 0.849x at 1 thread
and 0.474x at 4** — slower than FP32, exactly as claimed. The comment stands and now has a number
behind it.

### 2. The factorial design earned itself

ORT's documented trap is a *combination*, and the data shows why a one-axis-at-a-time sweep would
have missed it:

    QDQ       + S8S8   0.85x     main effect of S8S8 alone: bad but survivable
    QOperator + U8S8   1.58x     main effect of QOperator alone: fine, slightly best
    QOperator + S8S8   0.098x    the interaction: a 10x collapse neither main effect predicts

ORT emitted its own warning during the run — *"Please use QuantFormat.QDQ for activation type QInt8
and weight type QInt8. Or it will lead to bad performance on x64."* Had this been swept one axis at
a time, both axes would have looked acceptable and the cell that destroys performance would never
have been visited.

### 3. The shipped default is not the best configuration, and the reason is size, not speed

`qoperator + U8S8 + rr0` is **2,436,268 bytes against the default's 2,621,368 — 7.1% smaller** at
identical accuracy (−0.13 pt both) and identical SQNR (27.3 dB). Its speed advantage (1.128x vs
1.090x at 4T) is **3.5%, inside the 7% noise floor and not claimable.** The size difference is
deterministic and is claimable. Worth taking, and worth disclosing that the paper's numbers were
measured on a configuration that is not size-optimal.

`reduce_range` costs 4.5 dB of SQNR (27.3 -> 22.9), changes accuracy by less than the noise, and
changes speed by less than the noise. On this machine it buys nothing — it is a pre-VNNI saturation
guard and this CPU does not need it.

### 4. Accuracy is flat across all eight cells (−0.08 to −0.13 pt)

Configuration moves speed by 16x and size by 7% while leaving accuracy alone. The two axes of
experiment 18's synthesis stay separate here too.

---

## Experiment 19 is refuted before it runs, by the line above · 2026-09-16 05:10

Experiment 18 proposed that dynamic quantization's −11.07 pt on MobileNetV2 is explained by
calibration: *"dynamic is static pinned to the worst calibration method"* — exact per-inference
min/max where static clips with percentile. `configs/experiments/calibration_isolation.yaml`
pre-registered: **static + minmax should land closer to −11.07 pt than to −0.14 pt.**

The factorial answers it without running: **`auto` resolved to minmax on this model.** So the
shipped `int8` variant *is* static+minmax, and it scores **−0.13 pt**. Static minmax calibration is
completely fine here; dynamic, which is minmax computed per inference, is catastrophic. **The
calibration explanation is dead.**

A second candidate died with it. Both graphs were inspected directly: the dynamic model carries 54
one-dimensional float initializers at the same channel widths as the static model's 106 (384, 192,
576, 960, 32, 96), so **dynamic quantization is using per-channel weight scales too**. It is not a
per-channel-weights problem either.

What remains, and what experiment 19 must be rewritten to discriminate:

1. **Calibration set width.** Static minmax takes the range over 16 batches — *wider* than any one
   sample. Dynamic takes the current tensor's exact range — narrower, which should be *better*.
   That this inverts is itself the puzzle, and it points away from range width entirely.
2. **The excluded set.** Static runs `auto_keep_fp32: rules`, keeping the output tail in FP32.
   Dynamic receives the same `nodes_to_exclude`, but only quantizes [Conv, MatMul], so the tail Gemm
   is untouched in both. Probably not it, and cheap to confirm.
3. **The ConvInteger requantization path itself** — int32 output, Cast, per-channel Mul, Add —
   versus QLinearConv's fused int32 requantization. Different accumulation and rounding.
4. **DynamicQuantizeLinear's forced zero-point behaviour**, which adjusts the range to include zero
   and is uint8-only.

The honest position: **the −11.07 pt is real and reproducible, its architecture dependence is real
(ResNet50 −0.46, YOLOv8n ≤ −0.75 at three resolutions), and its mechanism is now unknown.** The two
explanations I had were both wrong, and the second one died to a graph inspection that cost nothing.
Experiment 19 should become a discriminating run over candidates 2–4 rather than a calibration
sweep; the calibration sweep is kept only because percentile and entropy arms cost little and pin
down candidate 1 for the record.

### The same factorial on MobileViT: configuration sensitivity is itself architecture-dependent

`auto` resolved to **percentile 99.9** here rather than minmax — the auto-calibrator does adapt per
model, which is worth recording since the MobileNetV2 run resolved differently.

    tag                  act    rr   top-1 drop   SQNR     bytes      t1      t4
    qdq_u8i8_rr0         uint8  no     -4.63     6.06 dB  1,796,122  0.983x  0.700x   <- shipped default
    qdq_u8i8_rr1         uint8  yes    -5.25     6.21 dB  1,796,100  1.009x  0.655x
    qdq_i8i8_rr0         int8   no     -4.60     6.07 dB  1,797,250  0.437x  0.387x
    qdq_i8i8_rr1         int8   yes    -5.24     6.20 dB  1,797,241  0.444x  0.387x
    qoperator_u8i8_rr0   uint8  no     -4.69     6.07 dB  1,498,542  1.047x  0.744x
    qoperator_u8i8_rr1   uint8  yes    -5.05     6.15 dB  1,498,542  1.050x  0.697x
    qoperator_i8i8_rr0   int8   no     -2.25    12.75 dB  1,499,544  0.158x  0.273x
    qoperator_i8i8_rr1   int8   yes    -1.94    13.85 dB  1,499,544  0.168x  0.263x

**Accuracy spans 3.3 points across the eight cells (−1.94 to −5.25).** On MobileNetV2 the same eight
cells spanned 0.05 points. So quantization configuration is an accuracy-relevant variable on one
architecture and an irrelevant one on the other — which means "we tuned the quantizer" is not a
transferable statement, and a paper that reports a single configuration has reported a
model-specific result whether it says so or not.

**QOperator + S8S8 recovers more than half the accuracy loss** — −1.94 pt against the default's
−4.63, with SQNR going 6.1 -> 13.8 dB — and is **0.263x**, four times slower than FP32. Unusable in
practice, but diagnostically decisive: MobileViT's accuracy collapse is **not inherent to INT8**.
A different format recovers it, so the damage is a property of the *representation chosen*, not of
8-bit arithmetic.

SQNR tracks the recovery cleanly (6.06 dB -> −4.63 pt; 13.85 dB -> −1.94 pt), which is the same
direction experiment 11 measured cross-task at rho +0.688/+0.703.

`reduce_range` interacts again rather than acting as a main effect: it **hurts** the U8S8 cells
(−4.63 -> −5.25) and **helps** the QOperator S8S8 cells (−2.25 -> −1.94). Two models, two
interactions, no usable main effect — the factorial has now justified itself twice.

**The practical conclusion for the deliverable:** static QDQ cannot be configured into acceptable
accuracy at acceptable speed on MobileViT. The variant that achieves both already exists and is
`int8_dyn_mm` (−0.02 pt, 1.12x at 4T). The factorial's contribution here is to close off the
alternative rather than to open one — which is worth the run, because "we tried the configuration
space and it does not rescue this" is exactly the question a reviewer asks.

Size: QOperator is **16.6% smaller** than QDQ on MobileViT (1,498,542 vs 1,796,122 bytes) against
7.1% on MobileNetV2, at identical accuracy in the U8S8 cells. The format change is free compression
on both models and larger where the graph is more heterogeneous.

---

## Experiment 4 — RAM attribution · 2026-09-16 05:20 · **THE PREMISE WAS A SPURIOUS DENOMINATOR, AND THE ARENA WAS NOT THE CAUSE**

The experiment was designed to explain a claim: *"MobileViT p50 INT8 uses 2.07x the peak RSS of
fp32 while also being slower. That is unexplained, and a reviewer will ask."* Three candidate causes
were pre-registered — the ORT allocator arena, int8/fp32 boundary buffers, extra scale/zero-point
initialisers — with the arena as the controlled one.

### The arena is not the cause

Isolated subprocess per configuration, peak RSS with `enable_cpu_mem_arena` on and off:

    run                       variant              arena ON   arena OFF
    stress_resnet50           p00_int8               0.36        0.35
    classification            p00_int8               0.81        0.80
    classification_deit       p00_int8_dyn_mm        0.83        0.84
    classification_deit       p00_int8               0.91        0.92
    classification_mobilevit  p00_int8_dyn_mm        1.01        0.99
    classification_mobilevit  p00_int8_search        1.07        1.06
    classification_mobilevit  p00_int8               1.10        1.08

Turning the arena off moves every ratio by 0.02 or less. **The pre-registered control fired and
eliminated its own candidate** — whatever the memory behaviour is, it is not allocator over-allocation.

### But the premise does not reproduce either

The worst ratio measured is **1.10x**, not 2.07x. Tracing it:

    MobileViT p50    peak_rss_mb_t4        7.0 -> 14.6 MB    2.07x    <- the published claim
                     rss_steady_mb_t4      7.2 -> 15.0 MB    2.09x
                     peak_rss_total_mb_t4 61.5 -> 69.0 MB    1.12x    <- the same measurement, untruncated

`peak_rss_mb_*` is **baseline-subtracted** — the empty-session RSS (~54.5 MB) is removed to attribute
memory to the model. The 2.07x is a ratio of two small remainders. In absolute terms INT8 costs
**7.5 MB more on a 61.5 MB process.**

This is the same error that produced "21 of 23 winners prune zero", the SQNR +0.395, and the Q/DQ
density correlation: **a ratio taken over a denominator that was constructed rather than measured.**
It is now the fourth instance, from a different part of the codebase each time, which makes it the
single most reliable failure mode in this project.

Neither number is wrong; the *sentence* was. "Uses 2.07x the RAM" is heard as "needs twice the
memory", and what is true is "adds 7.5 MB to a 61.5 MB footprint". The delta ratio answers
*attribute the cost to the model*; the total answers *size the device*. For an edge deployment the
second is the question, and it is the one the project was not reporting.

### The finding that replaces it

Measured on total process RSS, which is what an edge device has to provide:

> **INT8 reduces peak memory on plain CNNs — dramatically on ResNet50 (0.36x, 257 -> 93 MB) and
> clearly on MobileNetV2 (0.81x) — reduces it on DeiT (0.91x static, 0.83x dynamic), and increases
> it only on MobileViT, by 8-10%.**

That is a better result than the anomaly it was meant to explain, it aligns with the architecture
partition everything else has landed on, and it removes a "reviewer will ask" item by dissolving the
question rather than answering it.

**Action:** the Excel report and any deliverable RAM figure must state which denominator it uses.
`peak_rss_mb` (baseline-subtracted) and `peak_rss_total_mb` are both in `sweep.csv`; only the second
supports a sentence of the form "this model needs N MB".

---

## Experiment 10b — the fair-schedule control · 2026-09-16 06:37 · **THE FINDING SURVIVES; MY HEADLINE NUMBER WAS WRONG BY MORE THAN THE FINDING IS WORTH**

Both arms given the same epoch budget and the same learning rates, both quantized, and the unpruned
arm given a do-nothing option so the reference is "the best quantize-only deliverable available"
rather than "the model we happened to start from":

    arm       ratio  epochs  lr_mult   fp32     int8    int8 bytes   SQNR
    unpruned   0.00     0      0.0    0.9235   0.9236   2,621,369   27.71   <- do nothing
    unpruned   0.00    20      1.0    0.9234   0.9227   2,621,370   27.95
    unpruned   0.00    20      5.0    0.9303   0.9302   2,621,368   31.55   <- best unpruned
    unpruned   0.00    20     20.0    0.9301   0.9289   2,621,368   31.08
    pruned     0.25    20      1.0    0.9018   0.9010   2,050,263   25.10
    pruned     0.25    20      5.0    0.9131   0.9145   2,050,264   27.96   <- best pruned
    pruned     0.25    20     20.0    0.9114   0.9105   2,050,261   29.78

### The control was needed, and it changed the answer

**The unpruned arm improves too**: 0.9236 doing nothing -> **0.9302** at lr_mult 5, a gain of
+0.66 pt. The baseline was not converged, so experiment 10's comparison against it was measuring
"pruned model that got 20 extra epochs" against "model that got none". Exactly the Wang et al.
(arXiv:2301.05219) failure mode, exactly as feared.

Against the properly tuned reference the pruned model is **−1.57 pt, not −0.93 pt** — still inside
the 2.0-point budget, so **the finding survives**. But it costs 70% more of the budget than claimed.

### And the compression number was wrong, for a worse reason

Experiment 10 reported *"at ratio 0.25 the pruned INT8 model is 1,603,509 bytes ... 5.56x versus the
3.40x the paper currently reports as the ceiling."* The fair run measured the same configuration at
**2,050,264 bytes**. Chasing the 1,603,509:

    results/classification/models/p25_int8.onnx                 1,603,509 B   5.56x   acc -5.19 pt
    results/classification_prune_global/models/p25_int8.onnx    2,050,265 B   4.35x   acc -2.34 pt

**They are different pruning strategies.** Experiment 10 measured accuracy on
`classification_prune_global` (global pruning, 20 recovery epochs) and I quoted compression from
`classification` (uniform L2). And the model I took the compression from sits **5.19 points below
baseline** — two and a half times outside the budget, wholly inadmissible.

So the claim paired the compression of an inadmissible model with the accuracy of a different,
admissible one. That is not a subtle slip; it is the single worst error recorded in this log, it is
mine rather than inherited, and it propagated into the memory file, `FINDINGS.md`, and the published
report before the control caught it.

It is also the **fifth** instance of the same family tonight — a figure taken from a denominator or
a source other than the one the sentence is about. Previous four: "21 of 23 winners prune zero"
(wrong denominator), SQNR +0.395 (hand-picked layers), Q/DQ density (file graph, not executed
graph), RAM 2.07x (baseline-subtracted remainders). Five different files, five different weeks of
work, one mechanism.

### The corrected claim, internally consistent, from a single run

> Under the 2.0-point rule, global pruning at ratio 0.25 with a tuned recovery schedule reaches
> **4.35x** compression against quantize-only's **3.40x** — a **1.28x** advantage — at a cost of
> **1.57 points** of the 2.0-point budget. The schedule is what makes it admissible at all: at the
> default `recovery_lr_mult` of 1.0 the same configuration scores 0.9010 and is outside the budget.

Every number in that sentence comes from `results/paper/fair_schedule.csv`, one run, both arms
measured the same way.

### What it costs and what it is still worth

The advantage is 1.28x, not the 1.63x implied by 5.56/3.40. Whether that is worth the extra
training, the extra hyperparameter and the extra 1.57 points of budget is a judgement the paper
should put to the reader rather than answer for them. What is *not* in doubt is the methodological
point, and it is the part worth publishing: **a pruning method declared unprofitable under a default
recovery schedule can become profitable under a tuned one, and almost nobody reports the schedule.**
That claim needs no compression number at all.

### Still outstanding

`recovery_lr_mult` was swept at 1, 5 and 20 on both arms; both peaked at 5, and experiment 10's
earlier sweep peaked at 20 measuring FP32 top-1 rather than INT8 top-1. The interaction between the
recovery schedule and quantization is therefore real and unexplored — the best FP32 model is not the
best INT8 model. One more sweep at finer spacing around 5 would settle it, and it is cheap.

---

## Experiment 15 — ORT version bisect · 2026-09-16 06:48 · **THE PRE-REGISTERED FALSIFICATION FIRED. MOBILEVIT IS A REGRESSION, NOT AN ARCHITECTURE PROPERTY.**

Four runtimes fed the **same .onnx files** (the quantizer held fixed), one fresh subprocess per
measurement, randomised order under a logged seed, 3 sessions per cell, core-delivery gated —
1 of 144 measurements excluded.

    ort       family      thr   static   dynamic    fp32 ms   n
    1.16.3    mobilevit     4    0.951     0.980      1.553   3
    1.20.1    mobilevit     4    1.005     0.954      1.460   3
    1.28.0    mobilevit     4    0.655     1.014      1.341   3
    1.30.0    mobilevit     4    0.648     1.004      1.295   3

    1.16.3    deit          4    0.884     1.217      4.614   3
    1.20.1    deit          4    0.926     1.165      4.328   3
    1.28.0    deit          4    0.945     1.260      4.801   3
    1.30.0    deit          4    0.977     1.294      4.314   3

### MobileViT's static penalty appears between 1.20.1 and 1.28.0

**1.005 on ORT 1.20.1. 0.655 on 1.28.0.** A 35% collapse introduced by a runtime upgrade, on
byte-identical model files. The falsification condition written before the run was: *"if MobileViT
static is at or above 1.0 on an older release and below 1.0 only on recent ones, the architecture
claim is refuted and must be rewritten as an ONNX Runtime regression report."* **It fired.**

The challengers predicted exactly this from a pilot run under 24–52% background load, which they
correctly refused to quote. Measured cleanly, they were right.

### DeiT's is real, version-independent, and small

0.884 → 0.926 → 0.945 → 0.977 at 4 threads: below 1.0 on **every** version, and *improving* with
each release. So DeiT keeps an architecture-level static penalty — but it is 2–12%, not the headline
number, and ORT has been closing it.

**Dynamic MatMul is stable across all four versions** (0.954–1.014 on MobileViT, 1.165–1.294 on
DeiT). The regression is specific to static QDQ.

### What this does to the rest of the work

- **Axis 1 (accuracy) is untouched.** The bisect varies only the runtime; the −4.62 pt on MobileViT
  and −11.07 pt on dynamic-Conv MobileNetV2 are properties of the quantized files themselves.
- **Axis 2 (speed) splits.** MobileViT — which supplied the most dramatic numbers in this project —
  is largely an **ORT 1.28 regression**. DeiT's penalty survives as an architecture effect but is
  small.
- Every latency claim in this project was measured on 1.30.0 and must now name the version. A result
  that changes by 35% across a minor-version upgrade is not a property of the model.

### This is a better paper, not a worse one

"ONNX Runtime 1.28 introduced a ~35% static-INT8 regression on conv-heavy mobile architectures,
absent on 1.20.1, not present in the dynamic path, on byte-identical models" is concrete,
actionable, verifiable by anyone in ten minutes, and useful to the ORT maintainers. It is worth more
than the architecture claim it replaces.

**Immediate next step:** bisect 1.21–1.27 to name the exact release, and check the ORT changelog for
QDQ transformer or kernel changes in that window. `scripts/paper/ort_version_bisect.py` takes any
set of side venvs, so this is a `uv venv` loop and one re-run.

---

## Experiment 15b — the regression pinned to two releases · 2026-09-16 08:39 · **AND DEIT NEVER HAD A STATIC PENALTY**

Eleven runtimes, same .onnx files throughout, 3 sessions per cell, randomised order, 396
measurements. Re-computed with the core-delivery gate on (0.95) and off: **the pattern is identical
either way**, so nothing below depends on the gate. Exclusions (75/396) cluster on 1-thread DeiT —
the memory-bound case where sub-1.0 delivery is honest — and are **not version-correlated** (8-28%
per version, no trend), so the version comparison is unbiased.

### MobileViT static: two separate regressions, one threading-specific

    @4 threads   1.16.3 0.941 | 1.20.1 1.006  1.21.1 0.979  1.22.1 0.985  1.23.2 0.976
                 | 1.24.4 0.742 | 1.25.1 0.631  1.26.0 0.636  1.27.0 0.649  1.28.0 0.639  1.30.0 0.649

    @1 thread    1.16.3 0.935 | 1.20.1 1.020  1.21.1 1.012  1.22.1 1.019  1.23.2 1.021  1.24.4 1.016
                 | 1.25.1 0.903  1.26.0 0.893  1.27.0 0.921  1.28.0 0.924  1.30.0 0.917

Two distinct steps, and they behave differently:

- **1.23.2 -> 1.24.4 costs 24% at 4 threads and NOTHING at 1 thread** (0.976 -> 0.742 versus
  1.021 -> 1.016). A threading or parallel-scheduling change.
- **1.24.4 -> 1.25.1 costs a further 15% at 4 threads and ~10% at 1 thread** (0.742 -> 0.631;
  1.016 -> 0.903). A general per-op change.

Combined: ~35% at 4 threads. Stable from 1.25.1 through 1.30.0 — five releases with no recovery.

### DeiT static has no penalty at all, and my earlier reading was wrong

    @4 threads   0.978  0.988  0.977  0.927  0.941  1.049  0.956  1.008  0.994  0.978  1.028
                 (1.16.3 ................................................................. 1.30.0)

Oscillating around 1.0 with **no trend**, a spread of 0.93-1.05 against a 7% noise floor. Four hours
ago, on four versions, I read 0.884 -> 0.977 as "below 1.0 on every version and improving" and
called it an architecture effect. **With eleven versions it is noise around parity.** Four points
suggested a trend that eleven points deny — the same lesson as the 2-point Amdahl estimate, on the
same night.

### Dynamic quantization never regressed

MobileViT 0.94-1.03, DeiT 1.20-1.40, flat across all eleven runtimes. **The regression is specific
to the static QDQ path.**

### What is left of the speed half of this project

Almost nothing architectural. The headline "ORT static QDQ INT8 is slow on transformers and
conv-heavy mobile nets" reduces to:

> **A single ONNX Runtime regression affecting MobileViT static QDQ, introduced in two steps at 1.24
> (4-thread only) and 1.25 (general), worth ~35% at four threads, absent from the dynamic path, and
> unrepaired through 1.30. DeiT never had a static penalty. The accuracy findings are untouched,
> because the bisect varies only the runtime.**

The two-axis synthesis survives but becomes lopsided: axis 1 (accuracy, architecture-dependent,
depthwise-related) is real and unexplained; axis 2 (speed) is now mostly a bug report.

### This also undercuts the fusion-blocking mechanism

Fusion blocking is a property of the QDQ graph and ORT's optimiser. The node profile measured it on
1.30.0 and found MobileViT static spending 27.5% of node time on decomposed elementwise chains. But
**MobileViT static was FINE on 1.23.2 and earlier**, and the .onnx file never changed. So either
fusion blocking is not what makes it slow, or ORT's optimiser stopped fusing at 1.24/1.25 —
and those are distinguishable.

**Queued, and it is now the decisive experiment:** dump the optimized graph under 1.23.2 (fast) and
1.25.1 (slow) for the same MobileViT INT8 file and diff the op census. If the fused ops present at
1.23.2 are absent at 1.25.1, the regression IS fusion loss and the mechanism holds in a sharper
form. If both graphs fuse identically, fusion blocking is refuted as the cause and the regression is
in the kernels, not the graph.

---

## Experiment 16 — fusion blocking vs boundary density, head to head · 2026-09-16 08:54 · **FUSION BLOCKING LOSES ITS OWN TEST**

The two candidate mechanisms make opposite predictions about restricting static quantization to
`[Conv, MatMul]`, which is why this run was designed: boundary density says leaving more of the
graph in fp32 creates MORE int8/fp32 crossings and is **slower**; fusion blocking says letting the
elementwise ops stay fp32 lets Gelu/LayerNorm fuse again and is **faster**.

    variant               acc      bytes   vs fp32 t1   vs fp32 t4   order
    p00_fp32            0.9270  4,014,666    1.000        1.000        2
    p00_int8            0.8808  1,796,122    0.921        0.674        1
    p00_int8_static_cm  0.8806  1,573,672    0.838        0.618        0
    p00_int8_static_ex  0.8786  1,692,395    0.809        0.633        4
    p00_int8_dyn_mm     0.9272  2,701,375    1.088        1.113        3

    int8_static_cm relative to plain int8:   t1 0.910x   t4 0.916x

**Both arms built to restore fusion came out slower.** `static_cm` (restricted to Conv+MatMul) loses
9% at one thread and 8% at four; `static_ex` (elementwise op types held in fp32) loses more still.
Boundary density predicted this direction. Fusion blocking predicted the opposite. **Fusion blocking
is refuted as the dominant effect.**

### But this does not restore the original boundary-density result

The distinction matters and must be kept: the *mechanism* ("more int8/fp32 crossings cost time")
survives its first real prediction, while the *earlier analysis* offered as evidence for it stays
retracted — those counts came from the `.onnx` file, and ORT deletes ~90% of them before inference
(experiment 14). A mechanism can be right while the argument once made for it was wrong. What this
run adds is a prediction made in advance and confirmed; it adds nothing to the discredited
correlation.

### Honest limits, and they are real

- The margins are **8-9% against a 7% noise floor**, and the sweep runs **n=1 per cell**, not the 3
  repeats the bisect used. This is suggestive, not decisive. A repeat run would settle it cheaply.
- `measure_order` is populated (0,1,2,3,4) — the randomisation fix is live and recorded, so at least
  the ordering confound is gone from this one.
- **Measured on 1.30.0, inside the regressed regime.** Given experiment 15b, the same comparison on
  1.23.2 could give a different answer, and that is now the interesting version of this experiment
  rather than a footnote.

### Two incidental results worth keeping

**Restricting op types made the file SMALLER, not larger** — 1,573,672 against 1,796,122 bytes (39%
vs 45% of fp32) — which is backwards from the naive expectation that quantizing fewer ops leaves
more fp32 weights. The explanation is that MobileViT's elementwise ops carry almost no weights, so
quantizing them adds scale and zero-point initialisers while saving nothing. **Quantizing an op with
no parameters costs file size.** Anyone optimising for size should restrict `op_types_to_quantize`
to ops that actually hold weights.

**Accuracy is unmoved** — −4.64, −4.62, −4.84 pt across all three static arms. So the op-type axis
changes speed and size and leaves the accuracy damage exactly where it was, consistent with the
two-axis separation and with the MobileViT accuracy failure being about activation range rather than
about which ops are quantized.

**And `int8_dyn_mm` remains the only variant that is good on every axis** — +0.02 pt, 1.11x faster
than fp32 at four threads — at the cost of keeping 67% of the bytes.

### The same test on DeiT gives the OPPOSITE answer, and the split is by thread count

    variant               acc      bytes   vs fp32 t1   vs fp32 t4
    p00_fp32            0.9156 21,580,849    1.000        1.000
    p00_int8            0.9128  6,024,476    1.653        0.994
    p00_int8_static_cm  0.9143  5,935,253    1.641        1.193
    p00_int8_static_ex  0.9123  6,007,838    1.476        1.028
    p00_int8_dyn_mm     0.9155  5,869,604    1.949        1.433

    static_cm relative to plain int8:   t1 0.992x (neither)   t4 1.201x (FASTER)

**Restricting static quantization to [Conv, MatMul] is 20% faster at four threads and exactly
neutral at one.** It takes DeiT static INT8 from parity with fp32 (0.994x) to a 19% win (1.193x) at
four threads — and it does so with *better* accuracy (−0.13 vs −0.28 pt) at the same file size.
Better on all three axes than the shipped configuration.

**So fusion blocking wins on DeiT and loses on MobileViT.** That is not a contradiction; it is the
thread-count structure giving the mechanism away. The ops static QDQ quantizes beyond Conv and
MatMul are the memory-bound elementwise chains — `QLinearAdd`, `QLinearMul`, the decomposed
`Erf/Div/Sqrt` — which the node profile measured at 28.5% of DeiT static's node time. Removing them
from quantization costs nothing at one thread, where memory bandwidth is not the limit, and recovers
20% at four, where it is.

That is exactly the prediction registered before this batch ran: *"fusion blocking replaces
compute-dense fused kernels with memory-bound elementwise ones, which should cost latency AND
scaling together, so restricting op_types_to_quantize should improve BOTH."* The 1T/4T split is the
signature, and the thread sweep now running (16b) measures the curve directly.

Why MobileViT goes the other way has two candidate explanations and they are separable:
1. MobileViT is conv-dominated, so it has far fewer elementwise ops to recover from — and the extra
   int8/fp32 crossings dominate instead.
2. **MobileViT on 1.30.0 is inside the regressed regime** (experiment 15b). Its result may be about
   the 1.24/1.25 regression rather than about op types at all. Re-running this config under 1.23.2
   would separate them, and that is now a more interesting experiment than either arm here.

### The actionable recommendation, stated carefully

> **For transformer models under ORT static QDQ, set `op_types_to_quantize: [Conv, MatMul]`.** On
> DeiT it is 1.193x against fp32 at four threads instead of 0.994x, with better accuracy and the
> same size. Do NOT generalise it to conv-heavy models: on MobileViT the same change is 8-9% slower.

Limits, same as the MobileViT arm: n=1 per cell against a 7% noise floor, so the 20% is comfortably
outside it but the 0.992x and 1.034x readings are not distinguishable from parity. Repeats would
tighten it and cost minutes.

---

## Experiment 16b — the DeiT op-type win does not survive repeats, and my scaling prediction failed · 2026-09-16 09:15

The thread sweep re-measured the same five models with **3 repeats per cell** instead of the
sweep's n=1, across 1/2/4/6 threads. It disagrees with experiment 16 on the headline.

> **Correction (2026-09-19):** the "n=1" above is wrong. `results/fusion_blocking_deit/summary.json`
> records measure `repeats: 3` and `runs: 100`, so the experiment-16 sweep is also a 3-process
> harness median and a valid second run, not a single sample. The two runs genuinely disagree on
> DeiT-tiny: sweep static_cm vs plain int8 0.992x (1T) / 1.201x (4T), thread sweep 1.103x / 1.044x;
> static_cm vs dynamic 1.188x / 1.201x (sweep) and 1.107x / 1.298x (thread sweep). The DeiT
> fusion-blocking gain is therefore run-dependent (10% at 1T in one run, 20% at 4T in the other);
> the MobileViT-XXS refutation (restricting is slower) is unaffected. The paper (Secs. IV-C, V-C)
> now reports both runs.

    model                    1T            2T            4T            6T
    p00_fp32           11.448±6.6%    7.904±2.5%    4.888±5.3%    4.623±5.2%
    p00_int8            7.331±1.8%    5.434±4.1%    5.064±4.4%    4.806±5.6%
    p00_int8_static_cm  6.647±1.4%    5.449±2.6%    4.851±3.0%    4.584±6.1%
    p00_int8_static_ex  7.476±2.2%    5.564±2.5%    5.270±1.4%    5.141±2.8%
    p00_int8_dyn_mm     6.006±2.8%    4.486±1.5%    3.736±4.6%    3.554±4.2%

    static_cm vs plain int8    experiment 16 (n=1):  t1 0.992x   t4 1.201x
                               this run    (n=3):    t1 1.103x   t4 1.044x

**The 20% four-thread win is actually 4.4%, inside the 7% noise floor.** The n=1 sweep was wrong by
16 percentage points on the number I led with. I flagged the n=1 limitation when reporting it; the
correction is larger than that flag implied, and the recommendation built on it has to be withdrawn.

### And the registered scaling prediction failed

Registered before this ran: *"static_cm should have a HIGHER parallel fraction than plain int8 if
the memory-bound-elementwise mechanism is right."*

    p00_int8            p = 0.430   serial 4.177 ms
    p00_int8_static_cm  p = 0.366   serial 4.215 ms   (-0.064 p, +0.037 ms serial)
    p00_int8_dyn_mm     p = 0.498   serial 3.016 ms
    p00_fp32            p = 0.718   serial 3.233 ms

`static_cm`'s parallel fraction is **lower**, not higher, and its serial time is unchanged. The
mechanism predicted both the wrong sign and the wrong location: **the gain is at ONE thread (10%),
where bandwidth is not the limit, and vanishes into the noise at four.** That is the opposite of what
a memory-bandwidth story requires.

**So the memory-bound-elementwise mechanism is refuted by its own test, twice.** What `static_cm`
actually does is remove single-thread work — fewer quantized ops means less per-op Q/DQ overhead —
with no parallelism benefit at all. A plain "fewer ops, less work" reading fits; the parallelism
story does not.

### The corrected recommendation

> `op_types_to_quantize: [Conv, MatMul]` buys about **10% at one thread on DeiT** and nothing
> measurable above one thread. It also costs nothing in accuracy (−0.13 vs −0.28 pt) and nothing in
> size. Worth taking for single-threaded deployment; **not** the 20% four-thread win reported an
> hour ago, and not a scaling fix.

### What survives all of this, on the better measurement

    dyn_mm vs plain static int8:  1.220x (1T)  1.211x (2T)  1.356x (4T)  1.352x (6T)

**Dynamic MatMul-only quantization beats static QDQ by 21-36% at every thread count on DeiT**, with
better accuracy (−0.01 vs −0.28 pt) and the same size. That is the one DeiT result that has now
survived n=1, n=3, eleven ORT versions and an independent challenger pass. It is also the only
finding in this section that never needed a mechanism to be interesting.

And note `p00_fp32` vs `p00_int8` at 4T/6T: **1.036x / 1.040x — fp32 is marginally FASTER than
static INT8**, consistent with the bisect's 0.93-1.05 oscillation around parity. DeiT static QDQ
buys size and costs a little speed; it does not buy speed at all.

### The lesson, which is the same one as the Amdahl estimate and the 4-version bisect

Three times tonight a small-n measurement suggested a large effect that a larger-n measurement
dissolved: 2-point Amdahl vs 4-point curves, 4-version bisect vs 11-version bisect, n=1 sweep vs
n=3 thread sweep. In every case the direction of the error was **toward a more exciting result**.
The sweep harness runs n=1 by default and that default is now a known hazard: any claim from a
single sweep cell should be treated as provisional until a repeated measurement confirms it.

---

## Experiment 19 — calibration isolation, measured · 2026-09-16 09:22 · **CLOSES THE DOOR, AND DECOUPLES SQNR FROM ACCURACY**

The prediction was already refuted before this ran (the factorial showed `auto` resolves to minmax
on this model, so the shipped variant *is* static+minmax at −0.13 pt). Running it anyway was the
right call, because it converts an inference into a measurement and it found something else.

    variant            acc      drop    SQNR      bytes
    p00_int8         0.9220   -0.14 pt  27.3 dB  2,621,368   <- auto, resolved to minmax
    p00_int8_minmax  0.9220   -0.14 pt  27.3 dB  2,621,368   <- byte-identical, as predicted
    p00_int8_pct999  0.9218   -0.16 pt  23.6 dB  2,621,377
    p00_int8_entropy 0.9149   -0.85 pt   2.4 dB  2,621,356

**The entire calibration axis spans 0.71 points. The dynamic arm on this same model is −11.07 pt —
thirteen times the whole spread.** Calibration cannot explain dynamic quantization's damage, and
that is now measured rather than inferred.

`p00_int8` and `p00_int8_minmax` came out **byte-identical** with identical SQNR, independently
confirming the factorial's finding that `auto` selects minmax here. Two different experiments, same
conclusion, no shared code path.

### The incidental result is the interesting one: SQNR falls off a cliff and accuracy barely moves

Entropy calibration scores **2.4 dB** against minmax's **27.3 dB** — a 25 dB collapse, deep into
what `sqnr_warn_db: 15.0` calls broken — and costs **0.71 points of accuracy**. Meanwhile MobileViT's
static INT8 sits at 6.1 dB and costs **4.6 points**.

So SQNR is not monotone against task accuracy across models, and at the low end it is nearly
uninformative *within* a model. Experiment 11 measured Spearman +0.688 / +0.703 between per-layer
SQNR and the task metric, which stands — but it was measured *across layers within a model*, and
this is a different question: **SQNR does not calibrate across calibration methods, and a model can
sit at 2.4 dB while losing under a point.** The `sqnr_warn_db` threshold of 15 dB would have flagged
this model as broken when it is fine.

That is worth a line in the paper's limitations and a second look at the warning threshold: it
currently fires on a configuration that costs 0.7 pt and stays silent on nothing, so as a screening
test it has poor specificity at the bottom of its range.

---

## Experiment 21b — detection's 4-point curve · 2026-09-16 09:32 · **I OVER-GENERALISED THE 2-POINT RETRACTION**

Four hours ago I wrote that *"the 2-point serial decomposition is unusable"* and retracted the whole
corpus-wide table. Detection now has a real 4-point curve, and the two methods agree almost exactly:

    320_p00_fp32   2-point p=0.890 serial 3.109 ms  |  4-point p=0.887 serial 3.228 ms
    320_p00_int8   2-point p=0.643 serial 4.823 ms  |  4-point p=0.656 serial 4.847 ms

**So the retraction was too broad.** The failure is not in the 2-point method, it is in the 2-point
method *near p = 1*. ResNet50 sits at p = 0.965-0.996, where the map from speedup to serial time is
hypersensitive — my own demonstration used S(4) = 3.951, i.e. p ≈ 0.996 — and a 2.3% measurement
shift moves the estimate 4x. Detection sits at p = 0.89, comfortably off the cliff, and the estimate
is fine there.

**Corrected rule:** a 2-point (1T/4T) Amdahl estimate is usable below roughly p = 0.9 and unusable
above it. The corpus-wide serial table is therefore valid for the low-p runs and invalid only for
the ResNet50 family and anything else near-perfectly parallel. That is a much smaller retraction
than the one I made, and it was made in the direction of over-caution rather than over-claiming —
which is the better direction, but still an error worth naming.

### And detection joins DeiT, not ResNet50

    192px   fp32 serial 1.803 -> int8 2.435 ms   +35%
    256px   fp32 serial 2.042 -> int8 3.868 ms   +89%
    320px   fp32 serial 3.228 -> int8 4.847 ms   +50%
    base    fp32 serial 2.543 -> int8 5.141 ms  +102%

Static INT8 adds **35-102% of serial time** on YOLOv8n at every resolution, against DeiT's +48% and
ResNet50's **−5%**. Detection is a CNN, so "transformers pay, CNNs do not" is the wrong predictor.

The better predictor is **fusible activations**. DeiT has Gelu and LayerNorm; YOLOv8n has SiLU, which
ORT fuses; ResNet50 has ReLU, which folds into the convolution and leaves nothing to lose. All three
observations fit that, and it now rests on three architectures rather than the two I had.

    architecture   activation        serial cost of static INT8
    DeiT           Gelu / LayerNorm        +48%
    YOLOv8n        SiLU                    +35% to +102%
    ResNet50       ReLU (folded)            -5%

This also puts the earlier fusion refutation in its place. Experiment 16b showed that *restricting
op types does not recover* the loss and does not improve scaling — so fusion loss is not something
the user can configure away — but the serial-cost pattern still tracks which activations are
fusible. Those are compatible: the cost is real and structural, and the knob does not reach it.

**Pre-registered for the remaining curves:** classification (MobileNetV2, ReLU6) and anomaly (1-D
conv, ReLU) should look like ResNet50 — little or no serial cost. Imagenette is ResNet-family and
should also be near zero. If any of them shows a DeiT-sized serial increase, the activation-fusion
predictor is wrong and this table is a coincidence over three points.

---

## RESURRECTION — I re-published a claim this project withdrew three days ago · 2026-09-16 10:0x · **KILLED BY INDEPENDENT CHALLENGERS**

Asked whether pruning had a breakthrough, I audited every `results/*/sweep.csv`, found
`classification_prune_a20` (ratio 0.20, default schedule) at **−1.48 pt / 6.15x / faster than
quantize-only**, verified the numbers came from one run with a consistent baseline, and reported it
as an opening. Three independent challengers then killed it four separate ways.

### 1. It is a pre-fix artifact of a bug this repo already fixed

`allocate_ratios()` encoded "leave this layer alone" as **absence from the plan dict**, and
`prune_and_finetune` then realised that dict *together with* the uniform ratio — so `prune_model`
applied 0.20 to every layer the knapsack solver had deliberately spared. Verified directly:

    alloc_param_frac (what the plan asked for)   0.1969
    params  2,236,682 -> 1,128,554               49.5% removed
    MACs   26,641,920 -> 17,549,120              34.1% removed

**The plan spent 19.7%; the exported model lost 34.1% of MACs and 49.5% of parameters.** The 6.15x
is the product of the bug, not of the allocator. A third fingerprint: an 11-layer plan left **50 of
53** Conv/Gemm weight tensors physically narrower.

### 2. It was already withdrawn, by name, in this repo's own history

Commit **5eb0574**, 2026-09-13, *"Fix 9 defects from an adversarial hunt, one of which invalidated
my own results"*, says in its body:

> *"budget 0.20 -> 6.15x for −1.48 pt" was not a 0.20 budget*

The fix is the `eff_ratio` line at `core/prune.py:692`, whose comment names this run. Every
allocator run on disk — `a10`, `a15`, `a20`, `alloc` — predates it, provable from the CSV schema
independently of timestamps: post-fix code writes both `alloc_frac` and `alloc_param_frac`; these
files have only the latter. **None of the four is reproducible from its config on current code.**

### 3. Even taking the numbers at face value, it is outside the budget

−1.48 pt is measured against a **0-epoch** baseline (`train_epochs_run = 0` on the p00 rows, 20 on
p20). Experiment 10b established the tuned quantize-only reference at **0.9302**. Against that,
a20's deliverable is **0.9089 − 0.9302 = −2.13 pt — over the 2.0-point budget.**
`fair_schedule_comparison.py` prints exactly this quantity under the label *"the flattering number"*.

### 4. And "better on all three axes" was false on two of them

Put both arms on any single reference and tuned global-0.25 is **0.56 pt more accurate** (2x the
0.28 pt SD), not less. On speed they are a tie: a20 cuts 34.1% of MACs and global-p25 cuts 34.7%,
giving 0.41565 vs 0.41925 ms — **0.9% apart against a 7% floor.**

### The proximate cause, and the fix for it

**The 2026-09-13 retraction was never written into this log.** So an audit of the CSVs found a
number with nothing next to it saying "withdrawn", and I re-published it. Every other retraction
tonight is recorded here; that one lived only in a commit message, and a commit message is not
where anyone looks.

This is a sixth instance of the project's signature failure — a figure read from a source that did
not carry its own context — and the first where the missing context was *our own prior retraction*.

**Rule added: before quoting any number from `results/`, check whether the run predates a fix that
invalidates it.** Concretely: `git log --oneline -- core/prune.py core/quantize.py` against the
run's mtime, and check the CSV schema for columns a post-fix run would have written.

### What stands, unchanged

> Under the fixed 2.0-point rule, global magnitude pruning at ratio 0.25 with `recovery_lr_mult` 5
> reaches **4.35x** compression (2,050,264 B INT8 vs 8,919,751 B FP32) against quantize-only's
> **3.40x** — a **1.28x** advantage — at a cost of **1.57 points** against the best tuned
> quantize-only deliverable (0.9302). At the default `recovery_lr_mult` of 1.0 the same
> configuration scores 0.9010 and is outside the budget.

Every number from `results/paper/fair_schedule.csv`, one run, both arms measured the same way.
**And: there is currently no valid measured result for any DP-allocation budget at any ratio.**
The 6.15x / −1.48 pt sentence must not be written again in any form.


---

## Experiments 23-25 (2026-09-18): decision rules registered BEFORE any measurement

Designed by the independent review workflow's experiment gatekeeper (run wf_32870e1f-704,
`judge:experiments`, items K2-K4), not by the author of the claims, and copied here unchanged
before anything was timed. Its K1 (scope test of the 1.24 step on CNNs, MobileViT at 256 px and
DeiT at 224 px) is **not run, by the user's decision**; Branch B therefore stays scoped in the text
to MobileViT-XXS at 64 px, and the "fixed per-inference threading cost on a sub-2 ms model"
objection stays open and is stated as such.

Two corrections verified before registering (medians of 3 sessions, `ort_version_bisect.csv`,
MobileViT, 4 threads, ms):

| release | fp32 | static int8 | dynamic int8 |
|---|---|---|---|
| 1.23.2 | 1.517 | 1.553 | 1.506 |
| 1.24.4 | 1.512 | **2.037** | 1.507 |
| 1.25.1 | **1.281** | 2.028 | **1.269** |
| 1.30.0 | 1.321 | 2.034 | 1.280 |

* The static file regressed **once**, +31% at 1.24.4, at 4 threads only (1T: 1.472 -> 1.493).
  The 1.25 "second step" in the speedup ratio is FP32 and dynamic getting ~15% faster while static
  did not. "Static INT8 lost 35%" was wrong; the ratio fell 35%. Wrong-denominator failure, #7.
* CPUID on this machine: AVX-VNNI **present** (leaf 7.1 EAX[4]); AVX-512F and AVX512-VNNI absent.
  Every "no VNNI" sentence in the draft was false.

### Experiment 23 (K2): localise the 1.24 step

(a) Optimized-graph dump, no timing: `optgraph_version_diff.py`, releases 1.23.2 / 1.24.4 / 1.25.1,
MobileViT p00_int8, p00_fp32, p00_int8_dyn_mm; compare the op-type census AND the sorted list of
(domain, op_type) per node, pairwise.
* p00_int8 graphs differ 1.23.2 -> 1.24.4 => name the lost/changed fusion as the 1.24 mechanism.
* identical => write "identical optimized graph; the change is in kernels or threading".
* p00_fp32 graph changes 1.24.4 -> 1.25.1 => attribute the 1.25 ratio drop to an FP32-side
  transform the static path does not receive.

(b) Spin factor: `session.intra_op.allow_spinning` in {1, 0}, randomised with everything else;
MobileViT p00_fp32 / p00_int8 / p00_int8_dyn_mm, 4 threads, releases 1.23.2 / 1.24.4 / 1.25.1,
3 fresh sessions per cell (54 measurements). dW = spin-off p50 change of p00_int8 1.23.2 -> 1.24.4;
dC = its cpu_ms_per_inf change.
* dW < 7% => the step is the thread pool's spin policy; allow_spinning=0 is a mitigation
  (confirm the spin-on step reappears in the same run).
* dW >= 7%: dC/dW < 0.5 => waiting (sleep/wake, synchronisation); 0.5-2 => an extra serial
  region; > 2.5 => extra parallel work ("more executed work", now with valid evidence).

(c) Only if (a) finds p00_int8 identical at 1.23.2 and 1.24.4: per-op profile at 4 threads,
1.23.2 vs 1.24.4, 3 sessions each; node-time sum must be ~ wall time.
* >= 70% of the delta in one op type => "a 4-thread-only regression in <op> on an identical
  optimized graph, introduced in 1.24". Diffuse => "a thread-pool-level regression".

### Experiment 24 (K3): coverage-matched static arm on the hybrid

MobileViT arms: fp32, static (as in the paper), static restricted to MatMul, dynamic MatMul-only;
same calibration as the paper's p00_int8 (percentile 99.9). Accuracy from the sweep; latency under
the Section III protocol, releases 1.23.2 / 1.24.4 / 1.30.0, 1 and 4 threads, 3 sessions.
* static_mm within 0.28 pt of FP32 AND within 7% of dyn_mm at both thread counts on 1.23.2 and
  1.30.0 => the hybrid half of Branch A is a coverage result ("quantize only the MatMuls; dynamic
  MatMul-only is the simplest way to get that"); the method claim then rests on DeiT.
* static_mm > 7% slower than dyn_mm, or > 0.28 pt worse => the method matters at matched coverage
  on the hybrid too (strong form of Branch A).
* Side reading: static_mm shows no 1.24.4 4T step while full static does => the regression lives
  in the static conv/elementwise path, not in QDQ MatMul.

### Experiment 25 (K4): the dynamic-Conv cliff at deployment resolutions

MobileNetV2 at 224 px (Imagenette) and YOLOv8n at 640 px: fp32, static, dynamic-Conv; 1.30.0,
1 and 4 threads, 3 sessions, Section III protocol. Penalty = dynamic-Conv latency / FP32 latency.
* 1T penalty stays within 12-17x at both => "does not shrink with input size within one model,
  across 49x (MobileNetV2 32 -> 224 px) and 11x (YOLOv8n 192 -> 640 px) activation volume";
  replaces "two orders of magnitude".
* clearly below ~10x at the larger inputs => "shrinks slowly with activation volume"; show the
  within-model curve; the order-of-magnitude cliff remains.
* Accuracy: MobileNetV2-224 dynamic-Conv within ~1 pt of FP32 => the -11.07 pt is a 32-px/CIFAR
  result; if large => the depthwise-net damage generalises.

Noise floors used by every rule: latency 7% between sessions, accuracy 0.28 pt.

### Experiment 23a RESULT (2026-09-18): the static graph is identical; 1.25 is an FP32-side layout transform

`optgraph_version_diff.py --versions 1.23.2,1.24.4,1.25.1 --keep`, then a node-by-node comparison
of the kept dumps (ordered list of (domain, op_type, #in, #out, attributes), initializer dtypes and
shapes, opset imports):

| model | 1.23.2 -> 1.24.4 | 1.24.4 -> 1.25.1 |
|---|---|---|
| p00_int8 (static) | 698 -> 698, **ordered-identical** | 698 -> 698, **ordered-identical** |
| p00_fp32 | 574 -> 574, ordered-identical | 574 -> **520**: ReorderInput -25, ReorderOutput -28, Add -1 |
| p00_int8_dyn_mm | 535 -> 535, ordered-identical | 535 -> **481**: same 54 nodes |

* **1.24 rule fires: "identical optimized graph; the change is in kernels or threading".** The
  static file's executed graph is the same node for node at 1.23.2 and 1.24.4, yet it runs 31%
  slower at 4 threads. Fusion loss is excluded as the cause of the 1.24 step. Step (c), the
  per-op profile, is therefore triggered.
* **1.25 rule fires: an FP32-side transform the static path does not receive.** From 1.25.1, the
  NCHWc layout transformer keeps QuickGelu inside the blocked layout instead of leaving it
  (ReorderOutput -> QuickGelu -> ReorderInput becomes QuickGelu between nchwc Convs). That removes
  53 layout conversions from every graph that still has FP32 convolutions - FP32 and dynamic
  (which leaves convolutions in FP32) - and they get ~15% faster. Static QDQ runs its
  convolutions as integer kernels outside the NCHWc path, so its graph and its latency do not
  change. The 1.25 drop in the INT8/FP32 ratio is the baseline gaining an optimisation INT8 cannot
  use.
* The script's built-in verdict line ("the optimized graphs DIFFER ... fusion blocking survives")
  was written for the old two-step hypothesis and does not apply: the only graphs that differ are
  the ones that got faster.

### Experiment 23b RESULT (2026-09-18 15:01-15:05): not the spin policy; extra parallel work

`ort_version_bisect.py --spin 1,0 --threads 4 --versions 1.23.2,1.24.4,1.25.1`, MobileViT
p00_fp32 / p00_int8 / p00_int8_dyn_mm, 3 fresh sessions per cell, 54/54 cells completed, order
randomised (seed 23), the ICCAD WSL job stopped before the run. Written to
`results/paper/spin_version.csv`. Medians (min-max) in ms:

| spin | variant | 1.23.2 | 1.24.4 | 1.25.1 |
|---|---|---|---|---|
| on (default) | static | 1.574 (1.564-1.577) | **2.043** (2.017-2.062) | 2.026 |
| on | fp32 | 1.534 | 1.536 | 1.292 |
| on | dynamic | 1.535 | 1.504 | 1.276 |
| off | static | 2.366 (2.363-2.371) | **2.928** (2.895-2.982) | 2.917 (2.05-5.17) |
| off | fp32 | 2.711 (2.33-3.04) | 2.373 | 1.940 |
| off | dynamic | 2.210 | 2.006 | 2.260 |

* Spin on reproduces the original bisect: static +29.7% at 1.24.4, siblings flat; at 1.25.1
  fp32/dynamic -16%, static flat.
* **Rule: dW (spin off) = +23.7% >= 7% => not the spin policy.** Ranges do not overlap
  (max 2.371 before, min 2.895 after). allow_spinning=0 is not a mitigation.
* **dC/dW (spin off) = (5.651 - 3.892) / (2.928 - 2.366) = 3.13 > 2.5 => extra parallel work.**
  With spinning off, CPU time counts work rather than waiting, so "the regressed release does more
  executed work" is restored on valid evidence (spin on gives 3.36, which by itself proved nothing).
* Spin off costs ~0.4-1.2 ms on every variant of this 1.5 ms model (sleep/wake), and its spread
  is wider; core delivery 0.41-0.56 is expected there and is not a contamination signal.
* Step (c) is triggered by 23a (identical graph).

### Experiment 23c RESULT (2026-09-18 15:06-15:08): registered verdict "thread-pool-level"; Transpose is the lead

`node_profile_version.py --versions 1.23.2,1.24.4 --seed 24`: static MobileViT p00_int8, 4 threads,
ORT_SEQUENTIAL, default spinning, profiling on, 3 fresh sessions per release, 100 warmup + 300
profiled iterations, randomised order. `results/paper/node_profile_version.csv`.

* Profiling inflates this model ~9x (wall 13.1 / 15.3 ms profiled vs 1.57 / 2.04 unprofiled), so
  absolute per-op numbers describe the profiled regime, not deployment. Node-time coverage of wall
  is 0.72 / 0.76, but the DELTA is entirely inside node time: wall +2.14 ms, node sum +2.16 ms.
* Delta by op type: **Transpose 0.730 -> 2.074 ms (+184%), 62% of the delta.** Every other op type
  rises 5-14% (DequantizeLinear 5%, QLinearMatMul 5%, QLinearAdd 4% of the delta; Shape, Squeeze,
  Unsqueeze - ops with no parallel work - rise by similar percentages).
* **Registered rule: largest share 62% < 70% => "a thread-pool-level regression". That is the
  verdict and the only one the paper may state as a finding.**
* Exploratory, not a finding: the pattern has two components - one op type nearly tripling and a
  uniform per-node rise. Transpose in the static graph moves 8-bit tensors (DQ-Transpose-Q is
  collapsed to an integer Transpose); in the FP32 and dynamic graphs it moves float tensors, and
  those graphs did not regress. An 8-bit-Transpose threading change in 1.24 would fit the 4-thread-
  only step, the static-only step and dC/dW = 3.1 together. Untested; it would take a Transpose-only
  micro-model timed under both releases.
* A contamination check before this run printed a false "WSL job back": `pgrep -f` inside
  `sh -c "..."` matched its own shell. WSL had actually idled out (init elapsed 00:15 right after,
  load average 0.07/0.02/0.00), so 23b and 23c ran with WSL empty. Use `pgrep -f '[r]eport.py'`.

**Correction to 23c (2026-09-19, raised by the round-3 referee; recomputed from stored CSVs, no
timing).** The "~9x" profiling inflation above used 1.57 ms as the unprofiled 1.23.2 value. Against
the sweep (`ort_version_bisect.csv`, mobilevit p00_int8, 4 threads, median of three sessions:
1.5533 ms on 1.23.2, 2.0368 ms on 1.24.4) the profiled wall medians of `node_profile_version.csv`
(13.112 ms and 15.254 ms) are **8.4x (1.23.2) and 7.5x (1.24.4)**, replacing "~9x". This is a
cross-run ratio (profiled run over the sweep). The paper's "about eightfold" stands.

### Experiment 24: coverage check BEFORE timing (2026-09-18) - the first static arm was not matched

The registered design required checking that the static and dynamic MatMul arms quantize the same
nodes. Read from the stored files (the right graph for "what did the quantizer target"):

| arm | weight MatMuls (36) | activation x activation MatMuls (18: attn/MatMul, attn/MatMul_1) | head Gemm (1) |
|---|---|---|---|
| int8_static_mm (`op_types [MatMul]`) | quantized | **quantized** | FP32 |
| int8_dyn_mm (`op_types [MatMul, Gemm]`) | quantized | FP32 | quantized (as MatMulInteger) |

So int8_static_mm is NOT coverage-matched: it also quantizes the 18 attention-score MatMuls. Its
accuracy (0.9067, -2.03 pt vs FP32 0.9270; dyn_mm 0.9272) cannot be used for the registered rule.
Added the matched arm the design called for: `int8_static_mmw` = `op_types [MatMul, Gemm]` with
`keep_fp32: ["re:attn/MatMul(_1)?$"]` (regex verified: hits all 18 activation x activation
MatMuls and none of the 36 weight MatMuls). **The registered rule is applied to static_mmw vs
dyn_mm.** static_mm stays in the run as a separately reported arm (static quantization of the
attention-score MatMuls). Full static p00_int8 reproduced the paper's accuracy exactly (0.8808).
* Matched arm verified before timing: int8_static_mmw quantizes 36/36 weight MatMuls, 0/18
  activation x activation MatMuls, and the head Gemm - the same coverage as int8_dyn_mm.
  Accuracy (CIFAR-10 test, from the sweep): FP32 0.9270, static full 0.8808, static_mm 0.9067,
  **static_mmw 0.9263 (-0.07 pt, inside the 0.28 pt floor)**, dyn_mm 0.9272. The accuracy half of
  the registered rule therefore does NOT fire at matched coverage; the -2.03 pt of static_mm is the
  cost of statically quantizing the 18 attention-score MatMuls. Speed half: pending timing.
* Experiment 25 accuracy (Imagenette val, from the sweep): MobileNetV2-224 FP32 0.9824, static
  0.9806 (-0.18 pt), **dynamic-Conv 0.9789 (-0.35 pt)**. Registered rule fires: "within ~1 pt =>
  the -11.07 pt is a 32-px/CIFAR result", not a property of dynamic-Conv on depthwise nets in general.

### Experiments 24 and 25 RESULTS (2026-09-18, timing 18:51-19:09)

A first timing run (18:32-18:48) was discarded unread: a competing load on the machine put 90/90
exp-24 cells below the core-delivery gate (0.73-0.79) and failed 14 of the YOLOv8n-640 cells. The
gate caught it. The re-run added a precondition (machine-wide CPU load <= 20% over 5 s after a 20 s
settle; measured 8%). Invocations: `ort_version_bisect.py --repeats 3 --threads 1,4`,
exp 24 `--versions 1.23.2,1.24.4,1.30.0 --seed 24 --families cov=results/coverage_mobilevit/models:3,64,64`,
exp 25 `--versions 1.30.0 --seed 25`, one invocation per resolution family.

**Experiment 24 (K3)** - `results/paper/coverage_mobilevit.csv`, 90 cells, 2 below the gate
(0.93/0.94; gated and ungated medians agree to 3 decimals except one cell). Median ms:

| arm | 1.23.2 1T | 1.23.2 4T | 1.24.4 1T | 1.24.4 4T | 1.30.0 1T | 1.30.0 4T |
|---|---|---|---|---|---|---|
| fp32 | 1.602 | 1.681 | 1.543 | 1.739 | 1.403 | 1.477 |
| static, full | 1.499 | 1.610 | 1.596 | **2.180** | 1.563 | **2.315** |
| static_mm (+18 attention MatMuls) | 1.700 | 1.829 | 1.704 | 1.884 | 1.551 | 1.521 |
| **static_mmw (matched)** | 1.584 | 1.705 | 1.606 | 1.684 | 1.422 | 1.543 |
| **dyn_mm** | 1.380 | 1.619 | 1.433 | 1.550 | 1.263 | 1.343 |

* static_mmw / dyn_mm: 1.23.2 +14.8% (1T), +5.3% (4T); 1.24.4 +12.0%, +8.7%; 1.30.0 +12.6%, +15.0%.
* **Rule: the "coverage result" branch does NOT fire** (it needs <= 7% at both thread counts on
  1.23.2 and 1.30.0; 1T is 14.8% / 12.6%). **The "method matters at matched coverage" branch fires
  on speed** (> 7% slower in 5 of 6 cells; the exception, 1.23.2 at 4T, +5.3%, is inside the
  floor). On accuracy it does not (static_mmw -0.07 pt). Stated exactly: at matched coverage,
  dynamic INT8 is 9-15% faster than static QDQ on the hybrid, at equal accuracy, and static QDQ
  restricted to the same MatMuls buys no speed over FP32 at all (0.96-1.03x) while dynamic buys
  1.04-1.16x.
* **Side reading fires: the 1.24 step lives in the static conv/elementwise path, not in QDQ
  MatMul.** 4T, 1.23.2 -> 1.24.4: full static +35% (1.610 -> 2.180), static_mmw -1%,
  static_mm +3%. Consistent with 23c's exploratory Transpose lead (static_mmw keeps its
  Transposes in float), still untested as a mechanism.
* The step also reproduces on a freshly built full-static artifact (+35% at 4T, 1T +6%), the
  third independent observation after the 11-release sweep and experiment 23b.

**Experiment 25 (K4)** - `results/paper/dynconv_scale.csv`, 90 cells, 27 below the gate (23
single-thread and 4 four-thread cells) at 0.879-0.95 (corrected 2026-09-19; first written as
"0.90-0.95 (mild, ~5% of one core; per-session spread within a cell <= 2%)", which this CSV does not
support: the spread (max-min)/median between the three sessions of a cell has median 6.4% and
maximum 18.5%, 320_p00_fp32 at 4T).
The gated table loses every MobileNetV2-224 dynamic cell (1T and 4T) and three one-thread YOLOv8n
cells (192_p00_int8, 320_p00_fp32, 640_p00_fp32), so the ungated medians are reported;
no conclusion below depends on the difference. Penalty = dynamic-Conv / FP32 latency:

| model @ input | FP32 1T ms | dyn-Conv 1T ms | penalty 1T | penalty 4T | vs static 1T | static speedup 1T |
|---|---|---|---|---|---|---|
| MobileNetV2 @ 32 | 0.922 | 14.83 | 16.1x | 18.1x | 26.2x | 1.63x |
| MobileNetV2 @ 224 | 8.938 | 162.8 | **18.2x** | 29.1x | 29.7x | 1.63x |
| YOLOv8n @ 192 | 11.43 | 147.6 | 12.9x | 11.6x | 28.1x | 2.17x |
| YOLOv8n @ 320 | 28.76 | 453.0 | 15.8x | 13.0x | 33.2x | 2.11x |
| YOLOv8n @ 640 | 122.2 | 1602 | **13.1x** | 12.4x | 29.6x | 2.26x |

* **Rule: the "shrinks" branch does NOT fire** (nothing below ~10x). YOLOv8n-640 stays inside
  12-17x; MobileNetV2-224 is 18.2x, just above it. That is NOT evidence the penalty grew: 16.1 -> 18.2
  is +13%, a ratio of two medians carries roughly 10% of noise (7% per latency), the 224 px cells sat
  just under the gate, and YOLOv8n moves 12.9 -> 15.8 -> 13.1 across its three sizes with no trend.
  The supported statement is only: no downward trend. [Corrected the same day; the first version of
  this entry said "the penalty grew", which the table does not show.] Written as:
  "within one model the one-thread penalty does not shrink with input size: MobileNetV2
  16.1x -> 18.2x across 12.2x activation volume (32 -> 224 px; 49x input pixels), YOLOv8n 12.9x -> 13.1x across 11x
  (192 -> 640 px); 12.9-18.2x over all five settings". Replaces "two orders of magnitude".
* Against the correct choice (static), the wrong choice costs 26-33x at one thread, 15-41x at four.
* Static INT8 at deployment resolution: 1.63x (MobileNetV2-224, -0.18 pt) and 2.26x
  (YOLOv8n-640) at one thread.

### Correction (2026-09-19, found by the round-1 referee of the paper review; verified before editing)

Two statements in the Experiment 25 entry above were wrong and are corrected in place:
* "49x activation volume" for MobileNetV2 32 -> 224 px. 49x is the INPUT-PIXEL ratio. The CIFAR
  MobileNetV2 keeps a stride-1 stem (stem output 32x32x32 at 32 px vs 32x112x112 at 224 px), so the
  summed Conv-output volume is 545,152 vs 6,678,112 elements = 12.2x (onnx shape inference on the
  two FP32 files). YOLOv8n 192 -> 640 px is the same architecture at both sizes, so its 11x holds.
  The pre-registered rule text quoted "49x" from the design; the ruling is unaffected, since the
  penalty did not shrink across either span.
* "27 below the gate, all single-thread": 23 single-thread and 4 four-thread cells
  (`dynconv_scale.csv`, core_delivery < 0.95).

### Note (2026-09-19): which Experiment 24 medians the paper quotes

The Experiment 24 table above gives gated medians. The paper reports the ungated medians of
`coverage_mobilevit.csv` and says so. They differ in one cell: dyn_mm on 1.23.2 at one thread is
1.380 ms gated and 1.405 ms ungated. The paper's ranges therefore read "dynamic 1.04-1.14x" and
"matched static slower by 8.7-15.0%" (quoted as 9-15% in five of six cells).


---

## Experiment 26 (K1, 2026-09-19): scope of the 1.24 step - decision rules registered BEFORE measuring

Designed by the independent experiment gatekeeper on 2026-09-18 (run wf_32870e1f-704, item K1,
ranked first). Skipped at first by the author's decision; run now because an external review
(2026-09-19) raised exactly this objection: the 1.24 step rests on one 64-px model. The rules are
copied from the design unchanged. Artifacts: fresh MobileViT-XXS at 64 px (bridge) and 256 px from
the paper's weights; DeiT-tiny at 224 px/patch 16 with RANDOM weights (no cached ImageNet weights
exist and nothing is downloaded; latency only, accuracy not interpreted); the existing CNN files
(ResNet-50, MobileNetV2-32, YOLOv8n-256, 1-D AE). Releases 1.23.2 / 1.24.4 / 1.25.1 / 1.30.0,
1 and 4 threads, 3 fresh sessions, randomised order -> `results/paper/ort_version_scope.csv`.

* A "step": static 4T median absolute latency at 1.24.4 more than 7% above 1.23.2, still more than
  7% above at 1.25.1 and 1.30.0, with the FP32 and dynamic siblings moving less than 7% at 1.24.4.
* Bridge: if the fresh mv64 is not byte-identical to the paper's file and shows no step of at least
  15%, the mv256 series is not interpreted; then report that the step depends on the exact QDQ graph.
* (A) Any CNN static artifact (r50/mnv2/y256/ae) steps => Branch B is a static-QDQ-path regression
  across architectures; the title is earned; replace 'two models' with 'k of n static artifacts'.
* (B) No CNN steps but mv256 steps by >= 15% => a hybrid-op-mix regression that persists at
  deployment resolution with real parallel work; keep the title, name the hybrid class in scope.
* (C) mv256 step < 7% and no CNN steps => a fixed ~0.5 ms multi-thread overhead on sub-2 ms hybrid
  models; report absolute ms, demote Branch B from the title to a scoped finding, let the main axis
  and Branch A carry the title.
* Gap in the design, recorded rather than filled: mv256 step between 7% and 15% with no CNN step
  fires neither B nor C; the measured value is then reported as is, Branch B stays scoped to the
  hybrid at 64 px, and the title decision goes to the author.
* Independently: a CNN FP32 sibling >7% faster at 1.25.1 while its static sibling is not => state
  generally that reported INT8 speedups shrink across releases through the FP32 baseline.
* Branch A: 1T dynamic-Conv penalty within 12-17x on all four releases and static beating FP32 on
  ResNet-50 on all four => 'holds on every release tested'; otherwise report per release.
* DeiT-224: dynamic beats static by >7% at both thread counts on all four releases => the
  attention half holds at 197 tokens; static at least as fast as dynamic => restrict the attention
  half to short sequences.


---

## Experiment 27: Transpose isolation probe. Design and decision rule, registered 2026-09-19 BEFORE any timing

Responds to external review point (2): "the mechanism is localised but not explained". Designed by an
independent experiment-design agent, not by the author of the claim. Nothing was timed and onnxruntime
was not imported. Builder: `scripts/paper/transpose_probe.py` (onnx.helper only). Models:
`results/transpose_probe/models/*.onnx`. Pre-registered predictions: `results/transpose_probe/manifest.json`.

### 0. Hypothesis at risk

H_T: the 1.24.4 step of the static MobileViT-XXS file lives in ONNX Runtime's 8-bit Transpose under
multi-threading. The step is +31% at 4 threads, none at 1 thread, on an identical optimized graph, with
extra parallel work (23b). Evidence so far: 23c put 62% of the delta on Transpose under profiling, below
the registered 70% threshold. Experiment 24 found no step in the MatMul-only static arms, whose Transposes
stay in float. The paper may currently say only "thread-pool-level".

The desk research (section 2) makes this sharper and riskier:

H_M (code-level mechanism): commit 42869ca (PR #25680, first shipped in 1.24) added
`has_mlas_transpose<uint16_t>` to `core/framework/transpose_helper.cc`. From 1.24, a single-axis
Transpose whose moved block is 2 bytes wide no longer runs a scalar loop. It calls
`MlasTranspose<uint16_t>` once per outer loop, and every call goes through `MlasExecuteThreaded` on the
intra-op pool when M > 1. In the static graph, exactly four uint8 Transposes have 2-byte blocks. At 4
threads they issue **832 thread-pool dispatches per inference on 1.24.4 and 0 on 1.23.2**: the graph
goes from 32 to 864 dispatches in total. Their float twins in the FP32 graph have 8-byte blocks and stay
scalar on both releases (24 -> 24). With one thread there is no pool (`thread_pool_size <= 1` returns
nullptr), so nothing is dispatched. 0.48 ms / 832 = 0.58 us per dispatch.

### 1. What the executed graphs contain (step 1, onnx only)

Source: `_optdiff/opt_1.23.2_p00_int8.onnx` and `opt_1.23.2_p00_fp32.onnx`, input pinned to 1x3x64x64.
onnx shape inference cannot pass through the contrib ops (QLinearConv/QLinearAdd/QLinearMul/
QLinearSigmoid/QLinearSoftmax/QLinearConcat/QLinearGlobalAveragePool/QGemm, nchwc Conv/Reorder*,
QuickGelu). Each was replaced by standard ops that produce the same output shape and dtype, and
inference ran on that proxy. The ORT-written value_info in the dump is stale (pre-optimization names),
and matching nodes by name to the source file gives WRONG shapes, because the transpose optimizer
rewired the attention Transposes. Both were therefore rejected.
The table was re-derived from the 1.24.4 dumps and is identical (`transpose_probe.py --check-dump`).

* Static graph: **56 Transposes, all on UINT8**, 105,984 bytes moved per inference.
  * 48 have a FLOAT twin in the FP32 graph with the same (shape, perm), matched as a multiset.
  * 8 are NHWC layout Transposes around QLinearConv: rows 1, 2, 15, 16, 37, 38, 55, 56.
* FP32 graph: **48 Transposes, all FLOAT**, 80,896 elements (323,584 bytes). Twin indices are in the table.

The last three columns are the static prediction. `transpose_probe.predict` re-implements
`TransposeImpl`, `IsTransposeReshape`, `IsTransposeMovingSingleAxis` and
`TransposeSingleAxis{Outwards,Inwards}` from the v1.24.4 sources, where "loops" is the number of
MlasTranspose calls.

| # | static node (idx, name) | in dtype | in shape | perm | FP32 twin (idx, FLOAT) | path @4T 1.23.2 | path @4T 1.24.4 | dispatches 1.23.2 -> 1.24.4 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 `Transpose_token_191` | UINT8 | [1, 3, 64, 64] | [0, 2, 3, 1] | - (layout, QLinearConv) | MlasTranspose<8-bit> in loops=1 M=3 N=4096 | same | 1 -> 1 |
| 2 | 47 `Transpose_token_258` | UINT8 | [1, 8, 8, 64] | [0, 3, 1, 2] | - (layout, QLinearConv) | MlasTranspose<8-bit> out loops=1 M=64 N=64 | same | 1 -> 1 |
| 3 | 49 `s2/2.1/Transpose` | UINT8 | [256, 2, 4, 2] | [0, 2, 1, 3] | 71 | scalar out 16-bit loops=256 | **MlasTranspose<16-bit> out loops=256 M=2 N=4** | **0 -> 256** |
| 4 | 60 `s2/2.1/Transpose_1` | UINT8 | [1, 64, 16, 4] | [0, 3, 2, 1] | 73 | untyped (serial) | same | 0 -> 0 |
| 5 | 84 `Transpose_token_357` | UINT8 | [4, 16, 4, 16] | [0, 2, 1, 3] | 88 | scalar out 128-bit loops=4 | same | 0 -> 0 |
| 6 | 86 `Transpose_token_355` | UINT8 | [4, 16, 4, 16] | [0, 2, 1, 3] | 100 | scalar out 128-bit loops=4 | same | 0 -> 0 |
| 7 | 98 `s2/2.1/tf0/attn/Transpose_1` | UINT8 | [4, 16, 4, 16] | [0, 2, 3, 1] | 98 | MlasTranspose<8-bit> in loops=4 M=16 N=64 | same | 4 -> 4 |
| 8 | 103 `s2/2.1/tf0/attn/Transpose_2` | UINT8 | [4, 4, 16, 16] | [0, 2, 1, 3] | 105 | scalar out 128-bit loops=4 | same | 0 -> 0 |
| 9 | 140 `Transpose_token_361` | UINT8 | [4, 16, 4, 16] | [0, 2, 1, 3] | 130 | scalar out 128-bit loops=4 | same | 0 -> 0 |
| 10 | 142 `Transpose_token_359` | UINT8 | [4, 16, 4, 16] | [0, 2, 1, 3] | 142 | scalar out 128-bit loops=4 | same | 0 -> 0 |
| 11 | 154 `s2/2.1/tf1/attn/Transpose_1` | UINT8 | [4, 16, 4, 16] | [0, 2, 3, 1] | 140 | MlasTranspose<8-bit> in loops=4 M=16 N=64 | same | 4 -> 4 |
| 12 | 159 `s2/2.1/tf1/attn/Transpose_2` | UINT8 | [4, 4, 16, 16] | [0, 2, 1, 3] | 147 | scalar out 128-bit loops=4 | same | 0 -> 0 |
| 13 | 184 `s2/2.1/Transpose_2` | UINT8 | [1, 4, 16, 64] | [0, 3, 2, 1] | 161 | untyped (serial) | same | 0 -> 0 |
| 14 | 186 `s2/2.1/Transpose_3` | UINT8 | [256, 4, 2, 2] | [0, 2, 1, 3] | 163 | scalar out 16-bit loops=256 | **MlasTranspose<16-bit> out loops=256 M=4 N=2** | **0 -> 256** |
| 15 | 189 `Transpose_token_261` | UINT8 | [1, 64, 8, 8] | [0, 2, 3, 1] | - (layout, QLinearConv) | MlasTranspose<8-bit> in loops=1 M=64 N=64 | same | 1 -> 1 |
| 16 | 208 `Transpose_token_285` | UINT8 | [1, 4, 4, 80] | [0, 3, 1, 2] | - (layout, QLinearConv) | MlasTranspose<8-bit> out loops=1 M=16 N=80 | same | 1 -> 1 |
| 17 | 220 `s3/3.1/Transpose` | UINT8 | [160, 2, 2, 2] | [0, 2, 1, 3] | 206 | scalar out 16-bit loops=160 | **MlasTranspose<16-bit> out loops=160 M=2 N=2** | **0 -> 160** |
| 18 | 225 `s3/3.1/Transpose_1` | UINT8 | [1, 80, 4, 4] | [0, 3, 2, 1] | 208 | untyped (serial) | same | 0 -> 0 |
| 19 | 249 `Transpose_token_376` | UINT8 | [4, 4, 4, 20] | [0, 2, 1, 3] | 223 | scalar out 160-bit loops=4 | same | 0 -> 0 |
| 20 | 251 `Transpose_token_374` | UINT8 | [4, 4, 4, 20] | [0, 2, 1, 3] | 235 | scalar out 160-bit loops=4 | same | 0 -> 0 |
| 21 | 263 `s3/3.1/tf0/attn/Transpose_1` | UINT8 | [4, 4, 4, 20] | [0, 2, 3, 1] | 233 | MlasTranspose<8-bit> in loops=4 M=4 N=80 | same | 4 -> 4 |
| 22 | 268 `s3/3.1/tf0/attn/Transpose_2` | UINT8 | [4, 4, 4, 20] | [0, 2, 1, 3] | 240 | scalar out 160-bit loops=4 | same | 0 -> 0 |
| 23 | 305 `Transpose_token_380` | UINT8 | [4, 4, 4, 20] | [0, 2, 1, 3] | 265 | scalar out 160-bit loops=4 | same | 0 -> 0 |
| 24 | 307 `Transpose_token_378` | UINT8 | [4, 4, 4, 20] | [0, 2, 1, 3] | 277 | scalar out 160-bit loops=4 | same | 0 -> 0 |
| 25 | 319 `s3/3.1/tf1/attn/Transpose_1` | UINT8 | [4, 4, 4, 20] | [0, 2, 3, 1] | 275 | MlasTranspose<8-bit> in loops=4 M=4 N=80 | same | 4 -> 4 |
| 26 | 324 `s3/3.1/tf1/attn/Transpose_2` | UINT8 | [4, 4, 4, 20] | [0, 2, 1, 3] | 282 | scalar out 160-bit loops=4 | same | 0 -> 0 |
| 27 | 361 `Transpose_token_384` | UINT8 | [4, 4, 4, 20] | [0, 2, 1, 3] | 307 | scalar out 160-bit loops=4 | same | 0 -> 0 |
| 28 | 363 `Transpose_token_382` | UINT8 | [4, 4, 4, 20] | [0, 2, 1, 3] | 319 | scalar out 160-bit loops=4 | same | 0 -> 0 |
| 29 | 375 `s3/3.1/tf2/attn/Transpose_1` | UINT8 | [4, 4, 4, 20] | [0, 2, 3, 1] | 317 | MlasTranspose<8-bit> in loops=4 M=4 N=80 | same | 4 -> 4 |
| 30 | 380 `s3/3.1/tf2/attn/Transpose_2` | UINT8 | [4, 4, 4, 20] | [0, 2, 1, 3] | 324 | scalar out 160-bit loops=4 | same | 0 -> 0 |
| 31 | 417 `Transpose_token_388` | UINT8 | [4, 4, 4, 20] | [0, 2, 1, 3] | 349 | scalar out 160-bit loops=4 | same | 0 -> 0 |
| 32 | 419 `Transpose_token_386` | UINT8 | [4, 4, 4, 20] | [0, 2, 1, 3] | 361 | scalar out 160-bit loops=4 | same | 0 -> 0 |
| 33 | 431 `s3/3.1/tf3/attn/Transpose_1` | UINT8 | [4, 4, 4, 20] | [0, 2, 3, 1] | 359 | MlasTranspose<8-bit> in loops=4 M=4 N=80 | same | 4 -> 4 |
| 34 | 436 `s3/3.1/tf3/attn/Transpose_2` | UINT8 | [4, 4, 4, 20] | [0, 2, 1, 3] | 366 | scalar out 160-bit loops=4 | same | 0 -> 0 |
| 35 | 461 `s3/3.1/Transpose_2` | UINT8 | [1, 4, 4, 80] | [0, 3, 2, 1] | 380 | untyped (serial) | same | 0 -> 0 |
| 36 | 463 `s3/3.1/Transpose_3` | UINT8 | [160, 2, 2, 2] | [0, 2, 1, 3] | 382 | scalar out 16-bit loops=160 | **MlasTranspose<16-bit> out loops=160 M=2 N=2** | **0 -> 160** |
| 37 | 465 `Transpose_token_288` | UINT8 | [1, 80, 4, 4] | [0, 2, 3, 1] | - (layout, QLinearConv) | MlasTranspose<8-bit> in loops=1 M=80 N=16 | same | 1 -> 1 |
| 38 | 484 `Transpose_token_312` | UINT8 | [1, 2, 2, 96] | [0, 3, 1, 2] | - (layout, QLinearConv) | MlasTranspose<8-bit> out loops=1 M=4 N=96 | same | 1 -> 1 |
| 39 | 486 `s4/4.1/Transpose` | UINT8 | [96, 2, 1, 2] | [0, 2, 1, 3] | 422 | reshape-copy | same | 0 -> 0 |
| 40 | 497 `s4/4.1/Transpose_1` | UINT8 | [1, 96, 1, 4] | [0, 3, 2, 1] | 424 | untyped (serial) | same | 0 -> 0 |
| 41 | 521 `Transpose_token_403` | UINT8 | [4, 1, 4, 24] | [0, 2, 1, 3] | 439 | reshape-copy | same | 0 -> 0 |
| 42 | 523 `Transpose_token_401` | UINT8 | [4, 1, 4, 24] | [0, 2, 1, 3] | 451 | reshape-copy | same | 0 -> 0 |
| 43 | 535 `s4/4.1/tf0/attn/Transpose_1` | UINT8 | [4, 1, 4, 24] | [0, 2, 3, 1] | 449 | reshape-copy | same | 0 -> 0 |
| 44 | 540 `s4/4.1/tf0/attn/Transpose_2` | UINT8 | [4, 4, 1, 24] | [0, 2, 1, 3] | 456 | reshape-copy | same | 0 -> 0 |
| 45 | 577 `Transpose_token_407` | UINT8 | [4, 1, 4, 24] | [0, 2, 1, 3] | 481 | reshape-copy | same | 0 -> 0 |
| 46 | 579 `Transpose_token_405` | UINT8 | [4, 1, 4, 24] | [0, 2, 1, 3] | 493 | reshape-copy | same | 0 -> 0 |
| 47 | 591 `s4/4.1/tf1/attn/Transpose_1` | UINT8 | [4, 1, 4, 24] | [0, 2, 3, 1] | 491 | reshape-copy | same | 0 -> 0 |
| 48 | 596 `s4/4.1/tf1/attn/Transpose_2` | UINT8 | [4, 4, 1, 24] | [0, 2, 1, 3] | 498 | reshape-copy | same | 0 -> 0 |
| 49 | 633 `Transpose_token_411` | UINT8 | [4, 1, 4, 24] | [0, 2, 1, 3] | 523 | reshape-copy | same | 0 -> 0 |
| 50 | 635 `Transpose_token_409` | UINT8 | [4, 1, 4, 24] | [0, 2, 1, 3] | 535 | reshape-copy | same | 0 -> 0 |
| 51 | 647 `s4/4.1/tf2/attn/Transpose_1` | UINT8 | [4, 1, 4, 24] | [0, 2, 3, 1] | 533 | reshape-copy | same | 0 -> 0 |
| 52 | 652 `s4/4.1/tf2/attn/Transpose_2` | UINT8 | [4, 4, 1, 24] | [0, 2, 1, 3] | 540 | reshape-copy | same | 0 -> 0 |
| 53 | 677 `s4/4.1/Transpose_2` | UINT8 | [1, 4, 1, 96] | [0, 3, 2, 1] | 554 | untyped (serial) | same | 0 -> 0 |
| 54 | 679 `s4/4.1/Transpose_3` | UINT8 | [96, 1, 2, 2] | [0, 2, 1, 3] | 556 | reshape-copy | same | 0 -> 0 |
| 55 | 682 `Transpose_token_315` | UINT8 | [1, 96, 2, 2] | [0, 2, 3, 1] | - (layout, QLinearConv) | MlasTranspose<8-bit> in loops=1 M=96 N=4 | same | 1 -> 1 |
| 56 | 691 `Transpose_token_324` | UINT8 | [1, 2, 2, 320] | [0, 3, 1, 2] | - (layout, QLinearConv) | MlasTranspose<8-bit> out loops=1 M=4 N=320 | same | 1 -> 1 |

Name abbreviations: `sN/N.1/...` stands for `/stages/stages.N/stages.N.1/...`, and `tfK` for
`transformer/transformer.K`.

Paths of the FP32 graph's 48 float Transposes: the twins of rows 3, 14, 17 and 36 have 8-byte blocks,
so they run a scalar 64-bit loop on BOTH releases. The six `[0,2,3,1]` attention Transposes of stages
2-3 use `MlasTranspose<32-bit>` on both releases (4 dispatches each); the three in stage 4 are
reshape-copy. The rest are scalar, untyped or reshape-copy, as for uint8. Total dispatches at 4
threads: **24 -> 24**.

### 2. Desk research (step 3); only items actually seen are listed

* Tags compared with https://api.github.com/repos/microsoft/onnxruntime/compare/v1.23.2...v1.24.4
  (web view https://github.com/microsoft/onnxruntime/compare/v1.23.2...v1.24.4). Result: diverged,
  649 commits ahead and 21 behind, merge base 29c20cb (2025-07-25). The response's `files` list is
  truncated at 300, so relevant files were checked two ways. (a) The commits API filtered by path
  (`sha=v1.24.4`, `since` = merge base). (b) Both tags' raw files were diffed.
* **https://github.com/microsoft/onnxruntime/commit/42869cac54b6f870bbb33894bf0fddcc9b737646**
  ("FP16 inference performance improvement on CPU", PR **https://github.com/microsoft/onnxruntime/pull/25680**,
  merged 2025-10-24, merge sha 42869ca per the pulls API).
  * Its patch to `onnxruntime/core/framework/transpose_helper.cc` is exactly
    `template <> struct has_mlas_transpose<uint16_t> : std::true_type {};`. This is the whole
    `transpose_helper.cc` difference between v1.23.2 and v1.24.4. The commit message says the change
    was for fp16 Transpose on Arm (Graviton), but the specialisation is architecture-independent.
  * It applies to any single-axis transpose whose moved block is 2 bytes, including uint8 tensors
    whose moved block is 2 elements.
  * Absent at v1.22.1, v1.23.0 and v1.23.2. Present at v1.24.4, v1.25.1 and v1.30.0 (raw files).
    This matches the step persisting at 1.25.1 (2.028 ms) and 1.30.0 (2.034 ms).
* `MlasTranspose` (`onnxruntime/core/mlas/lib/transpose.cpp`) is the same threaded routine on v1.23.2,
  v1.24.4, v1.25.1 and v1.30.0. It sets `ThreadCountM = min(MlasGetMaximumThreadCount(tp), M)` and
  then calls `MlasExecuteThreaded`, which dispatches whenever iterations > 1.
  * The only v1.23.2 -> v1.24.4 change to this file is s390x SIMD code under
    `#if defined(MLAS_TARGET_S390X)`:
    https://github.com/microsoft/onnxruntime/commit/510dd14f6f (PR #25757, "S390x simd implementation").
  * `DegreeOfParallelism` = (NumThreads()+1) x TaskGranularityFactor (4) on hybrid CPUs. That is 16 at
    4 threads, so min(16, M) = M for the M = 2 and 4 calls above.
* `onnxruntime/core/providers/cpu/tensor/transpose.cc`: 3 commits in range.
  * https://github.com/microsoft/onnxruntime/commit/fe30e5c7506d220b51a4e5536ab3464f8fc0fa57
    (1.24.0 cherry-pick round 1, #27104). It adds INT2/UINT2 types and the Transpose-25 registration.
    For uint8 at opset 17 it adds only two type checks before `TransposeImpl`.
  * de398c4125 (ONNX 1.20.1 integration, #26579) and ecb26fb775 (ONNX 1.19 integration, #25678):
    titles seen only. The tag-to-tag diff shows only versioned kernel registrations (23, 24, 25) and
    the sub-byte refactor. `TransposeImpl`, `IsTransposeReshape` and the single-axis dispatch are
    unchanged.
* No change between the tags (tag-to-tag diff empty and/or no commits in range):
  * `core/common/threadpool.cc`
  * `include/onnxruntime/core/platform/EigenNonBlockingThreadPool.h`
  * `include/onnxruntime/core/platform/threadpool.h`
  * `core/util/thread_utils.cc`
  * `core/mlas/lib/threading.cpp`
  * `core/framework/sequential_executor.cc`
  * `core/framework/execution_frame.cc`

  The intra-op thread pool itself did not change. `core/platform/windows/env.cc` has one commit
  (568ad20, file memory-mapping fix, #25833), unrelated to threading. The x86 part of the
  `mlas/lib/platform.cpp` diff only adds a T-MAC LUT kernel pointer (`MlasLutGenKernelAvx2`); the rest
  is ARM, s390x or FreeBSD.
* A GitHub issue search found no upstream report of this regression.
* Consequence: the executed graph is identical and the thread pool is identical. The only code change
  on the static graph's Transpose path is the uint16 routing. That fits all three signatures: 4T-only
  (no pool at 1T), static-only (the float twins have 8-byte blocks) and extra parallel work
  (dC/dW = 3.1). It remains a hypothesis until timed.

### 3. Probe design (step 2)

All models are opset 17, IR 8. Each takes ONE float32 input `x` [1,3,64,64], which is what
`ort_version_bisect.py` feeds. Default spinning, ORT_SEQUENTIAL, ENABLE_ALL, the harness unchanged.

**Skeleton "full" (56 branches).** The flat source tensor feeds 56 branches. Branch b does
Slice(start=b, length n_b) -> Reshape(s_b) -> [Transpose(perm_b)] -> Reshape([n_b]). The branches then
go to Concat -> output. Branch 0 (12,288 elements) takes the whole source.

* The u8 arms wrap this in Q ... DQ: QuantizeLinear (scale 0.02, zp 128) at the input and
  DequantizeLinear on the concatenated output, so every Transpose runs on uint8 as in the executed
  static graph.
* The f32 arms have no wrapper.
* Distinct slice offsets stop CSE from merging identical branches.
* The branches never put two Transposes next to each other, so the optimizer cannot cancel or merge
  them.
* All arms of one dtype share the same skeleton and differ only in which branches hold their
  Transpose. One control per dtype therefore suffices.

**Skeleton "mech" (4 branches).** Same structure with 4 branches, no Q/DQ, and Cast to/from float16
for the f16 arm.

| variant | Transposes | purpose | predicted dispatches/inference at 4T, 1.23.2 -> 1.24.4 (1T: 0 -> 0 for all) |
|---|---|---|---|
| tp_u8_all | 56 uint8, rows 1-56 | (a) the static graph's 8-bit Transposes | 32 -> 864 |
| tp_u8_ctrl | none | (c) wrapper + skeleton control | 0 -> 0 |
| tp_f32_all | 48 float, the FP32 twins | (b) the same Transposes in float | 24 -> 24 |
| tp_f32_ctrl | none | float skeleton control | 0 -> 0 |
| tp_u8_w2 | 4 uint8, rows 3, 14, 17, 36 | mechanism: the 2-byte-block Transposes alone | 0 -> 832 |
| tp_u8_rest | the other 52 uint8 | mechanism: everything else | 32 -> 32 |
| tp_f16_w2 | 4 float16 [256,2,4], [256,4,2], [160,2,2] x2, perm [0,2,1] | same loops/M/N and bytes as tp_u8_w2, 2-byte ELEMENTS: dtype-independence | 0 -> 832 |
| tp_f32_w4 | the same 4 shapes in float32 | 4-byte elements: MLAS path on BOTH releases (positive dispatch-cost control) | 832 -> 832 |

Checks already done (onnx only):
* `onnx.checker.check_model(full_check=True)` passes for all 8 files.
* Each built graph's Transposes equal the design: shape, perm, dtype.
* The table matches the 1.23.2 and the 1.24.4 dumps.
* The `--verify-executed` comparison logic was exercised on the built models: reordering accepted,
  dropping a reshape-equivalent Transpose accepted, dropping an essential one rejected, wrong dtype
  rejected.
* `--analyse` was exercised on SYNTHETIC numbers only (deleted afterwards). Four scenarios gave the
  intended verdicts: localised + mechanism confirmed; flat -> refuted; partial -> mechanism refuted;
  step also at 1T -> inconclusive.

Not run by the designer: `--verify-executed`, because it needs the side venvs' onnxruntime.

Cost: 8 variants x 2 releases x 2 thread counts x 3 sessions = **96 cells**. At about 4.5 s per cell
(experiment 23b ran 54 cells in 4 min), that is about 7-8 min. The optional extension (section 4.5)
adds 12 cells (about 1 min) and the executed-graph dump takes under 1 min. Total is under 10 min.

### 4. Procedure and decision rule (step 4), fixed before timing

#### 4.1 Procedure

1. Precondition, same as the exp 24/25 re-run:
   * The ICCAD WSL job is stopped (`pgrep -f '[r]eport.py'` is empty).
   * Machine-wide CPU load is <= 20% over 5 s after a 20 s settle.
2. `python scripts/paper/transpose_probe.py --verify-executed`. This dumps each model's optimized graph
   under 1.23.2 and 1.24.4 at 4 threads and writes `results/transpose_probe/executed_check.json`. It
   passes only if all of the following hold:
   * Every designed Transpose that is not reshape-equivalent is executed with the designed shape, perm
     and dtype. Reshape-equivalent ones may become Reshape because the shapes are static.
   * No other Transpose appears.
   * The op census is identical between the two releases.

   **FAIL = do not time.**
3. Timing, from the project root:

   ```
   "C:/Users/User/Desktop/SEC/edge-ai-toolkit/.venv/Scripts/python.exe" scripts/paper/ort_version_bisect.py --scratch "C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-SEC/103ae351-d709-4d02-91fc-0b8da660fe58/scratchpad" --versions 1.23.2,1.24.4 --threads 1,4 --repeats 3 --seed 27 --families "tprobe=results/transpose_probe/models:3,64,64" --variants tp_u8_all,tp_u8_w2,tp_u8_rest,tp_u8_ctrl,tp_f32_all,tp_f32_ctrl,tp_f16_w2,tp_f32_w4 --out results/paper/transpose_probe.csv
   ```

   The harness's built-in speedup summary looks for p00_* names and prints nothing for these
   variants; that is expected.
4. `python scripts/paper/transpose_probe.py --analyse results/paper/transpose_probe.csv` applies the
   rule below mechanically and writes `results/transpose_probe/verdict.json`.

#### 4.2 Quantities

* m(v, r, T) is the median of the 3 per-session p50s of variant v on release r at T threads, **ungated**
  (as the paper reports).
* D(v, T) = m(v, 1.24.4, T) - m(v, 1.23.2, T).
* The isolated Transpose step is Δu8(T) = D(tp_u8_all, T) - D(tp_u8_ctrl, T).
  * Δf32(T) = D(tp_f32_all, T) - D(tp_f32_ctrl, T).
  * Δw2(T) = D(tp_u8_w2, T) - D(tp_u8_ctrl, T).
  * Δrest(T) = D(tp_u8_rest, T) - D(tp_u8_ctrl, T).
* Step reference S = **0.48 ms**, from `ort_version_bisect.csv` medians at 4T: 2.037 - 1.553. Shares
  are also reported against 0.469 (spin_version.csv, spin on) and 0.570 (coverage_mobilevit.csv) as
  sensitivity; they do not change the verdict.
* Floor: 7% (the between-session floor used by every registered rule). "Small" means
  < 0.07·S = **0.034 ms**. An arm's own floor is 0.07 x its 1.23.2 median.
* Envelope: the minimum and maximum of a Δ over all 3^4 = 81 combinations of one session p50 from each
  of its four cells. This is a conservative session-resampling range; it is not a confidence interval.

#### 4.3 Validity (a failure means INCONCLUSIVE, not a verdict)

* V1: `--verify-executed` passed.
* V2: controls stable. |D(tp_u8_ctrl, T)| and |D(tp_f32_ctrl, T)| are each <= max(own 7% floor,
  0.034 ms) at T = 1 and 4.
* V3: the verdicts from ungated and gated (core delivery >= 0.95) medians agree. If more than 1/3 of
  any decisive cell's sessions are below the gate, rerun those cells.
* V4: the precondition in 4.1 held.

#### 4.4 Primary verdict on H_T (bands against S)

| verdict | condition |
|---|---|
| **LOCALISED** | Δu8(4T) >= **0.70·S = 0.336 ms** (the 70% threshold registered in 23c), AND D(tp_u8_all, 4T) > its own 7% floor, AND Δu8(1T) < 0.034 ms (no 1-thread step, as in the full model), AND Δf32(4T) < 0.034 ms (the float twins do not step) |
| **PARTIAL** | 0.15·S = 0.072 ms <= Δu8(4T) < 0.336 ms, with the same 1T and float conditions |
| **REFUTED** | Δu8(4T) < **0.072 ms** (under 15% of the step, i.e. under a quarter of the 62% the profile attributed), OR D(tp_u8_all, 4T) within its own 7% floor |
| INCONCLUSIVE | any V fails. OR Δu8(1T) >= 0.034 ms: the probe does not reproduce the 4-thread-only signature. OR Δf32(4T) >= 0.034 ms while Δu8 >= 0.072 ms: not 8-bit-specific, so the full FP32 graph's flat latency is unexplained |

* Firmness: the verdict is FIRM when the Δu8(4T) envelope lies entirely in the same band as the median.
  Otherwise it is PROVISIONAL, which triggers the registered extension: 3 more sessions of tp_u8_all
  and tp_u8_ctrl at 4T on both releases, 12 cells, same command with
  `--variants tp_u8_all,tp_u8_ctrl --threads 4 --seed 28 --out results/paper/transpose_probe_ext.csv`.
  Then run `--analyse` on both CSVs, which gives 6 sessions per cell. If the envelope still straddles
  a band edge, report the result as lying between the two adjacent bands, with the envelope as its
  range.
* If Δu8 > 1.5·S, the verdict is LOCALISED but the paper says "reproduces the step" and states no share
  above 100%.

#### 4.5 Mechanism verdict on H_M (evaluated only if LOCALISED or PARTIAL)

| verdict | condition |
|---|---|
| **M-CONFIRMED** | Δw2(4T) >= 0.70·Δu8(4T), AND Δrest(4T) < 0.30·Δu8(4T), AND D(tp_f16_w2, 4T) >= 0.50·Δw2(4T) (2-byte fp16 elements step too: the step depends on block width, not on 8-bit), AND abs(D(tp_f32_w4, 4T)) < 0.034 ms (4-byte blocks were already on the MLAS path in 1.23.2, so they do not step), AND Δw2(1T) < 0.034 ms |
| **M-REFUTED** | Δw2(4T) < 0.30·Δu8(4T), OR abs(D(tp_f32_w4, 4T)) >= 0.50·Δw2(4T) (the 4-byte arm steps too, so the change is not the uint16 routing), OR D(tp_f16_w2, 4T) < 0.034 ms |
| M-MIXED | anything else: report the shares; the code-level mechanism stays a hypothesis |

Reported, not gating:
* Additivity: D(w2) + D(rest) - D(ctrl) against D(all).
* Per-dispatch cost c = [m(tp_f32_w4, 1.23.2, 4T) - m(tp_f32_w4, 1.23.2, 1T)] / 832, and the ratio
  Δw2 / (832·c). The expected ratio is 0.5-2 if dispatch overhead is the cost.

#### 4.6 What the paper may write

* LOCALISED + M-CONFIRMED: "The 1.24 step is reproduced by the static graph's 8-bit Transposes alone at
  4 threads (Δ = x ms, y% of the 0.48 ms step), not at 1 thread and not by the same Transposes in
  float. Four of them, whose moved block is two bytes wide, carry z%. From ONNX Runtime 1.24 (PR
  #25680), such transposes are routed to MLAS's multi-threaded transpose once per outer loop: 832
  thread-pool dispatches per inference at 4 threads instead of none." This replaces "thread-pool-level"
  with a named, code-level cause. Also report the ungated and gated values and the envelope.
* LOCALISED + M-MIXED/REFUTED: state the localisation to the 8-bit Transpose with its share. The code
  path stays unexplained; do not cite #25680 as the cause.
* PARTIAL: keep the registered 23c verdict "thread-pool-level", and add "an isolated replica of its
  8-bit Transposes reproduces y% (envelope) of the step".
* REFUTED: keep "thread-pool-level", and state that the Transpose lead was tested in isolation and not
  reproduced. Do not mention #25680 as a cause.
* INCONCLUSIVE: report the numbers and the failed condition; no mechanism sentence.

### 5. Limits stated in advance

* The probe removes the context of the full graph: the interleaving with QLinear kernels, cache state
  and pool state between nodes. REFUTED therefore rules out the Transpose kernel as a stand-alone
  cause, not a context-dependent interaction. LOCALISED shows that the Transposes are sufficient in
  isolation, not that nothing else contributes in the full graph.
* 23c's second component, a uniform 5-14% rise of every op type under profiling, is not addressed.
  PARTIAL leaves room for it.
* A direct causal test would rebuild 1.24.4 without the uint16 specialisation, or run the full model
  with the four Transposes rewritten to non-2-byte blocks. Neither is in scope; the f16/f32 arms are
  the in-harness substitute.
* The prediction uses the hybrid-CPU `DegreeOfParallelism` (x4). It does not change which calls
  dispatch, because every changed call has M <= 4.

(Copied verbatim from the independent designer's note transpose_probe_design.md before anything was timed; the designer, not the author of the claim, fixed every threshold.)

### Experiment 26 (K1) RESULT (2026-09-19, timing 13:53-14:36): rule (B) fires; the version effect is two-directional

`results/paper/ort_version_scope.csv`: 462 cells measured, 78 below the gate (all reported ungated;
every rule cell was re-checked gated and gives the same verdict). Two earlier launches were stopped
before producing data: 13:42 (quiet precheck aborted at 24%, the launching session was busy) and
13:47 (a FALSE overload alarm: Win32_Processor.LoadPercentage read 100% on K1's own 4-thread cells
while the machine was 22-32% busy; the guard now uses PerfOS_Processor _Total).

Change of median latency vs 1.23.2 (%), ungated; sessions disjoint where marked *:

| file | thr | arm | 1.24.4 | 1.25.1 | 1.30.0 |
|---|---|---|---|---|---|
| MobileViT-XXS 64 px (byte-identical to the paper's file) | 4 | static | +25.3* | +24.4 | +31.0 |
| MobileViT-XXS 256 px | 4 | static | **+46.4*** (9.56 -> 13.99 ms) | +36.7 | +39.0 |
| MobileViT-XXS 256 px | 4 | fp32 / dyn | -0.7 / -4.3 | -11.5 / -16.2 | -12.0 / -19.7 |
| MobileViT-XXS 256 px | 1 | static | +5.5 | +7.2 | +5.2 |
| ResNet-50 / MobileNetV2-32 | 4 | static | -1.7 / +0.3 | +2.1 / -1.2 | +0.3 / +1.5 |
| YOLOv8n 256 px | 4 | static | **-28.2*** | -31.2 | -23.5 |
| YOLOv8n 256 px | 1 | static | **-18.5*** | -17.3 | -16.4 |
| YOLOv8n 256 px | 4 | fp32 | +7.5 | -14.5 | -10.1 |
| 1-D AE | 4 | fp32 / static | -44.1 / -41.3 | -44.2 / -41.7 | -42.3 / -41.1 |

* **Bridge holds:** the rebuilt 64 px file is byte-identical (sha256 911de4ee...) and steps +25% at 4T
  (fourth independent observation). The 256 px series is interpretable.
* **(A) does not fire:** no CNN static file steps; ResNet-50 and MobileNetV2 are flat, YOLOv8n and
  the autoencoder get FASTER.
* **(B) FIRES:** MobileViT at 256 px steps +46.4% (gated +48.1%) at 4T, still +36.7/+39.0% at
  1.25.1/1.30.0, siblings within 7% at 1.24.4. The step is 4.43 ms at 256 px vs 0.48 ms at 64 px:
  it grows with the model (16x the pixels), so the "fixed ~0.5 ms multi-thread overhead on a sub-2 ms
  model" objection is refuted. Registered wording: a hybrid-op-mix regression that persists at
  deployment resolution with real parallel work; keep the title and name the hybrid class in scope.
* **Independent rule FIRES (with context):** YOLOv8n FP32 at 4T got 20.5% faster from 1.24.4 to
  1.25.1 while its static file moved 4.2%, so its INT8 speedup shrank through the FP32 baseline
  (1.58x -> 1.31x). But at 1.24.4 its static file had become 28% FASTER with FP32 flat, so its 4T
  speedup went 1.05x (1.23.2) -> 1.58x (1.24.4) -> 1.31x (1.25.1) -> 1.24x (1.30.0) on one file.
  The same release that made the hybrid's static file 46% slower made YOLOv8n's 28% faster: the
  version effect on INT8 speedups is two-directional and model-dependent.
* **Branch A, per release (rule: 'otherwise report per version'):** dynamic-Conv penalty vs FP32
  at 1T = ResNet-50 16.2/16.2/16.3x, MobileNetV2 17.0/17.0/17.3x, YOLOv8n 13.5/14.1/14.3x on
  1.24.4/1.25.1/1.30.0. On 1.23.2 the dynamic-Conv files do not run at all: NOT_IMPLEMENTED,
  'Could not find an implementation for ConvInteger(10)'. Static beats FP32 on ResNet-50 on all four.
* **DeiT-224 rule FIRES:** dynamic faster than static by 15-46% at both thread counts on all four
  releases (static/dynamic latency 1.17/1.22/1.30/1.28 at 1T, 1.43/1.26/1.46/1.27 at 4T; gated
  agrees). The attention half of Branch A holds at 197 tokens (random weights, latency only).
* The 1-D autoencoder (0.1-0.2 ms) got ~40-50% faster in every arm at 1.24.4: a per-call overhead
  change, visible only because the model is that small; its ratios are indicative only.

### Experiment 27 RESULT (2026-09-19, timing 17:41-17:47): registered verdict INCONCLUSIVE; ungated evidence strong

`results/paper/transpose_probe.csv` (96 cells, 19 below the gate), analysed mechanically by
`transpose_probe.py --analyse` -> `results/transpose_probe/verdict.json`. A first launch at 15:31
aborted before timing because the laptop was on battery.

* **Ungated (the paper's reporting convention):** PRIMARY LOCALISED, MECHANISM M-CONFIRMED.
  delta_u8 at 4T = +0.423 ms (81-combination envelope +0.396..+0.441), 88% of the 0.48 ms step;
  at 1T +0.008 ms; float Transposes at 4T +0.009 ms; the four 2-byte-block Transposes alone
  (tp_u8_w2) +0.427 ms = 101% of delta_u8; the other 52 uint8 Transposes -0.003 ms; the same four
  loop shapes in float16 (2-byte elements) +0.408 ms; in float32 (4-byte) +0.012 ms. Dispatch-cost
  prediction from tp_f32_w4 (0.51 us per dispatch x 832) = 0.420 ms vs 0.427 ms measured (1.01).
* **Gated:** every 4-thread cell of both controls and tp_u8_w2 at 1.23.2 falls below 0.95
  (core delivery 0.26-0.32), so the control subtraction is undefined; the registered final reads
  **INCONCLUSIVE (gated and ungated verdicts differ)**. tp_u8_all keeps 3/3 gated sessions on both
  releases (+0.422 ms).
* Author's reading, NOT a verdict (sent to an independent referee before any paper text changes):
  the excluded cells are the ones with no parallel work, whose idle workers sleep, so core delivery
  is ~1/threads by construction (0.26-0.32 at 4T), while cells with parallel work sit at 1.0-1.09.
  The same file tp_u8_w2 reads 0.28 at 1.23.2 (no dispatch) and 1.06 at 1.24.4 (832 dispatches):
  the gate reading itself switches with the predicted dispatch.


---

## Referee ruling on Experiment 27 (Transpose isolation probe), 2026-09-19

The referee did not design the probe and is not the author of the claim. Sources: FINDINGS.md Exp 27
sections 0-5 and its RESULT block; `results/paper/transpose_probe.csv` (96 rows, file order is run order);
`results/transpose_probe/verdict.json`, `manifest.json` and `executed_check.json`;
`scripts/paper/transpose_probe.py` (`analyse`); `scripts/paper/ort_version_bisect.py`; and the scratch
file `run_probe.sh`. Nothing was timed and onnxruntime was not imported.

Checked independently: every ungated number in the task matches the CSV. Δu8(4T) = +0.4228 ms (envelope
+0.396..+0.441), which is 88.1% of 0.48 ms. Δw2(4T) = +0.4265 ms (101% of Δu8), Δrest = −0.0025,
D(f16_w2, 4T) = +0.4077, D(f32_w4, 4T) = +0.0117, Δu8(1T) = +0.0083, Δf32(4T) = +0.0085, and 832·c = 0.4204
ms. There are 19 below-gate sessions. A re-implementation of rules 4.2-4.5 reproduces the ungated
verdict, LOCALISED (FIRM) + M-CONFIRMED.

### Q1. Contamination, or a gate that cannot apply to these cells?

**Ruling: the gate cannot apply to these cells. Their exclusion is set by the graph and says nothing
about contamination either way.**

1. **The groups separate perfectly, and the grouping was fixed before timing.** The classes come from
   manifest.json (13:35) `predicted_dispatches`, not from the observed values.

   | 4T cells by predicted thread-pool dispatches | sessions | core delivery | below 0.95 |
   |---|---|---|---|
   | 0 dispatches (both controls on both releases; tp_u8_w2 and tp_f16_w2 on 1.23.2) | 18 | 0.258-0.320 | 18/18 |
   | >0 dispatches (the other 10 cells) | 30 | 0.949-1.115 | 1/30 (tp_f16_w2 1.24.4 r1, 0.949) |
   | 1T, all variants (no pool) | 48 | 0.955-1.229 | 0/48 |

2. **The CPU time matches one busy core, not a starved one.** Busy cores = cpu_ms_per_inf / p50.
   * The 18 excluded sessions read 1.03-1.28. The 1T sessions read 0.96-1.23. Sessions with pool work
     read 3.80-4.46.
   * Contamination that took CPU from the only working thread would push this ratio BELOW 1.
   * Core delivery ≈ busy/4 ≈ 0.26-0.32 is the expected value when workers are never woken and so never
     spin.
   * The project already records this effect: `ort_version_bisect.py` lines 150-153, and Exp 23b says
     "core delivery 0.41-0.56 is expected [with sleeping workers] and is not a contamination signal".
     Both concern spin=0, not graphs with no pool work.
3. **The same effect appears on both releases.** tp_u8_ctrl and tp_f32_ctrl fail on 1.23.2 AND 1.24.4.
   Inside one file, the reading switches with the predicted dispatch count:
   * tp_u8_w2: 0.263-0.314 on 1.23.2 (0 dispatches), 1.061-1.066 on 1.24.4 (832 dispatches).
   * tp_f16_w2: 0.274-0.282, then 0.949-1.100.
   * Contamination cannot know which release is running.
4. **The 4T latency of the controls equals their 1T latency.** 4T/1T median ratios:

   | cell | 4T/1T |
   |---|---|
   | u8_ctrl | 1.015 / 1.034 |
   | f32_ctrl | 1.071 / 1.048 |
   | u8_w2 on 1.23.2 | 1.010 |
   | f16_w2 on 1.23.2 | 1.004 |

   These sessions ran on one core with no inflation, as a graph with no parallel work should.
   * D(ctrl, 4T) is −0.0012 ms (u8) and −0.0075 ms (f32), so V2 passes easily on ungated medians.
5. **Session spread gives no sign of contamination.**
   * Excluded cells: 4T spread 1.5-9.6%.
   * Gate-passing 4T cells: 1.0-20.3%. The widest cell in the run, tp_f32_w4 1.23.2 4T at 20.3%, PASSES
     the gate.
   * So the gate does not track dispersion.
6. **The excluded sessions are spread through the whole run, in shuffled order.** They sit at positions
   0-95 of 96, next to sessions with pool work that pass. Example: position 45, tp_f32_ctrl 1.23.2, reads
   0.258, between position 44 at 1.072 and position 46 at 1.102.

Caveat: points 2 and 4-6 are post-hoc evidence against contamination. They are not a registered check.
"Inapplicable" means the gate is uninformative for these 18 sessions. It does not mean they passed it.

### Q2. Verdict under the registered rule as written

**Registered verdict: INCONCLUSIVE. The failed condition is V3.** It is not V2 and it is not drift.

* V1 passes: `executed_check.json` shows all 8 variants executed as designed, with the same op census on
  both releases.
* V2 passes: section 4.2 defines m() as ungated.
* V4: `run_probe.sh` aborts unless the machine is on AC power, no WSL distro is running and machine-wide
  load is ≤ 15%. The CSV exists, so these gates were passed.
* V3 cannot be satisfied. Every 4T control session is excluded, so the gated Δu8, Δf32 and V2 are
  undefined.
* Label defect: in `verdict.json` the gated arm reads "INCONCLUSIVE (control drift at or above the
  floor)". That label comes from NaN comparisons evaluating False. No drift was measured, and the text
  must not say "drift".
* **The registered text does not anticipate this case.**
  * V3's second sentence ("If more than 1/3 of any decisive cell's sessions are below the gate, rerun
    those cells") assumes a below-gate session is transient contamination.
  * No clause exempts cells whose expected reading is ~1/threads.
  * The referee may not add such a clause now.
* **The registered rerun was never carried out.**
  * `--analyse` jumps straight to "gated and ungated verdicts differ".
  * FINDINGS must state this.
  * The registered path is to rerun the 6 cells with > 1/3 below the gate: u8_ctrl and f32_ctrl at 4T
    on both releases, and u8_w2 and f16_w2 at 4T on 1.23.2. tp_f16_w2 1.24.4, at exactly 1/3, does not
    trigger a rerun.
  * On the evidence above, the rerun cells are predicted, with near certainty, to fall below 0.95
    again.
* **So the verdict stays INCONCLUSIVE.**
  * It can change only if the registered rerun (Part A of the follow-up) puts ≥ 2/3 of every rerun
    cell's sessions above 0.95 and the gated verdict then equals the ungated one.
  * That outcome is not expected.
* The ungated result is FIRM (envelope entirely ≥ 0.336), so the registered seed-28 extension was never
  triggered.

### Q3. What the paper may write now (Exp 27 alone, verdict INCONCLUSIVE)

Rule 4.6 for INCONCLUSIVE says: "report the numbers and the failed condition; no mechanism sentence."
The mechanism wording stays at 23c's "thread-pool-level".

Permitted, sentence by sentence:

1. (standing claim, unchanged) The mechanism of the 1.24 step is described as "thread-pool-level"; no
   change.
2. (numbers, ungated, the paper's convention) "An isolated probe containing only the static graph's 56
   uint8 Transposes (quantize/dequantize-wrapped, net of a Transpose-free control) changed by +0.423 ms
   from ONNX Runtime 1.23.2 to 1.24.4 at 4 threads (session envelope +0.396 to +0.441 ms; 88% of the
   0.48 ms step) and by +0.008 ms at 1 thread; the same Transposes in float32 changed by +0.009 ms at 4
   threads."
3. (numbers, arms) "At 4 threads, the four of these Transposes whose moved block is two bytes wide
   changed by +0.427 ms and the other 52 by −0.003 ms (net of the control); the same four loop shapes
   changed by +0.408 ms in float16 and by +0.012 ms in float32 (raw)."
   * Only neutral verbs are allowed ("changed by"). "Carry", "reproduce", "explain" and "cause" are not.
4. (the failed condition) "The registered rule also requires the same verdict when sessions with core
   delivery (CPU time / (threads × median latency)) below 0.95 are excluded. That exclusion removes 19
   of 96 sessions, including every 4-thread session of both controls, so the gated control subtraction
   is undefined and the registered verdict is inconclusive; the all-Transpose arm kept all its sessions
   (+0.422 ms raw)."
5. (EXPLORATORY; it must carry this label and may appear only as a qualifier of sentence 4) "Post hoc
   and outside the registered rule: the 18 excluded 4-thread sessions are exactly those whose graph was
   predicted before timing to give the intra-op thread pool no work. Their workers are never woken, so
   CPU time equals one busy core (1.03–1.28 × latency, as at 1 thread) and core delivery sits near
   1/threads by construction (0.26–0.32). 29 of the 30 four-thread sessions with pool work, interleaved
   with them in the same randomised run, read 0.966–1.115. The gate therefore could not test these
   sessions for contamination, and this does not change the registered verdict."

NOT permitted under Exp 27 alone:

* PR #25680 or commit 42869ca in any sentence about the step, as "cause", "candidate" or "consistent
  with". Such a sentence is a mechanism sentence.
* The per-dispatch cost prediction (0.420 vs 0.427 ms). It exists only through H_M's count of 832
  dispatches.
* The within-file switch in core delivery from 1.23.2 to 1.24.4 (tp_u8_w2 0.28 -> 1.06). It is direct
  evidence for H_M's 0 -> 832 prediction, so it too is a mechanism sentence.
* The words LOCALISED or M-CONFIRMED, or "reproduced by the Transposes alone".

All of these remain fine in FINDINGS.md as the author's reading. The route to the 4.6 LOCALISED +
M-CONFIRMED sentence is the follow-up below.

### Q4. Follow-up, Experiment 27b. Registered here, BEFORE any timing, by the referee

**Purpose**
* (A) Carry out Exp 27's registered V3 rerun.
* (B) Re-test H_T and H_M at 4T on fresh sessions, with a validity check that means something for
  cells with no pool work.
* 27b is a new registered experiment. It does not rewrite Exp 27.

**Cells**
* All 8 Exp 27 variants, 4 threads only, 1.23.2 and 1.24.4, 3 fresh sessions each, seed 28.
* That is 48 cells. Exp 27 ran 96 in 6 min, so this takes about 3 min.
* Seed 28 is unused: the extension that would have used it was not triggered.
* The model files must be byte-identical to those Exp 27 verified (V1b), with these sha256 values:

| model | sha256 |
|---|---|
| tp_u8_all | af36d9c7e3f432885a23c1bccdf8152601bbeb24a4eae9354a941d1fba5e9562 |
| tp_u8_w2 | cf4c963beed312f87c470e70cb6c096bc64b948c4d4a3111dc2985e14e0c3500 |
| tp_u8_rest | a7d3b670ed527f07f5715077fc72c4b758384ba6584ef7ccea03491197597e5a |
| tp_u8_ctrl | 31d5509d54d5d01fd6e5f4b4dc7098cd38e2d6c0bc9efe0902ea4e4b9e75776b |
| tp_f32_all | b00b71939302b02150ee889cbf06955e7419773dc69c047ee23351822370f3f5 |
| tp_f32_ctrl | fc73f976339da9c69f04ac2ff42575fc3ff475a910c0a3461c4bd2db4353db97 |
| tp_f16_w2 | 964922b650d39233cbf434419efada13402f9dfdd62e4c294c8c4c060295c56e |
| tp_f32_w4 | a8b41525929c7c437edfe570ea988823edf8f564b79837a3105426b4fc2eea01 |

**Precondition (V4b)**: the same gates as `run_probe.sh`:
* on AC power;
* no WSL distro running;
* machine-wide PerfOS _Total load ≤ 15% (5 s average) within 10 one-minute checks;
* no other timing job. Otherwise abort; do not time.

**Command** (from the project root):

```
"C:/Users/User/Desktop/SEC/edge-ai-toolkit/.venv/Scripts/python.exe" scripts/paper/ort_version_bisect.py --scratch "C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-SEC/103ae351-d709-4d02-91fc-0b8da660fe58/scratchpad/" --versions 1.23.2,1.24.4 --threads 4 --repeats 3 --seed 28 --families "tprobe=results/transpose_probe/models:3,64,64" --variants tp_u8_all,tp_u8_w2,tp_u8_rest,tp_u8_ctrl,tp_f32_all,tp_f32_ctrl,tp_f16_w2,tp_f32_w4 --out results/paper/transpose_probe_followup.csv
```

**Analysis**
* Command: `python <scratchpad>/probe_followup_analyse.py results/paper/transpose_probe.csv results/paper/transpose_probe_followup.csv --out results/transpose_probe/verdict_followup.json`
* The script was written by the referee before timing.
* It reproduces Exp 27's ungated verdict and numbers exactly.
* Before any timing it was run on 7 SYNTHETIC follow-up scenarios (generator `probe_followup_synth.py`),
  and each gave the intended verdict: confirm; contaminated control; flat; control with pool work;
  rerun clears the gate; M-MIXED; not firm.
* Missing cells return INCONCLUSIVE. A NaN is never allowed to pass a check silently.

**Classes** (fixed now from `manifest.json` predicted_dispatches at 4T; observed values are never used):
* Z (0 dispatches): tp_u8_ctrl and tp_f32_ctrl on both releases; tp_u8_w2 and tp_f16_w2 on 1.23.2.
* P: the other 10 cells.

**Part A: Exp 27's registered V3 rerun, 0.95 gate unchanged**
* The rerun cells are Exp 27's cells with > 1/3 of their sessions below the gate, i.e. the 6 Z cells.
* If every rerun cell has ≥ 2/3 of its 27b sessions at cd ≥ 0.95:
  * recompute Exp 27's gated verdict with those passing 27b sessions in place of the excluded ones;
  * Exp 27's final is its ungated verdict if the two agree, and INCONCLUSIVE otherwise.
* If not, Exp 27 is final: INCONCLUSIVE (V3). This outcome is predicted.

**Part B: the 27b verdict**
* Quantities are exactly as in Exp 27 section 4.2 (S = 0.48 ms, 7% floor, 0.034 ms "small",
  81-combination envelope).
* 4T medians come from 27b and are ungated.
* 1T quantities come from Exp 27, fixed now: Δu8(1T) = +0.0083, Δw2(1T) = +0.0104, and 1T control drift
  −0.0033 (u8) and −0.0046 (f32). All 48 Exp 27 1T sessions passed the registered gate, so the gated and
  ungated values are the same.
* The per-dispatch cost (reported, not gating) uses the Exp 27 1T f32_w4 median.

Validity:
* V1b: the sha256 values above match.
* V2b: |D(tp_u8_ctrl, 4T)| and |D(tp_f32_ctrl, 4T)| ≤ max(7% of own 1.23.2 median, 0.034 ms).
* V3b: each session is checked by its class.
  * P session: cd = cpu_ms_per_inf / (4·p50) ≥ 0.95. This is the registered gate, unchanged.
  * Z session: busy = cpu_ms_per_inf / p50 must satisfy 0.95 ≤ busy < 1.5.
    * The lower bound is the same test the 1T cells pass: for one busy thread, cd at T = 1 equals busy.
      A main thread that other load preempts reads below 1.
    * The upper bound confirms that the workers stayed asleep, i.e. that the cell really had no pool
      work.
  * (i) No cell may have more than 1 of its 3 sessions failing its class check.
  * (ii) The verdict computed from class-valid sessions must equal the ungated verdict, both the
    primary band and the M class.
  * Otherwise INCONCLUSIVE. There are no further reruns.
* V4b: the precondition held.

Verdict:
* Apply Exp 27's section 4.4 and 4.5 tables unchanged.
* FIRM means the Δu8(4T) envelope lies in one band.
* There is no extension. If the verdict is not FIRM, report it as lying between the adjacent bands; the
  paper uses the LOWER band's 4.6 wording.

Reported, not gating (a test of this ruling's own explanation):
* All 18 Z sessions have cd < 0.5 and 0.95 ≤ busy < 1.5.
* ≥ 28 of the 30 P sessions have cd ≥ 0.95.
* If this fails, sentence 5 above must be withdrawn.

**Paper wording after 27b**
* The Exp 27 section 4.6 wording that matches 27b's verdict, with 27b's values and both its ungated and
  class-gated numbers plus the envelope.
* If 27b is LOCALISED (FIRM) + M-CONFIRMED, the 4.6 sentence may be used, including #25680, and ONLY
  together with this disclosure: "A first run (Exp 27) was inconclusive under its registered rule
  because its core-delivery gate cannot be met by cells whose graph gives the thread pool no work; the
  verdict is from a re-registered repeat (Exp 27b) that checks such cells for one fully delivered core
  instead."
* If 27b is INCONCLUSIVE, only sentences 1-5 above stand.
* If Part A unexpectedly clears, Exp 27's own 4.6 wording applies to Exp 27.

(Copied verbatim from the independent referee's ruling, written 2026-09-19 17:59, before Experiment 27b was timed.)

### Experiment 27b RESULT (2026-09-19, ~18:03): LOCALISED (FIRM) + M-CONFIRMED; Exp 27 stays INCONCLUSIVE

`results/paper/transpose_probe_followup.csv` (48 cells, 4 threads), analysed mechanically by
`scripts/paper/probe_followup_analyse.py` -> `results/transpose_probe/verdict_followup.json`.
* Part A (Exp 27's registered rerun): the six cells again read 0.23-0.31 core delivery, below the
  unchanged 0.95 gate, as the referee predicted. **Experiment 27 remains INCONCLUSIVE.**
* Part B (27b's own rule): validity V3b_i and V3b_ii hold (at most one failing session per cell:
  tp_u8_all@1.24.4 and tp_u8_ctrl@1.23.2); ungated and class-gated verdicts agree.
  **Final: LOCALISED (FIRM) + M-CONFIRMED.** Class-gated: delta_u8 at 4T +0.421 ms (envelope
  +0.392..+0.441), 88% of the 0.48 ms step; 1T +0.008 ms; float Transposes +0.009 ms; the four
  2-byte-block Transposes +0.417 ms, the other 52 +0.007 ms; float16 same shapes +0.435 ms; float32
  same shapes -0.016 ms; w2 at 1T +0.010 ms; dispatch-cost prediction 0.440 ms.
* The Z-class structural prediction did not hold in full (one Z session's CPU-time/latency ratio
  0.918 < 0.95), so, as the ruling requires, the post-hoc exploratory sentence 5 is WITHDRAWN.
* Verified by the author from the source before citing (2026-09-19): PR #25680 "FP16 inference
  performance improvement on CPU", commit 42869ca, adds `struct has_mlas_transpose<uint16_t> :
  std::true_type {};` to onnxruntime/core/framework/transpose_helper.cc.
* The paper may now use the registered 4.6 LOCALISED + M-CONFIRMED sentence naming #25680, with
  the referee's mandatory disclosure that Exp 27 was inconclusive and the verdict comes from the
  re-registered repeat 27b.

### Logged for the paper (2026-09-19, ~19:40): statistics the round-1 referee required to be on record

All recomputed by the author from stored CSVs; no timing.

1. **The 1.24 step of the byte-identical static 64 px MobileViT-XXS file, four independent timings**
   (4 threads, default spinning): sweep 1.311x, spin run 1.297x, coverage run 1.355x, scope run
   1.253x (median 1.24.4 / median 1.23.2). Pooled 12 v 12 sessions: every 1.24.4 session slower
   than every 1.23.2 session (max 1.737 < min 2.002 ms); exact Mann-Whitney p = 7.4e-7 (treats
   sessions as independent). Stratified by run, all four 3-v-3 strata are disjoint: exact p =
   (1/20)^4 = 6.25e-6 one-sided, 1.25e-5 two-sided, the minimum possible for four strata.
   Run-level factor mean 1.304, 95% t-interval 1.237-1.371 (n = 4 runs). Supersedes the earlier
   three-run pooled figure (factor 1.30, CI 1.27-1.35), which left out the scope run's 1.253x.
2. **Null gap behind the 7% floor** (`scripts/paper/stats_verify.py`, copied from the stats
   agent's scratch, seed as in the script): gap = |m_b/m_a - 1| between two 3-session medians
   resampled from the same cell. Sweep: p95 3.31%, p99 5.0%, P(gap > 7%) = 0.07%. Coverage run
   (noisier): p95 10.59%, P(gap > 7%) = 15.6%. The floor is conservative for the sweep, not for
   the coverage run.
3. **Harness between-run spread of byte-identical files** (results/*/sweep.csv, files matched by
   sha256; one-thread): latency max/min 1.004-1.174 for 17 of 18 file groups (e.g. ResNet-50 FP32
   1.174 over 4 runs), and 1.284 for MobileViT FP32, whose mobilevit_r64 build-run value (1.79 ms)
   is an outlier measured while other agents were active. Static speedup of byte-identical file
   pairs: ResNet-50 3.20-3.53x (4 runs), MobileNetV2-32 1.54-1.65x (14 runs), MobileViT-XXS 64 px
   0.92-0.95x (3 runs; 1.11x in the build run). A harness ratio carries up to about 10% between-run
   spread beyond the within-run floor.
4. **SHA-256 of the version-sweep files** (for the availability statement):
   MobileViT-XXS 64 px FP32 8bba9618966c5739448d67738b4a0051e99125d96d8bda2673746b40103289fa,
   static 911de4ee3257966074fe54952cf7ad5ed546798fe138fa8997efce9b11b1f629,
   dynamic 9c0025026d71ead82278eba5122264009deee1eb8b4c272a8422f786348794f4;
   DeiT-tiny 32 px FP32 41a49e3c3be8e24916e8ecb2c1506983e8acf51e984da42b2b7cdabe18c4a2e5,
   static 4e3cf32eadffc1cba4fc3cdfb63edf8157661c83189e79a21e9e79035aeb5d0c,
   dynamic e628f6c9ee4a6e5350822ef5f86ab2685518c82663a11efb69c2fb7538baf170;
   MobileViT-XXS 256 px static ae5d630c76ccc5d40983777666afc619250cfed3ae072ac9492ed354c68d32cd.

### Logged for the paper, round 2 (2026-09-19): values the round-2 referee required on record before the text changes

All recomputed by the author from stored CSVs; no timing. Numbered to continue the list above.

5. **MobileViT-XXS 256 px static file at one thread, scope run** (`results/paper/ort_version_scope.csv`,
   family mv256, threads 1, ungated medians of three sessions). Static sessions 20.044-20.300 ms on
   1.23.2, 21.055-22.061 ms on 1.24.4 (+5.5%, +1.12 ms, disjoint), 21.577-21.721 ms on 1.25.1
   (+7.2%), 21.239-22.739 ms on 1.30.0 (+5.2%): all nine later-release sessions are slower than all
   three 1.23.2 sessions. FP32 medians 20.878, 19.493, 18.343, 17.054 ms. One-thread static speedup
   **1.031x, 0.912x, 0.845x, 0.801x** on 1.23.2, 1.24.4, 1.25.1, 1.30.0. Below 7% at 1.24.4, so not a
   registered step; at 64 px the one-thread change of the static file stays small (sweep +1.4%,
   scope run -0.6%, probe +0.008 ms), so "needs more than one thread" holds at 64 px only.
6. **The four timings of the 64 px 1.24 step at four threads, and the Exp 27b probe as a share of
   each** (median 1.24.4 - median 1.23.2 of the static file, default-spin arm where there is one;
   probe = Exp 27b ungated delta_u8_4T 0.4374 ms, `results/transpose_probe/verdict_followup.json`):
   sweep (`ort_version_bisect.csv`) 0.4835 ms, share 0.905; spin-on arm (`spin_version.csv`, spin=1)
   0.4682 ms, 0.934; coverage run (`coverage_mobilevit.csv`) 0.5707 ms, 0.766; scope run
   (`ort_version_scope.csv`, mv64) 0.4082 ms, 1.072. The earlier "91%" divided by the rounded 0.48 ms;
   against the unrounded sweep step it is 90%, and the range over all four timings is 77-107%.
7. **Correction to item 3 (harness between-run spread).** Recomputed from results/*/sweep.csv, pairs
   matched by the sha256 prefixes of `model_file`, speedup = FP32 over INT8 `latency_p50_ms_t1`, every
   run's env onnxruntime 1.30.0. The "MobileNetV2-32 1.54-1.65x (14 runs)" pooled two different
   byte-identical pairs: c0980c89d299/ad770b998718, 6 runs, 1.564-1.642x (Table II's file, run
   classification_dyn), and 220419044659/7acbcd878bf5, 8 classification_prune_* runs, 1.537-1.646x.
   ResNet-50 19b355585c64/41a987f1bc46: 3.201-3.527x (4 runs, 10.2%). MobileNetV2 224 px
   16072166e095/e250141217af: 1.728x vs 1.522x (2 runs, 13.6%), the largest spread over all
   byte-identical convolutional pairs. YOLOv8n pairs: at most 7.0%. Item 3's "up to about 10%" is
   replaced by **up to 13.6% (MobileNetV2 224 px)**.

### Correction to 'Logged for the paper' item 3, and the probe-share record (2026-09-19, ~20:45)

Raised by the round-2 challengers and referee; verified by the author before editing.
* **Item 3 was imprecise.** It pooled two different MobileNetV2-32 file pairs and missed the
  largest spread. By sha256 pair (one-thread static speedup, onnxruntime 1.30.0, harness):
  MobileNetV2-32 c0980c89d299/ad770b998718 (Table II's files): 1.564-1.642x over 6 runs;
  MobileNetV2-32 220419044659/7acbcd878bf5 (pruning runs): 1.537-1.646x over 8 runs;
  ResNet-50 3.201-3.527x over 4 runs (10.2%); **MobileNetV2-224 16072166e095/e250141217af:
  1.728x vs 1.522x (13.6%)**; YOLOv8n pairs at most 7.0%. "Up to about 10%" is replaced by
  **up to 13.6% (MobileNetV2 at 224 px)**.
* **Probe share against each timing of the 64 px step** (27b ungated delta 0.4374 ms at 4T):
  sweep step 0.4835 ms -> 0.905; spin run 0.4682 -> 0.934; coverage run 0.5707 -> 0.766;
  scope run 0.4082 -> 1.072. The probe accounts for 77-107% of the step depending on the timing;
  quoting 91% against the rounded 0.48 ms mixed conventions.
* **256 px one-thread (from ort_version_scope.csv, recorded in the Exp 26 table):** the static file
  slowed 5.5% at 1.24.4 with disjoint sessions (20.04-20.30 vs 21.06-22.06 ms), +7.2% at 1.25.1,
  +5.2% at 1.30.0; its one-thread speedup fell 1.03x -> 0.91x -> 0.80x. "The regression needs more
  than one thread" holds at 64 px only.

### Logged for the paper, round 3 (2026-09-19): values the round-3 referee required on record before the text changes

All recomputed by the author from stored CSVs; no timing. Numbered to continue the lists above.

8. **Harness vs version protocol on 1.30.0, every convolutional cell of Table II** (harness: the
   Table II runs via `paper/figs/_fig_data.csv`; version protocol: the scope run,
   `ort_version_scope.csv`, 1.30.0, median of three sessions; the scope run uses the Table II model
   files). Autoencoder static at 4T: **1.066x harness (0.13815/0.12955 ms) vs 0.926x version protocol
   (0.1146/0.1238 ms)**, a 15% gap with the sign flipped. Under the version protocol static is below
   FP32 at 4T on all four releases (0.945/0.900/0.905/0.926x on 1.23.2/1.24.4/1.25.1/1.30.0), with
   disjoint sessions on 1.23.2 and 1.25.1. ResNet-50 dynamic at 1T: **14.79x slower (harness) vs
   16.28x (version protocol)**, a 10% gap. Every other convolutional cell differs by at most 6.5%
   (ResNet-50 static 4T 2.479x vs 2.639x the largest; the rest 0.1-4.5%). So the two protocols
   disagree by more than the 7% floor on three cells (MobileViT-XXS static 4T, recorded before;
   autoencoder static 4T; ResNet-50 dynamic 1T), not one.
9. **MobileViT-XXS 256 px, static over dynamic latency per release** (scope run, mv256, medians):
   1T 1.12/1.23/1.42/1.38, 4T 1.14/1.74/1.86/1.97 on 1.23.2/1.24.4/1.25.1/1.30.0 (1T 1.23.2 exactly
   20.2466/17.9980 = 1.125). On 1.23.2 the static and dynamic session ranges are disjoint at both
   thread counts: 4T 9.341-9.625 vs 8.277-8.467 ms, 1T 20.044-20.300 vs 17.952-18.076 ms. Dynamic
   over FP32: 1T 1.16/1.12/1.20/1.11x (1.11-1.20x), 4T 1.05/1.09/1.11/1.15x (1.05-1.15x). At 256 px
   dynamic led static on all four releases at both thread counts (1.12-1.97x); "static and dynamic
   within the 7% floor up to 1.23.2" holds at 64 px only.
10. **ResNet-50 static speedup at one thread, per release** (scope run, r50): 3.09x (90.316/29.260 ms)
    on 1.23.2, 3.02x on 1.24.4, 3.06x on 1.25.1, 3.07x on 1.30.0. Static beats FP32 by 3.02-3.09x on
    all four releases, including 1.23.2, the release of Shin's single-release contrast.
11. **1-D autoencoder, absolute latency cut at 1.24.4** (scope run, median 1.24.4 minus median
    1.23.2): FP32 1T -0.0937 ms (-38.7%), FP32 4T -0.0875 ms (-44.1%), static 1T -0.0849 ms (-51.5%),
    static 4T -0.0868 ms (-41.3%). A near-constant 0.085-0.094 ms cut in every arm; the relative cut
    (39-51%) varies only because the bases differ.
12. **YOLOv8n 256 px four-thread static speedup over the four releases** (scope run, y256): 1.055x
    (7.2046/6.8321 ms) on 1.23.2, 1.579x (7.7465/4.9062) on 1.24.4, 1.312x (6.1624/4.6978) on 1.25.1,
    1.239x (6.4748/5.2270) on 1.30.0, as in the Exp 26 RESULT. From 1.24.4 to 1.25.1 FP32 fell 20.4%
    and static 4.2% (the independent rule's registered consequence: the speedup shrank through the
    FP32 baseline). From 1.25.1 to 1.30.0 static rose 11.3% (sessions overlap) and FP32 5.1%.
13. **MobileViT-XXS 256 px at one thread, the two terms** (adds to item 5): FP32 20.878 -> 19.493 ms
    (-6.6%; sessions 19.376-21.082 vs 19.260-19.580 ms, overlapping); static 20.247 -> 21.366 ms
    (+5.5%, disjoint). The static term alone would give 1.031/1.055 = 0.977x; the FP32 term
    (x 0.934) takes it to 0.912x. At four threads the step is 13.994 - 9.562 = 4.43 ms.
14. **Table IV star rule.** The +3.8% four-thread step of DeiT-tiny 224 px dynamic has disjoint
    sessions (8.338-8.499 vs 8.648-9.243 ms) but is below the 7% floor; Table IV now stars a step
    only when it exceeds 7% and its sessions are disjoint (+25.3, +46.4, -28.2).
15. The Exp 23c profiling-inflation factor is corrected under Exp 23c above (8.4x / 7.5x, not ~9x).


---

## Experiment E2 -- full-model counterfactual (REGISTERED 2026-09-20T21:59:08+08:00, BEFORE ANY TIMING)

Designed by an independent agent, attacked by an independent reviewer (verdict MAY-RUN-WITH-FIXES,
six re-registrations required), revised by the designer. The claim's author designed none of it and
adjudicates none of it. Full protocol archived verbatim at `results/paper/e2_design_registered.md`
(sha256 387544129b6c4ade9c67ee917df5c82df4195688f443f1c2f3ddba8c3d8e3491); the paragraphs below are its registered rule, copied verbatim.

Models: `results/e2_counterfactual/models/` -- `p00_int8.onnx` (byte-identical to
`results/classification_mobilevit/models/p00_int8.onnx`, sha256 911de4ee...), `p00_int8_e2.onnx`
(sha256 8a9274e889c2...), `p00_int8_sham.onnx` (sha256 b2a3aec6bd81...), `p00_int8_plac.onnx`
(sha256 360d68688885...).

Pre-timing verification, already PASS: outputs bit-identical over 8 inputs x 3 variants x 2 releases
x 2 optimization levels (max|diff| = 0.000e+00); optimized graphs under both releases have the
identical 698-node name multiset for e2 and sham (placebo +1 Concat, CSE un-merge, recorded);
Transpose bytes 105,984 in all four; predicted 4-thread dispatches orig 864, e2 32, sham 864,
plac 864.

## 5. Registered decision rule (fixed BEFORE timing)

### 5.1 Cells

variants {p00_int8, p00_int8_e2, p00_int8_sham, p00_int8_plac} x releases {1.23.2, 1.24.4} x threads
{4 = primary, 1 = control} = **16 cells, 5 fresh sessions each = 80 sessions**, seed 29, randomised
order across everything, one fresh subprocess per session, P-core pinning (`pick_cpus(t,"auto")`),
1.5 s warmup then >= 2 s and >= 40 runs, p50 per session, batch 1, default spinning,
ORT_SEQUENTIAL, ORT_ENABLE_ALL. Expected wall time about 7-8 minutes.

Seeds: 29 = this run; 30 = reserved for the V7a single-cell re-run; 31 = reserved for the extension.
(27 = Exp 27, 28 = Exp 27b.)

### 5.2 Quantities

* `m(v,r,T)` = median of the session p50s of variant v on release r at T threads, **ungated** (the
  paper's reporting convention). Everything is also computed gated (V7b).
* `D(v,T) = m(v,1.24.4,T) - m(v,1.23.2,T)` - the release step of variant v.
* `R(v,T) = m(v,1.23.2,T) - m(p00_int8,1.23.2,T)` - the rewrite's own cost, on the release where
  nothing dispatches.
* **`f = (D(orig,4T) - D(e2,4T)) / D(orig,4T)`** - the share of the observed step that rewriting the
  four nodes removes. The estimand.
* `keep_sham = D(sham,4T) / D(orig,4T)`, `keep_plac = D(plac,4T) / D(orig,4T)`.
* `numerator_1T = D(orig,1T) - D(e2,1T)` - the estimand's numerator at one thread, where neither file
  dispatches on either release. It must be null.
* Envelope: min and max over all combinations of one session p50 per cell (5^4 = 625 for `f` with 5
  sessions per cell; 10^4 = 10,000 after an extension). A conservative session-resampling range, not
  a confidence interval - the device Exp 27/27b used. Combinations whose denominator `D(orig,4T)` is
  not positive are skipped and counted in `skipped_nonpositive_denominator`; they are never divided
  by. The 5th-95th percentile of the same combinations is computed and REPORTED, and may never
  upgrade a verdict or a firmness.

### 5.3 Constants

* `S = 0.48 ms`, the registered step (ort_version_bisect.csv 4T medians 2.037 - 1.553), i.e. +31%.
  `S` is used only in the fixed thresholds below; every sentence the paper writes about the size of
  the step uses **this run's** `D(orig,4T)`, never 0.48 (FINDINGS "Logged for the paper" item 6
  records the step measured at 0.408-0.571 ms across four runs).
* `FLOOR_REL = 0.07`, the study's between-session latency floor. `SMALL = 0.07·S = 0.034 ms`, an
  ABSOLUTE constant, exactly as Exp 27 section 4.5 and Exp 27b register it. A cell's own floor is
  `0.07 · m(v,1.23.2,T)`: about 0.109 ms at 4T and 0.103 ms at 1T for this file.
* `F_CONFIRM = 0.70` - the share of the step the rewrite must remove (the threshold registered in 23c
  and in Exp 27's LOCALISED band).
* `F_PARTIAL = 0.30` - from Exp 27's `Drest < 0.30·Du8` clause.
* `CTRL_KEEP = 0.70` and `CTRL_KEEP_HI = 1/0.70 = 1.43` - the two-sided band the sham and the placebo
  must keep the step within.
* **What CONFIRMED tolerates, stated plainly** (revision 1 had this backwards): `f >= 0.70` permits a
  residual `D(e2,4T)` of up to `0.30 · D(orig,4T)` ~ 0.144 ms, while the floor test permits
  `max(0.034, 0.07·m(e2,1.23.2,4T))` ~ 0.110 ms. **The FLOOR is the binding constraint, not the
  share.** So ROOT-CAUSE CONFIRMED on medians tolerates a residual step of up to 7% of the rewritten
  file's 1.23.2 latency, i.e. up to about 23% of the step. That is why firmness (5.5) now also
  carries the floor condition, and why W1 states the residual explicitly.

### 5.4 Validity conditions

Gating unless marked REPORTED. A gating failure gives INCONCLUSIVE (or CONFOUNDED for V3), never a
verdict.

* **V1 - pre-timing verification.** `manifest.json` present; the four files' sha256 still match it;
  `--verify-static` all-intended; `--verify-outputs` bit-identical everywhere; `--verify-executed`
  only-intended, same Transpose name set, and node ids identical across releases. All re-checked by
  the analysis script from the stored JSON and by re-hashing the files. **Already PASS.**
* **V2 - the step reproduced in this run.** `D(orig,4T) >= 0.70·S = 0.336 ms` AND
  `D(orig,4T) > 0.07·m(orig,1.23.2,4T)`. If the step is not there, the counterfactual tested nothing.
* **V3 - the controls keep the step, TWO-SIDED.**
  `0.70 <= keep_sham <= 1.43` AND `0.70 <= keep_plac <= 1.43`. Either breach - losing the step or
  AMPLIFYING it - gives **CONFOUNDED**: the edit, not the dispatch path, moved the number. The
  session envelopes of `keep_sham` and `keep_plac` are reported with the verdict.
* **V4 - the four-thread-only signature.**
  * **V4a**: `|D(v,1T)| < 0.034 ms` for all four variants. ABSOLUTE, as registered in Exp 27 4.5 and
    27b, not an own-floor bound. (Revision 1 used `max(0.034, 0.07·own median)` = 0.103 ms, three
    times looser than the precedent and ten times the largest 1-thread change ever recorded for this
    file: +0.010 ms in 27b, +1.4% in the sweep. The own-floor value is REPORTED beside V4a.)
  * **V4b**: `|numerator_1T| < 0.034 ms`. New in revision 2, and the sharper test: at one thread
    there is no pool on either release, so removing 832 would-be dispatches must remove nothing. This
    is the estimand's own negative control.
* **V5 - the rewrite's own cost on 1.23.2. REPORTED, NOT VOIDING.**
  `|R(e2,T)| <= 0.07·m(orig,1.23.2,T)` (about 0.109 ms at 4T, 0.103 ms at 1T) for T in {1,4}. A
  breach is recorded as a caveat in `verdict.json` and must be quoted in the paper sentence; it does
  not change the band. Reason (C3): `R` is `T_elt - T_simple`, which cancels identically out of `f`,
  so it is the rewrite's SIZE, not a bound on bias. Revision 1 made it voiding at
  [-0.034, +0.072] ms - a band inside the study's own noise, which could have destroyed a valid
  experiment.
* **V5b - thread-consistency of that cost. GATING.**
  `|R(e2,4T) - R(e2,1T)| <= max(0.034, 0.07·m(orig,1.23.2,4T))` = about 0.109 ms. The added serial
  cost is thread-independent by construction (`DoUntypedTranspose` never sees the pool), so a large
  thread-dependent difference means the rewrite did something thread-related that was not designed,
  and the run cannot be read. The bound is wide on purpose: it combines four session medians.
* **V6 - machine precondition.** AC power, no WSL distro running, machine-wide
  `Win32_PerfFormattedData_PerfOS_Processor` `_Total` <= 15% over 5 s within 10 one-minute checks, no
  other timing job. `run_e2.sh` aborts otherwise and writes `results/e2_counterfactual/
  precondition.json` only when all of them passed, AFTER the V1 verification commands, so the
  recorded load describes the machine the timing actually ran on. If that file is absent, the
  analysis returns INCONCLUSIVE.
* **V7 - core delivery.** `cd = cpu_ms_per_inf / (threads · p50)`, gate 0.95.
  * (a) GATING, and **only for the eight FOUR-THREAD cells**: at most 1 of 5 sessions of any such
    cell may read `cd < 0.95`. At one thread `cd = cpu_ms/(1·p50)` is a CPU-accounting ratio, not a
    core-delivery measure (this project has recorded 0.894-1.229 at 1T on healthy cells), so the
    eight 1-thread cells are REPORTED, not gated; more than one 1-thread session below 0.95 raises a
    caveat. A contaminated 1-thread cell can only void the run through V4; it can never create a
    verdict.
  * The referee's structural exemption from the Exp 27 ruling (cells whose graph gives the pool no
    work) is **explicitly not invoked**: every cell here is the full static MobileViT-XXS graph
    (35 QLinearConv, 54 QLinearMatMul, 57 QLinearAdd), i.e. abundant parallel work at 4 threads, and
    the same file already reads cd ~ 1.0 in four earlier runs.
  * If a 4-thread cell fails (a), it is re-run ONCE, restricted to that cell, with `--seed 30` and
    its OWN output file (section 6); the re-run REPLACES that cell's seed-29 rows via
    `--replace-cell`, it is never pooled with them. If it fails again the verdict is INCONCLUSIVE.
  * (b) GATING: the gated verdict must equal the ungated verdict - same band. The gated arm drops
    below-gate sessions from 4-thread cells only, for the reason in (a). Both arms are reported.
  * REPORTED, NOT GATING: the counterfactual removes 832 wakeups on 1.24.4, so `p00_int8_e2` at 4T on
    1.24.4 may read a LOWER core delivery than the original. That is the predicted mechanism showing
    in the gauge. It may be reported as such only if V7(a) still holds; it may never be used to
    excuse a cell that fails V7(a).
* **V8 - completeness.** All 16 cells present with >= 5 sessions each, no NaN. A missing cell or a
  NaN is INCONCLUSIVE; it never passes a comparison silently. The verdict records the exact file set,
  the per-file row count, the rows used and any replaced cell.

### 5.5 Verdict bands and firmness

| verdict | condition |
|---|---|
| **ROOT-CAUSE CONFIRMED** | `f >= 0.70` AND the rewritten file has no release step of its own: `|D(e2,4T)| < max(0.034 ms, 0.07·m(e2,1.23.2,4T))` (~0.110 ms) |
| **PARTIAL** | `0.30 <= f < 0.70`; or `f >= 0.70` while `D(e2,4T)` still exceeds that floor |
| **REFUTED** | `f < 0.30` |
| **CONFOUNDED** | V3 fails (a control lost the step or amplified it) |
| **INCONCLUSIVE** | any of V1, V2, V4a, V4b, V5b, V6, V7a, V7b, V8 fails |

**Firmness.** FIRM requires BOTH:

1. the 625-combination envelope of `f` lies entirely in the same band as the median `f`; and
2. **new in revision 2** - if the band is ROOT-CAUSE CONFIRMED, the envelope of `D(e2,4T)` also lies
   inside `max(0.034 ms, 0.07·m(e2,1.23.2,4T))` at BOTH ends.

Otherwise PROVISIONAL, and the paper uses the LOWER band's wording (the convention 27b used):
a PROVISIONAL ROOT-CAUSE CONFIRMED is written with W2, not W1. Condition 2 exists because the
CONFIRM band's floor test tolerates a residual of up to ~23% of the step on medians alone; without an
envelope, "step shrank rather than vanished" would read as FIRM about three times in four.

### 5.6 The two registered follow-up runs, and nothing else

Both are registered NOW with exact triggers, cells, seeds and commands. Neither may be run for any
other reason, and there is no third run.

* **V7a re-run** - trigger: exactly one 4-thread cell has more than 1 of its 5 sessions below the
  0.95 gate. Cells: that one cell. Seed 30. Output: `results/paper/e2_counterfactual_rerun_seed30.csv`
  (never the primary CSV). Analysis: both CSVs with `--replace-cell <variant>,<ort>,4`, which drops
  that cell's seed-29 rows. If the re-run cell fails again: INCONCLUSIVE.
* **Extension** - trigger: the run is otherwise valid and the firmness is PROVISIONAL. Cells: all
  eight 4-thread cells (so no arm becomes asymmetric), 5 more sessions each, seed 31, about 4
  minutes. Output: `results/paper/e2_counterfactual_ext_seed31.csv`. Analysis: both CSVs POOLED, 10
  sessions per 4-thread cell, envelope over 10^4 combinations. The band and firmness are recomputed
  on the pooled data and that is final. If it is still PROVISIONAL, the result is reported as lying
  between the two adjacent bands with the envelope as its range, and the paper uses the LOWER band's
  wording. This restores the extension clause Exp 27 section 4.5 registered and revision 1 wrongly
  removed; its trigger is firmness alone, never the band, so it cannot be used to fish for a
  preferred verdict.

### 5.7 Reported, not gating

`D(v,T)` and `m(v,r,T)` for all 16 cells, ungated and gated, with envelopes and the 5-95 percentile;
`R(e2,T)`, `R(sham,T)`, `R(plac,T)` with the own-floor values; the 1-thread own floors beside V4a;
`numerator_1T`; the implied cost per removed dispatch `(D(orig,4T) - D(e2,4T))·1000/832` in
microseconds, to compare with Exp 27b's 0.51-0.53 us; per-cell core delivery min/max and the count
below the gate, with 4-thread cells marked gating and 1-thread cells marked reported; the step in
percent of each file's own 1.23.2 median; the exact CSV file set and row counts.

### 5.8 The analysis script was tested on synthetic data, which was then deleted

`scripts/paper/e2_analyse.py` (revision 2) was run before any timing on SYNTHETIC scenarios
(generator `<scratchpad>/e2/synth2.py`, no real measurement involved). All 25 checks gave the
intended result:

| scenario | intended | got |
|---|---|---|
| step removed in full (f = 0.99) | ROOT-CAUSE CONFIRMED (FIRM) | as intended |
| step not removed (f = 0.00) | REFUTED | as intended |
| half the step removed (f = 0.49) | PARTIAL | as intended |
| sham LOSES the step (keep 0.11) | CONFOUNDED | as intended |
| sham AMPLIFIES the step (keep 1.62) | CONFOUNDED (new in rev 2; revision 1 passed it) | as intended |
| placebo loses the step | CONFOUNDED | as intended |
| the step does not reproduce (0.09 ms) | INCONCLUSIVE (V2) | as intended |
| a 0.30 ms 1-thread step | INCONCLUSIVE (V4a) | as intended |
| a 0.05 ms 1-thread step | INCONCLUSIVE (V4a) (revision 1 PASSED this) | as intended |
| D(orig,1T) +0.030, D(e2,1T) -0.030 (each within V4a) | INCONCLUSIVE (V4b) | as intended |
| the rewrite is 0.30 ms FASTER on 1.23.2 at both thread counts | verdict stands, with the V5 caveat (revision 1 voided it) | as intended, caveat present |
| rewrite cost +0.25 ms at 4T but +0.02 ms at 1T | INCONCLUSIVE (V5b) | as intended |
| 2 of 5 sessions of a 4-thread cell below the gate | INCONCLUSIVE (V7a) | as intended |
| 2 of 5 sessions of a 1-thread cell below the gate | verdict stands, with the 1-thread caveat | as intended, caveat present |
| residual 0.096 ms (f = 0.79), inside the floor on medians | CONFIRMED band but PROVISIONAL -> W2 | as intended |
| only 3 sessions per cell | INCONCLUSIVE (V8) | as intended |
| `precondition.json` absent | INCONCLUSIVE (V6) | as intended |
| foreign `family=mobilevit` rows mixed into the CSV | ignored by the default `--family e2`; identical verdict and row count | as intended (and `--family all` proves the filter is doing it) |
| a V7a re-run CSV passed with `--replace-cell` | the cell has 5 sessions, the failing ones dropped | as intended (5 rows dropped) |
| the same two CSVs WITHOUT `--replace-cell` | still INCONCLUSIVE (the failure is not silently diluted) | as intended |
| an extension CSV pooled with the primary | 10 sessions per 4-thread cell, no crash | as intended |
| a session pairing that makes `D(orig,4T)` exactly 0 | skipped and counted, no ZeroDivisionError | as intended (125 skipped) |

The synthetic CSVs, the synthetic verdicts and the synthetic `precondition.json` were DELETED after
the test (`<scratchpad>/e2/synth2/` removed). `results/e2_counterfactual/{verdict,precondition}.json`
and `results/paper/e2_counterfactual.csv` do not exist, so the real run must write its own.

---

## Referee ruling on Experiment E2 (full-model counterfactual), 2026-09-20

The referee (owner tag: Jess) did not design E2, did not run it and is not the author of the claim it
tests. This ruling is binding on what the paper may say about E2. Sources read: the registered
protocol `results/paper/e2_design_registered.md` (revision 2, registered 2026-09-20T21:59:08+08:00,
before any timing) and its copy in this file; `results/paper/e2_counterfactual.csv` (80 rows);
`results/e2_counterfactual/verdict.json`, `precondition.json`, `manifest.json`, `static_check.json`,
`output_check.json`, `executed_check.json`; `scripts/paper/e2_analyse.py`; and the runner output
`<scratchpad>/tasks/bd018d3ma.output`. Nothing was timed for this ruling, no benchmark was run and
onnxruntime was not imported. Every number below was recomputed from the CSV with an independent
script, not copied from verdict.json.

### 1. Numbers verified independently from the CSV

80 rows, family `e2` only, 16 cells x 5 sessions, seed 29, no NaN, no missing cell (V8 holds).

| variant | m(1.23.2,4T) | m(1.24.4,4T) | D(4T) ms | D(4T) % | D(1T) ms |
|---|---|---|---|---|---|
| p00_int8 | 1.6153 | 2.0525 | +0.4372 | +27.07% | +0.0112 |
| p00_int8_e2 | 1.5975 | 1.6027 | +0.0052 | +0.33% | -0.01275 |
| p00_int8_sham | 1.6011 | 2.1205 | +0.5194 | +32.44% | **+0.03485** |
| p00_int8_plac | 1.5730 | 2.0506 | +0.4776 | +30.36% | +0.02765 |

* `f = (D(orig,4T) - D(e2,4T)) / D(orig,4T)` = **0.98811**. Confirmed.
* `keep_sham` = 1.18801, `keep_plac` = 1.09241; both inside the registered two-sided 0.70-1.43
  window, so V3 holds on medians. Confirmed.
* `D(e2,4T)` = 0.00520 ms against the registered floor `max(0.0336, 0.07*1.5975)` = 0.11182 ms.
  Confirmed.
* step removed = 0.43200 ms; implied cost per removed dispatch = 0.43200*1000/832 = **0.51923 us**,
  which does sit inside Experiment 27b's independent 0.51-0.53 us. Confirmed.
* `numerator_1T` = D(orig,1T) - D(e2,1T) = **+0.02395 ms** against the 0.0336 ms bound: V4b holds.
  Confirmed.
* V2: D(orig,4T) = 0.4372 >= 0.70*0.48 = 0.336 and > own floor 0.11307. Holds.
* V5 (reported): R(e2,4T) = -0.01780 ms, R(e2,1T) = +0.03395 ms, both inside their own floors
  (0.11307 / 0.10689), so **no V5 caveat is recorded and W6 is not owed**. V5b: |R(e2,4T)-R(e2,1T)| =
  0.05175 <= 0.11307. Holds.
* V1 (four sha256 re-hashed, static / output / executed checks) and V6 (AC power, no WSL, machine-wide
  load 28% -> 17% -> 15% over three one-minute checks, recorded after the V1 commands) both pass.
* f envelope over 625 combinations: min 0.56910, max 1.38326, 0 skipped for a non-positive
  denominator; p05 0.69985, p95 1.30638. D(orig,4T) envelope 0.34950..0.62060; D(e2,4T) envelope
  -0.13395..+0.15060.
* Core delivery: **0 of 40 four-thread sessions** below the 0.95 gate (V7a holds, worst 4T cell 0/5).
  **9 of 40 one-thread sessions** below 0.95: p00_int8 1.23.2 2/5, p00_int8 1.24.4 1/5, sham 1.23.2
  1/5, **sham 1.24.4 3/5**, plac 1.23.2 2/5, the other three 1T cells 0/5. Reported, not gating, per
  V7(a).
* Session spread of the eight 1T cells (max-min of the five session p50s): orig 1.23.2 0.0551, orig
  1.24.4 0.0386, e2 1.23.2 0.0520, e2 1.24.4 0.0105, sham 1.23.2 0.0358, **sham 1.24.4 0.1525**, plac
  1.23.2 0.0415, plac 1.24.4 0.0345. The sham 1.24.4 1T cell is both the widest cell in the run and
  the cell with the most below-ratio sessions; its two slowest sessions (1.5972 and 1.6613 ms) are two
  of its three below-ratio sessions.

**Corrections to the summary this referee was given.** Four points, none of which change the ruling:

1. The sham's 1-thread step is **0.03485 ms**, which exceeds the operative bound `SMALL = 0.07*0.48 =
   0.0336 ms` by **0.00125 ms (3.7% over)**, not by 0.001 ms. Against the protocol's prose rounding
   ("0.034 ms") it is over by 0.00085 ms. **V4a fails under both readings**, so the 0.0336-vs-0.034
   ambiguity is immaterial.
2. The placebo's 1-thread step is +0.02765 ms, i.e. **82% of the bound**. It passes, but it is not
   "well inside"; two of the four variants sit above 0.027 ms at one thread. The one-thread cells of
   this run are not quiet.
3. The p05 of f is **0.69985**, marginally BELOW the 0.70 CONFIRM edge. Quoting the p05-p95 as
   "0.700-1.306" rounds across a decision boundary: the 5-95 percentile band straddles the band edge
   too. (It may never upgrade a verdict or a firmness in any case, per section 5.2.)
4. "Both controls kept the step" is true on medians only. The control envelopes, which section 5.4
   requires to be reported with the verdict, are keep_sham 0.50741..1.78956 and keep_plac
   0.54222..1.74192 - outside the 0.70-1.43 window at both ends.

Everything else in the summary is correct as stated.

### 2. Binding verdict: INCONCLUSIVE stands as registered, on V4a

**The verdict is INCONCLUSIVE (a 1-thread step appeared), firmness PROVISIONAL, paper wording W5.**
The mechanical verdict is upheld in full.

Reasoning:

1. **V4a fails on the plain registered text.** Section 5.4: "**V4a**: `|D(v,1T)| < 0.034 ms` for all
   four variants. ABSOLUTE, as registered in Exp 27 4.5 and 27b, not an own-floor bound." The sham
   reads 0.03485 ms. The test is a strict inequality against an absolute constant and it fails.
2. **The protocol admits no narrower reading of V4a.** The words "for all four variants" are in the
   registered text, and the scope was the subject of the reviewer's objection 1, which the designer
   recorded as "**ACCEPTED in full**, and strengthened: V4a is the absolute 0.034 ms of Exp 27/27b,
   and a new V4b requires the estimand's own numerator to be null at one thread". V4b exists precisely
   so that the estimand has its own one-thread control; that the protocol kept BOTH forecloses the
   argument that V4a was only ever meant to police the estimand's own pair. V4a is listed in the 5.5
   band table as an INCONCLUSIVE trigger, and the 5.4 preamble states: "Gating unless marked REPORTED.
   A gating failure gives INCONCLUSIVE (or CONFOUNDED for V3), never a verdict." There is nothing to
   quote that narrows it.
3. **The protocol anticipated this exact failure mode and deliberately left it without a remedy.**
   Section 5.4 V7(a): "A contaminated 1-thread cell can only void the run through V4; it can never
   create a verdict." The cell that fails V4a (sham, 1.24.4, 1T) is precisely a 1-thread cell with 3
   of 5 sessions below the 0.95 ratio. The designer wrote down, before timing, that such a cell voids
   the run through V4 - and registered no re-run for it. The registered V7a re-run is available only
   when "exactly one 4-thread cell has more than 1 of its 5 sessions below the 0.95 gate"; zero
   4-thread cells are below the gate, so that remedy is not triggered and may not be repurposed to a
   1-thread cell.
4. **The strength of the primary result is not a permitted input.** f = 0.98811, a residual of 0.0052
   ms against a 0.1118 ms floor, and a 0.519 us per-dispatch cost matching 27b are all real and all
   recorded. Under section 5.4 they cannot produce a verdict once a gating condition has failed, and
   this referee may not relax a registered bound because the result would otherwise be nice.
5. **No remedy would rescue the run in any case, and the tempting one would make it worse.** For the
   record only, and explicitly NOT as a decision input: if the 0.95 ratio were applied to the
   one-thread cells as well - which section 5.4 V7(a) and the analysis script both forbid ("The gated
   arm drops below-gate sessions from 4-thread cells only") - the sham's 1-thread step would fall to
   +0.0304 ms (pass) but the placebo's would rise to +0.0453 ms (fail), so V4a would still fail, on a
   different variant. Under single-session leave-one-out on the two sham 1T cells, 8 of the 10 drops
   still fail V4a. The failure is not one outlier session and there is no reading of the existing data
   that clears V4a.

The failure is most plausibly one-thread measurement noise rather than a real one-thread release step
- a real one would have to appear in the original file too, and D(orig,1T) = +0.0112 ms with V4b at
+0.0240 ms both say it does not. That plausibility is the reason the result is INCONCLUSIVE rather
than CONFOUNDED or REFUTED. It is not a reason to pass the condition. E2 as run returns no verdict.

### 3. The registered extension run is NOT triggered and may not be run

Section 5.6 registers its trigger as: "**Extension** - trigger: the run is otherwise valid and the
firmness is PROVISIONAL." Both conjuncts are required. The firmness is PROVISIONAL (the f envelope
0.569..1.383 crosses the 0.70 edge), but the run is **not otherwise valid**: V4a, a gating condition,
failed, and under 5.4 and 5.5 that makes the run INCONCLUSIVE. A run that a gating condition has
voided is not "otherwise valid" under any reading this protocol supports; the protocol does not define
the phrase in a glossary, and on the stricter reading the trigger is unmet. **Ruling: the extension
run is not triggered. It may not be run under the registration**, and section 5.6 closes the door on
anything else: "Neither may be run for any other reason, and there is no third run."

Two further points make this unambiguous:

* **The extension is incapable of curing the failure.** Section 5.6 fixes its cells as "all eight
  4-thread cells ... 5 more sessions each, seed 31". The one-thread cells are untouched, so
  D(sham,1T) would remain 0.03485 ms and V4a would fail identically on the pooled data. In
  `e2_analyse.py` the V4a check precedes every band assignment, so the pooled band would again be
  "INCONCLUSIVE (a 1-thread step appeared)" whatever the new four-thread sessions showed. The
  extension could only narrow f's envelope, which is a firmness statistic for a verdict this run does
  not have.
* **The analysis script's printed trigger line is an implementation defect, not a registration.**
  `e2_analyse.py` computes `ext = (u["firmness"] == "PROVISIONAL")` and reports
  `extension_run_triggered = bool(ext and hard)`, where `hard` covers only V1, V6, V8, V7a and
  gated/ungated agreement - it omits V2, V3, V4a, V4b and V5b. So the flag reads True on a run that
  the same script has just declared INCONCLUSIVE on V4a. The script's own docstring says it "only
  executes" the rule and "contains no judgement of its own"; where its code and the registered text
  disagree, the registered text governs. The printed line
  "> PROVISIONAL: the registered extension run (section 5.6) is triggered" is **not binding and is
  overruled**. The defect should be recorded but the seed-29 verdict string itself is unaffected: the
  final verdict is correctly INCONCLUSIVE.

If a seed-31 four-thread run is nonetheless executed out of curiosity, it is unregistered exploratory
data: it may not be pooled into `e2_analyse.py` for a reported number, it may not be used to restate
f, keep_sham, keep_plac or the firmness, and no sentence of the paper may rest on it.

**What must be reported either way** (section 5.7, unchanged by this ruling): the 16 cell medians and
D(v,T) ungated and gated, with the envelopes and the 5-95 percentile; R(e2,T), R(sham,T), R(plac,T)
with their own-floor values; the one-thread own floors beside V4a; numerator_1T; the implied
0.519 us per removed dispatch; per-cell core delivery min/max and the count below the gate, with the
four-thread cells marked gating and the one-thread cells marked reported; the step as a percent of
each file's own 1.23.2 median; and the exact CSV file set and row counts. The mandatory caveat that
verdict.json records - that a 1-thread cell has more than one session below 0.95 - must travel with
them, as must the V4a failure and its size (0.03485 ms against 0.0336 ms). These quantities are
REPORTED, not asserted: they may appear in an E2 data table or appendix carrying the INCONCLUSIVE
label, and may not be lifted into prose as findings (see section 4). Experiment 27b's mandatory
disclosure about Experiment 27 continues to travel wherever the 27b result is cited; E2 does not
remove it.

### 4. The exact sentences the paper may and may not use

**PERMITTED - W5, the only outcome sentence available.** Either of these two ASCII forms, and nothing
else in this slot:

> A full-model counterfactual was attempted and its registered validity condition V4a was not met, so
> it returns no verdict. The mechanism remains traced to a code-level cause.

or, with the "(state which)" clause of W5 filled in:

> A full-model counterfactual was attempted and its registered validity condition V4a (the one-thread
> release step of every variant must be smaller than 0.034 ms) was not met - the sham control's
> one-thread step measured 0.0349 ms - so it returns no verdict. The mechanism remains traced to a
> code-level cause.

**PERMITTED - the design-facts paragraph of section 7, verbatim, which is usable in every branch
because it is verified and timing-independent:**

> The rewritten file is an exact rewrite: its outputs are bit-identical to the original's on both
> releases and at both optimization levels, its optimized graph has the same 698 nodes with the same
> names under both ONNX Runtime 1.23.2 and 1.24.4, it moves the same 105,984 Transpose bytes, and the
> only difference is the shape and permutation of four uint8 Transposes, which by the dispatch rule
> in core/framework/transpose_helper.cc issue 832 fewer thread-pool dispatches per inference on
> 1.24.4 and the same number on 1.23.2.

**BARRED.**

* W1, W2, W3 and W4, in whole, in part or paraphrased. In particular the paper may NOT write "Part of
  the step is therefore carried by those four nodes and part is not" (W2), nor any sentence of the
  form "rewriting those four Transposes ... removes x ms of the y ms step at four threads".
* The words "root cause", "proves" and "the cause of", per the section 7 barred list, which permits
  them only with W1. **Because W1 is not available, the question of a scope limitation travelling with
  "root cause" does not arise: there is no permitted use of the phrase in connection with E2 or with
  the 1.23.2 -> 1.24.4 regression.** "Traced to a code-level cause" remains available and is what W5
  itself says.
* Presenting this run's measured quantities as findings in prose: f = 0.9881 or "98.8% of the step",
  the 0.4320 ms removed, the 0.0052 ms residual against the 0.1118 ms floor, keep_sham = 1.188 or
  keep_plac = 1.092 as evidence the counterfactual is controlled, and the 0.519 us per removed
  dispatch as confirmation of Experiment 27b's 0.51-0.53 us. They are reported under section 5.7 with
  the INCONCLUSIVE label; they may not do in numbers what W5 bars in words.
* W6 (the V5 caveat sentence). It is owed only in W1/W2/W3 and only when verdict.json records a V5
  breach; verdict.json records V5 as within floor, so W6 is neither owed nor available here.
* Any claim about releases other than 1.23.2 and 1.24.4; any claim that pull request #25680 is a bug;
  and any extension of an E2 statement to another input size, thread count, batch size, model file or
  machine.
* Quoting 0.48 ms as "the step" in an E2 sentence.
* Quoting the census's "scalar 128-bit / 160-bit" path labels for rows 5, 6, 8-10, 12, 19-34.

An INCONCLUSIVE E2 leaves the paper's claim exactly where Experiment 27b left it: the mechanism is
traced to a code-level cause, and the full-model counterfactual returned no verdict. That is the
ruling.

---


---

## Acknowledgement of Wei-Yao Chang -- provenance (recorded 2026-09-21)

The paper's credits block reads: "The authors thank Wei-Yao Chang for assistance with the experiments."

Provenance, recorded because the third pre-submission audit correctly flagged that this name appears
in no CSV, log or entry of this record and therefore could not be verified from the repository:

* The first author instructed it on 2026-09-21 and states that Wei-Yao Chang assisted with
  measurement work during the project.
* The wording was deliberately chosen NOT to claim that this person produced any measurement the
  paper reports: an earlier draft read "for running part of the measurements" and was changed to
  "for assistance with the experiments" precisely because the runs behind the paper's numbers were
  executed by the harness on the machine described in Section 3.1, and no run in results/ is
  attributed to another operator.
* Wei-Yao Chang is an acknowledged contributor, not an author. The first author confirmed the
  person exists, assisted, and is to be named.
* A separate authorization note (kept with the package given to that person) records the same
  contribution statement and the permitted use.

No claim in the paper depends on this acknowledgement.


---

## V0 -- environment validation for the rewrite (REGISTERED 2026-09-21T23:52:34+08:00, BEFORE TIMING)

The ten ONNX Runtime side environments behind the eleven-release sweep were lost (deleted from the
session scratchpad by an agent's clean-up). 1.23.2 and 1.24.4 were rebuilt from PyPI into
edge-ai-toolkit/.envs/ort1232 and .envs/ort1244 with numpy 2.4.6 (the sweep's version) and psutil;
their pip freeze is in results/env_freeze_ort1232.txt and results/env_freeze_ort1244.txt. They are not
byte-identical to the lost environments, so before any new result is compared with the paper's,
the paper's headline cell must reproduce in them.

Cells: MobileViT-XXS 64 px, files results/classification_mobilevit/models/p00_fp32.onnx, p00_int8.onnx
(sha256 911de4ee...), p00_int8_dyn_mm.onnx x {1.23.2, 1.24.4} x {1, 4} threads, version protocol,
5 sessions per cell, seed 41, out results/v2/v0_env_validation.csv.

Rule (the paper's own step rule, unchanged): the rebuilt environments REPRODUCE the paper if the static
file's four-thread latency rises by more than 7% from 1.23.2 to 1.24.4 with disjoint session ranges,
while the FP32 and dynamic siblings move by less than 7%. Reported, not gating: the size of the step
against the paper's four timings (1.25-1.35x per run, t-interval 1.24-1.37). If it does NOT reproduce,
no new measurement from these environments is compared with the paper's, and the cause is found first.


### V0 RESULT (2026-09-22T00:04:16+08:00) -- REPRODUCES

Timed 2026-09-21 23:59-00:03, precondition AC on, machine load 7%, no WSL distro running; 60 sessions,
seed 41, results/v2/v0_env_validation.csv (10 of 60 sessions below the 0.95 core-delivery ratio,
reported, not gating, as registered).

Four threads, median p50 [session range], ms:
- static p00_int8: 1.23.2 1.597 [1.583-1.653] -> 1.24.4 2.089 [2.050-2.360]: +30.8%, ranges DISJOINT
- FP32: 1.585 -> 1.581 (-0.2%); dynamic: 1.569 -> 1.558 (-0.7%)
One thread: static +3.4%, FP32 +0.3%, dynamic +1.7%, no range disjoint.

Registered rule: static step > 7% with disjoint sessions AND siblings < 7% -> REPRODUCES. Per-run
factor 1.308, inside the paper's four earlier timings (1.25-1.35). With five sessions a side and
disjoint ranges the exact two-sided Mann-Whitney p is 2/252 = 0.008 for this cell alone. The rebuilt
environments are equivalent to the lost ones for this result; this is a fifth independent timing.
