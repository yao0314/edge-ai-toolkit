# Experiment A / A2 — does the static check predict the regression on files we did not make?

**Measured 2026-09-23, Taipei.** Registration: `results/paper/a_design_registered.md`, with the A2
delta registration `results/paper/a2_design_registered.md`. Predictions were written and hashed
**before the first timing row** (`results/expA2/seal_exploratory.json`, frozen 2026-09-23T06:48:52Z,
predictions sha256 `4547ad0c…`, manifest `5503fa56…`, checker `c333fca2…`).

The question: the study explains one regression by a property readable from the file alone — an 8-bit
Transpose whose moved block is 2 bytes. **Does that property predict behaviour on models nobody here
built?**

---

## 1. The pool

| | |
|---|---|
| candidates assembled | **46** |
| downloaded | 39 |
| admitted | **18** |
| rejected | 28 |
| carried over from experiment A | 6 |

Rejections, by the first rule each candidate failed:

| rule | n | what it is |
|---|---|---|
| A2 | 12 | input shape or dtype the harness cannot drive |
| A3 | 6 | not statically 8-bit quantized |
| A8 | 4 | screening latency over 60 ms |
| A9 | 2 | family or publisher cap |
| A5 | 2 | dynamic control flow |
| A1 | 1 | does not parse |
| L4 | 1 | gated repository |

Coverage of the 18 admitted: **15 architecture families** (edgenext, efficientnet ×2,
emotion_ferplus, inception_v1, inception_v3, mobilefacenet, mobilenet_v2 ×2, resnet ×2, sesr,
shufflenet_v2, squeezenet, tinyvit, ultraface, xcit, yolo_fastest), **9 publishers** (onnx/models 6,
AMD 3, CoreEpoch 3, Arm, OpenCV, Thefalley, ketiswp, mohitsha, onnx on Hugging Face),
**11 export toolchains**, and **all 18 measured as statically quantized**.

The analysis unit is the architecture family, not the file (`results/expA2/analysis_unit_note.md`,
written before the results): three families hold two files each.

## 2. The result

Measured on ONNX Runtime 1.23.2 against 1.24.4, one and four threads, five fresh processes per cell.

| | four threads | one thread |
|---|---|---|
| **18 third-party models** | **+1.2 % to −4.8 %** | **+4.0 % to −6.0 %** |
| beyond the 7 % floor | **none** | **none** |
| MobileViT-XXS (our control) | **+32.8 %**, ranges disjoint | +1.8 %, ranges disjoint |
| DeiT-tiny (our control) | −4.5 % | +1.2 % |
| MobileNetV2 (our control) | −1.2 % | −0.4 % |

**The denominator, stated plainly. [corrected 2026-09-24]** All 18 admitted models were predicted
**negative** — the checker flagged none of them (`seal_exploratory.json`: `n_predict_step: 0`). The
first version of this section called that "specificity, 18/18" and "zero false alarms". Both hold by
construction once nothing is flagged — a timing run cannot produce a false alarm for a check that
raised none — so they carry no information. What the timing tested is whether any file the check
**called safe** moved:

| | n | held still | what it tests |
|---|---|---|---|
| called safe (predicted no step) | **18** | **18** | the safe calls: none moved |
| flagged (predicted a step) | **0** | — | the alarms: not testable in this scan |

The safe calls are not trivially safe: **15 of the 18 files execute eight-bit Transposes** (moved
blocks of 1, 32, 48, 49, 196 and 784 bytes, none of 2, plus reshape copies and multi-axis
permutations with no single moved block; `predictions_exploratory.json`, SC-A nodes), so
a check that flagged every eight-bit Transpose would have called 15 of them unsafe, and none moved — on
a machine and a release pair that moved our own MobileViT by a third. What this scan supports is that
the width condition, not the mere presence of an eight-bit Transpose, separates the files that step
from those that do not. It does not support the word **predictor**. **Call it a criterion, or a
screening rule; report the safe calls with their denominator; do not write "specificity" or "zero
false alarms" for this scan.**

The positive class is supplied elsewhere, by construction rather than by search: experiment W built
files that carry the flagged pattern and predicted them to move. Its base is small and is stated as such: the check
flags only eight-bit Transposes, so its alarms are the four constructed uint8 files with a two-byte
block (three distinct constructions; `d_u8_w2_L256` replicates `m_u8_w2`), eight decision cells at two
and four threads, and all eight moved, by 4.6× (`d_u8_w2_L64`, 256 dispatches, four threads) to 44×
(`d_u8_w2_L1024`, 4096 dispatches, two threads), while no 1-, 4- or 8-byte file moved slower
(`results/paper/W_RESULTS.md` §2, §7). The float16 file with the same two-byte block also moved but is
not among the check's alarms. **[corrected twice]** First written as "four distinct constructed files
(five with one replicate), ten decision cells", which counted the float16 file. **[corrected]** The alarms are therefore tested on files we made, and
the safe calls on files we did not. Neither half should be quoted as if it were both. *Replaces:*
"Sensitivity is therefore demonstrated on files we made, and specificity on files we did not."

That no published file carried the pattern is itself a finding about what the ecosystem ships, not a
gap in the scan.

## 3. What assembling the pool found

Most public "quantized" ONNX vision models are **dynamically** quantized — typically
`transformers.js`'s `quantize_dynamic`. Their Transposes never execute in 8 bits, so the mechanism
under study cannot reach them, and six candidates were rejected on exactly this (rule A3). This is
why statically quantized vision transformers under 60 MB are scarce: across the whole search, the
only publisher shipping them at that size was CoreEpoch.

Licence state was **recorded, not required** (A2's correction to the first round): 16 admitted files
declare a licence, 2 are silent, and nothing is redistributed. Seven admitted files are anonymous
re-exports — reported as such rather than dropped.

## 4. What this record must not overstate

* **The artefact still says EXPLORATORY, and that label is stale.** The seal's own computed fields
  say `n_models: 18` and `pool.json` says `pool_minimum_18_met: true`, but the seal's note reads
  "Pool minimum (18) not reached: 13 admitted. This run is EXPLORATORY". That sentence is a string
  literal in `scripts/paper/a2_exploratory_run.py`, written when 13 were admitted and never updated
  after the top-up round; `results/paper/a2_thirdparty_exploratory.csv` carries the same stale word in
  its filename. The measurement covered 18. The contradiction is recorded here rather than resolved
  silently in the result's favour.
* **Three of the four registered stratum minimums were never evaluated.** `manifest.json` records
  `|F|>=5`, `|U|>=8` and `|U-near|>=4` as `None`, with only `architecture_families>=8` computed
  (true). A full verdict under the registration needs those three; this record therefore reports the
  measurement and the false-alarm count, and does not claim the registered verdict.
* **A single-run verdict near the floor is unstable.** Experiment B re-measured the same files on
  another night with rebuilt interpreters: `a03` reads −8.0 % there against +0.0 % here, and `b02`
  +7.8 % against −2.9 %. Both gaps sit inside the between-run spread the B registration documents
  (ratios to 1.174). So "all 18 stayed inside the floor" is a statement **about this run**, not a
  property of those files. Neither case is a false alarm for the criterion, which flags slowdowns
  from a 2-byte moved block — and `a03` got faster.
  **Settled by the independent replication (2026-09-24):** both files were re-measured a third time
  in B's registered replication stage and both land back inside the floor — `a03` at −0.1 %, `b02`
  at −3.5 %. Two runs of three agree with A2; the map's reading was the outlier in both cases.

## 5. Where everything is

| | |
|---|---|
| pool, admissions, rejection reasons | `results/expA2/pool.json` |
| the 18 measured models and their metadata | `results/expA2/manifest.json` |
| predictions, frozen before measurement | `results/expA2/predictions_exploratory.json` |
| the seal | `results/expA2/seal_exploratory.json` |
| measurements | `results/paper/a2_thirdparty_exploratory.csv` |
| analysis unit, written before results | `results/expA2/analysis_unit_note.md` |
| figures | `results/qc/a2_dots.png`, `a2_pareto.png`, `a2_screen_control.png` |
| findings-log entries | `results/paper/FINDINGS.md` (prediction freeze, and the result) |
