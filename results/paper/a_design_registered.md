# Experiment A: the two-byte-block check as a PREDICTOR, tested on third-party quantized models

**REGISTERED 2026-09-23T01:35+08:00, BEFORE ANY MODEL WAS DOWNLOADED AND BEFORE ANY TIMING.**

Written by the independent designer of Experiment A. The designer did not produce the 1.24 claim,
did not write `transpose_probe.py`, will not choose the models and will not run the timing. Nothing
was downloaded for this document; no latency, benchmark or timing run was made. Three local model
files were loaded once under ONNX Runtime 1.24.4 to dump optimized graphs and to run 3 untimed
profiled iterations, and the two prototype checkers of section 2 were run on them; that is the whole
of the machine work behind this file, and every number below marked VERIFIED comes from it.

Sources read before writing: `results/paper/e2_design_registered.md` (house style of a registration),
FINDINGS.md Experiment 26 (K1), Experiment 27 sections 0-5 and its RESULT, the referee ruling on
Experiment 27 and its Experiment 27b registration and RESULT, the referee ruling on Experiment E2,
the V0 entry and its RESULT, `scripts/paper/ort_version_bisect.py`,
`scripts/paper/transpose_probe.py`, `scripts/paper/optgraph_version_diff.py`,
`scripts/paper/node_profile.py`, `results/transpose_probe/manifest.json`, and
`results/paper/CAMERA_READY_PLAN.md`.

The question, in one sentence: **can a static check on a quantized ONNX file say, before anything is
timed, which third-party models lose speed on ONNX Runtime 1.24.4 relative to 1.23.2 at four
threads?**

Everything below - how models may be chosen, what the checker is, what is frozen, how it is measured,
what counts as a hit, what the paper may say - is fixed now.

`<scratchpad>` is
`C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-SEC/<session>/scratchpad`.
All paths are relative to `C:/Users/User/Desktop/SEC/edge-ai-toolkit`.

---

## 0. What this experiment can and cannot deliver

The standing claim is Experiment 27b's: LOCALISED (FIRM) + M-CONFIRMED, i.e. the step is traced to a
code-level cause, PR #25680, which from 1.24 routes a single-axis Transpose with a two-byte moved
block to MLAS's threaded transpose. Experiment E2 returned INCONCLUSIVE on V4a, so the phrase "root
cause" is barred and E2's measured quantities may not appear in prose. Experiment A does not reopen
either ruling and does not depend on E2.

What A adds, if it works, is EXTERNAL VALIDITY of a checkable kind: the mechanism becomes a rule that
a reader can run on their own file. What A cannot do is prove causation on any third-party model - no
counterfactual is built for them. A measures whether a prediction made from the file alone, frozen
before the numbers exist, comes true.

A deliberately weak spot, registered here and not discovered later: a model can be slower on 1.24.4
for reasons that have nothing to do with #25680. Such a model is a false clear (FN) for this checker
and is reported as one; section 5.6 says what is then written.

---

## 1. The model set

### 1.1 How models may be sourced

* **Third-party quantized only.** The 8-bit quantization must have been performed and published by
  someone other than this project. A model this project quantizes is not admissible at any stage: the
  whole point is that the quantizer is not ours.
* **Publicly downloadable**, from a source recorded as a full URL that resolved on the day of the
  fetch: Hugging Face Hub repositories, the ONNX Model Zoo GitHub repository, or a hardware/framework
  vendor's own published model repository. No mirror, no re-upload of unknown provenance, no file
  whose publisher cannot be named.
* **Licence recorded** for every candidate: the licence identifier as the publisher states it, plus
  the URL where it is stated. A file with no stated licence is not admitted. No licence text is
  copied into this project; only the identifier and the URL.
* The fetch team records, per candidate, BEFORE downloading: publisher, repository, file path in the
  repository, URL, licence, declared architecture family, and the file size from an HTTP HEAD. After
  downloading: sha256 and the actual byte count. Any mismatch between the HEAD size and the file is
  recorded and the file is re-fetched once, then rejected.
* **Nothing is downloaded until the user approves the candidate list.** The list of URLs is produced
  by the fetch round and approved as a whole; this registration approves rules, not URLs.

### 1.2 Mechanical admission rules (applied before the checker's flag is computed)

A candidate is ADMITTED only if all of the following hold. Each is a yes/no test a script can make,
and none of them may refer to the checker's output.

| # | rule | why |
|---|---|---|
| A1 | The file parses with the project venv's `onnx` 1.22.0 and has exactly one graph input | the checker and the harness both need this |
| A2 | The single input is `float32` with rank 4 (N,C,H,W), N either 1 or a symbolic batch dim, and C,H,W all fixed positive integers | `ort_version_bisect.py`'s child feeds one float32 NCHW tensor; anything else makes the session fail and the row vanish silently |
| A3 | Static 8-bit quantization: the graph contains `QuantizeLinear`/`DequantizeLinear` with an 8-bit zero point (QDQ) or `QLinear*` operators (QOperator), and at least 8 of them | dynamic quantization keeps activations in float, so its Transposes are float by construction: a trivial negative that would pad the cleared class |
| A4 | No 4-bit or sub-byte tensor type anywhere (`INT4`, `UINT4`), and no `float16` weights | the moved-block rule is defined in whole bytes; sub-byte types are outside the dispatch table the checker implements |
| A5 | No data-dependent control flow or output shape: no `NonMaxSuppression`, `Loop`, `If`, `Scan`, `TopK` with a runtime K, and every graph output shape is static once the input is pinned | with random input data such models change how much work they do between sessions, which is latency variance with no mechanism |
| A6 | Loads without error under BOTH ONNX Runtime 1.23.2 and 1.24.4 at `ORT_ENABLE_ALL`, and one inference returns without error under both | a model only 1.24 can run has no 1.23.2 baseline. Exp 26 already hit this: `ConvInteger(10)` is NOT_IMPLEMENTED on 1.23.2 |
| A7 | File size <= 60 MB | keeps session load time and the 2.5 GB pool cap in hand |
| A8 | Screening latency `screen_ms` <= 60 ms (section 4.1) | a 100 ms model costs 5x the wall clock of a 20 ms one and buys no extra cell |
| A9 | Not a duplicate: no other admitted candidate has the same sha256, and at most 2 admitted candidates share one architecture family, at most 6 share one publisher | ten quantized MobileViT clones are one observation, not ten |

Rejections are recorded with the failing rule in `results/expA/pool.json`. A rejected file is deleted
after its sha256, size and reason are recorded; the pool directory is capped at 2.5 GB.

### 1.3 Sizes, strata, and what makes the result meaningless

* **Pool** (admitted candidates): target 24, minimum 18, maximum 40.
* **Strata**, from the frozen prediction of section 2:
  * **F** = models predicted to step (`PREDICT-STEP`).
  * **U** = models predicted not to step (`PREDICT-SAFE`).
  * **U-near** = the subset of U whose executed graph runs at least one 8-bit Transpose but none with
    a two-byte moved block. These are the informative negatives: a cleared model with no 8-bit
    Transpose at all is cleared trivially.
* **Selection into the measured set** (blind to anything but the flag and the hash): take every model
  in F, up to 12; if F has more than 12, take the 12 with the lexicographically smallest sha256. Then
  take models from U in ascending sha256 order, preferring U-near until 4 U-near are in, until the
  measured set reaches 25 models or U is exhausted, subject to the A9 caps.
* **Minimum sizes for any claim: |F| >= 5, |U| >= 8, |U-near| >= 4, and at least 8 distinct
  architecture families across the measured set.** If any of these is not met, the outcome is
  **INCONCLUSIVE-SPARSE** (section 5.5). A positive class of one, two or three models is not a
  criterion; it is a second anecdote, and this registration refuses it in advance.
* Three local files are measured in the same run as CONTROLS and are never in the confusion matrix:
  `results/classification_mobilevit/models/p00_int8.onnx` (positive control, sha256 911de4ee...),
  `results/classification_deit/models/p00_int8.onnx` and `results/classification/models/p00_int8.onnx`
  (negative controls). They are the checker's training cases; scoring them would be scoring the
  checker on its own worked example.

---

## 2. The checker

Two predictors are frozen at the same time. Both reuse `predict()`, `_single_axis()`, `_is_reshape()`
and `MLAS_BYTES` from `scripts/paper/transpose_probe.py` by import - not by re-implementation - so the
dispatch rule cannot drift between experiments. The implementation is
`scripts/paper/a_check.py`, to be written before the freeze; this section is its specification.

### 2.1 SC-B, the file-only checker. PRIMARY.

Reads one `.onnx` file and nothing else.

1. Pin the batch dimension of the single input to 1; leave C,H,W as published.
2. `onnx.shape_inference.infer_shapes(model, data_prop=True, strict_mode=False)`.
3. For every `Transpose` node with a fully static input shape, decide whether it executes in 8 bits:
   either its input comes from `DequantizeLinear` and its output feeds `QuantizeLinear` (the QDQ
   sandwich, which ORT's QDQ transformer turns into an 8-bit copy), or its input tensor type is
   already `INT8`/`UINT8`.
4. Apply `predict(dims, perm, 1, 4, version)` for version 1.23.2 and 1.24.4. A node is FLAGGED when
   it gains dispatches (1.24.4 > 1.23.2), which by the rule in `transpose_helper.cc` happens exactly
   when the moved block is two bytes wide and M > 1.
5. Model-level outputs: `sig_b` (any flagged node), `added_dispatches` (sum of the gains),
   and the per-node table (name, shape, perm, loops, M, N).

SC-B is PRIMARY because it is the tool the paper would offer a reader. It is also the weaker of the
two - it cannot see a Transpose that ORT's optimizer inserts - and the primary predictor should be
the falsifiable one.

**VERIFIED today on the three local files** (prototype, `<scratchpad>/expa-design/`, deleted):

| file | 8-bit-executing Transposes in the file | flagged | added dispatches |
|---|---|---|---|
| classification_mobilevit p00_int8 | 39 | **4** | **832** |
| classification_deit p00_int8 | 37 | 0 | 0 |
| classification p00_int8 | 0 | 0 | 0 |

and the four flagged nodes are exactly `/stages/stages.2/stages.2.1/Transpose` `[256,2,4,2]`,
`/stages/stages.2/stages.2.1/Transpose_3` `[256,4,2,2]`, `/stages/stages.3/stages.3.1/Transpose`
`[160,2,2,2]`, `/stages/stages.3/stages.3.1/Transpose_3` `[160,2,2,2]`, all perm `[0,2,1,3]` - the
four nodes Experiment 27's census rows 3, 14, 17, 36 and `e2_design_registered.md` section 1.2 name.

### 2.2 SC-A, the executed-graph checker. SECONDARY, frozen at the same time.

Answers the same question about the graph ONNX Runtime actually runs, which is not the file's graph
(FINDINGS, Experiment 27 section 1: the optimizer rewires and inserts Transposes).

1. In `.envs/ort1244`, create a session with `graph_optimization_level=ORT_ENABLE_ALL` and
   `optimized_model_filepath` set: ORT writes its optimized graph (the technique of
   `optgraph_version_diff.py`; the side venvs have no `onnx`, so only ORT writes, and the project venv
   parses).
2. In the same environment, a second session with `enable_profiling=True`, 1 intra-op thread, 3
   iterations on random input. Each `Node` `*_kernel_time` event carries `args.op_name` and
   `args.input_type_shape`, i.e. the executed dtype and shape of every Transpose.
3. Join profile events to optimized-graph nodes BY NODE NAME to get (dims, perm, dtype) per executed
   Transpose, then apply `predict()` as in SC-B with the element size of the executed dtype.
4. Outputs `sig_a`, `added_dispatches_a`, and the per-node table.

Profiling three iterations is not a latency measurement and its durations are never used.

**VERIFIED today on the three local files**: the join is complete (0 unjoined nodes in all three) and
gives 56 / 51 / 2 executed Transposes, all uint8, with graph totals at four threads of 32 -> 864
dispatches for MobileViT (**+832, the four nodes above**), 15 -> 15 for DeiT and 1 -> 1 for the third
file. The MobileViT numbers reproduce `results/transpose_probe/manifest.json` exactly.

The two checkers already disagree about the file that matters least: the third control's file has no
Transpose at all, yet ORT executes two (inserted layout transposes). That is SC-B's blind spot,
measured rather than asserted, and section 5 scores it.

### 2.3 From flag to prediction

* `DISPATCH_US = 0.52` microseconds per added dispatch. Source: **Experiment 27b's reported
  per-dispatch cost, 0.51-0.53 us** (FINDINGS, Exp 27b RESULT and Exp 27 section 4.5). E2's own
  0.519 us is NOT cited anywhere in Experiment A: the E2 ruling bars its numbers from prose.
* Predicted added time `A_ms = added_dispatches * 0.00052` ms.
* Predicted relative step `rho = A_ms / screen_ms`, with `screen_ms` from section 4.1.
* **`PREDICT-STEP` iff `sig_b` is true AND `rho >= 0.07`. Otherwise `PREDICT-SAFE`.**
  The 7% is the study's floor (`FLOOR_REL`, `e2_design_registered.md` section 5.3), the same number
  the outcome rule uses, so the prediction is made on the same scale as the test.
* Reported sub-class **SIG-SMALL**: `sig_b` true but `rho < 0.07`. These are predicted SAFE. A
  criterion that ignores model size would flag a 30 ms CNN for a 0.4 ms effect and be wrong on
  purpose; this sub-class is where that distinction is recorded.
* A separate, sharper, REPORTED-ONLY prediction per flagged model: the measured four-thread delta
  should land within a factor of two of `A_ms`, i.e. `0.5 <= measured_delta_ms / A_ms <= 2.0` - the
  band Experiment 27 section 4.5 registered for the dispatch-cost ratio. It can never change a
  verdict.

---

## 3. The frozen artefacts, and the order of freezing

Nothing may be measured until all of this exists, and nothing in it may change afterwards. Files go
into the project, never into the scratchpad: the scratchpad is not storage.

| # | artefact | path | written when |
|---|---|---|---|
| 1 | candidate pool: every URL, publisher, licence, HEAD size, sha256, admission verdict and failing rule | `results/expA/pool.json` | after the fetch, before the checker runs |
| 2 | measured set: sha256, bytes, stratum, `screen_ms`, input shape, opset, quantization format, architecture family, harness family dir and variant stem | `results/expA/manifest.json` | after admission and selection |
| 3 | predictions: per model `sig_b`, `sig_a`, `added_dispatches` (both), `A_ms`, `rho`, PRIMARY class, the per-node tables, plus the ORT versions used and the sha256 of `a_check.py` | `results/expA/predictions.json` | after the checker runs, before any timing |
| 4 | checker self-test against ground truth | `results/expA/selftest.json` | before 3 is accepted |
| 5 | human-readable frozen list: one line per model - sha256 prefix, publisher, architecture, `screen_ms`, flag, `added_dispatches`, `rho`, PREDICT-STEP/PREDICT-SAFE | `results/paper/a_predictions_frozen.md` | with 3 |
| 6 | the seal: the sha256 of artefacts 1-5, the UTC timestamp, and the count per stratum, appended to `results/paper/FINDINGS.md` | before the timing command is typed |
| 7 | quiet-machine record | `results/expA/precondition.json` | written by the runner, after the verification commands, before timing |
| 8 | timing output | `results/paper/a_thirdparty.csv` | the run |
| 9 | verdict | `results/expA/verdict.json`, by `scripts/paper/a_analyse.py` | after the run |

**The seal (artefact 6) is what makes this a pre-registration.** If FINDINGS.md does not carry the
sha256 of `predictions.json` before `a_thirdparty.csv` exists, the run is INCONCLUSIVE by VA1 and the
paper may not describe the predictions as frozen.

---

## 4. The measurement protocol

### 4.1 The screening pass (admission only, never quoted)

One session per candidate under 1.24.4, four threads, 3 warmup then 20 runs, median recorded as
`screen_ms`. It exists to apply A8 and to give `rho` a denominator. It is a screen, not a result:
`screen_ms` may appear in the manifest and in a table of the frozen list, and may NEVER be quoted as
a latency, a baseline or a speedup in the paper. It runs before the freeze, on the quiet machine, and
its values are frozen with everything else.

### 4.2 The primary run

* **Releases**: 1.23.2 (`.envs/ort1232`) and 1.24.4 (`.envs/ort1244`). These are the V0-validated
  environments; V0 (2026-09-22) confirmed the paper's headline step reproduces in them at +30.8% with
  disjoint session ranges.
* **Threads**: 4 (primary) and 1 (control, the mechanism's own negative: no pool, no dispatch).
* **Sessions**: 5 per cell, each a fresh subprocess. Cells = models x 2 releases x 2 thread counts.
* **Order**: one single invocation, one shuffle, `--seed 53`, so variant, release and thread count are
  decorrelated from position in time.
* **Per session**: batch 1, P-core pinning via `pick_cpus(t,"auto")`, `ORT_SEQUENTIAL`,
  `ORT_ENABLE_ALL`, default spinning, 1.5 s warmup then >= 2 s and >= 40 runs, p50 of the session.
* Seeds: 53 primary; 54 reserved for the single registered re-run (VA5); 55 reserved for the 1.30.0
  persistence arm. No other seed may be used.
* **Command** (one line, from the project root; `<FAMS>` and `<VARS>` are written out in full in
  `a_predictions_frozen.md` at the freeze):

```
"C:/Users/User/Desktop/SEC/edge-ai-toolkit/.venv/Scripts/python.exe" scripts/paper/ort_version_bisect.py --scratch "C:/Users/User/Desktop/SEC/edge-ai-toolkit/.envs" --versions 1.23.2,1.24.4 --threads 1,4 --repeats 5 --seed 53 --families "<FAMS>" --variants <VARS> --out results/paper/a_thirdparty.csv
```

  `--scratch` must be `.envs`, where `ort1232` and `ort1244` live; the harness resolves interpreters
  by the version they report, not by directory name. `<FAMS>` is `;`-separated `name=dir:c,h,w`
  entries, one per distinct input shape, plus the three control families
  `mobilevit=results/classification_mobilevit/models:3,64,64`,
  `deit=results/classification_deit/models:3,32,32`,
  `cls=results/classification/models:3,32,32`. `<VARS>` is the comma-separated list of every variant
  stem, which must be globally unique (`a01`..`a25`) plus `p00_int8`; the harness skips
  (family, variant) pairs whose file does not exist, so one invocation covers everything.

* **Analysis**: `scripts/paper/a_analyse.py results/paper/a_thirdparty.csv` -> `results/expA/verdict.json`.
  Like `e2_analyse.py`, it must be written before the timing and must be exercised on SYNTHETIC data
  first, with the synthetic files deleted afterwards. The registered synthetic scenarios, all of which
  must give the intended answer before the script is used: (1) clean CONFIRMED; (2) clean REFUTED;
  (3) PARTIAL; (4) a flagged model that also steps at one thread -> IMPURE, not TP; (5) a cleared
  model that steps -> FN counted, verdict capped per 5.4; (6) |F| = 3 -> INCONCLUSIVE-SPARSE;
  (7) |U-near| = 2 -> INCONCLUSIVE-SPARSE; (8) 30% indeterminate -> INCONCLUSIVE; (9) a 4-thread cell
  with 2 of 5 sessions below the gate -> that model excluded, count reported; (10) 4 models so
  excluded -> INCONCLUSIVE; (11) a model with 4 sessions in one cell -> dropped, counted; (12) gated
  and ungated verdicts differ -> INCONCLUSIVE; (13) the positive control fails to step ->
  INCONCLUSIVE; (14) a NaN p50 -> never silently passes a comparison; (15) a sha256 mismatch against
  the manifest -> INCONCLUSIVE.

### 4.3 The persistence arm (registered now, reported, never verdict-changing)

After the verdict is computed, every model in the matrix is re-run at four threads on ONNX Runtime
1.30.0 (the project venv), 3 sessions, `--seed 55`, `--out results/paper/a_thirdparty_130.csv`. It
answers "does the step persist three releases later", the way Experiment 26's rule did. It cannot
change any band, and if it is not run the verdict stands.

### 4.4 What makes a session invalid

A session is invalid if the subprocess exited non-zero (the harness writes no row), `iters < 40`,
`p50_ms` is NaN, or `core_delivery` is NaN. **No session is dropped for being slow, fast or
surprising**; there is no outlier rule, because the study has already been burnt by order and
selection effects. A cell with fewer than 5 valid sessions makes its model incomplete (VA7).

---

## 5. The outcome rule

### 5.1 "This model stepped", in exactly these words

For one model at four threads, with `m(r)` the median of its 5 session p50s on release `r`:

* **STEPPED** iff `m(1.24.4) / m(1.23.2) >= 1.07` **AND** the two cells' session ranges are DISJOINT
  (max of the five 1.23.2 p50s < min of the five 1.24.4 p50s).
* **NOT-STEPPED** iff `m(1.24.4) / m(1.23.2) < 1.07`.
* **INDETERMINATE** iff the ratio is >= 1.07 but the ranges overlap.

Where the rule is written: the 7% step rule is registered in FINDINGS.md under **"Experiment 26 (K1),
2026-09-19"** ("a 'step': static 4T median absolute latency at 1.24.4 more than 7% above 1.23.2");
7% is the study's between-session floor, `FLOOR_REL` in `e2_design_registered.md` section 5.3; the
disjoint-range form is the one registered for **V0** on 2026-09-21T23:52:34+08:00 and used in its
RESULT. **Deviation, stated in advance:** Experiment 26's full rule also requires the step to persist
at 1.25.1 and 1.30.0 and the FP32 and dynamic SIBLINGS to move less than 7%. Third-party quantized
files have no FP32 or dynamic sibling from the same quantizer, so that clause cannot be applied here;
persistence is measured instead by the 1.30.0 arm of section 4.3 and reported, not gated. No other
part of the rule is relaxed.

* **PURE** (the mechanism's four-thread-only signature) iff
  `|m(1.24.4,1T)/m(1.23.2,1T) - 1| < 0.07`. A model that also moves at one thread is moving for a
  reason the pool cannot explain.

### 5.2 The confusion matrix

Over the measured models only - never the three controls - after the exclusions of section 6:

|  | STEPPED and PURE | STEPPED, not PURE | NOT-STEPPED |
|---|---|---|---|
| **PREDICT-STEP** | TP | IMPURE | FP |
| **PREDICT-SAFE** | FN | FN-IMPURE | TN |

INDETERMINATE models appear in no cell and are listed by name with their ratios.

* **hit rate** `H = TP / (TP + IMPURE + FP)` - of the models the checker flagged, the share that
  stepped for the predicted reason.
* **false-clear rate** `M = (FN + FN-IMPURE) / (FN + FN-IMPURE + TN)` - of the models it cleared, the
  share that stepped anyway.
* **Fisher one-sided exact p** on `[[TP+IMPURE, FP], [FN+FN-IMPURE, TN]]`, computed from
  `math.comb` with the hypergeometric tail (no SciPy dependency), testing "flagged models step more
  often than cleared ones". With |F| = 5 all stepping and |U| = 8 none stepping, p = 1/C(13,5) =
  0.00078; with 4 of 5 and 1 of 8, p = 0.032. This is why 80% alone is not enough at these sizes.

### 5.3 Verdicts

| verdict | condition |
|---|---|
| **CONFIRMED** | `H >= 0.80` AND `M <= 0.10` AND Fisher `p < 0.05` AND `IMPURE <= TP` AND the sizes of 1.3 are met AND every validity condition of section 6 holds |
| **PARTIAL** | `H >= 0.60` AND Fisher `p < 0.05`, but not CONFIRMED |
| **REFUTED** | `H <= 0.40` OR `M >= 0.30` |
| **INCONCLUSIVE-SPARSE** | `|F| < 5` or `|U| < 8` or `|U-near| < 4` or fewer than 8 architecture families |
| **INCONCLUSIVE** | any validity condition of section 6 fails, or INDETERMINATE models exceed 20% of the measured set, or the gated and ungated verdicts differ |

Anything that falls in none of the first three bands while the sizes hold is PARTIAL's complement and
is reported as **MIXED**: the numbers are published and no criterion is claimed.

### 5.4 If models predicted safe step anyway

Every FN is investigated by a registered diagnostic, run after the verdict is recorded and never
allowed to change it: SC-A on that model, plus `scripts/paper/optgraph_version_diff.py` between the
two releases. Each FN is then classified:

* **(i) file-only blind spot** - SC-A finds the two-byte signature that SC-B could not see. The
  mechanism holds; the file-only tool is incomplete. `a_check.py` may be revised ONLY after the
  verdict is in `verdict.json`, and the revised checker is reported as a post-hoc revision and is
  never re-scored on this model set.
* **(ii) unexplained** - neither checker finds the signature. This is a second regression mechanism
  that #25680 does not cover.

Consequences, fixed now: **any FN at all** bars the paper from writing that a model without the
signature is safe, or from calling the check necessary; only the positive direction may be claimed.
**Three or more FNs of class (ii)** additionally force the sentence in section 7 that the 1.24 drift
has at least one further mechanism this study has not identified, and cap the verdict at PARTIAL
however good `H` is.

### 5.5 If the positive class is tiny

INCONCLUSIVE-SPARSE is a real outcome with its own permitted sentence (section 7). The measurement
still runs on whatever F exists plus 8 cleared models plus the 3 controls, and the numbers are
published, but no hit rate is claimed and the words "criterion", "rule" and "predicts" may not be
used of the result. Rarity is itself reportable: "in a pool of N admitted third-party quantized
models, k carried the signature" is a finding, and an honest one.

---

## 6. Validity conditions, and what voids the run

Gating unless marked REPORTED. A gating failure gives INCONCLUSIVE, never a verdict.

* **VA1 - the seal.** `predictions.json`, `manifest.json`, `pool.json` and `selftest.json` exist, and
  their sha256 values are in FINDINGS.md with a timestamp EARLIER than the first row of
  `a_thirdparty.csv`. Every measured file is re-hashed before and after the run and matches
  `manifest.json`.
* **VA2 - checker self-test.** SC-A and SC-B, run twice each, give identical output, and on the three
  local files reproduce exactly: MobileViT 4 flagged nodes and 832 added dispatches with the four
  names and shapes of section 2.1; DeiT 0; the third control 0 from SC-B and 0 added from SC-A. A
  checker that cannot reproduce its own worked example may not predict anything.
* **VA3 - the known step reproduces in THIS run.** The positive control steps by >= 7% at four
  threads with disjoint session ranges, and both negative controls do not step. This is the V0 rule
  applied inside the run; if the chain cannot reproduce the effect the paper is built on, nothing
  measured beside it can be read. One re-run of the control cells is permitted (`--seed 54`); a
  second failure aborts the experiment.
* **VA4 - quiet machine.** AC power; no WSL distro running; machine-wide
  `Win32_PerfFormattedData_PerfOS_Processor` `_Total` <= 15% over 5 s within 10 one-minute checks;
  no other timing job on the machine. Machine load is read from PerfOS `_Total`, never from
  `Win32_Processor.LoadPercentage`, which read 100% on a 22-32% busy laptop and stopped Experiment 26
  twice. `precondition.json` is written after the verification commands and before the timing; if it
  is absent the analysis returns INCONCLUSIVE.
* **VA5 - core delivery.** `cd = cpu_ms_per_inf / (threads * p50)`, gate 0.95, on every
  FOUR-THREAD cell: at most 1 of a cell's 5 sessions may read below the gate. A model breaching this
  on either release is moved to the EXCLUDED set and takes no place in the matrix; one registered
  re-run of that model's four-thread cells is allowed (`--seed 54`, own CSV, never pooled with
  seed 53, replacing that cell). If more than 3 models end up excluded, INCONCLUSIVE. One-thread
  cells are REPORTED, not gated: at one thread `cd` is a CPU-accounting ratio, not a core-delivery
  measure (E2 section 5.4 V7a, and the Exp 27 referee ruling on cells that give the pool no work).
* **VA6 - gated equals ungated.** The verdict computed from gated medians (below-gate four-thread
  sessions dropped) must equal the ungated verdict, band for band. Otherwise INCONCLUSIVE. Both are
  reported; the ungated numbers are the paper's reporting convention.
* **VA7 - completeness.** Every measured model has 5 valid sessions in all four of its primary cells.
  A model short of that is dropped and counted; more than 3 dropped, or any NaN reaching a
  comparison, gives INCONCLUSIVE.
* **VA8 - no re-selection after the freeze.** No model may be added, removed, re-classified or
  re-predicted after artefact 6 is written. A model that fails to run is reported as FAILED-TO-RUN
  and is not replaced. The checker is not touched until the verdict exists (section 5.4).
* **VA9 - one measurement at a time, and no orphans.** No other timing job runs on this machine
  during the run; no background process, no detached job, no whole-drive search; after the run,
  python/curl/find processes are swept and the pool directory is checked against its 2.5 GB cap.
* **VA10 - REPORTED, not gating.** Per-model medians and ratios at both thread counts on both
  releases with their session ranges; `screen_ms`; per-cell core-delivery minimum and the count below
  gate; the SC-A vs SC-B disagreement list; the magnitude ratios of section 2.3; the 1.30.0
  persistence arm; the INDETERMINATE, EXCLUDED, DROPPED and FAILED-TO-RUN lists by name.

---

## 7. Time budget and abort criteria

The whole result must exist by **2026-09-28**, and the measuring machine can run one measurement at a
time.

| milestone | deadline | if missed |
|---|---|---|
| candidate list (URLs, licences, HEAD sizes) put to the user | 2026-09-25 18:00 | abort to scan-only |
| user approves the list; download and admission complete | 2026-09-26 12:00 | measure whatever is admitted, if the 1.3 sizes hold; else INCONCLUSIVE-SPARSE |
| checker self-test, predictions, freeze, seal in FINDINGS.md | 2026-09-26 18:00 | no measurement may start; the round ends with the frozen predictions unmeasured, which is still a publishable pre-registration |
| primary run complete | 2026-09-27 12:00 | see the wall-clock rule below |
| analysis, independent ruling, FINDINGS entry | 2026-09-27 20:00 | report as unruled and do not put it in the paper |

**Wall-clock estimate for the measuring machine**, from this project's own rates (E2 timed 80 sessions
of a 1.6 ms model in 7-8 minutes, i.e. 5.3-6.0 s per session; third-party models are larger and load
more slowly, so 8-12 s per session is used here):

| stage | sessions | estimate |
|---|---|---|
| screening pass (section 4.1) | 24-40 | 4-8 min |
| primary run, 20 models + 3 controls | 460 | 61-92 min |
| primary run, 25 models + 3 controls | 560 | 75-112 min |
| 1.30.0 persistence arm | 60-75 | 8-15 min |
| quiet-machine waiting | - | 0-10 min |
| **total on the machine** | | **1.3-2.2 hours** |

Abort criteria, each mechanical:

1. **Positive control fails twice** (VA3): abort; no Experiment A result; the failure is recorded in
   FINDINGS.md, because a chain that cannot reproduce V0 is a finding about the chain.
2. **Pool under 18 admitted models**: no measurement; report the scan alone (how many public
   quantized models were examined, how many carried the signature).
3. **|F| < 5 after admission**: do not run 25 models; run F + 8 cleared + 3 controls and report
   INCONCLUSIVE-SPARSE.
4. **Primary run exceeds 150 minutes or is not finished by 2026-09-27 12:00**: stop it. The completed
   models count only if the section 1.3 minimums hold among them AND every counted model has all four
   cells complete; otherwise INCONCLUSIVE.
5. **Another measurement is running on the machine**: do not start; there is no parallel arm.
6. **Any download larger than 60 MB, or a licence that cannot be identified**: that candidate is
   dropped, never "included provisionally".
7. **The pool exceeds 2.5 GB**: stop downloading; the admitted set is whatever exists.

---

## 8. What the paper may write, one sentence per outcome

ASCII. `N` = measured models, `k` = flagged, `H`, `M`, `p` from `verdict.json`. Every sentence that
names the mechanism carries Experiment 27b's mandatory disclosure wherever 27b is cited, unchanged.

* **CONFIRMED** - "Applying the check to N third-party quantized ONNX models published by others, with
  every prediction frozen and hashed before any measurement, k of them were flagged as carrying an
  8-bit Transpose with a two-byte moved block; H of the flagged models lost more than 7% of their
  four-thread speed on ONNX Runtime 1.24.4 against 1.23.2 with disjoint session ranges and no
  one-thread change, against M of the cleared models (Fisher one-sided p = p)."
* **PARTIAL** - "Of N third-party quantized models with frozen predictions, H of the flagged models
  stepped against M of the cleared ones (p = p); the check is a useful positive screen on models we
  did not quantize, but it is not accurate enough to be called a criterion."
* **MIXED** - "The frozen predictions on N third-party models gave H on the flagged and M on the
  cleared; the file-level signature by itself does not separate the two groups, and we report the
  table rather than a rule."
* **REFUTED** - "Frozen predictions from the file-level signature did not identify which of N
  third-party quantized models lose four-thread speed on 1.24.4 (H flagged, M cleared), so the
  mechanism established for our file does not generalize as a file-readable criterion."
* **INCONCLUSIVE-SPARSE** - "Of N admitted third-party quantized models, only k carried the
  two-byte-block signature, too few to test a prediction rule; we report the scan and the k models'
  measurements without claiming a criterion."
* **INCONCLUSIVE** - "A pre-registered prediction test on third-party models was run and its
  registered validity condition VAn was not met (state which), so it returns no verdict."
* **Design facts, usable in every branch** (verified, timing-independent): "The check reads a
  quantized ONNX file, finds the Transposes that ONNX Runtime executes in 8 bits, and applies the
  dispatch rule in `core/framework/transpose_helper.cc`; on our own file it recovers exactly the four
  nodes whose moved block is two bytes wide and the 832 thread-pool dispatches they add per inference
  on 1.24.4."
* **Whenever there is at least one FN**: "The check did not clear every model correctly: c of the
  models it cleared lost speed anyway, so it must be read as a positive screen and not as a clean
  bill of health."
* **With three or more FNs of class (ii)**: "At least one further mechanism moves 1.24 latency on
  quantized models that this study has not identified."

### Barred in every branch

* "Root cause", "proves", "the cause of", for Experiment A or for any third-party model. A cannot
  build a counterfactual on a model it did not quantize, and the E2 ruling bars the phrase outright.
* Any of E2's measured quantities, including 0.519 us per dispatch, f = 0.9881 and the 0.4320 ms
  removed.
* Calling the check "necessary", "complete" or "a guarantee"; saying an unflagged model is safe.
* Presenting `screen_ms` as a latency result, a baseline or a speedup.
* Any claim about releases other than 1.23.2 and 1.24.4, except the 1.30.0 persistence arm, which may
  only be stated as "the step was still present / absent at 1.30.0 for j of the stepped models".
* Any claim about machines other than the i7-13620H under Windows 11, or about thread counts other
  than 1 and 4, or batch sizes other than 1. Cross-platform replication is item 4 of the camera-ready
  plan, not this experiment.
* Naming a model, publisher or licence that is not in `pool.json` with a URL that resolved on the day
  of the fetch.
* Any number that does not trace to `a_thirdparty.csv`, `predictions.json` or `verdict.json` from the
  same run.
