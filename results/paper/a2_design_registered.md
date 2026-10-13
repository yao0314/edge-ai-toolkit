# Experiment A2: the same frozen check, on an admission rule that does not delete the class under test

**REGISTERED 2026-09-23T12:10+08:00, BEFORE ANY MODEL WAS FETCHED OR BUILT FOR A2 AND BEFORE ANY
TIMING. REVISED 2026-09-23T13:40+08:00 after an independent review of this document, STILL BEFORE ANY
MODEL WAS FETCHED OR BUILT FOR A2 AND BEFORE ANY TIMING. Every change of the revision tightens or
labels; no threshold, minimum, abort criterion or ordering rule was relaxed, and the
freeze-then-measure order is unchanged.**

**Why this document exists.** Experiment A's sourcing rule *"A file with no stated licence is not
admitted"* (`results/paper/a_design_registered.md:63-65`, the rule is on line 64) caused nine of A's eleven rejections by
itself (`results/expA/pool.json:960-962`: `{"A2": 2, "section 1.1 (licence)": 9}`), and those nine
were exactly the vision transformers and hybrids - the only class in the candidate list that could
plausibly carry the pattern under test. What survived was six CNNs and detectors from one publisher
(`results/expA/manifest.json:192-194`: 6 measured, 6 architecture families, 1 publisher), which
cannot meet A's own minimums, so A returned INCONCLUSIVE-SPARSE and nothing was measured.

**Provenance of the change.** A's outcome was known before this document was written (A registered
2026-09-23T01:35, A2 at 12:10 the same day; A's status is at `CAMERA_READY_PLAN.md:51-56`). The
licence rule was questioned by the user after that outcome, not by the designer before it. No A2
flag, prediction or latency existed when these rules were written. The change is therefore
outcome-aware in its motivation and blind in its content, and the paper states both.

**Experiment A is closed and is not revised.** Its registration, its pool, its manifest and its
INCONCLUSIVE-SPARSE outcome stand exactly as written; this document changes no file of A and no
number of A. A2 is a *second, separately registered* experiment with its own artefacts, its own
seeds and its own verdict. **Both are reported**: the paper states that A's licence rule removed the
positive class before anything was measured, and that A2 was registered afterwards with a different
admission rule and re-run from the fetch onwards. A2's result never retro-fits A's.

**What A2 changes, in one sentence.** A2 does not decide what a silent licence permits and makes no
legal claim in either direction. It records what the publisher states - which for these files is
nothing - limits this project's use to fetching, reading and timing a file the publisher has itself
made publicly downloadable with no consent wall, never redistributes it (L3), and withdraws it on
request (L7).

**This document is a DELTA.** It is a delta on `results/paper/a_design_registered.md` as sealed at
sha256 `d62b6c0398cc8fd827c004c28148544668db6976544b673f1f66ef552b102444`
(`results/expA/pool.json:7`). Every section of A that is not named below applies to A2 unchanged,
with "Experiment A" read as "Experiment A2" and with artefact paths remapped by section 5. Where
this document and A disagree, this document governs A2 only.

All paths are relative to `C:/Users/User/Desktop/SEC/edge-ai-toolkit`.

---

## 0. The delta index

| # | A says | A2 says | section |
|---|---|---|---|
| D1 | no stated licence -> not admitted (A 1.1) | licence-silent may be admitted, never redistributed | 2 |
| D2 | licence identifier + URL recorded | licence STATE recorded, plus base model, export tooling, revision, date | 2.2 |
| D3 | one pool, one table | every artefact row and every paper row is labelled DECLARED or SILENT, with both subsets shown | 2.4 |
| D4 | A9: <= 2 per architecture family, <= 6 per publisher, first come | A9a/A9b: same caps, fixed family definition, family-first ordering when a cap binds; new A10: one matrix row per base checkpoint | 2.3 |
| D5 | one arm (third-party quantized) | **Arm 1** unchanged in purpose; **Arm 2** added: third-party architecture, our quantizer, separate rules, separate weaker claim, never pooled | 3 |
| D6 | abort 6: "a licence that cannot be identified" drops the candidate | abort 6 amended: an *indeterminate* licence state, or a gated repository, drops the candidate | 7 |
| D7 | seeds 53/54/55 | Arm 1: 63/64/65. Arm 2: 73/74/75 | 5 |
| D8 | - | the checker must be byte-identical to A's (VA11) | 6 |
| D9 | - | A12: the positive control's own architecture family is admitted, labelled `home_architecture`, and every rate is printed a second time without it | 2.3, 8 |
| D10 | architecture family as A's candidate list happened to label it | family = architecture, not width or depth variant; `mobilevit` and `mobilevit_xxs` are ONE family | 2.3 |
| D11 | nothing downloaded until the user approves the candidate list | the same gate for Arm 2: the checkpoint list is approved before the first build, and no checkpoint is replaced at any stage (B10, B9) | 3.1 |
| D12 | one set of verdict names | Arm 2 has its own verdict names (OWN-QUANT-*) on A's thresholds unchanged, and its own six outcome sentences, written out in full | 8 |
| D13 | one round in practice, not registered as one | one fetch round, registered: after the first A2 checker output exists, no candidate may be added | 2.1 |

Unchanged, and restated only to be explicit: the checker (SC-B primary, SC-A secondary) and its
constants, the prediction rule `PREDICT-STEP iff sig_b and rho >= 0.07`, the technical admission
rules A1-A8, the stratum definitions F / U / U-near, **the stratum minimums**, the
freeze-then-measure order, the step rule (>= 7% with disjoint session ranges), PURE/IMPURE, the
confusion matrix, the verdict bands and their thresholds (Arm 2 uses its own *names* for the same
thresholds, section 8), the FN diagnostic and its consequences, the validity conditions VA1-VA10,
the quiet-machine rules, and the 2.5 GB pool cap.

---

## 1. Why one arm cannot answer the question

The signature needs two things at once, and they come from two different places:

* the **architecture** supplies a single-axis Transpose whose moved block is two elements wide
  (MobileViT's unfold/fold: `[256,2,4,2]` perm `[0,2,1,3]` and its three siblings,
  `a_design_registered.md:153-156`);
* the **quantizer** supplies 8-bit execution of that Transpose, by wrapping it in
  DequantizeLinear/QuantizeLinear or by handing it an INT8/UINT8 tensor, which is what makes the
  moved block two *bytes* and puts it on the path 1.24.4 added (`MLAS_BYTES` 1.23.2 `[1,4]` vs
  1.24.4 `[1,2,4]`, `results/expA/selftest.json` constants).

A's design holds the quantizer foreign and lets the architecture vary - the right design, but its
supply of foreign quantized transformers is thin and, in this candidate list, licence-silent. A2's
Arm 2 holds the quantizer fixed (ours, one recipe, registered in advance) and lets the architecture
vary freely. Neither arm alone is the experiment: Arm 1 can claim the check works on files we did
not make; Arm 2 can claim the mechanism is not peculiar to one model of ours. The pair is the
argument, and the paper must present it as a pair (section 8).

---

## 2. Arm 1 - PUBLISHED-INT8: the amended admission rules

Arm 1 is A's experiment with A's question: *can a static check on a quantized ONNX file say, before
anything is timed, which third-party models lose speed on 1.24.4 against 1.23.2 at four threads?*
Everything in A section 1.1 stands except the third bullet (the licence rule), which is replaced in
full by L1-L8. The fetch round is re-run from the beginning under these rules; A's admitted six may
be re-used unchanged if their sha256 values still match `results/expA/pool.json`, and they are
re-recorded in A2's own pool file.

### 2.1 L1-L8, replacing A 1.1 bullet 3

| # | rule |
|---|---|
| **L1** | Every candidate is assigned a **licence state**, recorded before download: **DECLARED** - the publisher states a licence identifier and the URL where it is stated resolves on the day of the fetch; **SILENT** - the publisher's licence metadata field is empty *and* no licence file is present in the repository root; **INDETERMINATE** - anything else (conflicting statements, a licence naming third-party terms we cannot resolve, a page that does not resolve). |
| **L2** | DECLARED and SILENT are both admissible. **INDETERMINATE is not admitted**, and is recorded with what was found. |
| **L3** | **No file is ever redistributed by this project.** Not in the artifact repository, not in the paper's supplement, not in an archive, not as a derived `.onnx`, `.ort` or optimized-graph dump that contains weights. What is published is the URL, the repository revision, the sha256, the byte count, the publisher, the fetch date, the checker's output (node names, shapes, perms, dispatch counts) and the measured latencies. |
| **L4** | A repository that is **gated**, or that requires accepting terms, creating an account, or any click-through to obtain the file, is **not admitted**. We do not accept terms on anyone's behalf, and a file behind a consent wall is not "publicly downloadable" in the sense A 1.1 requires. |
| **L5** | A publisher who states a licence that forbids the use we make of it (reading and timing the file) is not admitted. A licence that restricts *redistribution* does not by itself block admission, because this project does not redistribute (L3). |
| **L6** | The fetch is **one HTTP GET per admitted URL**: no crawling, no bulk mirroring, at most one re-fetch after a HEAD/size mismatch (A 1.1 bullet 4, unchanged). |
| **L7** | If a publisher asks this project to stop using a file, that candidate is dropped and the fact is recorded in the artefact; after the freeze it is recorded as WITHDRAWN and reported as such, never silently removed (VA8 still forbids re-selection). |
| **L8** | No licence text is copied into this project; identifiers and URLs only (A 1.1, unchanged). |

**Still required, unchanged from A 1.1**: third-party quantization (the 8-bit quantization was
performed and published by someone other than this project - a model this project quantizes is not
admissible in Arm 1 *at any stage*), a named publisher, a full URL that resolved on the day of the
fetch, HEAD size before the download, sha256 and byte count after it, and **no download before the
user approves the candidate list**. This registration approves rules, not URLs.

**One fetch round.** The widened candidate list is approved once. After the first A2 checker output
exists, no candidate may be added, and there is no second fetch round: if the admitted pool is under
18, abort criterion 2 fires and the round ends with the scan reported and nothing measured.

### 2.2 What must be recorded per candidate, now that the licence field may be empty

The licence identifier used to carry the provenance. With it gone for part of the pool, provenance
must be reconstructed from things that can be checked, so every candidate row in
`results/expA2/pool.json` additionally carries:

| field | what it is | why it replaces the licence as provenance |
|---|---|---|
| `licence_state` | DECLARED / SILENT / INDETERMINATE (L1) | the reader's first filter |
| `licence`, `licence_url` | as stated, or `null` for SILENT | never inferred, never inherited |
| `base_model_named_by_repo` | the upstream checkpoint the repository itself names (README, `config.json`, model card) | says what architecture and weights this is a re-export of |
| `base_model_url`, `base_model_licence_stated` | the upstream repository and the licence *it* states | **recorded for the reader, never relied on**: a re-export does not automatically inherit the base licence, and this project makes no legal claim either way |
| `export_tooling`, `export_tooling_version` | the exporter the repository names (e.g. transformers.js / optimum-onnxruntime), with version when stated | the real confound: files from one exporter share one quantizer's op coverage |
| `quantization_mode_as_published` | the file's own name or variant (`model_quantized`, `model_uint8`, `-int8`, `-qdq`) plus the measured QDQ/QOperator format and the 8-bit Q/DQ count | distinguishes two files of the same checkpoint |
| `repo_revision` | commit sha or revision id at fetch time | makes "the same file" checkable when the branch moves |
| `fetched_utc`, `url_resolved_utc` | fetch timestamps | replication depends on the file remaining published |
| `sha256`, `bytes`, `head_bytes`, `head_size_matched` | as in A | the only identity that survives link rot |
| `architecture_family`, `architecture_family_as_labelled_in_A` | the family under the fixed definition of 2.3, and A's original label beside it | the caps are only as honest as the labelling |
| `home_architecture` | `true` iff the family is the positive control's (A12) | the reader's second filter |
| `carried_over_from_A` | `true` iff this row is one of A's admitted six, re-used on a sha256 match | its flag was computed in A, before A2 existed |
| `redistributed` | hard-coded `false` | the artefact carries its own compliance record |

A candidate whose `base_model_named_by_repo` cannot be determined is still admissible if it is
otherwise DECLARED or SILENT, but it is recorded as `null` and counted in a REPORTED-ONLY tally,
because an anonymous re-export is the weakest provenance in the pool.

### 2.3 A9 amended, and the new A10, A11, A12

A9's caps were written against "ten quantized MobileViT clones are one observation, not ten"
(`a_design_registered.md:88`). With the licence-silent hubs admitted, the caps bind for the first
time and their *order of application* matters: evaluated in list order on A's own candidates, the
per-publisher cap of 6 would admit a second export of a checkpoint already in the pool and reject a
distinct architecture family (`a15`, `swin-tiny`). That is the opposite of what the cap is for.
Therefore:

* **A9a (caps, unchanged)**: no two admitted candidates share a sha256; at most **2** admitted
  candidates per architecture family; at most **6** admitted candidates per publisher.

* **Family definition, fixed now.** Architecture family is the architecture, not its width or depth
  variant. `mobilevit` and `mobilevit_xxs` are ONE family; likewise any `-tiny`/`-small`/`-base`
  pair of one architecture. A's candidate list recorded them as two; A2 corrects the labelling
  before any A2 flag exists, and `results/expA2/pool.json` records the corrected label with A's
  original beside it.

* **A9b (ordering, new)**: when a cap binds, admission is decided **family-first and
  deterministically**: sort that publisher's passing candidates by (number of that architecture
  family already admitted ascending, number of that base checkpoint already admitted ascending,
  sha256 ascending) and admit down the list until the cap is reached. No human choice, and no
  re-ordering once any flag is known. Note the direction: this ordering drops a likely-flagged row
  of the study's own architecture and buys a likely-cleared row of a new one, i.e. it works against
  the hypothesis, not for it.

* **A10 (one matrix row per base checkpoint, new)**: two admitted files that quantize the **same base
  checkpoint** (same `base_model_named_by_repo`, or the same repository with two quantization
  variants) contribute **one row to the confusion matrix** - the one with the lexicographically
  smallest sha256. The second is measured anyway and reported as a **PAIR-REPLICATE**: a free,
  registered consistency check (does the same checkpoint quantized two ways get the same flag, and
  does it step the same way?). A PAIR-REPLICATE is **never** counted in `|F|`, `|U|`, `|U-near|`, the
  family count, or any rate.

* **A11 (export-tooling census, new, REPORTED-ONLY, never admission)**: the number of distinct
  `export_tooling` values in the measured set and in F is recorded in the manifest and printed in the
  paper. It gates nothing; it governs what may be *said* (section 8, bar P5).

* **A12 (home architecture, new).** A candidate whose architecture family is the positive control's
  (`mobilevit` / `mobilevit_xxs`, or any MobileViT variant) is admitted and measured, and is
  labelled `home_architecture: true` in every artefact. `|F|`, `|U|`, `H`, `M` and the Fisher p are
  computed and printed twice: over all matrix rows (the registered primary) and over the non-home
  rows alone (REPORTED-ONLY). If the two fall in different verdict bands, the paper must say so in
  the sentence that reports the verdict, exactly as §2.4 requires for DECLARED/SILENT.

  Symmetric labelling, so the same question cannot be asked of the other side and go unanswered: a
  candidate whose family is one of the two **negative** controls' (`deit`, and the third control's
  family) is labelled `control_architecture: "negative"` in every artefact and counted in a
  REPORTED-ONLY tally. This label gates nothing and no rate is recomputed without those rows; it
  exists so the reader can see them.

Technical rules A1-A8 are unchanged, including A2's fixed `C,H,W`. Considered and **rejected**:
relaxing rule A2 by pinning free spatial dimensions with `a_check.py --free-dim` (`a_check.py:886`),
which the checker supports. Rejected because the prediction is a function of the input shape; if we
choose the resolution, the prediction becomes ours rather than the publisher's, and the file-only
check stops being the thing a reader can run on their own file. `depth-anything-v2-small` therefore
stays excluded in A2 - for a shape reason, as in A, not a licence reason.

### 2.4 How DECLARED and SILENT are labelled, everywhere

* `results/expA2/pool.json`, `manifest.json`, `predictions.json` and `verdict.json` carry
  `licence_state` on **every** row, and `counts.by_licence_state` at the top level.
* `results/paper/a2_predictions_frozen.md` prints one line per model with a `lic` column (`D` plus
  the identifier, or `S` plus the literal words "not stated by the publisher"), and prints the
  per-stratum counts **twice**: once for the whole set, once split D / S.
* The paper's model table carries the same column and the same subtotals, and its caption states:
  *"No model file is redistributed by this work; each row is identified by its URL, repository
  revision and sha256."*
* The confusion matrix and the rates `H`, `M`, `p` are computed **on all matrix rows** - that is the
  pre-registered primary, fixed now so it cannot be chosen later - and the same table is printed for
  the DECLARED subset alone and for the SILENT subset alone, REPORTED-ONLY. If the two subsets fall
  in different verdict bands, the paper **must** say so in the sentence that reports the verdict.
  This split never gates: a difference between subsets is a finding about the pool, not a failure of
  the run.
* Every row carried over from A carries `carried_over_from_A: true` and A's own sha256.
  `a2_predictions_frozen.md` and the paper's model table state how many rows were carried over and
  that their flags had already been computed in A; the verdict sentence states the count in every
  band.
* The artifact repository (`Pric0123/taai2026-int8-cpu-artifact`) receives a
  `scripts/paper/a2_fetch_pool.py` that re-downloads from the recorded URLs and verifies each
  sha256, and **no `.onnx` file from the pool**. The test that enforces this is VA12, which is
  broader than a single directory and a single extension; VA12 governs, and no narrower test
  satisfies this bullet.

---

## 3. Arm 2 - OURS-INT8: third-party architecture, our quantizer

**Decision: Arm 2 is ALLOWED, and is registered here as a separate arm with its own admission rules,
its own freeze, its own verdict names and its own, weaker, claim.**

**Why it is allowed.** The question the mechanism poses is architectural - does a model whose layout
code moves two-element blocks lose four-thread speed on 1.24.4 when those Transposes run in 8 bits -
and Arm 1 can only ask it of architectures somebody else happened to quantize and publish under
60 MB. Arm 2 reaches the class directly. It also has a control Arm 1 *structurally cannot have*:
because we quantize, we hold the FP32 sibling from the same weights, so Experiment 26's full step
rule applies - the clause A had to drop (`a_design_registered.md:311-313`: "Third-party quantized
files have no FP32 or dynamic sibling from the same quantizer, so that clause cannot be applied
here"). A flagged Arm 2 model that steps while its FP32 sibling does not is a far tighter
observation than a flagged Arm 1 model that steps.

**Why it is not enough on its own, and must never be merged.** Our quantizer decides which
Transposes execute in 8 bits, so an Arm 2 result confounds architecture with our own op coverage: it
is Experiment 27's file with more shapes, not evidence about files we did not make. Pooling the arms
would launder that weakness through Arm 1's independence. They are reported side by side and never
summed.

### 3.1 Admission, B1-B10

The letters `B1`-`B10` in this section are **admission rules for Arm 2**. The letters `P1`-`P10` in
section 8 are **bars on what the paper may write**. The two series are distinct and are never
cross-referenced by number alone.

| # | rule |
|---|---|
| **B1** | The base checkpoint is public, its licence is **DECLARED** (identifier + URL), and that licence permits creating a derivative for research use. Arm 2 *builds* a derivative, which is exactly the case where the licence does matter, so the A-style licence requirement is kept here and is not relaxed. Recorded: source (`timm:<name>` or a named HF repository), revision, licence, URL, date, and the checkpoint's own hash where the hub exposes one. |
| **B2** | The quantized file we produce is **not redistributed either**. The artifact publishes `scripts/paper/a2_build.py`, the recipe, the base checkpoint id and the sha256 of the built file, so a reader rebuilds rather than downloads. |
| **B3** | Export is ours and fixed: batch 1, the model's own default input resolution as its `pretrained_cfg` states it (recorded per model), float32 NCHW, static shapes, via `core/export.py:export_onnx`, opset recorded. No dynamic axes. The per-model resize and normalization are implemented in `a2_build.py`; `adapters/classification.py` is **not** the implementation of this rule and is cited in B6 only for the calibration index list and the loader order (its own normalization is the fixed ImageNet MEAN/STD at `adapters/classification.py:21`, and `pretrained_cfg` does not appear in that file). |
| **B4** | **One quantization recipe for every Arm 2 model, fixed now**, from `core/quantize.py` `DEFAULT_QUANT_CFG`: `method: static` (:60), `per_channel: True` (:61), `activation_type: uint8` (:62), `weight_type: int8` (:63), `calibrate_method: minmax` (:64), `calib_batches: 16` (:78), `keep_fp32: []` (:79), `quant_format: qdq` (:88), `reduce_range: False` (:98), `op_types_to_quantize: None` (:99), and **`auto_keep_fp32: "none"`**, overriding the `"rules"` default (:81). No `search`, no `auto` calibration, no QAT, no fine-tuning, no per-model exclusion list, no second attempt with a different recipe. |
| **B5** | `auto_keep_fp32: "none"` is not a detail: `rules`/`search` make the set of nodes kept in FP32 depend on calibration data and SQNR, and a data-dependent exclusion set would make the QDQ topology - and therefore the flag - a function of the calibration images. With `none`, the calibration set can only change *scales*, never which Transposes execute in 8 bits. (`none` is a supported value: `core/quantize.py:14,579-581`.) |
| **B6** | **Calibration set, fixed and recorded**: the same source images and the same indices for every model - the 512 indices returned by `adapters/common.py:91-97` `subset(calib, 512, seed=1)` - a deterministic permutation under `torch.Generator().manual_seed(1)`, **not** the first 512 by index - consumed in the fixed order of `adapters/classification.py:97` (`batch_size=32, shuffle=False`), so B4's `calib_batches: 16` covers exactly those 512 images. Cited: `adapters/classification.py:41` (`calib_size: 512`), `:72` (the CIFAR-10 calibration dataset), `:92` (the `subset` call), `:97` (the loader); `adapters/common.py:91-97` (the `subset` function). The resolved index list and its sha256 go in `builds.json`. Images are resized and normalized per model by `a2_build.py` from that model's own `pretrained_cfg` (B3). An `ImageFolder` of natural images may be substituted **only** if chosen and recorded before the first build, and then it is used for every model. |
| **B7** | The built file must pass **A1-A8 unchanged** (the shape rules hold by construction) and must report `int8_quantized_nodes > 0`; a file where nothing was quantized is **FAILED-TO-BUILD**, not a cleared model. `int8_sqnr_db` is recorded, REPORTED-ONLY. |
| **B8** | **The three controls' checkpoints are barred as Arm 2 rows.** MobileViT-XXS at 64 px is the positive control (`results/classification_mobilevit/models/p00_int8.onnx`, sha256 `911de4ee...`, `a_design_registered.md:110-114`); re-quantizing the checker's own worked example and scoring it would be scoring the checker on its training case. The same bar applies to the DeiT and the third control checkpoint. They stay in the run as controls only. |
| **B9** | Caps as in A9a/A9b/A10 and the fixed family definition of 2.3, with "publisher" read as "model hub or library": at most 2 checkpoints per architecture family, one matrix row per checkpoint. A build that fails is recorded as FAILED-TO-BUILD and **is not replaced at any stage** (VA8). |
| **B10** | The Arm 2 checkpoint list is fixed **before the first build**. Every checkpoint, its source id, its revision and the reason it was chosen are written to `results/expA2/arm2_candidates.json`, hashed, and put to the user for approval before `a2_build.py` is run once. After that file exists, no checkpoint is added, substituted or removed for any reason other than B7's mechanical FAILED-TO-BUILD, and every FAILED-TO-BUILD is listed by name with its failure. The recorded reason may not be "expected to flag" or any synonym: checkpoints are chosen for architecture-family coverage, and that census is fixed before any checker runs. |

### 3.2 Arm 2's step rule - stronger than Arm 1's

Everything in A section 5.1 applies, **plus** the sibling clause A had to drop:

* Each Arm 2 model is measured as **files from the same weights**: `p_fp32` (FP32 export), `p_int8`
  (the B4 recipe), and REPORTED-ONLY `p_int8_dyn_mm` where it builds (`method: dynamic`,
  `op_types_to_quantize: [MatMul, Gemm]`).
* **STEPPED (Arm 2)** iff the static file steps by the A 5.1 rule (>= 7% at four threads with
  disjoint session ranges) **AND** its FP32 sibling moves by less than 7% between the two releases. A
  model whose FP32 sibling also moves is **NOT** counted as a step; it is listed as RELEASE-DRIFT and
  reported. This is the V0 rule (FINDINGS.md:3493-3494, V0 RESULT: static +30.8%, FP32 -0.2%,
  dynamic -0.7%) applied to every Arm 2 model.
* PURE (one-thread invariance) is required as in A 5.1. The dynamic sibling is reported, never
  gating.

### 3.3 Arm 2's cells and cost

Cells = models x {fp32, int8} x {1.23.2, 1.24.4} x {1, 4} threads, 5 sessions each = 20 sessions per
model. At this project's own 8-12 s per session for larger files (`a_design_registered.md:445-447`),
13 Arm 2 models is 260 sessions, 35-52 minutes on the machine, plus build time, which is not a
measurement and never runs while anything is being timed. Dynamic siblings, if built, add 10
sessions per model, run last, and are the first thing dropped if the budget of section 7 is short.

---

## 4. Strata, selection and minimums

**Decision: the minimums do not change, in either arm.** `|F| >= 5`, `|U| >= 8`, `|U-near| >= 4`, and
**at least 8 distinct architecture families**, per arm, counted on **matrix rows** (A10:
PAIR-REPLICATEs do not count) and under the fixed family definition of 2.3. Pool target 24 /
minimum 18 / maximum 40 per arm, unchanged. Selection into the measured set (A 1.3) is unchanged.

**Justification, given that A failed these very minimums.** They were fixed before any datum existed,
for a stated reason - "A positive class of one, two or three models is not a criterion; it is a
second anecdote, and this registration refuses it in advance" (`a_design_registered.md:106-109`) -
and the arithmetic behind them is arithmetic, not taste: with `|F| = 5` all stepping and `|U| = 8`
none stepping the one-sided Fisher p is `1/C(13,5) = 0.00078`, and with 4 of 5 and 1 of 8 it is
0.032 (`a_design_registered.md:336-339`). Lowering a threshold *because the first attempt did not
reach it* is the exact move a pre-registration exists to prevent, and it would be visible to any
reader who compares the two documents. The remedy for a thin positive class is a wider fetch and a
second arm - both registered here - not a smaller bar. The **cost** is accepted and stated in
advance: A2 may return INCONCLUSIVE-SPARSE again, in one arm or in both, and that is a publishable
outcome with its own sentence (A section 8, unchanged).

**Expected effect on the strata.** Arithmetic on A's existing 17-candidate list, which the fetch
round will widen; these are not predictions about what the checker will flag:

* The 9 licence rejections become admissible (`a07`-`a11`, `a13`-`a15`, `a17`), but the caps bite:
  Xenova published 8 of them, so A9a's per-publisher cap of 6 holds it to 6. Under the fixed family
  definition, Xenova's 8 fall in **5** families - mobilevit (four files: two exports of
  `mobilevit-small`, two of `mobilevit-xx-small`), deit, yolos, detr, swin - so A9b admits one file
  from each of the five families first, and spends the sixth slot on the second MobileViT
  *checkpoint* rather than on a second export of a checkpoint already admitted. The two files the
  cap drops are therefore the surplus export of each MobileViT checkpoint (which of the two pairs'
  members is dropped follows from the sha256 tiebreak and is not knowable before the fetch); either
  way they are same-checkpoint duplicates that A10 would have made PAIR-REPLICATEs anyway.
  `onnx-community/dinov2-small` (`a11`, also licence-silent) adds a 7th publisher-distinct row.
* That is **6 (onnx/models CNNs, unchanged) + 6 (Xenova) + 1 (onnx-community) = 13 matrix rows in 12
  architecture families** (shufflenet_v2, mobilenet_v2, resnet50, efficientnet_lite4,
  emotion_ferplus, ultraface, mobilevit, deit, yolos, detr, swin, dinov2) - the 8-family minimum is
  met for the first time, `|U| >= 8` is reachable, and `|F| >= 5` becomes possible instead of
  arithmetically impossible. **2 of those 13 rows are `home_architecture` (A12)**, both MobileViT,
  which is exactly why A12 requires every rate to be printed a second time without them.
* **13 is still below the pool minimum of 18**, and abort criterion 2 ("pool under 18 admitted
  models: no measurement; report the scan alone") is **not** relaxed. The fetch round must therefore
  widen the candidate list, preferring **distinct base checkpoints, distinct publishers and distinct
  export tooling** over more files from the hubs already represented, and it is **one** round (2.1).
  The user's standing decision of 2026-09-23 already covers this
  (`results/paper/CAMERA_READY_PLAN.md:37-39`: widen the list, keep files small, delete rejects,
  stay under the 2.5 GB cap; `results/expA/pool.json:13-14` records 64,991,667 bytes used of a
  2,500,000,000 cap).
* A2's checker output on A's six admitted CNNs was recorded as an indicative flagged count of 0
  (`results/paper/CAMERA_READY_PLAN.md:53`). A2 does not rely on that number: every model in A2,
  including the six, is re-checked and re-frozen inside A2's own artefacts, and every one of them
  carries `carried_over_from_A: true` in the artefacts and in the paper's table (2.4).
* Arm 2's 8-family minimum is reachable by construction, since we choose the architectures - which
  is precisely why B10 fixes and publishes the checkpoint list, with its reasons, before the first
  build. Its binding constraints are build time and rule A7 (<= 60 MB), which excludes most
  base-size transformers even at INT8 and keeps the arm to small and tiny variants.

---

## 5. Artefacts, paths, seeds

A's artefact table (section 3) applies with `expA` -> `expA2`, `a_` -> `a2_`, and one new column,
`arm`. Nothing of A is overwritten.

| # | artefact | path |
|---|---|---|
| 1 | candidate pool, both arms, one file, `arm` on every row | `results/expA2/pool.json` |
| 2 | measured set | `results/expA2/manifest.json` |
| 3 | predictions (`sig_b`, `sig_a`, dispatches, `A_ms`, `rho`, class, node tables) | `results/expA2/predictions.json` |
| 4 | checker self-test | `results/expA2/selftest.json` |
| 5 | human-readable frozen list, with the `lic` column, the D/S subtotals, `home_architecture` and `carried_over_from_A` | `results/paper/a2_predictions_frozen.md` |
| 6 | the seal: sha256 of 1-5 and of 10-11, UTC, per-stratum and per-arm counts, appended to `results/paper/FINDINGS.md` | before any timing |
| 7 | quiet-machine record | `results/expA2/precondition.json` |
| 8 | timing, Arm 1 | `results/paper/a2_published_int8.csv` |
| 8b | timing, Arm 2 | `results/paper/a2_ownquant.csv` |
| 9 | verdicts, one per arm, in one file, each under its own arm's band names (section 8) | `results/expA2/verdict.json` |
| 10 | Arm 2 build record: base checkpoint, revision, licence, export opset, recipe, the resolved calibration index list and its sha256, `int8_sqnr_db`, `int8_quantized_nodes`, sha256 of every built file | `results/expA2/builds.json`, written before the freeze |
| 11 | Arm 2 checkpoint list with the reason for each choice (B10), hashed and approved before the first build | `results/expA2/arm2_candidates.json`, written before any build |

**Seeds**: Arm 1 `--seed 63` primary, 64 for the single registered re-run, 65 for the 1.30.0
persistence arm; Arm 2 `--seed 73`, 74, 75. A's seeds 53/54/55 are not reused, so no A2 row can be
pooled with an A row by accident. **The two arms are measured in two separate invocations, written to
two separate CSVs, and are never concatenated.** The three controls run in **both** invocations
(VA3 applies to each arm's own run).

**`screen_ms` is re-measured for every A2 row, including the six carried over from A.** The
screening pass is itself a timed measurement (`a_design_registered.md:233`) and `screen_ms` is
`rho`'s denominator and therefore part of the prediction (`a_design_registered.md:191-192`); the
machine's state has changed since A, so no `screen_ms` value is carried over. Only the file identity
(sha256) is carried over.

`scripts/paper/a2_build.py` and the amended pool builder must exist, be hashed into the seal, and -
like `a_analyse.py` in A - the analysis script must be exercised on synthetic data first, with A's
fifteen registered synthetic scenarios (`a_design_registered.md:270-278`) plus five:
(16) a PAIR-REPLICATE disagreeing with its matrix row -> reported, verdict unchanged;
(17) an Arm 2 model whose FP32 sibling also moves -> RELEASE-DRIFT, not a step;
(18) rows from both arms present in one CSV -> refuse to analyse;
(19) the all-rows verdict and the non-home-rows verdict falling in different bands -> both printed
and the difference stated, verdict unchanged (A12);
(20) an Arm 2 verdict name drawn from Arm 1's vocabulary (CONFIRMED / PARTIAL / MIXED / REFUTED) ->
refuse to write `verdict.json`.

---

## 6. Validity conditions

VA1-VA10 apply unchanged, per arm. Five are added:

* **VA11 - the checker is byte-identical to A's.** `scripts/paper/a_check.py` must hash to
  `c333fca2e6f465d2344839f34ee8c245bc3106aa26b05f60b650f40cce527a18` and
  `scripts/paper/transpose_probe.py` to
  `fde66c9c550c7c55f681b6d78622e2bb37abc681fbcea1df80742a0e64fe0310` (`results/expA/selftest.json`,
  provenance). A2 changes *admission*, not the check. If the checker must change for any reason, A2
  is void and is re-registered. The self-test is re-run for A2 and must again return 59 of 59 PASS
  (`results/expA/selftest.json`: `n_checks` 59, `n_pass` 59, `result` PASS).
* **VA12 - no redistribution.** A test in the artifact repository asserts, **over the whole
  repository and its full git history**, that no path matching `*.onnx`, `*.onnx_data`,
  `*.onnx.data`, `*.ort`, `*.pb`, `*.bin`, `*.npz`, `*.safetensors`, `*.tflite`, or any archive
  containing one, is or has ever been tracked; and that every model row in every published artefact
  carries either URL + revision + sha256, or a rebuild recipe + sha256. Optimized-graph dumps
  written by SC-A (`a_check.py:311`, `optimized_model_filepath`) are deleted as soon as the per-node
  table is extracted and are named in no artefact. The test runs in CI and its failure blocks
  publication, not the verdict.
* **VA13 - one recipe.** Every Arm 2 model was built by the same `a2_build.py` with the same recipe
  hash recorded in `builds.json`. Any per-model deviation voids Arm 2 (and only Arm 2).
* **VA14 - the arms are never pooled.** `a2_analyse.py` refuses a CSV containing both arms and
  computes two verdicts, each under its own arm's band names. No rate, matrix, count or p-value is
  ever computed across arms.
* **VA15 - licence state is recorded before the download.** A row whose `licence_state` was written
  after its `fetched_utc` is invalid; the row is dropped and counted.

---

## 7. Schedule and abort criteria

Hand-in is 2026-09-28 (`results/paper/CAMERA_READY_PLAN.md:1`); one measurement at a time on this
machine.

| milestone | deadline | if missed |
|---|---|---|
| widened candidate list (URLs, licence states, HEAD sizes) put to the user; Arm 2 checkpoint list (B10) put to the user | 2026-09-24 18:00 | Arm 1 aborts to scan-only; Arm 2 may still run |
| user approves; downloads, admission and Arm 2 builds complete | 2026-09-25 18:00 | measure whatever is admitted if the minimums hold; else INCONCLUSIVE-SPARSE (Arm 1) / OWN-QUANT-SPARSE (Arm 2) for that arm |
| self-test, predictions, freeze, seal in FINDINGS.md (both arms) | 2026-09-26 12:00 | no measurement may start; the round ends with frozen unmeasured predictions, which is still a publishable pre-registration |
| Arm 1 primary run | 2026-09-26 20:00 | the wall-clock rule of A section 7 |
| Arm 2 primary run | 2026-09-27 12:00 | as above; if only one arm fits, Arm 2 is dropped first |
| analysis, independent ruling, FINDINGS entry | 2026-09-27 20:00 | report as unruled and keep it out of the paper |

Abort criteria 1-5 and 7 of A are unchanged and apply per arm. **Abort criterion 6 is amended** - the
only one touched, and only in its licence clause:

> 6'. Any download larger than 60 MB, any repository that is gated or requires accepting terms (L4),
> and any candidate whose licence state is INDETERMINATE (L1), is dropped - never "included
> provisionally". A SILENT licence state is **not** a reason to drop.

One addition, from the standing deadlock rule (`results/paper/CAMERA_READY_PLAN.md:43`): if either
arm needs more than three repair attempts on the same obstacle, an agent that has not worked on it
rules STUCK or CONTINUE; on STUCK that arm is recorded with what blocked it and the other arm
proceeds.

---

## 8. What the paper may write, and what A2 may not claim

**Arm 1** uses A section 8's sentences unchanged, with the arm named in each of them.

**Arm 2 uses its own verdict names and its own sentences, written out in full below.** Arm 2's bands
are OWN-QUANT-CONSISTENT / -PARTIAL / -MIXED / -REFUTED / -SPARSE / -INCONCLUSIVE, on A's thresholds
unchanged. The words CONFIRMED, PARTIAL, MIXED and REFUTED, unqualified, belong to Arm 1 and may not
be written of Arm 2 anywhere - not in `verdict.json`, not in FINDINGS.md, not in a table header, not
in a figure caption. **No Arm 2 verdict may appear in any sentence, abstract or caption that does not
also state Arm 1's verdict.**

Three fixed phrases:

* Arm 1 sentences say **"quantized and published by others"**, and for the SILENT subset **"publicly
  available files for which the publisher states no licence; no file is redistributed by this
  work"**.
* Arm 2 sentences say **"architectures published by others, quantized here by this project's own
  quantizer under a single fixed recipe"** - in full, every time, because "here" is the word a
  quotation drops.
* No sentence may report a rate, a count or a p-value computed across both arms, and no sentence may
  report a subtotal mixing DECLARED with SILENT without also giving the split.

### 8.1 Arm 2's six outcome sentences, fixed now

ASCII. `N` = Arm 2 matrix rows, `k` = flagged, `H`, `M`, `p` from `verdict.json`. Each sentence is
subject to the rule above: it may not stand without Arm 1's verdict beside it.

* **OWN-QUANT-CONSISTENT** (A's CONFIRMED thresholds, unchanged: `H >= 0.80` AND `M <= 0.10` AND
  Fisher `p < 0.05` AND `IMPURE <= TP` AND the section 4 sizes AND every validity condition) -
  "Applying the check to N architectures published by others, quantized here by this project's own
  quantizer under a single fixed recipe, with every prediction frozen and hashed before any
  measurement, k of them were flagged as carrying an 8-bit Transpose with a two-byte moved block; H
  of the flagged models lost more than 7% of their four-thread speed on ONNX Runtime 1.24.4 against
  1.23.2 with disjoint session ranges, no one-thread change and an FP32 sibling from the same
  weights that did not move, against M of the cleared models (Fisher one-sided p = p). Because this
  project performed the quantization, this result says nothing about files it did not make."
* **OWN-QUANT-PARTIAL** (A's PARTIAL thresholds) - "Of N architectures published by others,
  quantized here by this project's own quantizer under a single fixed recipe, with frozen
  predictions, H of the flagged models stepped against M of the cleared ones (p = p); under this one
  recipe the check is a useful positive screen across architectures, and this arm cannot say whether
  it is one on files this project did not quantize."
* **OWN-QUANT-MIXED** (A's MIXED condition) - "The frozen predictions on N architectures published
  by others, quantized here by this project's own quantizer under a single fixed recipe, gave H on
  the flagged and M on the cleared; under this one recipe the file-level signature by itself does not
  separate the two groups, and we report the table rather than a rule."
* **OWN-QUANT-REFUTED** (A's REFUTED thresholds) - "Frozen predictions from the file-level signature
  did not identify which of N architectures published by others, quantized here by this project's
  own quantizer under a single fixed recipe, lose four-thread speed on 1.24.4 (H flagged, M
  cleared), so the mechanism established for our one file does not generalize across architectures
  even when this project controls the quantizer."
* **OWN-QUANT-SPARSE** (A's INCONCLUSIVE-SPARSE condition) - "Of N architectures published by
  others, quantized here by this project's own quantizer under a single fixed recipe, only k carried
  the two-byte-block signature, too few to test a prediction rule; we report the scan and the k
  models' measurements without claiming a criterion."
* **OWN-QUANT-INCONCLUSIVE** (A's INCONCLUSIVE condition) - "A pre-registered prediction test on
  architectures published by others, quantized here by this project's own quantizer under a single
  fixed recipe, was run and its registered validity condition VAn was not met (state which), so it
  returns no verdict."

A's "design facts usable in every branch", its FN sentences and its "barred in every branch" list
apply to both arms unchanged.

### 8.2 The mandated paragraph about A

The paper must state, in the experiment's own description **and in the abstract** (the abstract
clause may be shortened but must name A's exclusion and A2's re-registration): *Experiment A was
registered with an admission rule that excluded files whose publisher states no licence; that rule
removed every vision transformer and hybrid from the candidate list, so A returned
INCONCLUSIVE-SPARSE on six CNNs from one publisher and nothing was measured. A2 re-registered the
admission rule so that a publicly downloadable file may be read and timed without being
redistributed. Both registrations and both outcomes are reported.*

The artifact carries **both registrations' full text, their sha256 and their registration
timestamps**, so a reader can diff them without taking this document's word for what changed.

### 8.3 Barred in A2, in addition to everything barred in A section 8

The letters `P1`-`P10` are bars on what the paper may write. They are a different series from Arm 2's
admission rules `B1`-`B10` (section 3.1).

* **P1** - any statement about the licence, the licensing status, the permissions or the legal
  standing of a SILENT model, including any suggestion that the base model's licence applies to the
  re-export. The paper reports what the publisher states, which for these files is nothing.
* **P2** - "we release", "we provide", "included in the artifact", for any model file of either arm.
  The artifact provides URLs, revisions, hashes, a fetch script and a build script.
* **P3** - any claim that the pool is a random, representative or exhaustive sample of published
  quantized models. It is filtered by <= 60 MB, static 8-bit, fixed rank-4 NCHW float input,
  loadable on both releases, and by what a few re-export hubs happen to have published. Every
  prevalence sentence is bounded to "of the N admitted models".
* **P4** - "the official export", "the publisher's model", or any implication that a re-export is
  endorsed by the owner of the base checkpoint.
* **P5** - the word **"independent"** of Arm 1's quantizers, unless the measured set carries at least
  **two distinct `export_tooling` values in F** (A11). With one exporter behind every flagged model,
  the finding may be about that exporter's op coverage, and the paper must say so instead.
* **P6** - any external-validity claim from Arm 2. Arm 2 may claim that the signature and its
  four-thread cost appear in architectures beyond our one model under one fixed recipe; it may
  **not** claim that the check works on files we did not make, nor that the effect is independent of
  our quantizer, nor anything about how often published models carry it.
* **P7** - any accuracy, quality or deployability claim about any model of either arm. No
  third-party file is evaluated on a task here, and Arm 2's files are built by a recipe chosen for
  uniformity, not for accuracy (`int8_sqnr_db` is reported for the record only; `core/quantize.py`'s
  own docstring warns at `:39` to read SQNR as "how far the outputs moved", not as accuracy).
* **P8** - presenting a PAIR-REPLICATE as an independent observation, or counting it in any rate.
* **P9** - claiming that a replication will succeed. A SILENT file can be withdrawn or altered by its
  publisher at any time, so the paper writes "the file whose sha256 is ..., as fetched on <date>",
  and replication is defined as matching that hash.
* **P10** - no sentence may state `|F|`, `H` or `p` without also stating how many of the flagged rows
  were MobileViT re-exports.
