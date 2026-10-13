# Experiment W: is the moved-block width the variable? — a synthetic width sweep

**REGISTERED 2026-09-24, Taipei, BEFORE ANY TIMING ROW EXISTS.**
Designed by an agent that is not the author of the claim under test and that ran no measurement while
writing this file (project rule: *"Any check meant to refute a claim is designed by a separate agent,
never by the claim's author."*, `CLAUDE.md`). Nothing in this document was chosen after seeing a
latency.

Harness: `scripts/paper/ort_version_bisect.py`, **unedited**. Dispatch rule: imported from
`scripts/paper/transpose_probe.py` (`predict`, `_single_axis`, `_is_reshape`, `MLAS_BYTES`) — the same
import path `scripts/paper/a_check.py` uses, so this design cannot drift from the rule Experiments
27/27b/A were registered with. Vocabulary (cell, pair-cell, MOVED / UNRESOLVED / STILL, the 7 % floor,
no outlier rule) is taken unchanged from `results/paper/b_design_registered.md` §4.

---

## 0. The claim, and what is actually unproven about it

The claim on the table:

> ONNX Runtime 1.24.4 (PR #25680, commit 42869ca, `has_mlas_transpose<uint16_t>`) routes 8-bit
> `Transpose` nodes whose **moved block is 2 bytes** to MLAS's multi-threaded transpose; on this
> laptop that adds thread-pool dispatches per inference and makes a byte-identical static-INT8
> MobileViT-XXS slower at four threads (+40.3 % at 64 px, +43.9 % at 256 px, `B_RESULTS.md` §3.1),
> while one thread, the FP32 sibling and the dynamically-quantized sibling do not move.

What is established: the regression is real, reproduced in fresh environments on a different night,
and localised to Transpose by Experiment 27b, whose micro-models reproduced **+0.417 ms of the full
model's +0.48 ms step** (`FINDINGS.md`, 27b RESULT).

What is **not** established, and is the entire object of Experiment W:

> **27b's micro-models are the regressing model's own Transposes.** Their shapes — `[256,2,4,2]`,
> `[256,4,2,2]`, `[160,2,2,2]`×2, perm `[0,2,1,3]` — were copied out of MobileViT-XXS
> (`transpose_probe.py:W2_IDX`). Every observation supporting the width law so far is an observation
> about four shapes that MobileViT happens to contain. The width has never been *set* by the
> experimenter and *swept*.

27b's coverage of the width axis, stated exactly:

| width | 27b evidence | confound |
|---|---|---|
| 1 byte | **none** | — |
| 2 bytes | `tp_u8_w2` +0.417 ms, `tp_f16_w2` +0.435 ms | both carry MobileViT's shapes |
| 4 bytes | `tp_f32_w4` −0.016 ms | different dtype **and** 2× the bytes of `tp_f16_w2` (the MECH skeleton drops the last axis, so the f32 arm moves 8192 B where the f16 arm moves 4096 B) |
| 8 bytes | **none** | — |

So the law rests on one positive width and one negative width, and the negative one differs from its
comparator in dtype and in size simultaneously. Experiment W sweeps width across **1, 2, 4 and 8
bytes**, reaches each width by **two or three independent dtype routes**, holds bytes / loops / node
count / M / N fixed within a width, and adds a **16× dose ladder on the dispatch count at constant
bytes moved** — which 27b has no equivalent of at all.

---

## 1. The dispatch rule, and where this design departs from `a_check.py`

### 1.1 The rule, as the frozen code states it

From `scripts/paper/transpose_probe.py:predict()` (not restated arithmetically anywhere in this
design; the builder and the analysis both call it):

```
MLAS_BYTES = {"1.23.2": {1, 4}, "1.24.4": {1, 2, 4}}

reshape-copy                      -> 0 dispatches
not single-axis                   -> 0 dispatches ("untyped (serial)")
single-axis, kind "out": loops = prod(dims[:to]),   block = prod(dims[from+1:])
single-axis, kind "in" : loops = prod(dims[:from]), block = prod(dims[to+1:])
nbytes = block * elem_bytes
nbytes in MLAS_BYTES[version] -> MlasTranspose, dispatches = loops if (threads > 1 and M > 1) else 0
else                          -> scalar,        dispatches = 0
```

Because `from > to` for "out" and `to > from` for "in", `block` is exactly
`prod(input_dims[max(from, to) + 1:])` and the moved-block width is
`elem_bytes * prod(input_dims[max(from,to)+1:])` — **the definition the task statement gives, and the
one `a_check.py` uses.** This design adopts it unchanged and computes it by calling the same function.

The rule has four consequences that make a clean 2×4 experiment possible, and they are the skeleton of
the whole design:

* width **2** is the *only* width whose routing differs between 1.23.2 and 1.24.4;
* widths **1** and **4** are MLAS on **both** releases — a null produced by "parallel before and after";
* width **8** is scalar on **both** releases — a null produced by "serial before and after";
* at **1 thread** the dispatch count is 0 at every width — a null produced by the predicate itself.

Two structurally *different* kinds of null, plus a thread-count null, all predicted by one rule.

### 1.2 Where this design departs from `a_check.py`, and why

Agreements (no departure): the width formula; `MLAS_BYTES`; the `threads > 1 and M > 1` predicate;
`FLOOR_REL = 0.07`; `DISPATCH_US = 0.52`; the REPORTED-ONLY magnitude band `0.5 ≤ ratio ≤ 2.0`.

Three deliberate departures, each stated here rather than discovered later:

1. **`ESIZE_8BIT = 1` does not apply to two of the three arms.** `a_check.py`'s SC-B admits only
   8-bit-executing Transposes and therefore hard-codes `esize = 1` (`a_check.py:74`). This design's
   fp16 and fp32 probes have `esize` 2 and 4. This is a **widening, not a contradiction**: `predict()`
   takes `esize` as a parameter, and `a_check.py`'s SC-A already passes 2 and 4 through
   `PROFILE_ELEM_BYTES`. Consequence for validity: **SC-B is not a valid checker for the fp16/fp32
   probes** and is run only on the 12 uint8 files (§7 V1). SC-A, which reads the executed dtype, is
   run on all 19.
2. **`THREADS = 4` is not fixed.** `a_check.py:73` evaluates the rule at 4 threads only. Experiment W
   evaluates it at 1, 2 and 4 by passing `threads` to `predict()`. The rule is unchanged; the
   consequence is new: since the predicate is `threads > 1`, **2 threads is predicted to add exactly
   the same dispatch count as 4 threads**. No measurement in this study has ever been made at 2
   threads (`B_RESULTS.md` is 1 and 4). §5 P1 registers it as a prediction.
3. **`prediction()` / PREDICT-STEP / PREDICT-SAFE is not used.** That classifier
   (`a_check.py:483-511`) needs a `screen_ms` read from a runner manifest and answers "should this
   *file* step". Experiment W's verdicts come from measured release deltas, not from a predicted
   class. Predicted dispatch counts and the derived `A_ms = dispatches × 0.00052 ms` are recorded as
   **REPORTED context only** and can never change a verdict.

---

## 2. The probe family

### 2.1 The one shape template

Every probe Transpose has

> **dims = `[L, A, N, K]`, perm = `[0, 2, 1, 3]`, A = 2.**

Verified by calling `_single_axis([0,2,1,3])` → `("out", 2, 1)`, so `from = 2`, `to = 1`, and therefore

* `loops   = prod(dims[:1]) = L`
* `block   = prod(dims[3:]) = K` elements → **moved-block width `W = K × elem_bytes`**
* `M = A = 2`, `N = dims[2]`
* total elements `= L·2·N·K`, **total moved bytes `= L·2·N·W`**

This is the same permutation family as the four MobileViT nodes (`[256,2,4,2]`, perm `[0,2,1,3]` →
loops 256, W = 2 B, M = 2, N = 4), so the probes sit in the regime the regression lives in — but the
`[L, A, N, K]` values are chosen by this design, not copied from any model. `A = 2` is required
(`M > 1`) for the dispatch predicate to fire at all; `N > 1` is required or `_is_reshape` returns True
and the node degenerates to a reshape-copy with zero dispatches. Every registered row is asserted
against both conditions at build time.

### 2.2 The confound this design exists to avoid, and how it is avoided

A naive sweep sets `K = 1, 2, 4, 8` and leaves the rest alone. Then the tensor grows 1 : 2 : 4 : 8
with the width, and the experiment measures **size**, not width.

The identity `bytes_per_loop = M · N · W` means width, N and bytes cannot all three be held fixed.
This design holds **bytes, loops, M, node count and skeleton** fixed and lets **N** absorb the width:

> **8192 bytes moved per Transpose node, 4 nodes per file, 32768 bytes per inference — at every
> width, in every dtype, in every variant of the W-MAIN family.** `N = 16 / W` at `L = 256`.

Four independent defences against the residual `N` confound:

**(a) The primary statistic is a within-file release difference, not a level.** The verdict quantity
is `Δ = 100 × (m_1.24.4 / m_1.23.2 − 1)` for one file at one thread count. Any effect of N, of aspect
ratio, of cache behaviour, of dtype kernels or of absolute size is present *identically on both sides*
of that ratio and cancels — because the MLAS transpose code, the CPU Transpose kernel and the intra-op
thread pool are **unchanged** between 1.23.2 and 1.24.4 except for the one added
`has_mlas_transpose<uint16_t>` line (verified at source by the author before 27b was cited). The
independent variable is width; the dependent variable is a difference; N is a nuisance that is
differenced away.

**(b) Within a width, dtype is fully crossed at identical (L, M, N) and identical bytes.** This falls
out of the construction and is the sharpest part of the design:

| width | uint8 | float16 | float32 | shared (L, M, N) | bytes/node |
|---|---|---|---|---|---|
| 1 B | `[256,2,16,1]` | — | — | (256, 2, 16) | 8192 |
| 2 B | `[256,2,8,2]` | `[256,2,8,1]` | — | (256, 2, 8) | 8192 |
| 4 B | `[256,2,4,4]` | `[256,2,4,2]` | `[256,2,4,1]` | (256, 2, 4) | 8192 |
| 8 B | `[256,2,2,8]` | `[256,2,2,4]` | `[256,2,2,2]` | (256, 2, 2) | 8192 |

Every file in one row has the same loop count, the same M, the same N, the same number of bytes moved
and the same number of Transpose nodes. They differ **only** in element type and therefore in element
count (`8192 / elem_bytes`) — which is not separable from dtype at fixed bytes, and is named as the
residual in §2.3. 27b never had this: its f16 and f32 arms differed in shape *and* in bytes.

**(c) The W-DOSE family carries its own N control.** DOSE varies `L ∈ {64, 256, 1024}` at fixed width
and fixed bytes, so `N = 8192 / (2·L·W)` sweeps 32 → 8 → 2 at W = 2 and 64 → 16 → 4 at W = 1. The
W = 1 arm has the *same* N ladder and the *same* bytes but **no release change at all** (MLAS on both
releases). Any N-driven artefact must appear in the W = 1 arm's release deltas. It is predicted not
to, and if it does, the dose result is void.

**(d) Two nulls of opposite mechanism.** W ∈ {1, 4} is MLAS on both releases; W = 8 is scalar on both.
A size or N artefact would not respect that partition. The width rule does.

### 2.3 The residual confounds, named

1. **Element count varies with dtype at fixed bytes** (8192 / 4096 / 2048 elements for u8 / f16 /
   f32). Unavoidable: bytes = elements × elem_bytes. Controlled by (a) — it is a level effect that
   differences away — and *measured* by the within-width dtype rows, which must agree on Δ.
2. **N varies with width at fixed bytes and loops** (16 / 8 / 4 / 2). Unavoidable by the identity
   above. Controlled by (a), (c) and (d).
3. **The skeleton's cost differs by dtype** (uint8 pays QuantizeLinear + DequantizeLinear, fp16 pays
   two Casts, fp32 pays neither). Controlled by the three no-Transpose control files
   `ctrl_u8`, `ctrl_f16`, `ctrl_f32`, which measure the skeleton's own release behaviour and are
   validity condition V3.
4. **The source buffer differs by dtype** (12288 elements = 12 / 24 / 48 KB). Level effect only;
   differences away.
5. **`L = 1024` is outside any real model's regime** (MobileViT's flagged nodes have L = 160–256). It
   is in the design to make the dose slope measurable, and the paper must say so (§8.5).

### 2.4 The skeleton

Identical in form to `scripts/paper/transpose_probe.py:build()`, which Experiments 27/27b already
validated end-to-end, so the harness times these files unchanged:

```
input x : float32 [1,3,64,64]            (12288 values — exactly what ort_version_bisect.py feeds)
  u8  : QuantizeLinear(scale=0.02, zp=128:uint8) -> Reshape [12288]
  f16 : Reshape [12288] -> Cast to FLOAT16
  f32 : Reshape [12288]
  for b in 0..R-1:                       (R = 4 branches; the control variants omit the Transpose)
      Slice(start=b, end=b+n, axis=0)    (distinct starts defeat common-subexpression elimination)
      Reshape -> dims
      Transpose(perm=[0,2,1,3])          (omitted in the control variants)
      Reshape -> [n]
  Concat(axis=0)
  u8  : DequantizeLinear -> y : float32 [R*n]
  f16 : Cast to FLOAT    -> y : float32 [R*n]
  f32 :                     y : float32 [R*n]
```

`n = L·2·N·K` elements per branch; `assert b + n <= 12288` for every branch (u8 n = 8192, f16 4096,
f32 2048 — all satisfied at R = 4). Transpose node names are `T{b}|{variant}`, unique within a file,
so `a_check.py`'s SC-A join by node name is unambiguous. opset 17, ir_version 8, producer
`w_probe.py`. **No RNG anywhere in the builder** — the same command must reproduce byte-identical
files, and their SHA-256s are frozen into `results/w_probe/manifest.json` at build time (V8).

`R = 4` is chosen to match the four flagged MobileViT nodes and 27b's `tp_u8_w2`, so the predicted
dispatch count at `L = 256` is 1024 — within 25 % of the model's own 832 and of 27b's probe.

---

## 3. The registered shape table

Every row below was produced by calling `predict(dims, [0,2,1,3], elem_bytes, threads, version)` from
`scripts/paper/transpose_probe.py` during the writing of this registration, and **no other way**. The
builder re-derives them at build time and aborts on any mismatch (V1).

### 3.1 Family W-MAIN — the width ladder — `results/w_probe/main/`

`L = 256`, `M = 2`, `R = 4` Transpose nodes, **8192 bytes moved per node, 32768 per inference, in every row.**

| variant | dtype | dims | W | N | elem/node | path 1.23.2 | path 1.24.4 | disp/node 4T | **added disp/file 4T** | 2T | 1T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `m_u8_w1`  | uint8   | `[256,2,16,1]` | 1 B | 16 | 8192 | MlasTranspose<8>  | MlasTranspose<8>  | 256 → 256 | **0** | 0 | 0 |
| `m_u8_w2`  | uint8   | `[256,2,8,2]`  | 2 B | 8  | 8192 | scalar 16-bit     | MlasTranspose<16> | 0 → 256   | **+1024** | +1024 | 0 |
| `m_u8_w4`  | uint8   | `[256,2,4,4]`  | 4 B | 4  | 8192 | MlasTranspose<32> | MlasTranspose<32> | 256 → 256 | **0** | 0 | 0 |
| `m_u8_w8`  | uint8   | `[256,2,2,8]`  | 8 B | 2  | 8192 | scalar 64-bit     | scalar 64-bit     | 0 → 0     | **0** | 0 | 0 |
| `m_f16_w2` | float16 | `[256,2,8,1]`  | 2 B | 8  | 4096 | scalar 16-bit     | MlasTranspose<16> | 0 → 256   | **+1024** | +1024 | 0 |
| `m_f16_w4` | float16 | `[256,2,4,2]`  | 4 B | 4  | 4096 | MlasTranspose<32> | MlasTranspose<32> | 256 → 256 | **0** | 0 | 0 |
| `m_f16_w8` | float16 | `[256,2,2,4]`  | 8 B | 2  | 4096 | scalar 64-bit     | scalar 64-bit     | 0 → 0     | **0** | 0 | 0 |
| `m_f32_w4` | float32 | `[256,2,4,1]`  | 4 B | 4  | 2048 | MlasTranspose<32> | MlasTranspose<32> | 256 → 256 | **0** | 0 | 0 |
| `m_f32_w8` | float32 | `[256,2,2,2]`  | 8 B | 2  | 2048 | scalar 64-bit     | scalar 64-bit     | 0 → 0     | **0** | 0 | 0 |
| `ctrl_u8`  | uint8   | `[256,2,8,2]`, **no Transpose** | — | — | 8192 | — | — | — | **0** | 0 | 0 |
| `ctrl_f16` | float16 | `[256,2,8,1]`, **no Transpose** | — | — | 4096 | — | — | — | **0** | 0 | 0 |
| `ctrl_f32` | float32 | `[256,2,4,1]`, **no Transpose** | — | — | 2048 | — | — | — | **0** | 0 | 0 |

`_is_reshape([0,2,1,3], dims) = False` and `_single_axis([0,2,1,3]) = ("out", 2, 1)` for all nine
probe rows. **Exactly two of twelve files gain dispatches, and they are the two 2-byte files.**

### 3.2 Family W-DOSE — the dispatch dose ladder — `results/w_probe/dose/`

uint8 only, `M = 2`, `R = 4` nodes, **8192 bytes moved per node in every row**; `L` sweeps 16× and
`N = 8192/(2·L·W)` compensates.

| variant | dims | W | L | N | disp/node 4T | **added disp/file 4T** | predicted `A_ms` @0.52 µs |
|---|---|---|---|---|---|---|---|
| `d_u8_w1_L64`   | `[64,2,64,1]`  | 1 B | 64   | 64 | 64 → 64     | **0** | 0 |
| `d_u8_w1_L256`  | `[256,2,16,1]` | 1 B | 256  | 16 | 256 → 256   | **0** | 0 |
| `d_u8_w1_L1024` | `[1024,2,4,1]` | 1 B | 1024 | 4  | 1024 → 1024 | **0** | 0 |
| `d_u8_w2_L64`   | `[64,2,32,2]`  | 2 B | 64   | 32 | 0 → 64      | **+256**  | 0.133 ms |
| `d_u8_w2_L256`  | `[256,2,8,2]`  | 2 B | 256  | 8  | 0 → 256     | **+1024** | 0.533 ms |
| `d_u8_w2_L1024` | `[1024,2,2,2]` | 2 B | 1024 | 2  | 0 → 1024    | **+4096** | 2.130 ms |
| `d_ctrl_u8`     | `[256,2,8,2]`, **no Transpose** | — | — | — | — | **0** | 0 |

**A 16× change in dispatch count at exactly constant bytes moved.** No prior experiment in this study
has this.

### 3.3 Two files appear in both families, deliberately

`d_u8_w1_L256` is identical in construction to `m_u8_w1`, and `d_u8_w2_L256` to `m_u8_w2`. They are
measured in two **separate harness invocations**, with different seeds and at different times. This is
the cross-batch replication check (V4) and it costs nothing extra: one replicate on a cell that must
move, one on a cell that must not.

---

## 4. The grid

### 4.1 Element types — the decision asked for

* **uint8 — essential, and the primary arm.** It is the only type that can reach width 1 (one element
  = one byte), it alone spans the whole ladder 1/2/4/8, and it is the type the regressed model
  executes (`a_check.py` SC-A on MobileViT: `executed_dtypes == ["uint8"]`).
* **float16 — meaningful, and the sharpest mechanism test.** It reaches width 2 with `K = 1`: a
  **single-element** moved block. If the governing quantity is bytes, an fp16 one-element block must
  regress exactly like a uint8 two-element block of the same total size. It also supplies widths 4
  and 8. It is also the type PR #25680 was written for ("FP16 inference performance improvement on
  CPU"), so a *failure* here would say the PR does not do what its title says.
* **float32 — meaningful only as a null.** Its minimum block is 4 bytes, so it **cannot reach width 1
  or 2 and cannot test the law's positive claim.** It contributes widths 4 and 8, where nothing is
  predicted to change, and its value is as a dtype control: a type the PR did not touch, in the same
  skeleton, must not move. This limitation is stated here so the paper cannot later present the fp32
  arm as support for the law.
* **8-byte element types (int64 / float64) — excluded.** They would reach width 8 with `K = 1`, but
  width 8 is already covered by uint8 and float16 at matched (L, M, N, bytes); 64-bit CPU Transpose is
  irrelevant to quantized inference, and a fourth dtype costs ~15 % of the wall clock for no new width.

### 4.2 Thread counts — 1, 2, 4

Pinned by `core.cpu_topology.pick_cpus(t, "auto")`, as the harness already does. 1 is the registered
null (the predicate is `threads > 1`). 4 is where `B_RESULTS.md` found the regression. **2 is new
information**: the rule says 2 threads adds *the same* dispatch count as 4, so the step should be
present at 2 threads; no measurement in this study has ever been taken at 2 threads. If the step is
absent at 2 threads the rule is incomplete, whatever happens at 4.

### 4.3 Releases

Six, all from `D:\sec_envs`: **1.21.1, 1.22.1, 1.23.2, 1.24.4, 1.25.1, 1.30.0.**

| pair | role |
|---|---|
| 1.21.1 → 1.22.1 | pre-step: must be STILL at every width |
| 1.22.1 → 1.23.2 | pre-step: must be STILL at every width |
| **1.23.2 → 1.24.4** | **the critical step — the decision set** |
| 1.24.4 → 1.25.1 | post-step: must be STILL (the new behaviour is not re-tuned) |
| 1.25.1 → 1.30.0 | **GAPPED** (skips 1.26.0/1.27.0/1.28.0); persistence only, reported apart |

W-DOSE runs **1.23.2 and 1.24.4 only** — its question is the mechanism at the critical step, not the
ladder.

**1.16.3 is excluded**: its compiled extension is built for numpy 1.x and every session fails under
numpy 2.4.6 (`B_RESULTS.md` §1) — it produced no data in Experiment B and is not attempted here.
**1.20.1, 1.26.0, 1.27.0, 1.28.0 are excluded** to keep the run under three hours; the consequence is
in §8.5.

`--drop-duplicate-envs` is used: the project `.venv` and `D:/sec_envs/ort1300` are byte-identical
1.30.0 builds, and without the flag 1.30.0 would be measured twice under two `env_dir` values.
`interpreters()` iterates `sorted(scratch.glob("ort*"))` first and appends `"project"` last, so the
surviving interpreter is deterministically `ort1300`.

### 4.4 Repeats — five sessions per cell, and why

**N = 5**, unchanged from `b_design_registered.md` §5.1. Five reasons, in order of weight:

1. **The MOVED rule's operating characteristic is a function of N.** The disjoint-range test's exact
   two-sided level is **0.0079 at n = 5**, 0.029 at n = 4, 0.100 at n = 3. Changing N changes what the
   word MOVED means and destroys comparability with Experiment B's 584 pair-cells — which is the
   entire reason for reusing the vocabulary rather than inventing new.
2. **The 7 % floor *is* a five-session four-thread floor.** `B_RESULTS.md` §6: median within-cell
   spread over five fresh processes is 3.5 % at one thread and **9.0 % at four**, and the floor was
   set from that. A floor and an N that were calibrated together cannot be separated.
3. **Power for the positive.** The predicted step at `m_u8_w2`, 4 threads, is `1024 × 0.52 µs =
   0.533 ms` on a skeleton expected near 0.6–1.2 ms — roughly **+45 % to +90 %, five to ten times the
   9 % four-thread spread.** Five sessions a side make disjoint ranges near-certain at that size.
4. **Power for the nulls, which matter more here than in B.** This design's central claim rests on
   thirty predicted-STILL cells. With a 9 % spread, five-draw ranges from the same distribution
   overlap almost always, so STILL is the expected outcome under a true zero and a MOVED null is
   informative. Three sessions would not do: the project's own resampled null puts `P(gap > 7 %)` for
   three-session medians at 0.07 % on a quiet run but **15.6 % on a noisy one** (`FINDINGS.md`,
   logged item 2).
5. **Cost.** N is the only linear knob on the wall clock, and against a 5–10 σ effect, N = 7 buys
   nothing and costs 40 % of the budget.

**False-move budget, pre-registered.** Expected chance-MOVED = `pair-cells × 0.0079`:

| denominator | N | expected chance MOVED |
|---|---|---|
| decision set (1.23.2 → 1.24.4, 19 files × 3 threads) | 57 | **0.45** |
| the 30 predicted-STILL null cells inside it | 30 | **0.24** |
| W-MAIN adjacent pair-cells (12 × 3 × 4) | 144 | 1.14 |
| W-DOSE pair-cells (7 × 3 × 1) | 21 | 0.17 |
| W-MAIN gapped (12 × 3 × 1) | 36 | 0.28 |
| **all adjacent pair-cells** | **165** | **1.30** |

`P(≥ 1 chance-MOVED among the 30 nulls) ≈ 21 %`. R2 in §6.2 is written against that number: a single
small MOVED null is not a refutation on its own.

### 4.5 Totals and wall clock

| batch | variants | releases | threads | reps | **sessions** |
|---|---|---|---|---|---|
| **W01 — W-MAIN** | 12 | 6 | 3 | 5 | **1080** |
| **W02 — W-DOSE** | 7 | 2 | 3 | 5 | **210** |
| | | | | **total** | **1290** |

Pair-cells: 144 adjacent + 36 gapped (W-MAIN) + 21 (W-DOSE) = **165 adjacent, 36 gapped**;
**decision set 57**.

**Wall clock, from this harness's own measured throughput.** Experiment B ran 736 cells × 5 = 3680
measured sessions in 5.71 h on this machine with this harness → **5.59 s per session**, and that
figure already includes B's batch boundaries, anchors and quiet checks. The child's floor is
1.5 s warm-up + `max(2.0 s, 40 iterations)` (`ort_version_bisect.py:78`), and every probe here is
sub-3 ms so the 2.0 s window binds in every cell.

| | |
|---|---|
| 1290 sessions × 5.6 s | **2.01 h** |
| pessimistic, 7.0 s/session | **2.51 h** |
| pre-flight (build, SC-B, SC-A on 19 files, hashes, quiet checks) | ~0.2 h |
| **registered estimate** | **2.2 h, worst case 2.7 h** |
| **hard stop** | **3.0 h**, checked only at a batch boundary |

W01 runs first (it carries the primary question). If the hard stop is reached before W02 starts, W02
is not run and §8.4 says what the paper may then claim.

---

## 5. The pre-registered predictions

All predictions are about the **1.23.2 → 1.24.4 step** unless stated. `Δ` and the verdict classes are
defined in §6.1. "Slower" means `Δ > 0`.

### P1 — PRIMARY: the law. The 57-cell decision set, every cell assigned in advance

| # cells | files | threads | **registered verdict** |
|---|---|---|---|
| 10 | the 5 files with W = 2 (`m_u8_w2`, `m_f16_w2`, `d_u8_w2_L64/L256/L1024`) | 2, 4 | **MOVED, slower** |
| 5 | the same 5 files | 1 | **STILL** |
| 30 | the 10 files with W ∈ {1, 4, 8} | 1, 2, 4 | **STILL** |
| 12 | the 4 control files (no Transpose) | 1, 2, 4 | **STILL** |
| **57** | | | |

P1 **holds** iff at least 9 of the 10 MOVED-slower cells are MOVED and slower, **and** at most 1 of
the 47 predicted-STILL cells is MOVED. (One slack cell on each side, against an expected chance-MOVED
of 0.45 over the whole set.) Anything else sends the result to §6.2.

### P2 — the byte-width identity, across element types

At 4 threads, `m_u8_w2` (two uint8 per block) and `m_f16_w2` (one float16 per block) — same L, M, N,
same 8192 bytes/node, same node count — must both be MOVED and their **absolute** deltas in ms must
agree within a factor of two (`0.5 ≤ Δms(f16)/Δms(u8) ≤ 2.0`, the band `a_check.py:78` already uses).
Symmetrically, at width 4 all three of `m_u8_w4`, `m_f16_w4`, `m_f32_w4` must be STILL, and at width 8
all three of `m_u8_w8`, `m_f16_w8`, `m_f32_w8` must be STILL.

**What P2 adds over 27b:** 27b's `tp_f16_w2` and `tp_f32_w4` had different shapes *and* different byte
counts, so it could not distinguish "the byte width decides" from "the element type decides". Here the
dtypes are crossed within a width at identical geometry and identical bytes.

### P3 — the dose. Δ scales with dispatch count at constant bytes moved

At 4 threads, over `d_u8_w2_L64 / L256 / L1024` (dispatch counts 256 / 1024 / 4096, bytes identical):

* **verdict-bearing:** the three absolute deltas `Δms` are strictly increasing in L, and
  `Δms(L=1024) / Δms(L=64) ≥ 4` (the dispatch ratio is 16; a factor-4 floor allows for thread-pool
  saturation);
* **verdict-bearing:** all three `d_u8_w1_L*` files are STILL at every thread count (the N-ladder
  control of §2.2(c));
* **REPORTED-ONLY:** the implied per-dispatch cost `Δms × 1000 / (4L)` for each rung lands in
  **0.26–1.04 µs**, i.e. 0.5×–2.0× of Experiment 27b's independently measured 0.51–0.53 µs.

### P4 — the step belongs to 1.24.4, and it stays

At **1.21.1 → 1.22.1** and **1.22.1 → 1.23.2**: every file, every thread count, **STILL**.
At **1.24.4 → 1.25.1**: every file, every thread count, **STILL**.
At the gapped **1.25.1 → 1.30.0**: the W = 2 files are **STILL** — the new behaviour persists rather
than being reverted, matching `B_RESULTS.md` §3.1's "it never comes back".

### P5 — SECONDARY, level cross-check inside one release

Read on file-to-file ratios **within a single release, a single thread count and a single shuffled
invocation** (a matched contrast, not a machine-state gauge):

* on **1.23.2**, 4 threads: `m_u8_w1` and `m_u8_w4` (MLAS, 256 dispatches/node) each sit **at least
  7 % above** `m_u8_w8` (scalar, 0 dispatches), and within 7 % of each other;
* on **1.24.4**, 4 threads: `m_u8_w2` joins them — above `m_u8_w8` by at least 7 %, and within 7 % of
  `m_u8_w1` and `m_u8_w4`;
* on **1.23.2**, 4 threads: `m_u8_w2` sits within 7 % of `m_u8_w8` (both scalar).

P5 is the same mechanism read from levels instead of from the release step. It is REPORTED, and one
refutation (R8) attaches to it. It never overrides P1.

---

## 6. The decision rule, and the refutations

### 6.1 The rule — taken unchanged from `b_design_registered.md` §4.1–4.3

* A **cell** is `(family, variant, threads, release)`. Its quantity is **`m` = the median of the
  cell's five session `p50_ms` values**. The five repeats are five **processes**, never more
  iterations inside one.
* **A cell is analysed at exactly five OK sessions or not at all.** Four, three or six make it
  **VOID**, and every pair-cell touching it VOID.
* A **pair-cell** is `(family, variant, threads, release_a → release_b)`. Its quantities are
  **`Δ = 100 × (m_b/m_a − 1)` percent** and the two `[min, max]` five-session ranges. **Absolute
  medians in ms are reported beside every ratio.**
* **MOVED** iff `|Δ| ≥ 7 %` **AND** the two five-session ranges are **disjoint**.
* **STILL** iff `|Δ| < 7 %`. Range overlap is not consulted.
* **UNRESOLVED** iff `|Δ| ≥ 7 %` but the ranges overlap.
* **No outlier rule, ever.** Every session that returned a number is in its cell's median and range.
* Shares are reported as a **bracket**: lower = `MOVED/N`, upper = `(MOVED+UNRESOLVED)/N`.
* **`core_delivery` is recorded per session and gates nothing** — the operational rule was withdrawn
  on 2026-09-24 (`results/expB/v6_decision.md`) and is not reintroduced here in any form.

The floor is **not** changed in either direction. 7 % is a four-thread floor (`B_RESULTS.md` §6); at
one and two threads it is conservative, and that asymmetry is stated wherever a one-thread STILL is
reported.

### 6.2 What refutes the law — each stated so it can be observed

**R1 — the law is false.** Fewer than 9 of the 10 W = 2 cells at 2/4 threads are MOVED-slower.
*Strongest form:* none of the five W = 2 files moves at 4 threads while the MobileViT file's own
+40 % remains reproducible. Then the moved-block width does not govern synthetic graphs, and the paper
may state only the file-specific observation about MobileViT — not a law.

**R2 — width is not the variable.** Two or more of the 30 W ∈ {1,4,8} null cells are MOVED in the same
direction, **or** a single one is MOVED with `|Δms|` within a factor of 2 of the W = 2 files'. (One
lone small MOVED null is expected at 21 % probability by chance and is re-measured once — §7 V9 —
before being counted.) Then something other than the 2-byte predicate changed between 1.23.2 and
1.24.4 and the attribution to PR #25680 is unsafe.

**R3 — it is not the byte width, it is the element type. → THIS MAKES THE PAPER'S CURRENT EXPLANATION
WRONG.** `m_u8_w2` and `m_f16_w2` land in different verdict classes, or their absolute deltas differ
by more than 2×, at matched (L, M, N) and matched bytes. Then the governing quantity is not
`elem_bytes × prod(dims[max(from,to)+1:])`, and every sentence of the form "whose moved block is 2
bytes" must be withdrawn and replaced by a dtype-conditional statement — even though the regression
itself would remain real. 27b's f16 result argues against R3 but cannot exclude it, because 27b never
measured a uint8 twin at equal bytes and equal geometry.

**R4 — it is not dispatch count. → THIS ALSO MAKES THE PAPER'S CURRENT EXPLANATION WRONG.** All three
`d_u8_w2_L*` files are MOVED, but `Δms(L=1024)/Δms(L=64) < 2` — a 16× change in dispatch count at
*exactly constant bytes moved* produced no proportional change in cost. Then the cost is not
per-dispatch thread-pool overhead, and every number in this study that multiplies a dispatch count by
0.52 µs is wrong: 27b's "dispatch-cost prediction 0.440 ms", `a_check.py`'s `A_ms`, its whole §2.3
PREDICT-STEP machinery, and E2's arithmetic. The regression would survive; its explanation would not.

**R5 — it is not the thread pool.** Any W = 2 file is MOVED at **1 thread**. Then the "moves only at
four threads" signature — which is what separates the MobileViT case from the YOLOv8n case in
`B_RESULTS.md` §4 — does not follow from the width rule, and the two-signature taxonomy needs
rewriting.

**R6 — wrong release.** A W = 2 file is MOVED at 1.21.1 → 1.22.1 or 1.22.1 → 1.23.2, or its level on
1.21.1/1.22.1 already sits within 7 % of its 1.24.4 level. Then the step is not introduced at 1.24.4
and commit 42869ca is the wrong attribution.

**R7 — the instrument is invalid.** A control file (no Transpose at all) is MOVED at the critical step
in the same direction as the probes. Then the skeleton, not the Transpose, carries the release
difference. See V3.

**R8 — the mechanism does not show in levels.** P5 fails: on 1.23.2 at 4 threads, `m_u8_w1` and
`m_u8_w4` are **not** above `m_u8_w8`. Then dispatching 256 MLAS loops costs nothing measurable at
this size, and a release step at W = 2 of the size P1 predicts cannot be a dispatch cost, whatever
else it is.

**R9 — the direction is wrong.** A W = 2 file is MOVED **faster** at 4 threads. Then routing 2-byte
blocks to MLAS is a *win* in that regime and "the 2-byte width makes models slower" is false as
stated; see O4.

### 6.3 The outcome that is neither confirmation nor refutation, and is the most likely surprise

**O4 — width changes the code path; whether that costs depends on the loop count.** `d_u8_w2_L1024`
is MOVED-slower and large, `d_u8_w2_L256` is MOVED-slower, but `d_u8_w2_L64` is STILL or MOVED-faster.
This is entirely possible: MLAS's inner loop is vectorized and at `L = 64, N = 32` it has 32-element
rows to work on, where scalar code may lose. If observed, the law survives in the form *"a 2-byte
moved block changes the code path on 1.24.4; the sign and size of the change depend on the loop count
and the block geometry"* — weaker, more accurate, and requiring the paper's headline to move from
"makes it slower" to "changes how it is executed, which on MobileViT's geometry makes it slower".
Registered now so it cannot be presented later as confirmation.

---

## 7. Validity conditions — what must hold for the run to count

Designed against the failure recorded in `results/expB/v6_decision.md`: V6 there was registered on
**`core_delivery`, a level statistic**, which turned out to be duration-biased
(`rho(p50, core_delivery) = −0.78` against `rho(p50, u_true) = +0.03` over 346 cells), fired on 7 of
15 batches, reproduced at 65.1 % on a *quiet-machine* re-run of the batch it fired on (so it was not a
machine-state signal at all), and was uncorrelated with whether a batch produced findings — the
batches it fired on contained the four largest effects in the map.

**The lesson applied here:** every validity condition below is either (a) a structural fact about the
files, checkable before any timing, or (b) a **differential** check that fails exactly when the
attribution fails. None is a level statistic computed from the measurement itself. `core_delivery`,
`other_load_pct` and `LoadPercentage` are recorded where the harness already records them and **gate
nothing**.

**V1 — the static table is reproduced (pre-flight, no timing).**
`scripts/paper/w_probe.py --verify-static` re-derives every row of §3 by importing `predict`,
`_single_axis`, `_is_reshape` and `MLAS_BYTES` from `scripts/paper/transpose_probe.py` and must match
`(dims, perm, moved_block_bytes, loops, M, N, dispatches@1.23.2, dispatches@1.24.4)` at threads 1, 2
and 4 exactly. Additionally `a_check.py sc-b` is run on the **12 uint8 files** as a second,
independently written implementation and must agree on `moved_block_bytes`, `loops`, `M`, `N`,
`n_flagged` and `added_dispatches`. (SC-B is not run against the fp16/fp32 files — §1.2 departure 1.)
*Can fail:* a typo in a single dims entry turns a probe into a reshape-copy or moves N off its
registered value. **Failure ⇒ fix the builder, re-register the table, restart.**

**V2 — the executed graph is the graph this design is about (pre-flight, no timing loop).**
`a_check.py sc-a` on all 19 files, run **twice**: once with `--sc-a-python D:/sec_envs/ort1244/...`
and once with `--sc-a-python D:/sec_envs/ort1232/...`. Required, for every probe file, on **both**
releases: `n_transposes_executed == 4` (0 for the four control files), `n_unjoined == 0`,
`duplicate_transpose_names_in_graph == []`, `dtype_or_shape_varied_between_iterations == {}`, and every
executed Transpose's `(dtype, shape, perm)` equal to the registered row.
This is the check `MEMORY.md` demands — *"ONNX file graph is not the executed graph; ORT deletes ~90 %
of Q/DQ before inference; never count nodes from the file."* *Can fail, and realistically:* ORT's
transpose optimizer can cancel a Transpose against a neighbouring Reshape; the fp16 arm could be
executed in float32 after a Cast fusion; a branch could be constant-folded. **Most dangerous form: the
executed graph differs between 1.23.2 and 1.24.4** — the comparison would then not be like-for-like
and the file is void.
*Failure for one dtype ⇒ that dtype's arm is void and is reported as void. Failure for the uint8
W = 2 files ⇒ the whole experiment is void.*
SC-A runs 3 profiled iterations at 1 thread purely to read `args.input_type_shape`; its durations are
read by nothing (`a_check.py` docstring). It is a graph-identification step, not a measurement.

**V3 — the control differential.** For each release pair `P` and thread count `t`: if a control file of
dtype `D` is **MOVED**, every probe verdict of dtype `D` at `(P, t)` is reported as **CONFOUNDED** and
excluded from the primary count. If the uint8 control is MOVED at `(1.23.2 → 1.24.4, 4 threads)`, the
primary question is **void** and W01 is re-measured once as a whole invocation (no cell spans two
invocations). *Can fail:* ORT changed `QuantizeLinear`/`DequantizeLinear`/`Slice`/`Concat` across a
release, or the machine drifted between the two sides of a pair. A control that moves for a real
reason is itself a finding and is published either way.

**V4 — cross-batch replication.** `m_u8_w2` vs `d_u8_w2_L256`, and `m_u8_w1` vs `d_u8_w1_L256`: same
graph, two invocations, two seeds, two times. Each pair must land in the **same verdict class** at
every thread count on the critical step, and their medians must agree within Experiment B's documented
between-run spread (`max/min ≤ 1.174`, `FINDINGS.md` logged item 3). *Can fail:* the between-run spread
at this magnitude is larger than assumed. **Failure ⇒ every verdict in the study is downgraded to "one
run, not replicated" and the paper says so.**

**V5 — quiet machine, as a machine statistic.** Immediately before and immediately after each harness
invocation, sample `Win32_PerfFormattedData_PerfOS_Processor` `_Total` `PercentProcessorTime` five
times over 20 s (the `PS_LOAD` probe of `b_run.py:268-283`). Required at both ends: **median ≤ 15 %**,
**on AC power**, no other measurement running, no leftover `python`/`find` process from an earlier
agent. `LoadPercentage` and `other_load_pct` are barred by name (`MEMORY.md`: `LoadPercentage` read
100 % while the laptop was 22–32 % busy; `other_load_pct` fires on 61 % of measurements at
`rho = −0.06` with real core delivery).
*Why this one can fail in the way it claims to catch:* it reads whole-machine busy-ness directly, at
the invocation boundary, with no dependence on model duration. A failure **before** an invocation means
the invocation does not start; a failure **after** means that invocation's CSV is re-measured, not
voided on the spot. It is evaluated twice per run, not once per data batch, so it cannot fire on half
the batches while being unrelated to the findings.

**V6 — code frozen.** SHA-256 of `w_probe.py`, `ort_version_bisect.py`, `transpose_probe.py`,
`a_check.py`, `w_analyse.py` and **this registration** are logged before the first timing row and
re-checked after the last. Any change mid-run voids every CSV written after it. The harness is used
**unedited**; the only flags used beyond the defaults are the three opt-in flags it already carries
(`--drop-duplicate-envs`, `--status-rows`, `--exclude`).

**V7 — completeness.** `--status-rows` is used, so every attempted session writes a row and an empty
cell is distinguishable from a cell never attempted. Expected row counts: **W01 = 1080**,
**W02 = 210**. A cell with ≠ 5 `status == OK` rows is VOID. **No cell spans two invocations.** The
count of VOID cells is published.

**V8 — file identity.** SHA-256 of all 19 `.onnx` files is frozen into `results/w_probe/manifest.json`
at build time and re-checked after the last timing row. Any difference not traceable to a recorded
action voids the run. The same file bytes go to every release (the harness guarantees this: one
`--families` directory, all interpreters).

**V9 — the one permitted re-measurement.** A *single* MOVED null cell among the 30 (expected by chance
at 21 %) is re-measured once, as a fresh whole invocation of its family restricted to that file
(`--variants <one>`), and the re-measurement replaces the first — never pooled with it
(`b_design_registered.md` §3.4(a) bars cross-run pooling). Both results are published. This is the
**only** permitted re-measurement and it is available only once in the run.

### 7.1 What voids the whole experiment

Any one of:

* V2 fails for the uint8 W = 2 files on either release;
* V3 fires on the uint8 control at `(1.23.2 → 1.24.4, 4 threads)` twice;
* V6 fails (the harness or builder was edited mid-run);
* V8 fails;
* fewer than 5 of the 6 releases complete W01;
* more than 10 % of W01's cells are VOID under V7.

A voided experiment is reported as voided, with the pre-flight logs and both quiet-machine samples
published. It is not partially rescued.

---

## 8. What the paper may and may not say, per outcome

### 8.1 If P1, P2, P3 and P4 all hold

**May say:** *"The moved-block width, and not the model, governs the behaviour. In synthetic graphs
built solely to set that width — matched on bytes moved, loop count, node count, M and N — ONNX
Runtime 1.24.4 slows a 2-byte moved block by X % at four threads and Y % at two, and does not move 1-,
4- or 8-byte blocks; the same 2-byte behaviour appears whether the block is two uint8 elements or one
float16 element; it is absent at one thread; it is absent at every release step before 1.24.4 and
present at every step after; and at constant bytes moved the cost grows with the dispatch count over a
16× range."*
**May say:** the law predicts a step at **two** threads, and the prediction was tested and held.
**May not say:** "root cause" (the phrase is barred for this mechanism throughout this study), nor
extend the law to permutations that are not single-axis, to `M = 1`, to other block geometries, to
other CPUs, to GPU, or to batch sizes other than 1.
**May not say:** anything about widths reached only by element types not measured here.

### 8.2 If P1 holds but P3 fails (R4)

**May say:** the width law holds — the 2-byte width is the variable.
**Must say:** the *cost model* is refuted: a 16× change in dispatch count at constant bytes moved did
not change the cost proportionally, so per-dispatch thread-pool overhead is not the mechanism.
**Must withdraw:** every quantitative use of 0.51–0.53 µs per dispatch, including 27b's "dispatch-cost
prediction 0.440 ms", `a_check.py`'s `A_ms`/ρ and its PREDICT-STEP class, and E2's arithmetic.
**May not say:** "N dispatches × 0.52 µs explains the step" anywhere in the paper.

### 8.3 If P1 holds but P2 fails (R3)

**Must withdraw** every sentence containing "whose moved block is 2 bytes" and replace it with a
dtype-conditional statement naming exactly which element types were observed to move.
**May still say** that the MobileViT regression is real, reproduced and localised to Transpose —
Experiment B and 27b stand on their own.

### 8.4 If W02 does not run (hard stop)

**May say** the law (P1, P2, P4 from W01). **May not say** anything about dispatch count, dose or the
per-dispatch cost, and must state that V4's cross-batch replication was not performed.

### 8.5 In every outcome, the paper must state

* The ladder is **1.21.1 … 1.30.0 with 1.20.1, 1.26.0, 1.27.0 and 1.28.0 absent**, so the study cannot
  say the behaviour was absent before 1.21.1 nor trace 1.25.1 → 1.30.0 step by step; that pair is
  **GAPPED** and reported apart.
* **1.16.3 produced no data** (numpy 2.x incompatibility) and was not attempted.
* `L = 1024` is outside any real model's regime and exists to make the dose slope measurable;
  MobileViT's own flagged nodes have `L = 160–256`, which the `L = 256` rung matches.
* Widths 1 and 2 are unreachable in float32 by construction, so the float32 arm is a control and
  supports no positive claim.
* The 7 % floor is a **four-thread** floor; one- and two-thread STILL verdicts are conservative.
* **One run.** Only the two duplicated files (V4) are replicated across invocations.
* `core_delivery` is published per session and gated nothing.

---

## 9. The commands, exactly

`PY = .venv/Scripts/python.exe`. All paths relative to `C:\Users\User\Desktop\SEC\edge-ai-toolkit`.
`SCR = C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-SEC/<session>/scratchpad/w_exec`.

### 9.1 Pre-flight — no timing

```sh
# 1. build the 19 probe files + manifest (deterministic; no RNG; V1 asserts fire here)
$PY scripts/paper/w_probe.py --build --out-root results/w_probe

# 2. V1: the registered table, re-derived through transpose_probe.predict()
$PY scripts/paper/w_probe.py --verify-static --json results/expW/preflight/static.json

# 3. V1 second implementation, uint8 files only
$PY scripts/paper/a_check.py sc-b results/w_probe/main/m_u8_w1.onnx \
    results/w_probe/main/m_u8_w2.onnx results/w_probe/main/m_u8_w4.onnx \
    results/w_probe/main/m_u8_w8.onnx results/w_probe/main/ctrl_u8.onnx \
    results/w_probe/dose/d_u8_w1_L64.onnx results/w_probe/dose/d_u8_w1_L256.onnx \
    results/w_probe/dose/d_u8_w1_L1024.onnx results/w_probe/dose/d_u8_w2_L64.onnx \
    results/w_probe/dose/d_u8_w2_L256.onnx results/w_probe/dose/d_u8_w2_L1024.onnx \
    results/w_probe/dose/d_ctrl_u8.onnx \
    --json results/expW/preflight/sc_b.json

# 4. V2: the EXECUTED graph, on both sides of the critical step (19 files each, listed in full
#    by w_run.py; globs shown here for brevity and work under Git Bash)
$PY scripts/paper/a_check.py sc-a results/w_probe/main/*.onnx results/w_probe/dose/*.onnx \
    --work-dir "$SCR/exec_1244" \
    --sc-a-python D:/sec_envs/ort1244/Scripts/python.exe \
    --json results/expW/preflight/sc_a_1244.json
$PY scripts/paper/a_check.py sc-a results/w_probe/main/*.onnx results/w_probe/dose/*.onnx \
    --work-dir "$SCR/exec_1232" \
    --sc-a-python D:/sec_envs/ort1232/Scripts/python.exe \
    --json results/expW/preflight/sc_a_1232.json

# 5. V6 + V8: freeze the hashes
$PY scripts/paper/w_probe.py --freeze --json results/expW/preflight/hashes.json
```

**Stop here if any of V1, V2, V5 fails.** No timing row is written until all three pass.

### 9.2 W01 — the width ladder. One invocation, output to a FILE

```sh
$PY scripts/paper/ort_version_bisect.py \
  --scratch D:/sec_envs \
  --threads 1,2,4 \
  --repeats 5 \
  --seed 1101 \
  --batch 1 \
  --versions 1.21.1,1.22.1,1.23.2,1.24.4,1.25.1,1.30.0 \
  --families "wmain=results/w_probe/main:3,64,64" \
  --variants m_u8_w1,m_u8_w2,m_u8_w4,m_u8_w8,m_f16_w2,m_f16_w4,m_f16_w8,m_f32_w4,m_f32_w8,ctrl_u8,ctrl_f16,ctrl_f32 \
  --drop-duplicate-envs \
  --status-rows \
  --exclude "" \
  --out results/expW/batches/w01_main.csv \
  > results/expW/logs/w01_main.log 2>&1
```

**1080 rows expected.** Estimated 1.7 h; runner timeout 3.0 h.

### 9.3 W02 — the dose ladder. One invocation, output to a FILE

```sh
$PY scripts/paper/ort_version_bisect.py \
  --scratch D:/sec_envs \
  --threads 1,2,4 \
  --repeats 5 \
  --seed 1102 \
  --batch 1 \
  --versions 1.23.2,1.24.4 \
  --families "wdose=results/w_probe/dose:3,64,64" \
  --variants d_u8_w1_L64,d_u8_w1_L256,d_u8_w1_L1024,d_u8_w2_L64,d_u8_w2_L256,d_u8_w2_L1024,d_ctrl_u8 \
  --drop-duplicate-envs \
  --status-rows \
  --exclude "" \
  --out results/expW/batches/w02_dose.csv \
  > results/expW/logs/w02_dose.log 2>&1
```

**210 rows expected.** Estimated 0.35 h; runner timeout 1.0 h.

### 9.4 The runner

`scripts/paper/w_run.py`, built on `scripts/paper/b_run.py`'s pattern and reusing its helpers verbatim
where possible:

* **`run_to_file`** (`b_run.py:242-258`): stdout **and** stderr to a file via `subprocess.Popen`,
  **never a pipe** — *"a pipe nobody drains is what hung an earlier run"* — with a timeout and
  `kill_tree` on expiry.
* **`batch_complete`** (`b_run.py:748-757`): **resumable.** A batch whose CSV already holds exactly the
  expected row count is logged `complete already - not re-run` and skipped; an incomplete CSV causes
  the **whole** batch to be re-run, never a partial top-up (V7: no cell spans two invocations). To
  force a re-run, **move** the CSV to `results/expW/batches/rerun1/`, never rename it in place —
  `v6_decision.md` records why: the analysis globs `w[0-9][0-9]_*.csv` and a suffixed name still
  matches, which would read both runs into the same cells.
* **`quiet_machine`** / **`cpu_samples`** / **`on_ac_power`** (`b_run.py:268-300`) for V5, before and
  after each invocation.
* Order: **W01 then W02.** Hard stop 3.0 h, checked only at the batch boundary.
* `--stage build | preflight | map | analyse`, so each stage is separately resumable.
* Every command line, exit code, row count, elapsed time and SHA-256 appended to
  `results/expW/w_run.log`.
* **Leaves no background process**: `kill_tree` on timeout, and a final sweep for stray
  `python`/`find` processes (`CLAUDE.md` working rules; `MEMORY.md` "Workflow agents leave orphan
  processes").

### 9.5 Analysis

```sh
$PY scripts/paper/w_analyse.py \
  --batches results/expW/batches \
  --manifest results/w_probe/manifest.json \
  --out results/expW/analysis/w_analysis.json
```

Mechanical only. It computes per-cell medians and `[min,max]` ranges over the five sessions; per
pair-cell `Δ`, absolute `Δms` and the MOVED/UNRESOLVED/STILL verdict of §6.1; the 57-cell decision
table against P1's registered assignment; P2's dtype ratios; P3's dose slope and implied µs/dispatch;
P4's pre/post ladder; P5's within-release levels; V3, V4 and V7's counts; and prints which of R1–R9 and
O4 the data matches. It applies **no outlier rule** and **no core-delivery gate**, and it re-derives
every predicted dispatch count by calling `transpose_probe.predict()` rather than reading a number.

---

## 10. Artefacts

| path | what |
|---|---|
| `results/paper/w_design_registered.md` | **this file** — hash-stamped into every run log |
| `scripts/paper/w_probe.py` | builder + `--verify-static` + `--freeze` |
| `scripts/paper/w_run.py` | runner (pattern of `b_run.py`) |
| `scripts/paper/w_analyse.py` | analysis; applies §6.1 mechanically |
| `results/w_probe/main/*.onnx` (12), `results/w_probe/dose/*.onnx` (7) | the probes |
| `results/w_probe/manifest.json` | registered table + SHA-256 of all 19 files |
| `results/expW/preflight/{static,sc_b,sc_a_1244,sc_a_1232,hashes,quiet}.json` | V1, V2, V5, V6, V8 |
| `results/expW/batches/w01_main.csv`, `w02_dose.csv` | the measurements |
| `results/expW/logs/*.log`, `results/expW/w_run.log` | harness output and the run log |
| `results/expW/analysis/w_analysis.json` | verdicts |
| `results/paper/W_RESULTS.md` | written after, in `B_RESULTS.md`'s form |

**Seeds:** W01 = 1101, W02 = 1102. Distinct, so the two invocations' shuffles are independent — which
is what makes V4's cross-batch replication of the two duplicated files meaningful.

---

## 11. Barred in Experiment W

* Editing this file after the first timing row exists. If a decision has to be taken under it, it goes
  in a dated `results/expW/w_decision.md`, as `v6_decision.md` did — the registration is hash-stamped
  and is not altered.
* Any outlier rule, any trimmed mean, any gate on `core_delivery`, `other_load_pct` or
  `LoadPercentage`.
* Changing the 7 % floor, N = 5, the release list, the variant list or the shape table after seeing a
  latency.
* Pooling two runs of the same cell.
* Analysing a cell at fewer than five sessions.
* Adding a probe shape after the run to explain a result. A post-hoc shape is a **new registration**.
* The phrase "root cause" for this mechanism.
* Quoting a W-DOSE `L = 1024` number as if it described a real model.
