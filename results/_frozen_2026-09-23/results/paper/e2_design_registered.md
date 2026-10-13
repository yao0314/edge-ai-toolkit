# Experiment E2: the full-model counterfactual for the 1.24 Transpose step

**FINAL REGISTERED PROTOCOL, revision 2.** Written 2026-09-20 by the experiment designer, after an
independent reviewer attacked revision 1 (verdict MAY-RUN-WITH-FIXES) and before ANY timing. The
designer is not the author of the claim and does not run the timing. Nothing has been timed;
onnxruntime was imported only to load each model once, to dump optimized graphs and to compare
outputs. Revision 1 is kept unchanged as `e2_design_draft.md`.

Everything below - models, cells, seed, thresholds, command lines, analysis script, permitted
wording - is fixed now. This file is the whole protocol: someone who reads only this file can run it.

Artifacts:

| what | path |
|---|---|
| builder / verifier | `scripts/paper/e2_counterfactual.py` (onnx only for the build) |
| analysis | `scripts/paper/e2_analyse.py` (revision 2) |
| models | `results/e2_counterfactual/models/{p00_int8,p00_int8_e2,p00_int8_sham,p00_int8_plac}.onnx` |
| pre-timing evidence | `results/e2_counterfactual/{manifest,static_check,output_check,executed_check}.json` |
| runner | `<scratchpad>/run_e2.sh` |
| timing output | `results/paper/e2_counterfactual.csv` |
| verdict | `results/e2_counterfactual/verdict.json` |

`<scratchpad>` is
`C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-SEC/103ae351-d709-4d02-91fc-0b8da660fe58/scratchpad`.

Reused from the dead earlier attempt: only the CHOICE OF PLACES TO LOOK. `e2_look.py`, `e2_look2.py`
and `e2_look3.py` list the four candidate node names and show that each sits between a Reshape and a
Reshape inside a Q/DQ sandwich. Every fact used below was re-derived from the files (fresh scripts in
`<scratchpad>/e2/`). `e2src/transpose*.cc` (the v1.23.2 and v1.24.4 sources of `transpose.cc` and
`transpose_helper.cc`) was read directly and is quoted below.

---

## 0. What the review changed, objection by objection

The reviewer verified the structural half of the design independently (byte-identity of the timed
original, the 8 changed node protos per variant, the executed-graph dumps, the bit-identical outputs,
and a hand trace of `IsTransposeMovingSingleAxis` / `DoUntypedTranspose` against all four shapes) and
found no unlisted confound that moves four-thread latency. Every objection was about the decision
rule and the wording. Dispositions:

| # | objection | disposition |
|---|---|---|
| 1 | V4's 1-thread bound was silently loosened from the registered absolute 0.034 ms to `max(0.034, 0.07·own median)` = 0.103 ms | **ACCEPTED in full**, and strengthened: V4a is the absolute 0.034 ms of Exp 27/27b, and a new V4b requires the estimand's own numerator to be null at one thread (section 5.4) |
| 2 | V5 can void a valid experiment while policing a quantity that cancels out of `f`; release-asymmetry, the thing that would bias `f`, is not measurable on 1.23.2 | **ACCEPTED**, with the split the reviewer proposed: V5 is widened to the study's own floor and is REPORTED, not voiding; a new gating V5b tests thread-consistency of the rewrite's cost. Release symmetry of `DoUntypedTranspose` is stated to rest on the source diff, not on V5 (sections 3 C3, 5.4) |
| 3 | V3 was one-sided: a control that AMPLIFIES the step passed | **ACCEPTED in full**: `0.70 <= keep <= 1.43` for both controls, breach gives CONFOUNDED, and the already-computed control envelopes are now reported (section 5.4) |
| 4 | FIRM was decided from the `f` envelope alone, so "no residual step" got no envelope | **ACCEPTED in full**: in the CONFIRM band, FIRM also requires the whole `D(e2,4T)` envelope inside the floor (section 5.5) |
| 5 | V7(a) was applied to 1-thread cells where core delivery is not a core-delivery measure; the re-run command would overwrite the primary CSV; the analysis would pool re-run rows and foreign families | **ACCEPTED in full**: V7a gates the eight 4-thread cells only, the re-run writes its own CSV, and `e2_analyse.py` gained `--family` (default `e2`) and `--replace-cell` (sections 5.4, 5.6, 6) |
| 6 | W1's "root cause of this model's 1.24 regression" over-reaches beyond one file, size, thread count and machine | **ACCEPTED in full**: W1 is narrowed to the named file, sha256, input size, thread count and machine; the 256 px sibling's 4.43 ms four-thread and +5.5% one-thread steps are named as outside E2's scope in W1 and W2; extension to other sizes, thread counts, batch sizes or machines is added to the barred list (section 7) |
| 7 | `run_e2.sh` could not abort on a failed V1 (piped `|| exit`), and the quiet check was recorded before ~128 verification inferences | **ACCEPTED in full**: `set -o pipefail`, and the quiet loop moved after the three verification commands (section 6) |
| 8 | Two arithmetic statements were wrong: "24 comparisons" (it is 12) and "the 0.70 share, not the floor, binds" (the floor binds) | **ACCEPTED in full**: both corrected, and section 5.3 now states plainly that CONFIRMED tolerates a residual of up to ~23% of the step on medians - which is why objection 4's envelope condition matters |
| 9 | min/max over 625 combinations has no coverage property, and the design banned the extension run that the cited precedent (Exp 27 4.5) allows; no guard against a zero denominator | **ACCEPTED in part.** Guard added (combinations with a non-positive denominator are skipped and counted). The extension run is re-admitted, registered now with its exact trigger, cells, seed and command (section 5.6). FIRM stays on min/max: the alternative offered (5th-95th percentile) would LOOSEN the gate to the "root cause" wording, which is the one direction this design may not move. The 5-95 percentile is computed and reported as a secondary statistic and can never upgrade a verdict |

Nothing in the review was rejected outright. The only partial rejection is the firmness statistic in
objection 9, and the reviewer's alternative (the extension run) was taken instead.

---

## 1. Locating the four two-byte-block Transposes in the SOURCE graph

### 1.1 The problem

Experiment 27's census (FINDINGS "Experiment 27", section 1) is a table of the OPTIMIZED graph: 698
nodes, 56 uint8 Transposes. The four that carry the step are rows 3, 14, 17 and 36, at optimized node
indices 49, 186, 220 and 463. The source file
`results/classification_mobilevit/models/p00_int8.onnx` has 1851 nodes and only 39 Transposes, all
FLOAT, and FINDINGS explicitly warns that "matching nodes by name to the source file gives WRONG
shapes, because the transpose optimizer rewired the attention Transposes". So node names alone are
not an acceptable mapping.

### 1.2 Three independent bridges, all of which agree

**Bridge A - name.** The four optimized nodes carry the names `/stages/stages.2/stages.2.1/Transpose`,
`/stages/stages.2/stages.2.1/Transpose_3`, `/stages/stages.3/stages.3.1/Transpose` and
`/stages/stages.3/stages.3.1/Transpose_3`. Nodes with exactly these names exist once each in the
source, at source indices 577, 821, 905 and 1369. A HINT only.

**Bridge B - shape and perm.** The source graph is pure standard ONNX (opset 17), so
`onnx.shape_inference.infer_shapes(..., data_prop=True)` runs through it once the batch dimension is
pinned to 1. It gives, for those four source nodes, input shapes `[*,2,4,2]`, `[*,4,2,2]`,
`[*,2,2,2]`, `[*,2,2,2]` with perm `[0,2,1,3]` (the leading dim stays symbolic because the Reshape
target is computed at run time from Shape/Gather/Concat). The optimized census gives `[256,2,4,2]`,
`[256,4,2,2]`, `[160,2,2,2]`, `[160,2,2,2]`, same perms. The trailing three dims and the perm match
one to one, and no other source Transpose has a trailing dim of 2 with perm `[0,2,1,3]`. Running the
probe's `predict()` on these shapes reproduces the census's 256+256+160+160 = 832 dispatches on
1.24.4 and none on 1.23.2.

**Bridge C - decisive.** Each candidate was edited in the source file and the edited file's OPTIMIZED
graph was dumped under BOTH releases and diffed against the unmodified file's, node name by node
name. Result (`results/e2_counterfactual/executed_check.json`): the optimized node-name multiset is
IDENTICAL and exactly four optimized Transposes change - census rows 3, 14, 17, 36 - with the shapes
and perms the edit was designed to produce, still UINT8. This does not depend on names or on shape
inference.

| census row | optimized idx | executed uint8 shape | perm | SOURCE idx | source node name |
|---|---|---|---|---|---|
| 3  | 49  | [256, 2, 4, 2] | [0,2,1,3] | 577  | `/stages/stages.2/stages.2.1/Transpose` |
| 14 | 186 | [256, 4, 2, 2] | [0,2,1,3] | 821  | `/stages/stages.2/stages.2.1/Transpose_3` |
| 17 | 220 | [160, 2, 2, 2] | [0,2,1,3] | 905  | `/stages/stages.3/stages.3.1/Transpose` |
| 36 | 463 | [160, 2, 2, 2] | [0,2,1,3] | 1369 | `/stages/stages.3/stages.3.1/Transpose_3` |

They move 4096+4096+1280+1280 = 10,752 bytes per inference, 10.1% of the graph's 105,984 Transpose
bytes. Each sits in `... -> Reshape(shape = Concat(a,b,c,d)) -> Q -> DQ -> TARGET Transpose -> Q ->
DQ -> Reshape -> ...` with fan-out 1 throughout and with the shape Concat feeding only its own
Reshape (the builder's `_site()` aborts if any of this is untrue).

---

## 2. The rewrite

### 2.1 What has to change, from the ORT source

From `e2src/transpose_helper_v1.24.4.cc`, `TransposeSingleAxisOutwards` computes

```
block_size      = input_shape.SizeFromDimension(from + 1)
bytes_per_write = block_size * element_size
switch (bytes_per_write) { case 1: ... case 2: ... case 4: ... case 8: ... default: DispatchStridedCopy }
```

The `case 1/2/4` arms call `MlasTranspose` (threaded) because `has_mlas_transpose<uint8_t|uint16_t|
uint32_t>` is true; `case 8` (uint64_t) is a plain scalar loop; `default` is `DispatchStridedCopy`.
PR #25680 (commit 42869ca, first shipped in 1.24) added ONLY `has_mlas_transpose<uint16_t>`, so a
2-byte block is scalar on 1.23.2 and threaded from 1.24. `TransposeImpl` reaches that code only when
`IsTransposeMovingSingleAxis` is true; otherwise it falls to `DoUntypedTranspose`, which **takes no
thread pool argument at all** and is serial on every release.

### 2.2 The chosen rewrite: "rank lift", perm [0,2,1,3] -> [0,2,1,4,3]

Per site, two changes and no new node:

1. append a rank-1 int64 initializer `[1]` to the Reshape's dedicated shape `Concat` (axis 0), so the
   tensor reaching the Transpose is `[N,D1,D2,2,1]` instead of `[N,D1,D2,2]`;
2. change the Transpose's `perm` from `[0,2,1,3]` to `[0,2,1,4,3]`.

`np.transpose(x.reshape(d+[1]), [0,2,1,4,3])` flattens to exactly the same byte sequence as
`np.transpose(x, [0,2,1,3])` (checked for all four shapes, and bit-exactly in ORT), and the
downstream Reshape is untouched: an exact algebraic identity, not an approximation.

`[0,2,1,4,3]` fails both halves of `IsTransposeMovingSingleAxis`: `check_moved_outwards(2,2)` stops at
`perm[3] = 4 != 3`, and `check_moved_inwards(1)` stops at `perm[3] = 4` with `started_at = 1 != 4`.
So `TransposeImpl` falls to `DoUntypedTranspose`, where the prefix/suffix partition gives
`suffix_blocksize = 1` and it runs `DoTransposeEltWise<uint8_t>`: 10,752 one-byte copies with an
index increment, serial, on both releases.

### 2.3 Alternatives considered and rejected

| alternative | why rejected |
|---|---|
| Make the moved block 1 or 4 bytes | Those are `has_mlas_transpose` types on BOTH releases, so the rewrite would ADD 1.23.2 dispatches. Also unreachable without changing the data layout. |
| Make the moved block 8 bytes (the uint64 scalar case the FP32 twins use) | `block_size` is the product of the dims after the moved axis; here that is the trailing 2 and nothing can be appended, because D1 and D2 swap. Unreachable at uint8. |
| Keep the four Transposes in float32 | Requires breaking the QDQ fusion: two real kernels per site (dequantize, quantize) and 4x the transpose's traffic. Far more unrelated work than the effect. |
| Split each Transpose into two single-axis Transposes | Doubles the bytes moved and adds 4 nodes; the 8-byte variant (`perm [1,0,2,3]`) changes the outer order and needs a fixup transpose. |
| Insert the length-1 axis in the MIDDLE (`[N,D1,1,D2,2]`, perm `[0,3,2,1,4]`) | Valid and about 0.01 ms cheaper at run time. REJECTED because on that shape no single-axis perm keeps the 2-byte block, so the SHAM could not be built on an identical shape with only the perm differing. A perfectly matched sham is worth more than 0.01 ms. |
| Delete the four Transposes, or replace them with Reshape/Gather/ScatterND | Changes the output, or swaps in an entirely different kernel. |
| Replace the dynamic shape Concat with a constant shape vector | Would let the optimizer constant-fold and delete a whole Shape/Gather/Concat chain per site: a free, unrelated speedup. |
| Rebuild ORT 1.24.4 without `has_mlas_transpose<uint16_t>` | The cleanest counterfactual in principle; out of scope (no toolchain here) and it would not be the shipped binary the paper's claim is about. |

### 2.4 The four timed variants

| file | edit | nodes edited | predicted dispatches/inference at 4T, 1.23.2 -> 1.24.4 |
|---|---|---|---|
| `p00_int8.onnx` | none, byte-identical copy of the paper's file | - | **32 -> 864** |
| `p00_int8_e2.onnx` | rank lift, perm `[0,2,1,3]` -> `[0,2,1,4,3]` | the four targets | **32 -> 32** |
| `p00_int8_sham.onnx` | rank lift, perm `[0,2,1,3]` -> `[0,2,1,3,4]` | the four targets | **32 -> 864** |
| `p00_int8_plac.onnx` | rank lift, perm `[0,3,2,1]` -> `[0,3,2,1,4]` | four NON-target Transposes | **32 -> 864** |

sha256 (re-verified 2026-09-20 after the review, all four unchanged; the first is byte-identical to
`results/classification_mobilevit/models/p00_int8.onnx`, the paper's file):

```
p00_int8       911de4ee3257966074fe54952cf7ad5ed546798fe138fa8997efce9b11b1f629
p00_int8_e2    8a9274e889c2e833803f10f76b3614c9f394c3dca2ae19d3784281e1e3d6bd57
p00_int8_sham  b2a3aec6bd81f145bbf2075b00b6e20ce05544bfa159cac3601cf3d94f635f57
p00_int8_plac  360d686888855c36df34568af5424905ce7845b1e4d1b16cb6ba08dc0c67344f
```

The unmodified file is COPIED into the same directory, so all four are timed by the same harness
invocation, from the same directory, with the same shape string, in one randomised order.

---

## 3. The central threat, and the controls that answer it

The threat: the rewrite could change speed for a reason unrelated to the two-byte dispatch path.

| # | confound | handled by |
|---|---|---|
| C1 | **Extra nodes.** | Verified absent: the optimized graph of `p00_int8_e2` and of `p00_int8_sham` has the SAME 698 nodes with the SAME names as the original, under BOTH releases (`nodes_added_vs_original: []`). The one new initializer is a 1-element int64 folded into the shape Concat. The placebo has +1 node (section 4b). |
| C2 | **Extra memory traffic.** | Verified absent: total Transpose bytes moved is 105,984 in all four models, delta 0. No copy, pad, slice or cast is added. |
| C3 | **A different kernel.** REAL and deliberate: the four nodes move from `SimpleTransposeSingleAxisOutwards<uint16_t>` to `DoTransposeEltWise<uint8_t>`. | **Corrected in revision 2.** Write the per-release cost: `D(orig,4T) = (base_new - base_old) + T_mlas - T_simple` and `D(e2,4T) = (base_new - base_old) + T_elt,new - T_elt,old`. If `T_elt` is release-symmetric, `D(orig,4T) - D(e2,4T) = T_mlas - T_simple`: **the rewrite's own cost cancels identically out of the estimand.** So the substitution can bias `f` only if `T_elt` is release-ASYMMETRIC. That is excluded by SOURCE DIFF, not by measurement: `transpose.cc` differs between the tags only in sub-byte (INT2/UINT2) types and the opset-23/24/25 kernel registrations, and `transpose_helper.cc` only by the three-line `has_mlas_transpose<uint16_t>`; `DoUntypedTranspose` and `DoTransposeEltWise` are byte-identical. No cell of this experiment can test that, and V5 never could: it is reported as the rewrite's SIZE (section 5.4), not as a bound on bias. |
| C4 | **A changed fusion.** | Verified absent for e2 and sham: identical optimized node-name multiset under both releases, identical op census, and only the four intended Transposes differ in shape or perm. |
| C5 | **Stale value_info.** | The builder drops exactly the 12 entries the rank change invalidates and puts back exactly those 12, re-derived by onnx shape inference. The same operation is applied to the sham and the placebo, so any residual effect is in the controls too. |
| C6 | **Changed thread-pool partitioning affecting OTHER nodes.** | Not separable from the claim - it IS part of "these dispatches cost time" - and reported, not controlled. The sham bounds how much of the difference could come from the edit rather than from the dispatches. |
| C7 | **Numerical divergence.** | Verified absent: bit-identical outputs (section 4a). |
| C8 | **Harness-level asymmetry** (directory, file size, run position, thermal drift). | One directory, one invocation, one seed, randomised order, fresh subprocess per session, P-core pinning. File sizes differ by <= 160 bytes out of 1.80 MB. |
| C9 | **The edit itself, whatever it is, changes latency.** | The SHAM. |
| C10 | **The step is a diffuse property of the uint8 Transpose kernel, so perturbing ANY four Transposes would relieve it.** | The PLACEBO. |

### 3.1 SHAM control (primary control)

`p00_int8_sham.onnx` applies the SAME edit to the SAME four nodes and produces the SAME rank-5 shapes
`[N,D1,D2,2,1]`. The only difference from the counterfactual is two entries of the perm:
`[0,2,1,3,4]` instead of `[0,2,1,4,3]`. `[0,2,1,3,4]` IS a single-axis move (out, from=2, to=1) with
`block_size = 2*1 = 2` elements, so `bytes_per_write` is still 2 and the four nodes still issue
832 `MlasTranspose<uint16_t>` calls on 1.24.4 and none on 1.23.2 - verified in the executed-graph
dump (864 predicted dispatches for the sham, 32 for the counterfactual).

The tightest possible control: same nodes, same graph, same shapes, same bytes, same executed node
count, same output, differing only in whether the moved block is two bytes wide. **Prediction: the
sham keeps the step (keep_sham ~ 1.00).** If it loses it OR amplifies it, the edit and not the
dispatch path is responsible: CONFOUNDED.

### 3.2 PLACEBO control, with a stated limitation

`p00_int8_plac.onnx`: the same rank-lift edit on the four `[0,3,2,1]` fold Transposes of the same two
MobileViT blocks - source indices 583, 815, 911, 1363, census rows 4, 13, 18, 35, shapes
`[1,64,16,4]`, `[1,4,16,64]`, `[1,80,4,4]`, `[1,4,4,80]`. They move 10,752 bytes, EXACTLY the same
total as the four targets. It tests node-specificity from the other side and costs 20 sessions.

Limitation, registered so it cannot be claimed later: these four are ALREADY on the untyped path on
both releases, and the rank lift leaves them there with identical kernel parameters (same
`num_blocks` 4096/4096/1280/1280, same `num_axes_in_prefix = 4`, same strides, because the appended
axis has length 1 and maps to itself). The placebo is therefore a NULL edit at the kernel level: it
controls the graph-level confounds (C1, C4, C5) and node-specificity, but NOT the kernel-substitution
confound C3, which is the sham's job and, for bias, the source diff's. No source Transpose in this
file is both outside the two-byte class and on a fast typed path AND fed by a dedicated shape Concat,
so a placebo that also substitutes the kernel cannot be built without adding nodes; checked, not
assumed. **Prediction: the placebo keeps the step and its own latency is unchanged.**

---

## 4. Verification without timing (done; re-checked after the review)

```
.venv/Scripts/python.exe scripts/paper/e2_counterfactual.py --build
.venv/Scripts/python.exe scripts/paper/e2_counterfactual.py --verify-static
.venv/Scripts/python.exe scripts/paper/e2_counterfactual.py --verify-outputs  --scratch <scratchpad>
.venv/Scripts/python.exe scripts/paper/e2_counterfactual.py --verify-executed --scratch <scratchpad>
```

Re-run after the review (2026-09-20): the four sha256 above still match `manifest.json`, the timed
original is still byte-identical to the paper's file, and `--verify-static` still prints PASS. The
two ORT-loading checks were not repeated: the files are byte-identical by sha256 to the ones they
were run on, and nothing in revision 2 touches a model.

### (a) Outputs match the original - PASS, bit-identical

8 random float32 inputs of shape 1x3x64x64 (`numpy` default_rng seed 2026) are run through each model
and compared as one stacked array per (variant, release, optimization level): 3 variants x 2 releases
x {ORT_DISABLE_ALL, ORT_ENABLE_ALL} = **12 comparisons** (96 input-model pairs), which is the record
count in `output_check.json`. (Revision 1 said "24 comparisons"; that was an arithmetic error, fixed
here.)

**max |difference| = 0.000e+00 in every one; `np.array_equal` is True in every one.**

Registered tolerance: **exactly zero, bitwise**. Justification, not convention: a Transpose is a pure
permutation of bytes, appending a length-1 axis moves nothing, and the perm change is an exact
algebraic identity, so the rewritten graph must produce the identical sequence of uint8 values at
every tensor. Any nonzero difference would mean the optimizer treated the two graphs differently,
which is what (b) tests for.

### (b) Optimized graph differs from the original's only where intended, under BOTH releases - PASS

Each model was loaded once per release with `ORT_ENABLE_ALL`, `intra_op_num_threads=4`,
`optimized_model_filepath` set, and the dumps were diffed. Shapes were re-derived with the probe's
`graph_transposes()` (contrib ops replaced by shape-equivalent standard ops), not read from the
dump's stale value_info.

| model | release | optimized nodes | nodes added vs original | op census delta | Transposes differing | Transpose bytes |
|---|---|---|---|---|---|---|
| p00_int8 | 1.23.2 / 1.24.4 | 698 / 698 | - | - | - | 105,984 |
| p00_int8_e2 | 1.23.2 / 1.24.4 | 698 / 698 | none / none | {} | the 4 targets, exactly | 105,984 |
| p00_int8_sham | 1.23.2 / 1.24.4 | 698 / 698 | none / none | {} | the 4 targets, exactly | 105,984 |
| p00_int8_plac | 1.23.2 / 1.24.4 | 699 / 699 | +1 `Concat /stages/stages.3/stages.3.1/Concat_3` | {Concat: +1} | the 4 placebo nodes, exactly | 105,984 |

Every model's optimized node-name multiset is IDENTICAL between 1.23.2 and 1.24.4, so the property
that made the original's step interpretable ("an executed graph that is identical between the
releases") holds for the rewritten files too.

The placebo's +1 node: `Concat_3` and `Concat_5` of stage 3 had identical inputs in the original and
were merged by common-subexpression elimination; editing `Concat_5` un-merges them, so one extra
Concat of five 1-element int64 vectors executes per inference. Sub-microsecond, in the CONTROL arm,
whose prediction is that a ~0.48 ms step survives. Recorded now so it is not discovered later. The
primary pair (original / e2 / sham) has no such asymmetry.

Executed Transposes at the four target sites on 1.24.4 (uint8 in every model):

```
original  [256,2,4,2]   [0,2,1,3]      [256,4,2,2]   [0,2,1,3]      [160,2,2,2] x2 [0,2,1,3]
e2        [256,2,4,2,1] [0,2,1,4,3]    [256,4,2,2,1] [0,2,1,4,3]    [160,2,2,2,1] x2 [0,2,1,4,3]
sham      [256,2,4,2,1] [0,2,1,3,4]    [256,4,2,2,1] [0,2,1,3,4]    [160,2,2,2,1] x2 [0,2,1,3,4]
plac      unchanged
```

### (c) Dispatch prediction, and everything else that changed - PASS

| model | dispatches on 1.23.2 | dispatches on 1.24.4 | change vs original |
|---|---|---|---|
| p00_int8 | 32 | **864** | - |
| p00_int8_e2 | 32 | **32** | **-832 on 1.24.4, 0 on 1.23.2** |
| p00_int8_sham | 32 | 864 | 0 |
| p00_int8_plac | 32 | 864 | 0 |

The rewritten file issues no two-byte-block dispatches on either release; its 32 remaining dispatches
are the eight layout Transposes around QLinearConv and the six `[0,2,3,1]` attention Transposes, all
1-byte blocks, present identically in all four models and on both releases.

Everything else that changed, quantified:

* executed node count 698 -> 698 (e2, sham); 698 -> 699 (placebo).
* executed op census unchanged (e2, sham); +1 Concat (placebo).
* Transpose bytes moved 105,984 -> 105,984 in all three.
* file size +128 / +144 / +158 bytes out of 1,796,122.
* inner-loop work at the four sites: 1.23.2 goes from 5,376 two-byte strided writes to 10,752
  one-byte writes with an index increment; 1.24.4 goes from 832 `MlasTranspose<uint16_t>` calls to
  the same 10,752 serial writes. Release-symmetric by source diff (C3).

### (d) A correction to the census table's mechanism labels

`transpose_helper.cc`'s `switch (bytes_per_write)` has cases 1, 2, 4 and 8; anything else falls to
`default:`, which calls `DispatchStridedCopy(tp, ...)` - a routine that DOES receive the thread pool.
The probe's `predict()` labels any non-{1,2,4} block "scalar N-bit" and predicts 0 dispatches. For
8-byte blocks that is right (`uint64_t` is not an `has_mlas_transpose` type). For the census's
"scalar out 128-bit" and "scalar out 160-bit" rows (16- and 20-byte blocks: rows 5, 6, 8-10, 12,
19-34) the label is wrong. It does not affect Exp 27, 27b or E2, because those rows are identical
between the releases in every model here, but the census's path column must not be quoted for them.

---

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

## 6. The exact commands

Everything in one script: **`bash <scratchpad>/run_e2.sh`**. It runs, in this order: AC-power and WSL
gates -> the three V1 verification commands (with `set -o pipefail`, so a failure really aborts) ->
the machine-quiet loop, which writes `precondition.json` -> the timing -> the analysis.

The timing command it runs, from the project root, is exactly:

```
"C:/Users/User/Desktop/SEC/edge-ai-toolkit/.venv/Scripts/python.exe" scripts/paper/ort_version_bisect.py --scratch "C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-SEC/103ae351-d709-4d02-91fc-0b8da660fe58/scratchpad" --versions 1.23.2,1.24.4 --threads 1,4 --repeats 5 --seed 29 --families "e2=results/e2_counterfactual/models:3,64,64" --variants p00_int8,p00_int8_e2,p00_int8_sham,p00_int8_plac --out results/paper/e2_counterfactual.csv
```

and the analysis is exactly:

```
"C:/Users/User/Desktop/SEC/edge-ai-toolkit/.venv/Scripts/python.exe" scripts/paper/e2_analyse.py results/paper/e2_counterfactual.csv
```

which writes `results/e2_counterfactual/verdict.json`. The harness's built-in speedup summary needs a
`p00_fp32` sibling and will print nothing for these variants; that is expected.

**V7a re-run** (only on its trigger; `<V>`, `<R>` are the failing cell's variant and release):

```
"C:/.../python.exe" scripts/paper/ort_version_bisect.py --scratch "<scratchpad>" --versions <R> --threads 4 --repeats 5 --seed 30 --families "e2=results/e2_counterfactual/models:3,64,64" --variants <V> --out results/paper/e2_counterfactual_rerun_seed30.csv
"C:/.../python.exe" scripts/paper/e2_analyse.py results/paper/e2_counterfactual.csv results/paper/e2_counterfactual_rerun_seed30.csv --replace-cell <V>,<R>,4
```

**Extension** (only if the verdict is otherwise valid and PROVISIONAL):

```
"C:/.../python.exe" scripts/paper/ort_version_bisect.py --scratch "<scratchpad>" --versions 1.23.2,1.24.4 --threads 4 --repeats 5 --seed 31 --families "e2=results/e2_counterfactual/models:3,64,64" --variants p00_int8,p00_int8_e2,p00_int8_sham,p00_int8_plac --out results/paper/e2_counterfactual_ext_seed31.csv
"C:/.../python.exe" scripts/paper/e2_analyse.py results/paper/e2_counterfactual.csv results/paper/e2_counterfactual_ext_seed31.csv
```

The primary CSV is never overwritten by either follow-up: `ort_version_bisect.py` writes with
`df.to_csv(out)`, so every run needs its own `--out`.

---

## 7. The exact sentences the paper may use, per outcome

ASCII only. `x`, `y`, `z`, `a`, `b`, `c`, `d` are filled from `verdict.json`; `y` is **this run's**
`D(orig,4T)`, never 0.48. In every case the paper keeps Experiment 27b's mandatory disclosure about
Exp 27 wherever the 27b result is cited; E2 does not remove it.

### W1 - ROOT-CAUSE CONFIRMED (FIRM)

> Rewriting those four Transposes in the full static file so that their moved block is no longer two
> bytes wide - an exact rewrite that leaves the executed graph otherwise identical, moves the same
> 105,984 bytes and produces bit-identical outputs - removes x ms of the y ms step at four threads
> (z% of it; session envelope a% to b%), leaving the rewritten file within the study's 7% floor of
> its ONNX Runtime 1.23.2 latency (a residual of up to 7% of that latency, about 23% of the step, is
> what this test tolerates). A sham rewrite that applies the same edit to the same four nodes but
> leaves the moved block two bytes wide keeps c% of the step, and the same edit applied to four
> Transposes outside that class keeps d% of it. We therefore report the two-byte transpose routing
> added in ONNX Runtime 1.24 (pull request #25680, commit 42869ca) as the root cause of the ONNX
> Runtime 1.23.2 -> 1.24.4 four-thread regression of this 64x64 static-INT8 MobileViT-XXS file
> (sha256 911de4ee...) on our test machine. The same architecture at 256 px shows a 4.43 ms
> four-thread step and a +5.5% one-thread step; a one-thread step cannot be caused by thread-pool
> dispatches, and neither is within the scope of this experiment.

Only with W1 may the paper use the words "root cause". "Traced to a code-level cause" may of course
remain.

### W2 - PARTIAL, or ROOT-CAUSE CONFIRMED but PROVISIONAL

> Rewriting those four Transposes in the full static file so that their moved block is no longer two
> bytes wide removes x ms of the y ms step at four threads (z%; session envelope a% to b%), while a
> sham rewrite that keeps the block two bytes wide keeps c% of it. Part of the step is therefore
> carried by those four nodes and part is not; we continue to report the mechanism as traced to a
> code-level cause and do not claim it as the root cause. The same architecture at 256 px shows a
> 4.43 ms four-thread step and a +5.5% one-thread step, which this experiment does not address.

### W3 - REFUTED

> Rewriting those four Transposes in the full static file so that their moved block is no longer two
> bytes wide changed the step at four threads by x ms (z% of y ms), so in the full model the step
> does not follow those four nodes. The isolation result of Experiment 27b therefore shows
> sufficiency in isolation only, and we report the mechanism as traced to a code-level cause and not
> as the root cause.

### W4 - CONFOUNDED

> A full-model counterfactual was attempted: the four Transposes were rewritten so that their moved
> block is no longer two bytes wide. Its matched control, which applies the same edit but leaves the
> block two bytes wide, did not keep the step (it kept c% of it), so the counterfactual cannot be
> interpreted and we report no result from it. The mechanism remains traced to a code-level cause.

### W5 - INCONCLUSIVE

> A full-model counterfactual was attempted and its registered validity condition V<n> was not met
> (state which), so it returns no verdict. The mechanism remains traced to a code-level cause.

### W6 - the V5 caveat sentence, mandatory in W1/W2/W3 whenever `verdict.json` records it

> On ONNX Runtime 1.23.2, where neither file issues a two-byte-block dispatch, the rewritten file
> differs from the original by R ms at four threads and R1 ms at one thread; this is the cost of the
> substituted serial kernel, which cancels out of the share reported above.

### Design facts, usable in every branch (verified, timing-independent)

> The rewritten file is an exact rewrite: its outputs are bit-identical to the original's on both
> releases and at both optimization levels, its optimized graph has the same 698 nodes with the same
> names under both ONNX Runtime 1.23.2 and 1.24.4, it moves the same 105,984 Transpose bytes, and the
> only difference is the shape and permutation of four uint8 Transposes, which by the dispatch rule
> in core/framework/transpose_helper.cc issue 832 fewer thread-pool dispatches per inference on
> 1.24.4 and the same number on 1.23.2.

### Barred

* Unless the verdict is W1: "root cause", "proves", "the cause of".
* In every branch: any claim about releases other than 1.23.2 and 1.24.4; any claim that #25680 is a
  bug (it is a documented optimization that is a pessimization here); and **any extension of the
  result to another input size, thread count, batch size, model file or machine** - E2 measures one
  file (sha256 911de4ee...), 1x3x64x64, batch 1, four threads pinned to four P-cores, default
  spinning, on one laptop.
* Quoting 0.48 ms as "the step" in an E2 sentence: use this run's `D(orig,4T)`.
* Quoting the census's "scalar 128-bit / 160-bit" path labels for rows 5, 6, 8-10, 12, 19-34
  (section 4d).
