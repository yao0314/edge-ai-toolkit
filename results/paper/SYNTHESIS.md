# Where the paper stands after A, B and W

Written 2026-09-24, after the three camera-ready experiments finished. The submitted PDF is frozen
(`results/_frozen_2026-09-23/`). This file is the input to deciding what the camera-ready version
says.

---

## Part 1 — What the submitted paper already establishes

Title: *What INT8 Post-Training Quantization Buys on a CPU: Runtime Version Moves the Speedup, Model
Class Picks the Method.* Six models, one i7-13620H laptop, ONNX Runtime, batch 1, one and four
threads.

**1. The method must follow the model class, and the wrong choice is expensive.** On 1.30.0 at one
thread, static U8S8 INT8 makes convolutional models 1.6–3.2× faster; dynamic INT8 on the same
convolutions is **more than 12× slower** than FP32. On the attention model the ranking reverses. On
the hybrid, INT8 buys mainly size. Files land at 26–32 % of FP32 size for at most 0.86 pt of quality.

**2. The runtime release moves the speedup of byte-identical files, in both directions.** Over eleven
releases and 396 measurements, MobileViT-XXS's static file went from 0.976× to 0.631× of its FP32
sibling: 1.24.4 made the **static file itself 31 % slower** at four threads (1.553 → 2.037 ms, disjoint
sessions) while its FP32 and dynamic siblings and its own one-thread latency moved under 2 %; then
1.25.1 sped up the FP32 baseline by 15–16 %. The regression grew to **+46 % at 256 px** and persisted
through 1.30.0. The same release cut **YOLOv8n's** static four-thread latency by 28 %.

**3. The regression is traced to a code-level cause on an identical executed graph.** The optimized
graph is node-for-node identical at 1.23.2, 1.24.4 and 1.25.1 (698 nodes). It is not the spin policy,
not the quantized MatMuls, not QDQ density, not fusion blocking, not bandwidth — each tested against a
rule fixed in advance and each failing. A probe holding only the 56 uint8 Transposes reproduces 87 %
of the step; within it, the **four Transposes whose moved block is two bytes wide** carry it and the
other 52 do not, the same loop shapes carry it in float16 and not float32, and at one thread none of
them does. PR #25680 adds `has_mlas_transpose<uint16_t>`, routing exactly those to MLAS's
multi-threaded transpose: **832 thread-pool dispatches per four-thread inference instead of none.**

**4. What the paper says it has NOT established** (§VI, verbatim in substance):

* The mechanism is shown **"in isolation, not by removal"**. The full-model counterfactual failed its
  own validity condition and returns no verdict; a 1.24.4 build without the change was never made.
* **One machine, one OS.** The dispatch cost is expected to be platform-specific.
* **One model per non-convolutional class**, so the regression is established for the hybrid class
  only to the extent MobileViT-XXS represents it.
* **YOLOv8n's gain was not examined.**
* The 256 px **one-thread** slowdown (5.5 %) is explicitly said to be unexplained by the mechanism,
  because "a one-thread slowdown cannot come from thread-pool dispatches".
* Energy was not measured.

---

## Part 2 — What A, B and W add

### A — the mechanism as a screening criterion, checked on files we did not make

`results/paper/A_RESULTS.md`. A static reader flags 8-bit Transposes whose moved block is two bytes;
predictions were hashed before any timing. 46 candidates, **18 admitted**, spanning **15 architecture
families, 9 publishers, 11 export toolchains**, all statically quantized.

**[corrected 2026-09-24 — see `A_RESULTS.md` §2: "zero false alarms" and "specificity 18/18" hold by
construction when nothing is flagged; the result is that all 18 safe calls held still, 15 of them on
files with eight-bit Transposes of other widths.]** On the release pair that moved our own MobileViT
by a third, every third-party model the check called safe stayed inside the 7 % floor (+1.2 % to −4.8 %
at four threads).

**The denominator matters, and it is one-sided.** The checker flagged none of the 18, so the scan has
**18 negatives and zero positives**: it measures specificity (18/18) and says nothing about
sensitivity. With an empty positive class, "the check is accurate" and "the check never fires" are not
separated by this scan, so the paper should call it a **criterion** or a **screening rule**, not a
predictor, and always give the denominator. The positive class comes from W instead, by construction:
four distinct constructed files carry the pattern (five with a replicate), ten decision cells at two
and four threads, and all ten moved, by 4.6× to 44×.
Sensitivity demonstrated on files we made, specificity on files we did not; neither half may be quoted
as if it were both.

Ecosystem observation worth a sentence: most public "quantized" ONNX vision models are **dynamically**
quantized, so their Transposes never execute in 8 bits and the mechanism cannot reach them.

### B — one case becomes a rate

`results/paper/B_RESULTS.md`. 38 files × 10 releases × 2 thread counts × 5 fresh processes, 5.71 h,
736 complete cells, denominators exactly as registered (584 pair-cells).

* **10.1 %–15.6 % of single-release steps move an unchanged file by ≥7 %.** Against a fixed 1.23.2
  baseline across the ladder: 23.3 %–29.0 %.
* **What that rate survives, measured against a real null.** The `r50` batch was measured twice, in
  two independent invocations hours apart — 52 cells of the same files on the same releases with no
  release change between them. Their medians differ by 2.6 % (median), 6.9 % (p90), 10.0 % (max); 10 %
  of them clear the 7 % floor and 10 % have disjoint ranges. But the registered rule needs **both**,
  and only **1 of 52 (2 %)** does: that is the measured false-MOVED rate of the rule as registered.
  Applied to N = 584 it predicts about 12 of the 59 MOVED cells could be run-to-run noise, leaving
  roughly 47, or **8 %**. The floor-sensitivity curve, computed but **not substituted for the
  registered figure**: 7 % → 10.1–15.6 % (as registered), 8 % → 9.8–12.7 %, 10 % → 8.6–9.9 %,
  12 % → 7.0–7.5 %. Moving the floor after seeing the data is the error this project already paid for
  once; the paper reports the registered number, this curve and this null, and lets the reader choose.
* **MobileViT reproduced at both sizes in a fresh environment**: +40.3 % at 64 px and **+43.9 % at
  256 px**, four threads, with FP32 and dynamic siblings and one thread flat. The paper had one size.
* **It never comes back**: +37.6 % against 1.23.2 still at 1.30.0, the newest release measured.
* **YOLOv8n's gain reproduced and, separately, explained**: −25.1 % at four threads, −20.6 % at one.
  Moving at both thread counts rules out the dispatch mechanism; a per-operator profile then names it.
  At one thread under `ORT_SEQUENTIAL`, 283 nodes timed individually, **99 % of the −1.785 ms delta is
  `Resize`**: 2.046 → 0.282 ms, a **7.3× faster kernel**. Second is `QLinearConcat` at −0.208 ms;
  everything else is flat. The registered rule reads ≥70 % in one operator as operator-specific, so
  this is a kernel change with a name. §VI's "we did not examine the mechanism of YOLOv8n's gain" is
  answered.
* **Third-party files move too**, up to +96.4 % (one release breaking one file, recovered by the
  next), +80.0 % at both thread counts, +70.5 % at four threads only, −43.1 %.
* **The largest of them is not a bare number, and it is a fallback.** `c3_299x299/b05` at 1.21.1 → 1.22.1,
  profiled per operator at one thread under `ORT_SEQUENTIAL`: the executed graph grows from **427 to 491 nodes**
  and every `QLinearConv`, `NhwcMaxPool` and layout `Transpose` disappears. The quantized convolutions stop being
  fused into integer kernels and run as float `Conv` inside Quantize/Dequantize pairs — hence `Conv` +103.2 ms,
  `QLinearConv` −25.1 ms (gone), float `MaxPool` 45× the time of the NHWC one, Q/DQ up by ~5 ms. 1.23.2 restores
  the fusion. An earlier draft of this file called it a Conv kernel slowdown; the node count shows it is not.
* Two releases stand out: 1.26.0 → 1.27.0 slows many files, 1.28.0 → 1.30.0 speeds many up.
* Measurement fact the paper needs: four-thread session-to-session spread is **9.0 %** against **3.5 %**
  at one thread, on the same machine and files. The 7 % floor is a four-thread floor.

### W — the mechanism becomes a law, by construction

`results/paper/w_design_registered.md`, designed by an independent designer before any timing; probes
built by `scripts/paper/w_probe.py`; 1290 sessions.

Nineteen probe files on one shape template, `dims = [L, 2, N, K]`, `perm = [0,2,1,3]`, so the moved
block is `K × elem_bytes` and **width is the only thing that varies**: 8192 bytes moved per node, four
nodes per file, L and M identical in every variant, with `N = 16/W` absorbing the width. Two
independent static implementations agreed on the predicted routing, and the executed graph was checked
on both releases before timing.

**W1 — only the two-byte files move, and hugely.**

| probe | 4 threads, 1.23.2 → 1.24.4 |
|---|---|
| uint8, 2-byte block | 0.0399 → 0.4511 ms, **+1031 %** |
| float16, 2-byte block | 0.0328 → 0.4308 ms, **+1213 %** |
| uint8 1-, 4-, 8-byte | −3.3 % to +1.6 % |
| float16 / float32 4-, 8-byte | inside the floor |
| same shape, no Transpose | −4.8 % |

**W2 — the byte width, not the element count, governs.** uint8-2-byte (two elements per block) and
float16-2-byte (one element per block) have identical geometry and identical bytes and agree within
**1.2×**; the registered refutation needed a disagreement beyond 2×. Every "the moved block is two
bytes" sentence in the paper survives a test designed to break it.

**W3 — the dose response.** At constant bytes moved, 16× the dispatch count gives **13.2×** the added
latency, and the per-dispatch cost is flat across that range at **0.40–0.52 µs**. The registered
refutation needed a ratio below 2.

**W4 — the strongest single line.** `d_u8_w1_L1024`, already on the MLAS path, costs **1.813 ms** on
both releases. `d_u8_w2_L1024`, after being re-routed, costs **1.817 ms**. At matched dispatch count
the re-routed file is indistinguishable from the file that was always there — **0.25 % apart**. The
two-byte file did not become slow; it was moved onto the slow path.

**W5 — a one-thread cost exists, and it is not big enough to explain the model.** **[corrected
2026-09-24 — see `W_RESULTS.md` §5: the serial cost is 0.030 ms for 4096 loops, not 29 ns per loop,
and the model's step is 33–47× that, not 8.3–11.8×. The passage below is left as first written.]** The two-byte probes
also slowed at one thread (+48 % and +56 %), and in the dose ladder the one-thread penalty grows with
L (+0.003 / +0.019 / +0.030 ms ≈ **29 ns per loop**). MLAS's transpose is slightly slower than the
scalar loop even serially. **Checked against the models, it does not account for them.** The executed
graph of MobileViT-XXS has 4 newly-routed Transposes and 832 loops at 64 px, 6 and 4096 at 256 px
(`a_check.py sc-a` on 1.24.4):

| | measured one-thread step | 29 ns × loops | ratio |
|---|---|---|---|
| 64 px static INT8 | +0.049 ms (FP32 sibling +0.065 ms, so **no specific effect**) | 0.024 ms | — |
| 256 px static INT8 | **+1.401 ms** (FP32 sibling −0.445 ms, so the effect is real and specific) | 0.119 ms | **11.8×** |

Direction right, magnitude twelve times short. The paper's sentence that the 256 px one-thread
slowdown is not explained by the dispatch mechanism **stands and must be kept**.

**W6 — the per-dispatch constant is a lower bound, not a calibration.** **[superseded 2026-09-24 —
see `W_RESULTS.md` §6: the four Transposes timed alone cost 0.51 µs per dispatch, the probes' level;
the "twice" figure charges the whole model step to the dispatches.]** The same check at four
threads: 64 px measured +0.664 ms against 832 × 0.43 µs = 0.358 ms (ratio 1.85); 256 px +4.223 ms
against 4096 × 0.43 µs = 1.761 ms (ratio 2.40). A dispatch inside a real model costs 0.80–1.03 µs,
about twice what it costs in a sub-millisecond probe. The probes establish that the cost is
proportional to the dispatch count; they under-state its size in a real graph, and the paper must
quote 0.43 µs as a floor rather than as the constant.

One honest complication: 12 of 36 W1 cells crossed the 7 % floor as *faster* at 1.24.4, including
probes with **no Transpose at all**. Every sub-0.04 ms file gained 5–10 %: a release-level improvement
in the skeleton, worth about 1.3 µs per inference, which crosses a floor calibrated on 1–100 ms
models. Control-adjusted, all of them are inside the floor. This is a property of the probes' size,
not of the law, and belongs in the write-up rather than being counted as fifteen failed predictions.

---

## Part 3 — What the camera-ready version should say

**The centre of gravity moves.** The submitted paper is a measurement study of what INT8 buys, with
the release effect as its main contribution and one traced case. After A, B and W the release effect
is no longer a case: it is a rate (B), a screening criterion with zero false alarms (A) and a demonstrated law with a dose
response (W). The paper should keep its baseline measurements and let the release effect carry the
claim.

**Section by section:**

* **Abstract and title.** Keep the topic first. The claim to lead with is no longer "1.24.4 made one
  file 31 % slower" but "a release moves 10–15 % of unchanged files past a 7 % floor, and one such
  move is explained to the microsecond". Both MobileViT sizes now carry the regression (+40.3 %,
  +43.9 %) in a fresh environment.
* **§V-C, Localizing the regression.** This is where W goes, and it upgrades the section from
  *sufficiency in isolation* to *a law with a dose response*: width is the only variable, only 2 bytes
  moves, the byte width and not the element count governs it, cost scales with dispatch count at
  0.43 µs each, and the re-routed file lands exactly on the always-routed file. §VI's "in isolation,
  not by removal" can be narrowed: a build-level revert is still absent, but the causal structure is
  now shown by construction.
* **New subsection, or §V-E: how often this happens.** B's rate, its bracket, the persistence to
  1.30.0, the third-party cases up to +96 %, and the two releases that move many files at once. This
  is the part that turns a case study into a measurement study, and it is the strongest new claim.
* **§V-D and §VI, YOLOv8n.** No longer unexamined: it moves at both thread counts, so it is a kernel
  change and not the Transpose routing. Say that and say what remains unknown.
* **§VI, threats.** Three items shrink and one grows. The one-model-per-class limit is answered by A
  (15 families, 9 publishers) and B (38 files). The one-thread puzzle is answered by W5. The "in
  isolation" limit is narrowed by W. What grows: the four-thread measurement floor (9.0 % against
  3.5 %) should be stated as a property of the platform, not buried.
* **Methods.** The 7 % floor now has a measured justification, not a historical one — and the measured
  null (1 false MOVED in 52 re-measured cells) belongs beside it.
* **§IV, Table IV, and the submitted paper's own small numbers.** The four-thread measurement floor
  applies backwards as well: the smallest four-thread static gains in the submitted paper (1.07× and
  1.12×) sit inside a spread of 9.0 % and can no longer be read as gains. One sentence in the
  camera-ready must say so, in the table's note as well as in §VI.
* **What must not be claimed.** B is a single run — the 91 moved cells were not independently
  re-measured. A's registered stratum minimums were only partly evaluated, so it reports a measurement
  and a false-alarm count, not the registered verdict. ORT 1.16.3 is absent from B. W's probes are
  sub-millisecond, so their absolute magnitudes do not transfer to models.

**What is still open, in priority order:** the build-level revert (plan item 5, needs a compile),
cross-platform replication (item 4, needs the teammate's machine), the fp16 trade-off (item 2, on
hold), and whether a release changes model *outputs* and not only speed — nobody has checked, and the
sentence "upgrading buys 25 % for free" depends on it.
