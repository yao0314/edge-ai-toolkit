# Experiment W — the moved-block width as a law, on files we built

**Measured 2026-09-24, 02:04 to 03:31 Taipei. 1290 sessions, none failed.** Registration:
`results/paper/w_design_registered.md`, written by an independent designer before any timing existed
and not by the author of the claim. Builder: `scripts/paper/w_probe.py`. Harness:
`scripts/paper/ort_version_bisect.py`, unedited.

**Corrected 2026-09-24, after the independent review of the paper rewrite** (re-derived from
`w01_main.csv`, `w02_dose.csv` and `mv_exec.json` before editing): §2's reading of the 8-byte files,
§4's dispatch count for the `L = 1024` files, §5's serial cost, §6's per-dispatch reading, and §7's decision-cell count and its range. Each
corrected passage is marked **[corrected]** and states what it replaces.

The question: the paper explains one regression by a property readable from the file — an 8-bit
Transpose whose moved block is two bytes. So far that width is a property of the models that happened
to regress. **Is the width itself what the runtime responds to?**

---

## 1. The instrument

Nineteen probe files on one shape template:

```
dims = [L, 2, N, K],  perm = [0, 2, 1, 3]
```

For this permutation the moved block is `K × elem_bytes`, the loop count is `L` and the inner
dimension is `M = 2`. Width is therefore set by `K` and the element type alone.

**The confound this design exists to avoid.** A naive width sweep changes the tensor size along with
the width and then measures size. Here **8192 bytes are moved per node, four Transpose nodes per
file, `L = 256` and `M = 2` in every variant**, with `N = 16/W` absorbing the width. The ladder is
iso-byte, iso-loop and iso-node-count; within each width, uint8, float16 and float32 land on
identical `(L, M, N)` and identical bytes and differ only in element count. The residual confound is
`N` — `bytes/loop = M·N·W` is an identity and cannot be escaped — and it cancels because the verdict
quantity is a **within-file release delta**, in which any size, aspect-ratio or dtype-kernel effect
appears on both sides.

**Checked before any timing:**

| check | result |
|---|---|
| bytes moved per node, every file | **8192**, asserted at build time |
| predicted routing, re-derived through `transpose_probe.predict()` | matches the registered table in every row |
| second independent implementation (`a_check.py sc-b`) | agrees on all twelve uint8 files |
| executed graph on **both** releases (`a_check.py sc-a`) | 4 Transposes execute in every probe on 1.23.2 and 1.24.4 — none cancelled against a neighbouring Reshape |
| file hashes | frozen before the first timing row |
| files predicted to gain dispatches | **exactly the two 2-byte files**, +1024 each at four threads |

## 2. Only the two-byte files move

W01, 12 variants × 6 releases × 1/2/4 threads × 5 fresh processes = 1080 sessions.

| probe | block | 4 threads, 1.23.2 → 1.24.4 | verdict |
|---|---|---|---|
| `m_u8_w2` | **2 B** | 0.0399 → 0.4511 ms | **+1031 %** |
| `m_f16_w2` | **2 B** | 0.0328 → 0.4308 ms | **+1213 %** |
| `m_u8_w1` | 1 B | 0.5471 → 0.5289 ms | −3.3 % |
| `m_u8_w4` | 4 B | 0.4677 → 0.4753 ms | +1.6 % |
| `m_u8_w8` | 8 B | 0.0356 → 0.0331 ms | −7.0 % |
| `m_f16_w4` / `m_f16_w8` | 4 B / 8 B | — | +5.2 % / −8.1 % |
| `m_f32_w4` / `m_f32_w8` | 4 B / 8 B | — | +4.2 % / −9.8 % |
| `ctrl_u8`, same shape, **no Transpose** | — | 0.0270 → 0.0257 ms | −4.8 % |

**[corrected]** Two structurally different nulls both hold in the sense that matters: **no 1-, 4- or
8-byte file slows.** 1 and 4 bytes are on MLAS's path on **both** releases; 8 bytes is scalar on both.
Under the MOVED rule taken literally, however, the three 8-byte files moved *faster* at every thread
count (−7.0 to −10.2 %, disjoint ranges), and so did the Transpose-free `ctrl_f32` (−7.3 to −7.8 %):
this is the release-wide saving of §7, and net of the controls they are inside the floor. The only
files that slow are the two whose moved block is two bytes. *Replaces:* "8 bytes is scalar on both and
does not move", which is false under the rule as written.

## 3. The byte width governs, not the element count

`m_u8_w2` carries two uint8 per block; `m_f16_w2` carries one float16. Identical geometry, identical
bytes, different element counts. They agree within **1.2×** (+1031 % and +1213 % at four threads;
+1171 % and +1399 % at two). The registration's refutation R3 required a disagreement beyond **2×**.
Every sentence in the paper of the form "the moved block is two bytes wide" survives a test that was
designed to break it.

## 4. The dose response

W02, uint8 only, `L ∈ {64, 256, 1024}` at widths 1 and 2, **8192 bytes moved per node in every row**,
so only the dispatch count changes — by 16×.

| added dispatches per file | added latency, 4 threads | per dispatch |
|---|---|---|
| 256 | +0.134 ms | 0.523 µs |
| 1024 | +0.411 ms | 0.402 µs |
| 4096 | +1.775 ms | 0.433 µs |

**16× the dispatches, 13.2× the added latency** (the total latency grows 10.6×), a log–log slope of
0.93 over the three points, with a per-dispatch cost flat across the range.
Refutation R4 required a ratio below 2. The 1-byte rows, already routed to MLAS and gaining no
dispatches, show no systematic change (−0.02 to +0.04 ms); the Transpose-free control is flat to
0.001 ms.

**The strongest single comparison.** `d_u8_w1_L1024`, which has always been on the MLAS path with 4096
dispatches per inference (`L = 1024` × four nodes; **[corrected]**, first written as 1024), costs
**1.81 ms** on both releases (1.814 and 1.813). `d_u8_w2_L1024`, after being re-routed onto it, costs
**1.817 ms**. At matched dispatch count the re-routed file is **0.25 %** from the file that was always
there. The two-byte file did not become slow; it was moved onto the slow path.

Cross-invocation replication (V4): `m_u8_w2` and `d_u8_w2_L256` are the same construction measured in
two separate invocations with different seeds — 0.4511 against 0.4502 ms, 0.2 % apart.

## 5. A one-thread cost exists, and it does not explain the models

The two-byte probes also slowed at one thread (+48.4 % and +56.3 %), where no dispatch occurs, and in
the dose ladder the one-thread penalty grows with `L`: +0.003, +0.019, +0.030 ms for 256, 1024 and 4096
loops (four nodes × `L`). **[corrected]** Per loop that is 13, 19 and 7 ns — not a constant, so no
per-loop figure is quoted. MLAS's transpose is slightly slower than the scalar loop even serially.

Checked against the real models — executed graphs from `a_check.py sc-a` on 1.24.4 give MobileViT-XXS
4 newly-routed Transposes and **832 loops** at 64 px, 6 and **4096** at 256 px. The 256 px model has
exactly the loop count of `d_u8_w2_L1024`, so that probe's serial cost is the comparison, with no
per-loop extrapolation:

| | measured one-thread step | probe with the same loops | ratio |
|---|---|---|---|
| 64 px static INT8 | +0.049 ms, but its FP32 sibling moved +0.065 ms — **no specific effect** | — | — |
| 256 px static INT8 | **+1.401 ms**, FP32 sibling −0.445 ms — real and specific | 0.030 ms | **47×** |
| 256 px static INT8, independent replication | **+0.983 ms** (+5.2 %) | 0.030 ms | **33×** |

Direction right, magnitude more than an order of magnitude short in both independent runs. *Replaces:*
"about 29 ns per loop" (0.030 ms divided by 1024, the per-node loop count, instead of the file's 4096
loops) multiplied by the model's 4096 loops, which gave 0.119 ms and ratios of 11.8× and 8.3×. The
conclusion is unchanged and stronger. **The paper's statement that the 256 px one-thread
slowdown is not explained by this mechanism stands.**

## 6. What the model's step contains **[corrected]**

The same check at four threads, charging the model's whole step to its new dispatches:

| | measured | dispatches × 0.43 µs | ratio |
|---|---|---|---|
| MobileViT 64 px | +0.664 ms | 832 → 0.358 ms | 1.85× |
| MobileViT 256 px | +4.223 ms | 4096 → 1.761 ms | 2.40× |

Charged that way, a dispatch inside the model would cost 0.80–1.03 µs, about twice the probes'
figure. The first version of this section read that as a property of dispatch in a real graph. The
isolation data say otherwise:

* **The Transposes alone sit at the probes' level.** The Transpose isolation probe timed the four
  two-byte Transposes of the 64 px model on their own: +0.417 ms over 832 dispatches in its
  registered repeat (Experiment 27b), **0.50 µs** each, inside the probes' 0.40–0.52 µs. (The first
  run, Experiment 27, read +0.427 ms and stayed INCONCLUSIVE under its gate.)
* **The rest of the step is outside the Transposes.** The four-thread profile puts 62 % of the added
  time in Transpose and spreads the rest over the other operator types (+9 % together), which the
  dispatch count does not predict.
* **Two runs are mixed in the 1.85× ratio.** The 64 px step above is the drift map's (+0.664 ms,
  +40.3 %); the release sweep measured +0.484 ms (+31 %). 832 × 0.402–0.523 µs = 0.334–0.435 ms is 69–90 %
  of the sweep's step and half to two-thirds of the map's.
* **Work per dispatch explains nothing.** Across the dose probes a loop moving 128, 32 or 8 bytes costs
  0.52, 0.40 or 0.43 µs per dispatch; the models' flagged loops move 8–16 bytes at 64 px (12.9 on
  average) and 16–64 at 256 px (45).

So the probes' 0.40–0.52 µs is the per-dispatch cost, and the model's larger step holds that plus a
remainder outside the Transposes that this experiment does not explain. *Replaces:* "a dispatch
inside a real graph costs 0.80–1.03 µs … the paper should quote 0.43 µs as a lower bound".

## 7. The floor artefact, stated rather than buried

**[corrected]** Cell by cell, 34 of the 57 registered decision cells (19 files × 1, 2 and 4 threads)
matched the pre-registered assignment: 21 of the 36 of W01 and 13 of the 21 of W02 (first written as
"21 of 36", W01 only). Twelve of the 23 that did not are one phenomenon: **every sub-0.04 ms file without a two-byte block got 3–10 %
faster at 1.24.4, including probes with no Transpose at all** (`ctrl_f32` −7.8 %) **[corrected]**,
first written as 5–10 %: `m_f16_w4` at one thread gained 2.8 %. That is a release-level improvement in the
skeleton worth about 1.3 µs per inference, and on a 0.02 ms file it crosses a floor calibrated on
1–100 ms models. Control-adjusted, all twelve are inside the floor. This is a property of the probes'
size, not of the law, and it is reported as one effect rather than as fifteen failed predictions. The
thirteenth, `m_f16_w4` at two threads (−34.3 %, ranges overlapping, UNRESOLVED), is a noisy cell and
is listed by name, as are four one-byte dose cells that cleared the floor with overlapping ranges
(`d_u8_w1_L1024` 2T +9.4 %, `d_u8_w1_L256` 2T −8.8 % and 4T +9.2 %, `d_u8_w1_L64` 4T −8.5 %) and
one that moved faster at one thread (`d_u8_w1_L1024` 1T −9.1 %). The remaining five are the
one-thread cells of the two-byte files (§5: `m_u8_w2`, `m_f16_w2` and the three `d_u8_w2` files),
which are a finding.

## 8. What this does not claim

* **The probes are sub-millisecond.** Their absolute magnitudes (+1031 %) do not transfer to models;
  what transfers is the law and the proportionality, not the size of the effect.
* **No build-level revert.** The causal structure is shown by construction — width is the only
  variable, and the re-routed file lands exactly on the always-routed file — but a 1.24.4 built
  without PR #25680 was still not made.
* **Six releases, not ten**, for the time budget: 1.21.1, 1.22.1, 1.23.2, 1.24.4, 1.25.1, 1.30.0. The
  1.25.1 → 1.30.0 pair is gapped and must be labelled so.
* **One machine, one OS**, as everywhere else in this study. The dispatch cost in particular is a
  scheduler property.

## 9. Where everything is

| | |
|---|---|
| registration | `results/paper/w_design_registered.md` |
| builder and static verification | `scripts/paper/w_probe.py` |
| probe files and manifest | `results/w_probe/` (19 files, hashes frozen) |
| pre-flight records | `results/expW/preflight/` (static, sc_b, sc_a on both releases, hashes) |
| measurements | `results/expW/batches/w01_main.csv`, `w02_dose.csv` |
| model-side checks | `results/expW/preflight/mv_exec.json`, `results/expW/y256_profile_t1.csv`, `b05_profile_t1.csv` |
