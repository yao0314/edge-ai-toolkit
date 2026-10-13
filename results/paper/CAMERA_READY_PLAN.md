# Five-day plan for the camera-ready version (agreed 2026-09-23, hand-in 2026-09-28)

The submitted PDF is frozen (`results/_frozen_2026-09-23/`, git `e26f7a5`). Everything below is for the
camera-ready version and for the record. Only one measurement may run at a time on this machine; every
experiment registers its decision rule, designed by someone other than the author of the claim, before any
timing.

| # | Item | Value if it works | Cost | Risk | Status |
|---|---|---|---|---|---|
| **A** | **Turn the mechanism into a predictive check and validate it on third-party quantized models**: a static reader flags 8-bit Transposes whose moved block is two bytes; the prediction list is frozen, then measured | a checkable rule plus a tool, instead of one anecdote: generalization | 1 day | low | **CLOSED 2026-09-24** -> `results/paper/A_RESULTS.md` |
| B | Release-drift map: operator × dtype × thread count × 11 releases | turns a case study into a measurement study with a public dataset | tool 0.5 d + overnight run + 0.5 d | medium | **CLOSED 2026-09-24** -> `results/paper/B_RESULTS.md` |
| 1 | Moved-block width sweep (1/2/4/8 bytes × uint8/fp16/fp32 × 1/2/4 threads) on generated probes | states the rule as a law with a pre-registered prediction | 1–2 d | low | queued (design drafted, not run) |
| 2 | The optimization's real trade-off: how much the same change speeds up float16 | "it buys X% on fp16 and costs Y% on INT8" — a complete trade-off, strong for the upstream issue | 0.5–1 d | low | queued, user asked to hold |
| 3 | Why YOLOv8n got 28% faster in the same release | both directions explained by code, not just the slow one | 1 d | medium | queued |
| 4 | Cross-platform replication (AMD Zen 2 without AVX-VNNI, Linux) | removes the study's biggest limitation: one laptop, one OS | 0.5 d + teammate's machine | low | queued |
| 5 | Build 1.24.4 twice, once with the routing reverted | causal proof: removing it removes the step | 1–2 d | high | design only; no build, no download without the user's word |

Rules that apply to all of them: predictions and decision rules are frozen before the numbers exist; the
machine must be quiet and on AC power while measuring; nothing is measured while another measurement runs;
every reported number traces to a CSV of the same run; agents get their own scratch directory, never search a
whole drive, never leave a background process, and the orphan sweep runs after each round.

## Decisions taken 2026-09-23 (user)

- **The submitted paper stays frozen.** If the new experiments produce results, the paper is redesigned from
  scratch rather than patched: no more writing a paragraph per experiment as they land. So page budget is not
  a constraint on what we measure now.
- **Energy work (item 3)**: the battery-mode runs happen with the user sitting at the machine, at a time they
  will set. Latency measurements stay on AC power; the two data sets are labelled separately and never mixed.
  More than CPU time is recorded: effective frequency, core class, dispatch count, MACs, peak RSS and tensor
  volume, quantized/float op mix, with the battery discharge rate as the only ground-truth energy signal.
- **The ten missing ONNX Runtime environments may be rebuilt** for item B. They go on D: (the wheels are in the
  local uv cache, so no re-download), not on C:, which is short of space.
- **Literature survey (item 5)**: the second independent coder is the teammate already credited in the paper;
  the acknowledgement is adjusted accordingly. Survey protocol registered before coding; counts only, no
  naming, no re-measurement of other people's experiments.
- **If experiment A's positive class is short of the registered minimum**, the candidate list is widened and
  more models are fetched, with disk use kept bounded: small files preferred, rejected files deleted as the
  registration requires, pool capped at 2.5 GB.

## Standing rules added 2026-09-23 (user)

- **Deadlock watchdog.** If one experiment needs more than three repair attempts on the same obstacle, an
  independent agent that has not worked on it reviews the state and rules STUCK or CONTINUE. On STUCK the user
  is notified, the attempt is recorded with what blocked it, and work moves to the next item instead of
  grinding. No item is abandoned silently.
- **Safety monitoring while anything runs.** Whole-machine CPU and free RAM are checked before and during each
  measurement block. The CPU temperature sensor needs administrator rights on this laptop and is not readable,
  so sustained clock-frequency drop is used as the throttling proxy instead. Our own processes stop on
  overload; the user's own jobs are only reported.
- **Item A status 2026-09-23: INCONCLUSIVE-SPARSE by its own registration.** 6 of 17 candidates admitted; 9 of
  the 11 rejections were "no licence stated" and all of them were the vision transformers and hybrids, the very
  class that could carry the signature. Indicative flagged count 0, architecture families 6 (minimum 8), so the
  registered abort criterion fired and nothing was measured. Two ways forward, queued rather than abandoned:
  widen the search to licence-declared quantized vision transformers, and/or register a separate arm that keeps
  the third-party ARCHITECTURES but uses our own quantizer, with the weaker external validity stated.

## Item A closed 2026-09-23, both rounds

- **A (as registered)**: 6 of 17 candidates admitted, indicative flagged count 0, 6 architecture families.
  Abort criterion 2 fired; nothing measured. INCONCLUSIVE-SPARSE.
- **A2 (amended admission, licence state recorded instead of required, no redistribution)**: 36 candidates,
  13 admitted, 8 matrix rows in 8 families, 5 pair-replicates. Abort criterion 2 fired again (13 < 18);
  nothing measured. Rejections: 12 for a non-fixed or non-float32 input, 6 for being dynamically quantized
  rather than static, 2 over the 60 ms screening cap, 3 other. 22 rejected files deleted after their hashes
  and reasons were recorded; pool down to 121 MB.
- **What the two rounds did establish, and what is worth writing up:** the static checker reproduces the
  mechanism's fingerprint from the file alone (the 832 added dispatches, validated against the probe manifest);
  public 8-bit ONNX vision models are overwhelmingly DYNAMICALLY quantized, so their Transposes never execute
  in 8 bits; and static-INT8 vision transformers under 60 MB come from a single publisher on the hub. The
  sampling frame, not the mechanism, is what stopped A.
- **Next**: item B, whose sample size is under our own control (generated probes plus our six models across
  eleven releases).

## Pending after item B (recorded 2026-09-23 22:00, do not lose)

1. **Output equivalence across releases.** B shows the same file changing speed between releases; nobody has checked whether the OUTPUTS change. Run the same inputs on two releases, compare tensors. Until this is done, "upgrading buys 25% for free" needs a caveat. User decision 2026-09-23: after B.
2. **V6 re-runs**, as registered and as the user decided (no amendment): b01 r50, b06 c3_320x320, b07 mv256, b08 c3_299x299 - one re-run each, void on a second failure. Procedure in results/expB/v6_decision.md.
3. **Within-batch drift test**, to be designed by an INDEPENDENT agent, not by me: does latency drift correlate with execution position inside a batch? That, not the below-gate share, is what would break exchangeability. Additive to V6, changes no threshold.
4. **Wording fix**: "all 18 third-party models stayed inside the +/-7% floor" must be scoped to the A2 run. In B, a03 reads -8.0% (MOVED, faster) and b02 +7.8% (UNRESOLVED) at the same transition; the gap is within the documented run-to-run spread.
5. **Memory to write**: report facts and numbers, no alarm words ("serious", "collapse", "contaminated"). User instruction 2026-09-23.
