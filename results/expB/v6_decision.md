# V6 fired on batch 1 — decision record

**2026-09-23, Taipei.** Written after batch 1 completed and before batch 2's data was analysed.
The registration (`results/paper/b_design_registered.md`) is **unchanged**; this file records a
decision taken under it, not an amendment to it.

## The rule, verbatim (§8.1, line 937)

> **V6 — core delivery reported, not gated** (§5.4). If a batch's below-0.95 rate exceeds **45 %**
> (roughly double the worst historical rate of 25.2 %), the batch is re-run once; a second failure
> voids that batch.

## What fired

Batch 1 (CONTRAST, ResNet-50 224×224, three variants, eleven releases, one and four threads, five
sessions per cell), `results/expB/batches/b01_contrast_r50.csv`, 330 rows, 260 measured:

| figure | value |
|---|---|
| below-0.95 rate | **179 / 260 = 68.8 %** |
| below-0.90 rate | 25.4 % |
| median core delivery | 0.936 |
| one thread / four threads | 81 % / 59 % below gate |

Denominator: sessions with a recorded `core_delivery`, both thread counts pooled — the definition
that reproduces the registration's historical figures (75/396, 78/462, 106/420 are whole-file counts).

## What was considered and rejected

An amendment was drafted and **rejected by the user on 2026-09-23**. It would have replaced the
trigger with "median core delivery over the batch's sessions with p50 ≥ 10 ms below 0.90", on the
grounds that the gauge is duration-biased. The evidence for that bias is real and is recorded here
because it belongs in the paper's limitations either way:

`core_delivery = u_true × (mean_wall / p50)`. The child's loop is time-limited at 2.0 s with a
40-iteration floor (`scripts/paper/ort_version_bisect.py:78`), so `u_true = cd × p50 × iters / 2000`
is exact whenever `iters > 40`. Over 346 cells of the three stored runs (spin default):

| statistic | value |
|---|---|
| Spearman rho(p50, core_delivery) | **−0.78** |
| Spearman rho(p50, u_true) | **+0.03** |

The whole slope is the inflation factor: a sub-2 ms model is flattered by 5–10 % because interference
lands in a tail the median ignores; above ~10 ms the factor is 1.00 and the gauge is honest.

The rejection is deliberate and is the stricter choice: the threshold is not moved to fit the data.

## The decision

1. **V6 stands as registered.** No amendment, no substituted statistic.
2. **Every fired batch is re-run once**, after the fifteen batches complete, as a fresh invocation of
   the whole batch (V7: no cell spans two invocations). The first run's CSV is **moved to
   `results/expB/batches/v6run1/`** first, because `b_run.py` skips a batch whose CSV already has the
   expected row count.

   The procedure needs no new code, because that same skip is what isolates the re-runs:

   ```
   mkdir -p results/expB/batches/v6run1
   mv results/expB/batches/b01_contrast_r50.csv results/expB/batches/v6run1/
   .venv/Scripts/python.exe scripts/paper/b_run.py --stage map --anchor-reps 3
   ```

   Batches that already have a complete CSV are logged as "complete already – not re-run" and
   `continue` before their anchor call (`b_run.py:793-799`), so the invocation measures only the
   fired batches. The anchors are not re-measured either: `run_anchor` skips a boundary whose CSV
   already holds enough rows (`b_run.py:764-766`), so the map's anchor series stays intact as the
   record of the night.

   **A subdirectory, not a suffix.** An earlier version of this record said to rename the file to
   `b01_contrast_r50.v6run1.csv` and claimed the analysis would ignore it. That is wrong: the
   analysis globs `b[0-9][0-9]_*.csv`, which that name still matches, so both runs would be read into
   the same cells and every one of them would become VOID at ten sessions. The first run is moved out
   of the globbed directory instead. Corrected 2026-09-24 before the re-runs were launched; the
   original wording is recorded here rather than quietly deleted.
3. **If the re-run's below-0.95 rate again exceeds 45 %, batch 1 is void** and the ResNet-50 CONTRAST
   arm is reported as void, with both runs' rates published.
4. **Every other batch is judged by the same rule as it lands.** Batches 2 (224×224), 6 (320×320) and
   8 (299×299) are the likely candidates.
5. **Interpretation recorded** (the registration does not spell it out): if the re-run passes, the
   re-run's sessions are the batch's data and the first run is reported as the failed attempt. The
   two runs are never pooled — §3.4(a) rejects cross-run comparison.

## What determines whether the re-run passes

Almost entirely the machine, not the models. In the honest regime (p50 ≥ 10 ms, which is 240 of
batch 1's 260 measured sessions) the stored runs sit at 12.5–20.5 % below gate; batch 1 tonight sat
at 61–73 %. The gap is background load: five resident `claude` processes, the largest at 9537 s of
cumulative CPU, and the same cells run 5–7 % slower than the stored scope run. A re-run on a quiet
machine is expected to land near the historical rate and pass. A re-run with the same background
load is expected to fail and void the batch.

Operationally, therefore: the re-run is scheduled only after the desktop sessions are closed and a
quiet check passes at the registered threshold.

## Published regardless of the outcome

* Every batch's below-gate rate, as §5.4 already requires.
* The duration bias above, in the limitations, with the two Spearman figures and the note that
  `u_true` is unrecoverable for cells that stopped at the 40-iteration floor.
* This decision, including the rejected amendment, so the record shows the threshold was not moved
  after seeing the data.

---

## WITHDRAWN as an operational rule — 2026-09-24, by the user

This rule is no longer applied. No batch is re-run for it, no batch is voided by it, and it is not
a reporting topic. The analysis pipeline no longer computes it: `scripts/paper/b_gate_check.py` is
deleted.

The registration document itself is NOT edited. It is hash-stamped into every run log
(`b_design_registered.md` sha256 5bb5d73d14f5) and altering it after seeing data is the one thing a
pre-registration exists to prevent. What changed is what we DO with it, and that is recorded here,
with its date, rather than done quietly.

Grounds, all established before this decision: the statistic is a level, not a difference, while
every claim in this study is a difference measured under matched conditions; its slope is the
gauge, not the machine (rho(p50, core_delivery) = -0.78 against rho(p50, u_true) = +0.03 over 346
stored cells); a quiet-machine re-measurement of batch 1 reproduced it (66.5% against 68.8%; first written as
65.1%, corrected 2026-09-24 from `batches/r50_repeat/`: 173 of 260 sessions), so it
is not a machine-state signal; and the batches it fired on contained the largest findings in the
map (+96.4%, +81.1%, +80.0%, +43.9%) while one batch it fired on contained no movement at all.

The measured values stay in the artefact for anyone who wants them: every batch CSV carries
`core_delivery` per session, as it always has.
