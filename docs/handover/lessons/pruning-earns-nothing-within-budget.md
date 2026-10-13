---
name: pruning-earns-nothing-within-budget
description: Pruning DOES pay inside the 2.0-pt budget (4.35x vs 3.40x, costing 1.57 pt) but only under a tuned recovery schedule; the earlier 5.56x figure was mine and was wrong
metadata:
  type: project
---

This memory used to assert that pruning contributes nothing inside the 2.0-point rule and that
3.40–3.91x (INT8 alone) is the ceiling. **Both halves are now known to be wrong**, and the way
each fell is worth more than the claim ever was.

**What broke it, in order:**

1. *Experiment 9 (measurement variance)* killed the evidence. The "21 of 23 winners prune zero"
   figure counted ratio variations of one strategy as separate strategies and mixed `_quick` smoke
   runs into the denominator. 22 of 48 cells were also decided inside a 0.28 pt combined SD — the
   argmax was noise, not a winner.
2. *Experiment 10 (finetuning schedule control)* killed the claim itself. `prune.recovery_lr_mult`
   had been left at its default of 1.0 for every run in this project. Sweeping it at ratio 0.25,
   `classification_prune_global`, 20 epochs:

       lr_mult  1.0 → −2.08 pt (over budget)      lr_mult  5.0 → −1.11 pt
       lr_mult  2.0 → −1.76 pt                    lr_mult 10.0 → −1.10 pt
                                                  lr_mult 20.0 → −0.93 pt

   Monotone and saturating, four times the noise floor.

3. *The fair-schedule control (10b), 2026-09-16* then corrected my own correction. Giving BOTH arms
   the same epochs and learning rates — and the unpruned arm a do-nothing option — the unpruned
   model also improves, 0.9236 → **0.9302**. So the pruned model costs **−1.57 pt against a properly
   tuned reference**, not −0.93. It still fits the 2.0-pt budget, so pruning still pays.

   **And the 5.56x I quoted was wrong.** It came from `results/classification/models/p25_int8.onnx`
   (uniform L2) while the accuracy came from `classification_prune_global` — two different
   strategies — and that 1,603,509-byte model is **5.19 points below baseline**, wholly
   inadmissible. The real figure for the tuned, admissible model is **4.35x against quantize-only's
   3.40x: a 1.28x advantage.**

**Why:** this is exactly the failure mode Wang et al. (arXiv:2301.05219) describe — a method
declared not to pay when what was actually measured was one untuned hyperparameter. We reproduced
it on ourselves, in writing, for three days.

**How to apply:** never quote "pruning earns nothing" or "3.4–3.9x is the ceiling" again. The
honest current statement is *"under the default recovery schedule pruning does not fit the budget;
under a tuned one it does, and the schedule is the finding."* One control still outstanding before
even that is safe: `recovery_lr_mult` applies only after pruning, so the unpruned arm never got the
same extra training — `scripts/paper/fair_schedule_comparison.py` settles it, and until it runs the
gain could still be "we tuned one arm". See [[adversarial-tests-need-a-separate-agent]]: every one
of these reversals came from a control written down before looking, not from re-reading the data.
