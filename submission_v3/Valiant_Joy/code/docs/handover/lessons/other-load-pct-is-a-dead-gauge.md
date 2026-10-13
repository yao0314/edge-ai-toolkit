---
name: other-load-pct-is-a-dead-gauge
description: core/measure.py's competing-load alarm does not measure competing load; use cpu_ms_per_inf/(threads*latency) instead
metadata:
  type: reference
---

`other_load_pct` in edge-ai-toolkit's `core/measure.py` is **not a valid contamination alarm.**
Measured over 232 stored measurements on 2026-09-16: it fires (>8%) on **141 of them** while
correlating with actual core delivery at **Spearman -0.057, p = 0.39** - indistinguishable from
zero. A useful alarm would be strongly negative.

The instrument that works is computable from columns already recorded:

    core_delivery = cpu_ms_per_inf / (threads * latency_p50_ms)      1.0 = every core delivered

    median 0.981 at 1 thread, 0.995 at 4 threads; below 0.95 in only 14 of 129 4-thread cells

**Why:** for three days this project treated a 61% false-alarm rate as evidence of a noisy machine,
and on the night the machine really was contaminated the gauge said nothing useful either. Both
errors came from the same place.

**How to apply:** gate on `core_delivery >= 0.95`, not on `other_load_pct`. And when a measurement
looks wrong, check the SD first - the contamination on 2026-09-16 was visible as
`2.289 ms (sd 2.441)`, an SD larger than the measurement, long before any alarm mattered.

Related: [[adversarial-tests-need-a-separate-agent]], [[clean-up-orphan-files]].
