# The unit of analysis is the architecture family, not the file

Written 2026-09-23 15:05 (UTC+8), while the measurement was still running and before any result
existed: `results/paper/a2_thirdparty_exploratory.csv` had 3 of 11 batches written and no analysis
had been run.

The signature under test is a property of the architecture: it is the shape and permutation of a
Transpose inside the network that decides whether its moved block is two bytes wide. Two files of the
same architecture quantized by different publishers are therefore not two independent observations of
the rule, and counting them as two inflates the sample.

The A2 pool has **18 admitted files in 15 architecture families**. Three families hold two files each:
efficientnet (a04, c01), mobilenet_v2 (a02, c02), resnet (a03, c05).

- **Primary unit: the architecture family, n = 15.** Hit rate, miss rate and any test statistic are
  computed over families. A family counts as predicted-to-step if any of its files is flagged, and as
  having stepped if any of its files stepped; both are reported per file underneath.
- **Second file of a family: a replicate, reported separately.** It answers a different question -
  whether two quantizers of the same architecture behave the same - and is never counted as an
  independent row of the confusion matrix.
- The registration's own cap rule already argued this ("ten quantized MobileViT clones are one
  observation, not ten", `a_design_registered.md:88`) and then counted matrix rows per file. The
  count, not the rule, was wrong, and this note fixes it in the direction that makes the sample
  smaller.
