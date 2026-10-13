# Quarantined: measured under competing load, 2026-09-16 03:04-03:18

These three `*_dyn` runs (experiment 1b, static vs dynamic quantization on CNNs) were measured while
independent challenger agents were running their own latency probes on the same machine. Two latency
measurements competing for the same P-cores inflate both, and the challengers' own output shows it -
one of their cells reads `2.289 ms (sd 2.441)`, an SD larger than the measurement.

- `classification_dyn`   complete, but its tail overlaps the contamination window
- `stress_resnet50_dyn`  1 row, killed mid-run
- `detection_dyn`        models exported, no measurements

Kept rather than deleted because the ACCURACY and SIZE columns are unaffected - those are
deterministic - so if a re-run disagrees on accuracy that is a real signal worth chasing. The
LATENCY columns here must never be quoted.

Re-run clean with: bash scripts/paper/run_remaining.sh

## Models deleted 2026-09-16 03:45

The `.onnx` files (116 MB, mostly ResNet50's 94 MB fp32) were deleted; the `sweep.csv` files, the
Pareto plots and this note stay. The models are byte-reproducible from the configs at any time, and
keeping 116 MB of weights whose only unique content is a set of unusable latency numbers is the
kind of quiet accumulation that put 4.7 GB of debris in this repo once already.
