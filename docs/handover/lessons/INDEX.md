# Lessons (copied from the previous machine's Claude memory, 2026-09-22)

Each file is one lesson, written when it was learned. They mention the previous machine's paths
(`C:\Users\User\Desktop\SEC\...`) and a TAAI 2026 paper written with the same toolkit; the lessons
themselves carry over. `[[name]]` links point at other files in this folder (some linked notes were
not copied because they concern only the paper).

Read the one that matches the work at hand:

| file | when it matters |
|---|---|
| `sec2026-competition-unknowns.md` | anything about competition rules, scoring, energy, the sandbox. Written 2026-09-14; its "we only have detection, no MOTA/IDF1" point is outdated: `adapters/tracking.py` and `adapters/mot_metrics.py` now exist |
| `onnx-canonical-output.md` | choosing an output or submission format |
| `pruning-earns-nothing-within-budget.md` | deciding whether to prune; the title is the OLD claim, the body corrects it (on MobileNetV2/CIFAR-10, pruning pays with global pruning, 20 recovery epochs and `prune.recovery_lr_mult` 5; raising the LR did not help YOLOv8n). Its "one control still outstanding" line is stale: point 3 of the same note reports that control as done |
| `onnx-file-graph-is-not-the-executed-graph.md` | counting nodes, explaining a speed-up or slowdown |
| `cpu-has-avx-vnni.md` | describing a CPU; it is about the previous laptop (i7-13620H), so check the new machine's CPUID yourself |
| `loadpercentage-is-not-machine-load.md`, `other-load-pct-is-a-dead-gauge.md` | deciding whether the machine is quiet enough to measure |
| `run-the-suite-on-linux.md` | before trusting a green test run on only one OS. Its WSL venv paths (`~/eatk`, `~/eatk-v2`) existed only on the old machine: use any Linux Python 3.11/3.12 venv and run with `-rs`. Current baseline: 167 passed on Windows (2026-09-22) |
| `adversarial-tests-need-a-separate-agent.md` | any check meant to refute a claim |
| `workflow-agents-leave-orphan-processes.md`, `scratchpad-is-not-storage.md`, `clean-up-orphan-files.md` | before and after running agents or long jobs |
