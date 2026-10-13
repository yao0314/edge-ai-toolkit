---
name: sec2026-competition-unknowns
description: "What is known/unknown about the SEC 2026 Software Track rules; don't hard-code assumptions"
metadata: 
  node_type: memory
  type: project
  originSessionId: 103ae351-d709-4d02-91fc-0b8da660fe58
  modified: 2026-09-12T18:06:56.722Z
---

SEC 2026 Minimalist Edge AI Challenge, Software Optimization Track. Held 2026-10-13..15 at Santa Clara Univ.

**Known from the official page (re-checked 2026-09-13):**
- Runs in a "virtual development environment" / sandbox. Hardware is only for the *other* track
  (Hardware Deployment, which uses the Qualcomm Arduino UNO Q), so this team probably never touches
  a physical board.
- The four items ARE published verbatim, but as OBJECTIVES, not as a rubric. Official page:
  "Teams will optimize model accuracy, model compression, resource efficiency, and energy
  parsimony." vTools: "Your mission is to optimize for the 'Green Trifecta':" + the four numbered
  items. Both say "optimize", never "judged on" / "scored on" / "criteria".
- **Re-verified 2026-09-14: NEITHER source contains the words judge/judging/criteria/scored/
  scoring/rubric/winner/selected/ranked/evaluation.** The single evaluation-adjacent sentence
  anywhere is "Pitch your solution to a jury of world-leading edge computing experts from academia
  and industry." There is NO published statement of how teams are evaluated or winners chosen.
  Prizes are $1000/$750/$500 per track.
- So "the four judging criteria" is OUR INFERENCE, not a quoted rule. Do not state it as fact to
  the user. The only published evaluation mechanism is the jury pitch itself.
- "Green Trifecta" names FOUR things while trifecta means three — the copy is loose, which is
  itself weak evidence the four are marketing framing rather than a specified rubric.
- **The organisers' own example workloads are "real-time multi-object tracking or environmental
  anomaly detection"** — quoted as "such as", so they are illustrative rather than binding, but
  they are the strongest signal available about the task.
- Registration deadline was 2026-09-06 (already passed as of 2026-09-13 — confirm with the user
  that they registered).

**Unknown:** the score weights and formula, the specific model or data handed out (PyTorch, ONNX,
TFLite?), how energy is measured, the time limit, the sandbox's RAM/CPU spec, and whether own code
may be brought.

**Why it matters:** the handover forbids hard-coding any of these, and the two named example tasks
should drive effort allocation.

**How to apply:**
- Keep every such choice a YAML parameter; suggest asking on the official Discord.
- **Multi-object tracking is named first, and we only have detection.** Compression targets the
  detector (tracking is cheap post-processing), so the compression side is covered — but the METRIC
  would be MOTA/IDF1, and `adapters/detection.py` only computes COCO mAP. That evaluate() gap is
  the most defensible remaining adapter work.
- "Environmental" anomaly detection is the same shape as the existing ECG5000 adapter (1D sensor
  series + reconstruction error): swap the data, keep the adapter.
- Image classification is NOT among the named examples, yet it has the most architectures (5).
  Those were chosen to cover pipeline failure modes rather than to guess the task, and each one
  found a bug now fixed in core/ — but do not add more.
- AI Hub has no UNO Q/QRB2210 device and reports no energy.

**Energy, re-verified 2026-09-14 (the user asked whether UNO Q is hardware-track-only — it is):**
- Official page, Software Optimization Track, verbatim: "Compete in a virtual development
  environment", "Optimize models without dedicated hardware". Hardware Deployment Track, verbatim:
  "Build with Qualcomm Arduino UNO Q hardware", "Work with microcontrollers, C++, and systems
  engineering". The UNO Q kit is "shipped to participating teams" of THAT track.
- The IEEE vTools page and the Facebook/blog copy describe one unified challenge "deploying ...
  onto provided low-power hardware (Qualcomm Arduino UNO Q)" with NO track split. That is
  marketing copy; the acm-ieee-sec.org page is authoritative and does split them. Expect this
  contradiction to come up again — do not re-litigate it, cite the official page.
- **How Joules-per-inference is measured in the software track is published NOWHERE** — not the
  official page, not vTools, not the blog. Only the objective is stated.
- Tested 2026-09-14: WSL2 exposes NO RAPL counters (`/sys/class/powercap` empty, 0 `energy_uj`
  files). Virtualised environments generally cannot read energy MSRs; containers on bare metal
  often can. So "measure energy ourselves in their sandbox" is probably NOT available, and adding
  a RAPL provider is speculative until the sandbox is seen.
- `core/energy.py` providers are none | csv | command | aihub. `command` is the right plug for
  whatever tool they hand out on the day; that design choice still looks correct.

Related: [[onnx-canonical-output]], [[clean-up-orphan-files]]
