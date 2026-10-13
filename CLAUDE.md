# edge-ai-toolkit: instructions for Claude

This project is the team's toolkit for the SEC 2026 Minimalist Edge AI Challenge, Software
Optimization Track (Santa Clara University, 2026-10-13 to 10-15). It was built on another machine;
this machine has no earlier conversation and no Claude memory of it. Start every session by reading
`docs/handover/HANDOVER.md` (Traditional Chinese: status, competition facts, results, lessons for
the day, checklist), then `README.md` as needed. Lessons learned the hard way are in
`docs/handover/lessons/` (see its `INDEX.md`); read the matching one before that kind of work.

New machine: follow section 4 of `docs/handover/HANDOVER.md` (Python 3.11, Git for Windows, a
`.venv` in the project, `pip install -r requirements.txt`, then `python -m pytest -q` must report
173 passed). On Windows call `.venv\Scripts\python` directly instead of activating the venv. If the
suite is not green, fix that before any other work.

## People
- The user (楊元蓁, Yuan-Zhen Yang) directs remotely from Taiwan (UTC+8). A teammate is on site in
  California (PDT, UTC-7): Taiwan is 15 hours ahead.
- Reply in plain Traditional Chinese, briefly. Lead with a recommendation, explain any jargon, and
  when a decision is needed ask a yes/no question. The user is token-sensitive: no repeated audit
  rounds; one independent check of an important conclusion is enough unless numbers changed.

## Hard rules of the code
- `core/` contains no task-specific code; everything task-specific lives in `adapters/`
  (`tests/test_core_is_task_agnostic.py` enforces it).
- Output is always ONNX (the user's decision). For another submission format, convert the FP32
  ONNX (`best_fp32.onnx`), never the INT8 QDQ file.
- Never hard-code an unconfirmed competition fact (scoring weights, formats, time limits, energy
  method): make it a YAML parameter. Never invent a number; energy stays empty until the organisers
  say how it is measured.

## Quantization defaults that were learned, not guessed
These were measured on the previous laptop (Intel i7-13620H, AVX-VNNI); re-measure on the target.
- CNNs: static INT8, QDQ, uint8 activations + int8 weights (U8S8). Never dynamic INT8 on
  convolutions (12.6-16.8x slower than FP32 at one thread). Avoid S8S8 on x86 (slower than FP32 on
  both models tried).
- Attention and hybrid models: dynamic INT8 on MatMul/Gemm weights (`int8_dyn_mm`).
- The ONNX Runtime version changes INT8 speed-ups (1.23.2 -> 1.24.4 made a static hybrid file 31%
  slower at four threads and a static YOLOv8n file 28% faster). Record the sandbox's ORT version and
  re-measure there (`scripts/remeasure.py`).
- Report one-thread and target-thread-count numbers side by side; for static INT8 on CNNs the
  one-thread speed-up overstates the multi-threaded one (not always true for dynamic INT8).

## Measurement discipline
- Measure only on a quiet machine: whole-machine CPU <= 15% (Windows:
  `Win32_PerfFormattedData_PerfOS_Processor` `_Total`; Linux: `mpstat 1 5` or `top`, idle >= 85%;
  never `LoadPercentage` or `other_load_pct`), on AC power, one measurement at a time; medians over
  repeated sessions; every reported number must trace to a CSV of the same run.
- After each run check the `core_delivery` column: the toolkit warns below 0.90; treat sessions
  below 0.95 as suspect and report them.
- Check hardware claims with CPUID before writing them down.

## Working rules
- Stop your own processes if the CPU overloads or overheats; if the user's own jobs cause it, only
  tell the user.
- Any check meant to refute a claim is designed by a separate agent, never by the claim's author.
- Agents: never search `/`, `C:\` or a whole drive; never leave background processes; each agent
  gets its own scratch subdirectory and deletes only that; afterwards check for leftover
  `python`/`find` processes. Keep anything that cannot be regenerated inside the project, not in a
  scratch directory.
- Ask before downloading files, publishing anything (GitHub, web), or sending messages. Commit or
  push only when asked.
