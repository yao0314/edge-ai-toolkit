#!/usr/bin/env bash
# Batch 2: the experiments the independent challengers' triage ranked, in their order.
#
# These exist because five challenger agents attacked the two INT8 performance claims and killed
# both as worded (results/paper/FINDINGS.md, experiment 14). Three of their kills needed no machine
# time at all. What is left below is the part that does.
#
#   bash scripts/paper/run_challenger_followups.sh
#
# Shares run_remaining.sh's lock ON PURPOSE. Two drivers measuring latency at once inflate both, and
# on 2026-09-16 that is exactly what happened; a shared lock makes it impossible rather than
# unlikely.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
LOG=results/paper/challenger_followups.log
LOCK=results/paper/.run_remaining.lock
SCRATCH="${SCRATCH_DIR:-C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-SEC/103ae351-d709-4d02-91fc-0b8da660fe58/scratchpad}"
mkdir -p results/paper

if ! (set -o noclobber; echo "$$ followups started $(date '+%F %T')" > "$LOCK") 2>/dev/null; then
  echo "another driver holds $LOCK:" >&2; cat "$LOCK" >&2
  echo "refusing to measure against a running batch." >&2
  exit 1
fi
trap 'rm -f "$LOCK"' EXIT INT TERM
: > "$LOG"

say() { printf '\n========== %s  [%s] ==========\n' "$1" "$(date +%H:%M)" | tee -a "$LOG"; }
FAILED=""
stage() {
  local title="$1"; shift
  say "$title"
  "$@" 2>&1 | tail -60 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { printf '  !! STAGE FAILED (exit %s)\n' "$rc" | tee -a "$LOG"
                       FAILED="${FAILED}\n  - $title (exit $rc)"; }
}

# --- 15: is the penalty one ORT release? -------------------------------------------------------
# The challengers' top priority, and the one that can invalidate everything else. Their pilot (run
# under 24-52% background load, so not quotable) found MobileViT static is NOT slow on 1.16.3 or
# UPDATED: eleven runtimes now - 1.16.3, 1.20.1, 1.21.1, 1.22.1, 1.23.2, 1.24.4, 1.25.1, 1.26.0,
# 1.27.0, 1.28.0 and the project's 1.30.0. The coarse run put the MobileViT static regression
# between 1.20.1 (1.005x) and 1.28.0 (0.655x); this pins it to an exact release.
# 1.20.1. If that holds clean, the finding is a regression report, not an architecture claim.
stage "15  ORT version bisect - 11 runtimes, 1.16.3 through 1.30.0" \
      "$PY" scripts/paper/ort_version_bisect.py --scratch "$SCRATCH" --repeats 3

# --- 1-corrected: count Q/DQ on the graph that actually executes -------------------------------
# The original density figures counted nodes ORT deletes before the first inference. This prints
# the size of that error per model, which is the most useful thing the script now produces.
# Already completed in the interrupted run (results/paper/qdq_density.csv). Re-run only if that
# file is missing, so a restart does not redo an hour of graph dumps for nothing.
if [ ! -s results/paper/qdq_density.csv ]; then
  stage "1c  Q/DQ density on the EXECUTED graph" \
        "$PY" scripts/paper/qdq_boundary_density.py --graph both
else
  say "1c  Q/DQ density - already done, skipping"
fi

# --- 16: fusion blocking, and its opposite prediction ------------------------------------------
# Boundary density predicts restricting static quantization to [Conv, MatMul] is SLOWER (more
# crossings). Fusion blocking predicts FASTER (Gelu/LayerNorm fuse again). They cannot both be right.
for c in fusion_blocking_mobilevit fusion_blocking_deit; do
  stage "16  fusion blocking  -  $c" "$PY" scripts/sweep.py --config "configs/experiments/${c}.yaml"
done

# --- 16b: does restoring fusion also restore SCALING? ------------------------------------------
# The second prediction from the same run, free. If int8_static_cm scales better than plain int8
# and not merely runs faster, fusion blocking explains the latency AND the parallel-scaling deficit,
# and the two-axis framing collapses into one mechanism with two symptoms.
for r in fusion_blocking_mobilevit fusion_blocking_deit; do
  stage "16b thread sweep on the fusion-blocking variants  -  $r"         "$PY" scripts/paper/thread_sweep.py --run "$r" --threads 1,2,4,6 --spill 8         --out "results/paper/thread_sweep_$r.csv"
done

# --- 19: is dynamic's accuracy damage just minmax calibration? ---------------------------------
# The decisive test for experiment 18's unification. Vary ONLY the calibration method on
# MobileNetV2 and see whether static+minmax reproduces the dynamic arm's -11.07 pt. Cheap, and it
# can falsify the explanation in one run.
stage "19  calibration isolation (minmax vs percentile vs entropy)"       "$PY" scripts/sweep.py --config configs/experiments/calibration_isolation.yaml

# --- 21b: four-point scaling curves for the runs that only have two -----------------------------
# A 2-point Amdahl estimate is hypersensitive near p -> 1: a 2.3% shift in one measurement moves the
# serial-time estimate by 4x, and the two methods disagreed by 2.9 ms on ResNet50. Any serial/parallel
# decomposition needs a real curve, so the remaining headline runs get one.
for r in detection classification anomaly stress_imagenette; do
  stage "21b thread sweep (4-point curve)  -  $r"         "$PY" scripts/paper/thread_sweep.py --run "$r" --threads 1,2,4,6 --spill 8         --out "results/paper/thread_sweep_$r.csv"
done
stage "21c parallel-fraction fits over every 4-point curve" "$PY" scripts/paper/parallel_fraction.py

# --- 17: the dynamic-quantization cost model ---------------------------------------------------
# Stage 1b measured int8_dyn_conv at 0.054x on a plain CNN - 18x slower than fp32. Dynamic
# quantization recomputes each activation's min/max every inference, so its cost scales with
# activation volume. This tests that against runs whose resolutions differ by an order of magnitude.
stage "17  dynamic quantization cost model" "$PY" scripts/paper/dynamic_quant_cost_model.py

# --- re-analysis over everything measured tonight ----------------------------------------------
stage "re-analysis  scaling deficit (now including the CNN runs)" "$PY" scripts/paper/scaling_deficit.py
stage "re-analysis  static vs dynamic decomposition" "$PY" scripts/paper/static_vs_dynamic.py
stage "variant inventory (de-duplicated)" "$PY" scripts/paper/variant_inventory.py

# --- deliverables ------------------------------------------------------------------------------
stage "refresh the Excel workbook" "$PY" scripts/paper/append_paper_sheets.py

say "done"
if [ -n "$FAILED" ]; then printf 'STAGES THAT FAILED:%b\n' "$FAILED" | tee -a "$LOG"
else echo "all stages exited 0" | tee -a "$LOG"; fi
echo "log: $LOG"
