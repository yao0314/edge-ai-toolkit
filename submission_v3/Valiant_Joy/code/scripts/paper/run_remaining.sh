#!/usr/bin/env bash
# Run the remaining paper experiments, strictly one at a time.
#
# Every experiment below measures LATENCY, and latency measurements contaminate each other: two
# runs competing for the same P-cores inflate both. The harness detects competing load and warns,
# but a warned measurement is a wasted measurement.
#
#   bash scripts/paper/run_remaining.sh
#
# On 2026-09-16 a copy of this script was left alive in its idle-wait for half an hour while another
# batch ran, and would have started a SECOND full batch the moment the machine went quiet. The wait
# loop below was written to prevent exactly that and instead made the duplicate patient enough to be
# invisible. Hence the lock: a second instance now refuses to start rather than queueing behind the
# first. A driver that waits politely for hours is indistinguishable from a driver that has hung.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
LOG=results/paper/run_remaining.log
LOCK=results/paper/.run_remaining.lock
mkdir -p results/paper

if ! (set -o noclobber; echo "$$ started $(date '+%F %T')" > "$LOCK") 2>/dev/null; then
  echo "another driver holds $LOCK:" >&2
  cat "$LOCK" >&2
  echo "refusing to start a second batch. Remove the lock if that process is gone." >&2
  exit 1
fi
trap 'rm -f "$LOCK"' EXIT INT TERM
: > "$LOG"

say() { printf '\n========== %s  [%s] ==========\n' "$1" "$(date +%H:%M)" | tee -a "$LOG"; }

# Run one stage and record whether it actually succeeded. `| tail | tee` swallows the exit status,
# so PIPESTATUS is read explicitly - a stage that dies at import time otherwise scrolls past as a
# short, calm-looking block of output and gets mistaken for a stage with little to say.
stage() {
  local title="$1"; shift
  say "$title"
  "$@" 2>&1 | tail -40 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    printf '  !! STAGE FAILED (exit %s): %s\n' "$rc" "$*" | tee -a "$LOG"
    FAILED="${FAILED}\n  - $title (exit $rc)"
  fi
}
FAILED=""

# --- wait for the currently running jobs -------------------------------------------------------
say "waiting for the machine to go idle"
idle=0
for _ in $(seq 1 120); do   # 60 minutes, then give up rather than wait forever
  n=$(powershell -NoProfile -Command "(Get-Process python* -ErrorAction SilentlyContinue | Measure-Object).Count" 2>/dev/null | tr -d '\r ')
  [ -z "${n:-}" ] && n=0
  if [ "$n" -le 1 ]; then echo "  idle (python processes: $n)" | tee -a "$LOG"; idle=1; break; fi
  echo "  $n python process(es) still running, waiting 30s" | tee -a "$LOG"
  sleep 30
done
if [ "$idle" -ne 1 ]; then
  echo "  STILL BUSY after 60 min - refusing to measure against competing load" | tee -a "$LOG"
  exit 1
fi

# --- 1b: does dynamic quantization beat static QDQ on CNNs too? -------------------------------
# Turns the static-vs-dynamic finding from n=2 architectures (both transformers) into n=5.
for c in classification stress_resnet50 detection; do
  stage "1b  dynamic vs static QDQ  -  $c" \
        "$PY" scripts/sweep.py --config "configs/experiments/${c}_dyn.yaml"
done

# --- 2: per-node profiling ---------------------------------------------------------------------
# Attribution for the static-vs-dynamic gap. 1 thread only, separate from the headline harness.
for r in classification_mobilevit classification_deit stress_resnet50; do
  stage "2  node profiling  -  $r" \
        "$PY" scripts/paper/node_profile.py --models "results/$r/models/p00_int8*.onnx" \
        --out "results/paper/node_profile_$r.csv"
done

# --- 3: thread sweep ---------------------------------------------------------------------------
# The parallel-scaling finding currently stands on two points (1T and 4T).
for r in classification_deit classification_mobilevit stress_resnet50; do
  stage "3  thread sweep  -  $r" \
        "$PY" scripts/paper/thread_sweep.py --run "$r" --threads 1,2,4,6 --spill 8,12 --out "results/paper/thread_sweep_$r.csv"
done

# --- 5+6+7: quantization configuration factorial ----------------------------------------------
# quant_format x type pair x reduce_range. Run together: ORT's documented trap is a COMBINATION.
for c in classification classification_mobilevit; do
  stage "5+6+7  quant config factorial  -  $c" \
        "$PY" scripts/paper/quant_config_factorial.py --config "configs/${c}.yaml"
done

# --- 4: RAM attribution ------------------------------------------------------------------------
# Several slow INT8 variants use MORE peak RAM than their fp32 sibling; the arena on/off control
# decides whether that is quantization or just the allocator.
stage "4  RAM attribution (arena on vs off)" "$PY" scripts/paper/ram_attribution.py

# --- 10-followup: the control that could take the pruning finding away again --------------------
# recovery_lr_mult applies only AFTER pruning, so experiment 10 tuned one arm and left the other
# alone. This gives both arms the same epochs and the same learning rates and quantizes both.
# It must run before anything about pruning is written down.
stage "10b  fair schedule comparison (pruned vs unpruned, both tuned)" \
      "$PY" scripts/paper/fair_schedule_comparison.py --lr-mults 1,5,20 --epochs 20

# --- inventory ---------------------------------------------------------------------------------
stage "variant inventory (de-duplicated)" "$PY" scripts/paper/variant_inventory.py

say "done"
if [ -n "$FAILED" ]; then
  printf 'STAGES THAT FAILED:%b\n' "$FAILED" | tee -a "$LOG"
else
  echo "all stages exited 0" | tee -a "$LOG"
fi
echo "log: $LOG"
