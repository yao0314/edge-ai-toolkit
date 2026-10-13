#!/usr/bin/env bash
# Two experiments the version bisect made necessary, to run AFTER run_challenger_followups.sh.
#
# They exist because experiment 15b changed what the other results mean: MobileViT static QDQ is
# fine on ORT 1.23.2 and ~35% slower from 1.24/1.25 onward, on byte-identical files. Everything
# measured on 1.30.0 was therefore measured inside a regressed regime, and two conclusions need
# re-checking against a runtime where the regression does not exist.
#
#   bash scripts/paper/run_post_batch.sh
#
# Shares run_remaining.sh's lock, so it cannot overlap either batch.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
LOG=results/paper/post_batch.log
LOCK=results/paper/.run_remaining.lock
SCRATCH="${SCRATCH_DIR:-C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-SEC/103ae351-d709-4d02-91fc-0b8da660fe58/scratchpad}"
mkdir -p results/paper

if ! (set -o noclobber; echo "$$ post-batch started $(date '+%F %T')" > "$LOCK") 2>/dev/null; then
  echo "another driver holds $LOCK:" >&2; cat "$LOCK" >&2; exit 1
fi
trap 'rm -f "$LOCK"' EXIT INT TERM
: > "$LOG"

say() { printf '\n========== %s  [%s] ==========\n' "$1" "$(date +%H:%M)" | tee -a "$LOG"; }
FAILED=""
stage() {
  local title="$1"; shift
  say "$title"
  "$@" 2>&1 | tail -70 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { printf '  !! STAGE FAILED (exit %s)\n' "$rc" | tee -a "$LOG"
                       FAILED="${FAILED}\n  - $title (exit $rc)"; }
}

# --- 20: did ORT stop FUSING, or did its kernels get slower? -----------------------------------
# The decisive test for the regression's mechanism, and it needs no timing at all. The same .onnx
# file dumped through the fast runtime (1.23.2) and the slow one (1.25.1): if the optimized graphs
# differ, the regression is lost fusion and the diff names which fusion; if they are identical, the
# regression is in the kernels or the threading and fusion blocking is refuted as the CAUSE.
stage "20  optimized-graph diff, 1.23.2 (fast) vs 1.25.1 (slow)" \
      "$PY" scripts/paper/optgraph_version_diff.py --scratch "$SCRATCH" --versions 1.23.2,1.25.1

# --- 22: is MobileViT's op-type result about op types, or about the regression? -----------------
# Experiment 16 found restricting op_types_to_quantize to [Conv, MatMul] is 20% FASTER on DeiT at
# 4 threads and 8-9% SLOWER on MobileViT - but both were measured on 1.30.0, inside the regressed
# regime. Timing the SAME files under 1.23.2 separates the two explanations. No torch or adapter is
# needed in the side venvs: the .onnx files exist and this only has to time them.
#
# PRE-REGISTERED: if static_cm beats plain int8 on MobileViT under 1.23.2 but loses under 1.30.0,
# MobileViT's result was the regression, not the op types, and the DeiT recommendation generalises.
# If static_cm loses on MobileViT under BOTH runtimes, the architecture difference is real and the
# recommendation must stay transformer-only.
stage "22  fusion variants across the regression boundary (1.23.2 vs 1.30.0)" \
      "$PY" scripts/paper/ort_version_bisect.py --scratch "$SCRATCH" --repeats 3 \
      --families "mvit=results/fusion_blocking_mobilevit/models:3,64,64;deit=results/fusion_blocking_deit/models:3,32,32" \
      --variants p00_fp32,p00_int8,p00_int8_static_cm,p00_int8_static_ex,p00_int8_dyn_mm \
      --versions 1.23.2,1.30.0 \
      --out results/paper/fusion_across_versions.csv

stage "refresh the Excel workbook" "$PY" scripts/paper/append_paper_sheets.py

say "done"
if [ -n "$FAILED" ]; then printf 'STAGES THAT FAILED:%b\n' "$FAILED" | tee -a "$LOG"
else echo "all stages exited 0" | tee -a "$LOG"; fi
echo "log: $LOG"
