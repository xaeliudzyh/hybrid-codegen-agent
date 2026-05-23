#!/usr/bin/env bash
# Run visualize_diffusion.py for every remasking strategy with the same
# task/seed/temperature, so the resulting JSONs are directly comparable.
#
# All command-line knobs of visualize_diffusion.py can be overridden via
# environment variables:
#
#   PYTHON          python interpreter (default: python)
#   MODEL           model name             (default: GSAI-ML/LLaDA-8B-Instruct)
#   DEVICE          cuda / cuda:0 / cpu   (default: auto)
#   TASK_PRESET     geometry_two | geometry_three | arithmetic_two
#                                          (default: geometry_two)
#   STEPS           diffusion steps        (default: 128)
#   BLOCK           block_length           (default: 128)
#   MAX_TOKENS      generation budget      (default: 256)
#   TEMPERATURE     sampling temperature   (default: 0.2)
#   SAVE_EVERY      snapshot stride        (default: 1)
#   FC_BOOST        fc_priority parameter  (default: 0.15)
#   SB              structural_boost param (default: 0.15)
#   SB_WIN          structural_window      (default: 2)
#   SEED            torch seed             (default: 42)
#   OUTDIR          output directory       (default: scripts/visualize/out/run_<ts>)

set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-python}
MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}
DEVICE=${DEVICE:-auto}
TASK_PRESET=${TASK_PRESET:-geometry_two}
STEPS=${STEPS:-256}
BLOCK=${BLOCK:-256}
MAX_TOKENS=${MAX_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.2}
SAVE_EVERY=${SAVE_EVERY:-1}
FC_BOOST=${FC_BOOST:-0.5}
SB=${SB:-0.2}
SB_WIN=${SB_WIN:-2}
SEED=${SEED:-42}

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR=${OUTDIR:-scripts/visualize/out/run_${TS}}
mkdir -p "$OUTDIR"

STRATEGIES=("low_confidence" "random" "entropy" "fc_priority" "structural_boost")

echo "============================================================"
echo "VISUALIZE LLaDA DENOISING  (all 5 remasking strategies)"
echo "  model       : $MODEL"
echo "  device      : $DEVICE"
echo "  task_preset : $TASK_PRESET"
echo "  steps       : $STEPS  block: $BLOCK  max_tokens: $MAX_TOKENS"
echo "  temperature : $TEMPERATURE  save_every: $SAVE_EVERY  seed: $SEED"
echo "  outdir      : $OUTDIR"
echo "============================================================"

for REM in "${STRATEGIES[@]}"; do
    echo
    echo "----- ${REM} -----"
    "$PYTHON" scripts/visualize/visualize_diffusion.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --remasking "$REM" \
        --task_preset "$TASK_PRESET" \
        --steps "$STEPS" \
        --block_length "$BLOCK" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --fc_boost "$FC_BOOST" \
        --structural_boost "$SB" \
        --structural_window "$SB_WIN" \
        --save_every "$SAVE_EVERY" \
        --seed "$SEED" \
        --out "$OUTDIR/visualize_${REM}.json"
done

echo
echo "============================================================"
echo "All runs complete."
echo "JSONs in: $OUTDIR"
echo
echo "Next step (locally):"
echo "  for f in $OUTDIR/*.json; do"
echo "      python scripts/visualize/plot_visualization.py \"\$f\""
echo "  done"
echo "============================================================"
