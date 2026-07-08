#!/bin/bash
# Serve the 0630 pickup LoRA checkpoint (18-D EEF, VLM LoRA, RTC OFF) with the
# LOCK-BASE server: base velocity forced to 0 and torso height held at
# --upright-height, no matter what the policy predicts (fixed-base, like bottle).
# The ckpt was merged offline (scripts/merge_lora_ckpt.py) into plain weights, so
# no LoRA/PEFT code runs at inference. Augmentation is NOT applied at inference
# (server only does resize+center_crop).
#
# Usage:
#   ./scripts/deploy/serve_psi0_lora_pickup.sh                 # defaults below
#   ./scripts/deploy/serve_psi0_lora_pickup.sh <RUN_DIR> <CKPT_STEP> <PORT>
set -e

PSI0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PSI0_DIR"

# --- config (override via args or env) ---
# Default checkpoint (uncomment the one you want; both merged + lock-base):

# RUN_DIR="${1:-.runs/psi0_legs_0630_pickup_lora}"   
# 0630 pickup LoRA (10hz), best practice: $ ~/run_psi0_realrobot_bridge.sh --rate 10 --prefetch-frac 0.3 --exec-horizon 15 --enable-publish 
 
RUN_DIR="${1:-.runs/psi0_legs_0701_pickup_lora}"     # 0701 pickup LoRA (30hz)
CKPT_STEP="${2:-17500}"          # merged plain weights (raw LoRA = ckpt_17500_lora_raw)
PORT="${3:-22085}"
HOST="${HOST:-0.0.0.0}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-30}"
UPRIGHT_HEIGHT="${UPRIGHT_HEIGHT:-0.78}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== Psi0 pickup LoRA server (18-D, RTC off, merged, no aug, LOCK-BASE) ==="
echo "    run_dir   : $RUN_DIR"
echo "    ckpt_step : $CKPT_STEP   (merged plain weights)"
echo "    port      : $PORT   (host $HOST)"
echo "    horizon   : $ACTION_EXEC_HORIZON   upright_height: $UPRIGHT_HEIGHT"
echo "    GPU       : $CUDA_VISIBLE_DEVICES"
echo "    base/height are HARD-locked server-side (vx,vy,vyaw->0, h->$UPRIGHT_HEIGHT)"
echo "    health    : curl -s localhost:$PORT/health"
echo "    NOTE: RTC is off (config); run the bridge in sequential chunk playback."
echo "==========================================================================="

exec .venv-psi/bin/python src/psi/deploy/psi0_serve_real_lockbase.py \
    --host "$HOST" \
    --port "$PORT" \
    --policy psi0 \
    --run-dir "$RUN_DIR" \
    --ckpt-step "$CKPT_STEP" \
    --action-exec-horizon "$ACTION_EXEC_HORIZON" \
    --upright-height "$UPRIGHT_HEIGHT"
