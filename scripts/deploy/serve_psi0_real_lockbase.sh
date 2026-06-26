#!/bin/bash
# Terminal 1: Psi0 policy server for REAL-ROBOT deployment (lower body locked).
# Base velocity forced to 0 and torso height held at --upright-height, no matter
# what the policy predicts. Replaces the GR00T inference server.
#
# Usage:
#   ./scripts/deploy/serve_psi0_real_lockbase.sh                 # defaults below
#   ./scripts/deploy/serve_psi0_real_lockbase.sh <RUN_DIR> <CKPT_STEP> <PORT>
set -e

# Resolve repo root (this script lives in submodules/Psi0/scripts/deploy/)
PSI0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PSI0_DIR"

# --- config (override via args or env) ---
RUN_DIR="${1:-.runs/finetune/bottle-pickup-500.2606241614}"
CKPT_STEP="${2:-20000}"
PORT="${3:-22085}"
HOST="${HOST:-0.0.0.0}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-30}"
UPRIGHT_HEIGHT="${UPRIGHT_HEIGHT:-0.78}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== Psi0 real-robot server (lock-base) ==="
echo "    run_dir   : $RUN_DIR"
echo "    ckpt_step : $CKPT_STEP"
echo "    port      : $PORT   (host $HOST)"
echo "    horizon   : $ACTION_EXEC_HORIZON   upright_height: $UPRIGHT_HEIGHT"
echo "    GPU       : $CUDA_VISIBLE_DEVICES"
echo "    base/height are HARD-locked server-side (vx,vy,vyaw->0, h->$UPRIGHT_HEIGHT)"
echo "    health    : curl -s localhost:$PORT/health"
echo "=========================================="

exec .venv-psi/bin/python src/psi/deploy/psi0_serve_real_lockbase.py \
    --host "$HOST" \
    --port "$PORT" \
    --policy psi0 \
    --run-dir "$RUN_DIR" \
    --ckpt-step "$CKPT_STEP" \
    --action-exec-horizon "$ACTION_EXEC_HORIZON" \
    --upright-height "$UPRIGHT_HEIGHT"
