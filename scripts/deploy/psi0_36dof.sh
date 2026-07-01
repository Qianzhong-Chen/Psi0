#!/bin/bash
# Terminal 3: Psi0 36-D loco-manip policy server (NO locking, NO WBC).
# Serves the bend-pick-wristcam model over HTTP /act for the dry-run bridge.
#
# Usage:
#   ./scripts/deploy/psi0_36dof.sh                          # defaults below
#   ./scripts/deploy/psi0_36dof.sh <RUN_DIR> <CKPT_STEP> <PORT>
set -e

# Resolve repo root (this script lives in submodules/Psi0/scripts/deploy/)
PSI0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PSI0_DIR"

# --- config (override via args or env) ---
RUN_DIR="${1:-.runs/bend-pick-wristcam-v1}"
CKPT_STEP="${2:-40000}"
PORT="${3:-8014}"
HOST="${HOST:-0.0.0.0}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-30}"
CONFIG_MODULE="${CONFIG_MODULE:-finetune_simple_psi0_config}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== Psi0 36-D loco-manip server (no lock, no WBC) ==="
echo "    run_dir   : $RUN_DIR"
echo "    ckpt_step : $CKPT_STEP"
echo "    port      : $PORT   (host $HOST)"
echo "    horizon   : $ACTION_EXEC_HORIZON   config: $CONFIG_MODULE"
echo "    GPU       : $CUDA_VISIBLE_DEVICES"
echo "    health    : curl -s localhost:$PORT/health"
echo "    (~70s to load; ready at 'Uvicorn running on http://$HOST:$PORT')"
echo "===================================================="

exec .venv-psi/bin/python src/psi/deploy/psi0_serve_locomanip.py \
    --host "$HOST" \
    --port "$PORT" \
    --policy psi0 \
    --run-dir "$RUN_DIR" \
    --ckpt-step "$CKPT_STEP" \
    --action-exec-horizon "$ACTION_EXEC_HORIZON" \
    --config-module "$CONFIG_MODULE"
