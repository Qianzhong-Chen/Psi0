#!/bin/bash
# Serve the 0703 LOCO-MANIP LoRA checkpoint (18-D EEF, VLM LoRA, RTC OFF) with the
# base + height UNLOCKED, so the policy's predicted vx/vyaw drive SONIC locomotion
# while the left arm manipulates. Same server binary as the pickup ckpts
# (psi0_serve_real_lockbase.py) but with --no-lock-base --no-lock-height, so nothing
# is clamped: the full 18-D action passes through.
#
# The ckpt was merged offline (scripts/merge_lora_ckpt.py) into plain weights, so no
# LoRA/PEFT code runs at inference. Augmentation is NOT applied at inference (server
# only does resize+center_crop).
#
# Loco-manip vs pickup: the 0703 training stats show vx in [0, 0.22] and vyaw in
# [-0.2, 0.2] (robot walks forward + turns toward the target); right arm + height
# are effectively fixed in the data. Height is left UNLOCKED here so the policy owns
# it; pass LOCK_HEIGHT=1 to pin it to --upright-height if you want a fixed torso.
#
# Usage:
#   ./scripts/deploy/serve_psi0_lora_locomani.sh                 # defaults below
#   ./scripts/deploy/serve_psi0_lora_locomani.sh <RUN_DIR> <CKPT_STEP> <PORT>
#   LOCK_HEIGHT=1 ./scripts/deploy/serve_psi0_lora_locomani.sh   # keep torso height fixed
set -e

PSI0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PSI0_DIR"

# --- config (override via args or env) ---
# Default checkpoint (uncomment the one you want; all merged + base-unlocked):
# RUN_DIR="${1:-.runs/psi0_legs_0706_locomani_lora}"; CKPT_STEP="${2:-30000}"  # 0706 loco-manip LoRA @30k (prev)

# RUN_DIR="${1:-.runs/psi0_legs_0715_locomani_lora_chunk30}"; CKPT_STEP="${2:-22500}"    # 0715 loco-manip chunk30 @22.5k (current)
## Best practice: $ run_psi0_realrobot_bridge.sh  --rate 10 --prefetch-frac 0.0  --exec-horizon 30 --enable-publish

RUN_DIR="${1:-.runs/psi0_legs_0715_locomani_lora_chunk20}"; CKPT_STEP="${2:-22500}"  # 0715 loco-manip chunk20 @22.5k
# Best @ 0715: $ PROMPT="walk to the table and pick up the bottle" ./run_psi0_realrobot_bridge.sh --rate 10 --prefetch-frac 0.0 --exec-horizon 10 --enable-publish 

PORT="${3:-22085}"
HOST="${HOST:-0.0.0.0}"
# Empty by default -> server uses the checkpoint's own action_exec_horizon (20 for
# chunk20, 30 for chunk30). Do NOT hardcode 30: the server asserts exec_horizon <=
# chunk_size, so 30 crashes a chunk20 ckpt. Override via env only if you know the size.
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-}"
UPRIGHT_HEIGHT="${UPRIGHT_HEIGHT:-0.78}"

# Base is ALWAYS unlocked for loco-manip. Height unlocked by default; LOCK_HEIGHT=1
# pins it to UPRIGHT_HEIGHT server-side.
if [ "${LOCK_HEIGHT:-0}" = "1" ]; then
    HEIGHT_FLAG="--lock-height"
    HEIGHT_DESC="LOCKED -> $UPRIGHT_HEIGHT"
else
    HEIGHT_FLAG="--no-lock-height"
    HEIGHT_DESC="unlocked (policy owns height)"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== Psi0 loco-manip LoRA server (18-D, RTC off, merged, no aug, BASE UNLOCKED) ==="
echo "    run_dir   : $RUN_DIR"
echo "    ckpt_step : $CKPT_STEP   (merged plain weights)"
echo "    port      : $PORT   (host $HOST)"
echo "    horizon   : ${ACTION_EXEC_HORIZON:-<from ckpt config>}"
echo "    base      : UNLOCKED (policy vx/vy/vyaw drive SONIC locomotion)"
echo "    height    : $HEIGHT_DESC"
echo "    GPU       : $CUDA_VISIBLE_DEVICES"
echo "    health    : curl -s localhost:$PORT/health"
echo "    NOTE: RTC is off (config); run the bridge in sequential chunk playback."
echo "==============================================================================="

# Build optional args: only pass --action-exec-horizon if explicitly set (else the
# server uses the ckpt's own value, which is always <= chunk_size and thus safe).
EXTRA_ARGS=()
if [ -n "$ACTION_EXEC_HORIZON" ]; then
    EXTRA_ARGS+=(--action-exec-horizon "$ACTION_EXEC_HORIZON")
fi

exec .venv-psi/bin/python src/psi/deploy/psi0_serve_real_lockbase.py \
    --host "$HOST" \
    --port "$PORT" \
    --policy psi0 \
    --run-dir "$RUN_DIR" \
    --ckpt-step "$CKPT_STEP" \
    "${EXTRA_ARGS[@]}" \
    --upright-height "$UPRIGHT_HEIGHT" \
    --no-lock-base \
    "$HEIGHT_FLAG"
