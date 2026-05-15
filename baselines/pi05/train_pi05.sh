#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

if [ ! -f .venv-openpi/bin/activate ]; then
    echo "ERROR: .venv-openpi not found at $(pwd)/.venv-openpi" >&2
    echo "       Build it first per baselines/pi05/README.md:" >&2
    echo "       uv venv .venv-openpi --python 3.10 && ..." >&2
    exit 2
fi
source .venv-openpi/bin/activate

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [extra args...]"
    echo "Example: $0 G1WholebodyXMoveBendPickTeleop-v0"
    exit 1
fi

task="$1"
shift
EXTRA_ARGS="$@"

# Caller can override SAVE_INTERVAL via env. Default 10000 preserves prior
# behavior for callers that don't set it; longrun sbatches lower this to 2500
# so each ~4h slurm segment lands at least one ckpt for --resume to pick up.
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE src/openpi/train_pytorch.py \
        $task \
        --exp_name=$task \
        --save_interval=$SAVE_INTERVAL \
        --checkpoint_base_dir=.runs/openpi-05 \
        ${EXTRA_ARGS}