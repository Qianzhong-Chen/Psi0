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

# Fail fast if the OpenPI transformers patch is missing.
# Otherwise PI0Pytorch.__init__ raises ValueError per-rank under torchrun,
# which is much harder to read than this single-line message.
if ! python -c "from transformers.models.siglip import check; \
    import sys; sys.exit(0 if check.check_whether_transformers_replace_is_installed_correctly() else 1)" 2>/dev/null; then
    echo "ERROR: transformers_replace patch is not applied in .venv-openpi." >&2
    echo "       Run (from $(pwd)):" >&2
    echo "         cp -r src/openpi/models_pytorch/transformers_replace/* \\" >&2
    echo "               .venv-openpi/lib/python3.10/site-packages/transformers/" >&2
    exit 3
fi

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