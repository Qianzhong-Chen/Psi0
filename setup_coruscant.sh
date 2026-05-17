#!/bin/bash
# Source this file on coruscant before training/eval:
#   source setup_coruscant.sh

cd /data/hojune/Psi0
source .venv-openpi/bin/activate
source .env

export HF_LEROBOT_HOME=/data/hojune/lerobot
export HF_HOME=/data/hojune/Psi0/cache/hf_home
export HF_DATASETS_CACHE=/data/hojune/Psi0/cache/datasets
export CUDA_HOME=/opt/miniconda3/pkgs/cuda-nvcc-11.8.89-0
export PATH=$CUDA_HOME/bin:$PATH
export WANDB_API_KEY=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

echo "Coruscant env ready. Set CUDA_VISIBLE_DEVICES before training."
