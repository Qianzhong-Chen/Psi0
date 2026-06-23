#!/usr/bin/env bash
# Multi-GPU (DDP, data-parallel only) Psi0 fine-tune on GPUs 4-7.
# Launched via `accelerate launch --multi_gpu`; each rank gets its own data shard.
set -euo pipefail

cd "$HOME/LEGS/submodules/Psi0"

# Load API keys / paths
set -a; source .env; set +a

export CUDA_VISIBLE_DEVICES=4,5,6,7        # use GPUs 4-7
export CUDA_HOME="$HOME/miniconda3/envs/legs"   # deepspeed/nvcc check
export WANDB_MODE=online

# 4 processes = 4 GPUs (1 process per GPU). DDP => data splitting only.
.venv-psi/bin/accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --main_process_port=29510 \
    scripts/train.py finetune_real_psi0_config \
    --exp=bottle-pickup-4gpu \
    --train.name=finetune \
    --train.data_parallel=ddp \
    --train.mixed_precision=bf16 \
    --train.train_batch_size=4 \
    --train.num_workers=0 \
    --train.max_training_steps=20000 \
    --train.warmup_steps=500 \
    --train.warmup_ratio=None \
    --train.checkpointing_steps=2500 \
    --train.validation_steps=1000 \
    --train.max_grad_norm=1.0 \
    --train.learning_rate=1e-4 \
    --train.lr_scheduler_type=cosine \
    --train.resume_from_checkpoint=latest \
    --log.report_to=wandb \
    --data.root_dir="$HOME/LEGS/data/lerobot/wigs" \
    --data.train_repo_ids=bottle_pickup_30_uni \
    --data.transform.repack.action-chunk-size=30 \
    --data.transform.repack.state-key=observation.state_psi0 \
    --data.transform.repack.action-key=action.psi0_18 \
    --data.transform.field.stat-path=meta/stats.json \
    --data.transform.field.stat-action-key=action.psi0_18 \
    --data.transform.field.stat-state-key=observation.state_psi0 \
    --data.transform.field.action_norm_type=bounds \
    --data.transform.field.no-use-norm-mask \
    --data.transform.field.normalize-state \
    --data.transform.model.img-aug \
    --data.transform.model.resize.size 240 320 \
    --data.transform.model.center_crop.size 240 320 \
    --model.model_name_or_path=cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
    --model.pretrained-action-header-path=cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
    --model.noise-scheduler=flow \
    --model.train-diffusion-steps=1000 \
    --model.n_conditions=0 \
    --model.action-chunk-size=30 \
    --model.action-dim=18 \
    --model.action-exec-horizon=30 \
    --model.observation-horizon=1 \
    --model.odim=15 \
    --model.view_feature_dim=2048 \
    --model.no-tune-vlm \
    --model.no-use_film \
    --model.no-combined_temb \
    --model.rtc \
    --model.max-delay=8
