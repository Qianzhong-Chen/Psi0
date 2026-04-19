#!/bin/bash

# Fine-tune Psi0 with 18-dim EEF actions for Sonic C++ deploy (IK mode).
# Usage: ./finetune-real-psi0-18act-eef.sh [task] [exp]
#
# Action (18): L_eef_6d(6) + L_grip(1) + R_eef_6d(6) + R_grip(1) + vx + vy + vyaw + height
# State  (15): L_eef_6d(6) + L_grip(1) + R_eef_6d(6) + R_grip(1) + height
#
# Deploy path: VLA → EEF poses → Sonic C++ (IK internally) → motor commands
# Dataset: hojjunekim/humanoid_18act_15state_psi (or custom task)

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source .venv-psi/bin/activate

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

# Default dataset and experiment name
DEFAULT_REPO="hojjunekim/humanoid_18act_15state_eef_psi"
export task="${1:-$DEFAULT_REPO}"
export exp="${2:-psi0-18act-eef}"

export DATA_ROOT="${DATA_ROOT:-${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}}"

echo "Task/Dataset: $task"
echo "Experiment name: $exp"
echo "Data root: $DATA_ROOT"

args="
finetune_real_psi0_config \
--seed=292285 \
--exp=$exp \
--train.name=finetune \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=64 \
--train.num_workers=0 \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=40000 \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=2500 \
--train.validation_steps=1000 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--data.root_dir=$DATA_ROOT \
--data.train_repo_ids=$task \
--data.transform.repack.action-chunk-size=30 \
--data.transform.field.stat-path=meta/stats.json \
--data.transform.field.stat-action-key=action \
--data.transform.field.stat-state-key=states \
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
"

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=29500 scripts/train.py \
    ${args}
