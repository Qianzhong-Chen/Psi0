#!/bin/bash

# Fine-tune Psi0 with joint-space actions aligned to original 36-dim format.
# Usage: ./finetune-real-psi0-36act-joint-aligned.sh [task] [exp]
#
# Dataset must be pre-built with 36-dim vectors where dims match original Psi0 ordering:
#   Original: hand_joints(14) + arm_joints(14) + RPY(3) + height(1) + vx(1) + vy(1) + vyaw(1) + target_yaw(1)
#
#   Action (36, aligned):
#     [0-13]  = 0          (hand joints, unused)
#     [14-20] = L_arm(7)   (matches original left arm)
#     [21-27] = R_arm(7)   (matches original right arm)
#     [28-30] = 0          (RPY, unused)
#     [31]    = height
#     [32]    = vx
#     [33]    = vy
#     [34]    = vyaw
#     [35]    = 0          (target_yaw, unused)
#
#   State (36, aligned):
#     [0-13]  = 0          (hand joints, unused)
#     [14-20] = L_arm(7)
#     [21-27] = R_arm(7)
#     [28-30] = 0          (RPY, unused)
#     [31]    = height
#     [32-35] = 0
#
# Pretrained weights transfer semantically for arm joints and locomotion dims.
# Deploy path: VLA → extract dims 14-27,31-34 → joint targets + loco → Sonic C++
# NOTE: C++ deploy needs modification to accept joint-space upper body targets.
# Dataset: hojjunekim/humanoid_36act_aligned_joint_psi (or custom task)

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source .venv-psi/bin/activate

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

# Default dataset and experiment name
DEFAULT_REPO="hojjunekim/humanoid_36act_aligned_joint_psi"
export task="${1:-$DEFAULT_REPO}"
export exp="${2:-psi0-36act-joint-aligned}"

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
--model.action-dim=36 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=36 \
--model.view_feature_dim=2048 \
--model.no-tune-vlm \
--model.no-use_film \
--model.no-combined_temb \
--model.rtc \
--model.max-delay=8
"

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=29500 scripts/train.py \
    ${args}
