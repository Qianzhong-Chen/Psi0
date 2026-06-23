#!/usr/bin/env bash
# Single-GPU batch-size sweep to find max per-GPU BS before OOM.
# Runs a few training steps per BS on GPU 1 (free). Stops at first OOM.
set -uo pipefail
cd "$HOME/LEGS/submodules/Psi0"
set -a; source .env; set +a

export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME="$HOME/miniconda3/envs/legs"
export WANDB_MODE=disabled

run_bs () {
  local BS=$1
  echo "##### TESTING BATCH SIZE = $BS #####"
  CUDA_VISIBLE_DEVICES=1 CUDA_HOME="$HOME/miniconda3/envs/legs" WANDB_MODE=disabled \
  .venv-psi/bin/python scripts/train.py finetune_real_psi0_config \
    --exp=bs-sweep-$BS \
    --train.name=finetune \
    --train.data_parallel=ddp \
    --train.mixed_precision=bf16 \
    --train.train_batch_size=$BS \
    --train.num_workers=0 \
    --train.max_training_steps=3 \
    --train.warmup_steps=1 \
    --train.warmup_ratio=None \
    --train.checkpointing_steps=100000 \
    --train.validation_steps=100000 \
    --train.max_grad_norm=1.0 \
    --train.learning_rate=1e-4 \
    --train.lr_scheduler_type=cosine \
    --log.report_to=tensorboard \
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
    --model.max-delay=8 > /tmp/bs_sweep_$BS.log 2>&1
  local rc=$?
  if grep -qiE "out of memory|CUDA out of memory|OutOfMemory" /tmp/bs_sweep_$BS.log; then
    echo "BS=$BS -> OOM"
    return 1
  elif [ $rc -ne 0 ]; then
    echo "BS=$BS -> FAILED (rc=$rc, non-OOM). See /tmp/bs_sweep_$BS.log"
    return 2
  else
    local peak=$(grep -oE "Traing steps:[^]]*loss=[0-9.]+" /tmp/bs_sweep_$BS.log | tail -1)
    echo "BS=$BS -> OK  ($peak)"
    return 0
  fi
}

for BS in "$@"; do
  run_bs $BS || { echo "Stopping sweep at BS=$BS"; break; }
done
echo "##### SWEEP DONE #####"
