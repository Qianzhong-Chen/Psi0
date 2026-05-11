#!/bin/bash
cd /home/jiankais/lustre_jiankais/Programs/hojune/Psi0-hojune/.runs/finetune/ || exit 1

d=glue2basket-teleop.real.flow1000.cosine.lr5.0e-05.b512.gpus8.2605081506
out=stanford:Hojune/ckpts/glue2basket-teleop-40000

# Throttle to avoid Drive API 403 rateLimitExceeded.
# rclone's default shared client_id is rate-limited globally; keep tpslimit low.
# --retries / --low-level-retries / --drive-pacer-* handle transient 403s with backoff.
RCLONE_FLAGS=(
  -P
  --tpslimit 4
  --tpslimit-burst 1
  --transfers 2
  --checkers 2
  --drive-pacer-min-sleep 200ms
  --drive-pacer-burst 1
  --retries 20
  --retries-sleep 30s
  --low-level-retries 30
)

echo "Uploading $d -> $out"

rclone copy "$d/checkpoints/ckpt_40000" "$out" "${RCLONE_FLAGS[@]}"
sleep 5
rclone copy "$d/argv.txt"         "$out" "${RCLONE_FLAGS[@]}"
sleep 5
rclone copy "$d/envs.txt"         "$out" "${RCLONE_FLAGS[@]}"
sleep 5
rclone copy "$d/run_config.json"  "$out" "${RCLONE_FLAGS[@]}"
