#!/bin/bash
cd /home/jiankais/lustre_jiankais/Programs/hojune/Psi0-hojune/.runs/ || exit 1

# d=finetune/glue2basket-teleop.real.flow1000.cosine.lr5.0e-05.b512.gpus8.2605081506
# out=stanford:Hojune/ckpts/glue2basket-teleop-40000
# d=finetune/locomani-18act-eef-full.real.flow1000.cosine.lr5.0e-05.b512.gpus4.2605102318
# out=stanford:Hojune/ckpts/locomani-18act-eef-full-40000
# d=finetune/locomani-18act-eef-head.real.flow1000.cosine.lr1.0e-04.b512.gpus4.2605102306
# out=stanford:Hojune/ckpts/locomani-18act-eef-head-40000
# d=finetune/mani-18act-eef-head.real.flow1000.cosine.lr1.0e-04.b512.gpus4.2605102246
# out=stanford:Hojune/ckpts/mani-18act-eef-head-40000
# d=finetune/mani-18act-eef-full.real.flow1000.cosine.lr5.0e-05.b512.gpus4.2605102246
# out=stanford:Hojune/ckpts/mani-18act-eef-full-40000
# d=finetune/turn-18act-eef-full.real.flow1000.cosine.lr5.0e-05.b512.gpus4.2605131504
# out=stanford:Hojune/ckpts/turn-18act-eef-full-40000
# d=finetune/turn-18act-eef-head.real.flow1000.cosine.lr1.0e-04.b512.gpus4.2605131504
# out=stanford:Hojune/ckpts/turn-18act-eef-head-40000
# d=finetune/turn-18state-eef-full.real.flow1000.cosine.lr5.0e-05.b512.gpus4.2605132302
# out=stanford:Hojune/ckpts/turn-18state-eef-full-40000
# d=finetune/turn-18state-eef-head.real.flow1000.cosine.lr1.0e-04.b512.gpus4.2605132301
# out=stanford:Hojune/ckpts/turn-18state-eef-head-40000
# d=finetune/teleop-locomani-uni-18state-eef-head.real.flow1000.cosine.lr1.0e-04.b256.gpus4.2605142241
# out=stanford:Hojune/ckpts/teleop-locomani-uni-18state-eef-head-20000
# d=finetune/wigs-locomani-18state-eef-head.real.flow1000.cosine.lr1.0e-04.b256.gpus4.2605151940
# out=stanford:Hojune/ckpts/finetune-wigs-locomani-18state-eef-head-20000
d=openpi-05/teleop_locomani_uni_pi05
out=stanford:Hojune/ckpts/openpi-05-teleop_locomani_uni_pi05-20000

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

# rclone copy "$d/checkpoints/ckpt_20000" "$out" "${RCLONE_FLAGS[@]}"
rclone copy "$d/teleop_locomani_uni_pi05/20000" "$out" "${RCLONE_FLAGS[@]}"
# sleep 5
# rclone copy "$d/argv.txt"         "$out" "${RCLONE_FLAGS[@]}"
# sleep 5
# rclone copy "$d/envs.txt"         "$out" "${RCLONE_FLAGS[@]}"
# sleep 5
# rclone copy "$d/run_config.json"  "$out" "${RCLONE_FLAGS[@]}"
