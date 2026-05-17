#!/bin/bash
cd /home/jiankais/lustre_jiankais/Programs/hojune/Psi0-hojune/checkpoints/ || exit 1

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

# Outer retry loop: --retries only covers transfers, not failures during
# filesystem init (e.g. "couldn't find root directory ID" when Drive returns
# 403 rateLimitExceeded on the very first call). Keep re-invoking rclone
# until it exits successfully.
RETRY_SLEEP="${RETRY_SLEEP:-60}"
RETRY_MAX="${RETRY_MAX:-0}"  # 0 = infinite

rclone_retry() {
  local attempt=1
  while true; do
    echo "[rclone_retry] attempt #$attempt: rclone $*"
    if rclone "$@"; then
      echo "[rclone_retry] success on attempt #$attempt"
      return 0
    fi
    local rc=$?
    if [ "$RETRY_MAX" -gt 0 ] && [ "$attempt" -ge "$RETRY_MAX" ]; then
      echo "[rclone_retry] giving up after $attempt attempts (last exit=$rc)"
      return "$rc"
    fi
    echo "[rclone_retry] attempt #$attempt failed (exit=$rc); sleeping ${RETRY_SLEEP}s..."
    sleep "$RETRY_SLEEP"
    attempt=$((attempt + 1))
  done
}

upload_ckpt() {
  local d=$1
  local out=$2
  echo "Uploading $d -> $out"

  # Sharded safetensors weights + index
  rclone_retry copy "$d/model-00001-of-00002.safetensors" "$out" "${RCLONE_FLAGS[@]}"
  sleep 5
  rclone_retry copy "$d/model-00002-of-00002.safetensors" "$out" "${RCLONE_FLAGS[@]}"
  sleep 5
  rclone_retry copy "$d/model.safetensors.index.json"     "$out" "${RCLONE_FLAGS[@]}"
  sleep 5

  # Top-level model config
  rclone_retry copy "$d/config.json"                      "$out" "${RCLONE_FLAGS[@]}"
  sleep 5

  # GR00T experiment / runtime config (modality, dataset stats, etc.)
  rclone_retry copy "$d/experiment_cfg"                   "$out/experiment_cfg" "${RCLONE_FLAGS[@]}"
  sleep 5

  # Processor (tokenizer / image processor / embodiment id / statistics)
  rclone_retry copy "$d/processor"                        "$out/processor"      "${RCLONE_FLAGS[@]}"
  sleep 5
}

# upload_ckpt gr00t_n16_teleop_locomani_uni       stanford:Hojune/ckpts/gr00t-n16-teleop-locomani-uni
# upload_ckpt gr00t_n16_wigs_locomani_align_uni   stanford:Hojune/ckpts/gr00t-n16-wigs-locomani-align-uni
upload_ckpt gr00t_n16_wigs_locomani_adjust      stanford:Hojune/ckpts/gr00t-n16-wigs-locomani-adjust
