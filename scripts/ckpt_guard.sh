# Source me from a longrun_*.sbatch right after the SIGTERM trap is installed
# and before srun. Assumes PWD = project root (where .runs/finetune lives).
#
# Why this exists:
#
# A segment killed mid-save (timelimit / preempt / OOM) can leave an empty or
# truncated model.safetensors. The trainer auto-resumes the newest ckpt and
# then dies with:
#
#   safetensors_rust.SafetensorError: Error while deserializing header:
#   header too small
#
# That repeats every requeue, jobs all exit < 2min, and after 5-6 consecutive
# short jobs the defunct-chain monitor sets JobHeldUser on the next segment.
# This guard quarantines the broken newest ckpt before srun so auto-resume
# falls back to the previous good one.
#
# Behaviour: for each .runs/finetune/<run>/checkpoints/, repeatedly rename the
# newest ckpt_<digits> to ckpt_<digits>.broken.<jobid> while it looks
# corrupted, until the newest one is healthy (or none left). Quarantined dirs
# are kept on disk for postmortem - the regex below skips them on next pass.

validate_checkpoints() {
    local runs_dir="${PWD}/.runs/finetune"
    if [[ ! -d "${runs_dir}" ]]; then
        echo "[ckpt-guard] No ${runs_dir}; skipping."
        return 0
    fi

    local quarantine_suffix=".broken.${SLURM_JOB_ID:-manual}"
    local total_broken=0

    while IFS= read -r -d '' ckpt_root; do
        while :; do
            local newest_ckpt
            newest_ckpt="$(ls -1dt "${ckpt_root}"/ckpt_* 2>/dev/null \
                            | awk '/\/ckpt_[0-9]+$/' \
                            | head -n1 || true)"
            if [[ -z "${newest_ckpt}" || ! -d "${newest_ckpt}" ]]; then
                break
            fi

            local broken=0 reason=""
            local mf="${newest_ckpt}/model.safetensors"
            if [[ -f "${mf}" ]]; then
                local size
                size=$(stat -c%s "${mf}" 2>/dev/null || echo 0)
                # A real model.safetensors is many MB; <1KB means truncated.
                if [[ "${size}" -lt 1024 ]]; then
                    broken=1
                    reason="model.safetensors size=${size}B (<1024)"
                fi
            else
                if ! compgen -G "${newest_ckpt}/model-*.safetensors" > /dev/null; then
                    broken=1
                    reason="no model.safetensors / model-*.safetensors"
                fi
            fi

            if [[ "${broken}" -eq 1 ]]; then
                local dst="${newest_ckpt}${quarantine_suffix}"
                echo "[ckpt-guard] Quarantining ${newest_ckpt} -> $(basename "${dst}") (${reason})"
                mv "${newest_ckpt}" "${dst}"
                total_broken=$((total_broken + 1))
                continue
            fi

            echo "[ckpt-guard] OK: ${newest_ckpt}"
            break
        done
    done < <(find "${runs_dir}" -mindepth 2 -maxdepth 2 -type d -name checkpoints -print0)

    if [[ "${total_broken}" -gt 0 ]]; then
        echo "[ckpt-guard] Quarantined ${total_broken} broken checkpoint(s); auto-resume will fall back."
    else
        echo "[ckpt-guard] All newest checkpoints look healthy."
    fi
}

echo "[ckpt-guard] Validating checkpoints under .runs/finetune ..."
validate_checkpoints
