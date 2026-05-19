#!/bin/bash
#SBATCH -A coreai_dlalgo_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@600
#SBATCH -J pi05-sam-locomani-adjust-full-20000
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# OpenPI pi05 FULL fine-tune on the SAM mesh-only baseline universal dataset.
# TrainConfig: sam_locomani_adjust_pi05 (registered in src/openpi/training/config.py).
#
# Counterpart to pi05-wigs-locomani-adjust-full-20000.sbatch — same episode
# JSONs, but rendered through offline_renderer_baseline.py (no gsplat, ivory
# ground, raw SAM3D vertex colors). This is the no-photorealism A/B against
# the wigs gsplat dataset.
#
# Dataset must be materialized at
# $HF_LEROBOT_HOME/hojjunekim/sam_locomani_adjust_uni/
#
# sbatch pi05-sam-locomani-adjust-full-20000.sbatch
#
# Arg1 SEG — chain segment index (1..MAX_SEG). Passed through by sbatch requeue.
SEG="${1:-1}"

############################
# User config
#
# Optional pi05 train.py extras (appended after the config-name):
#   export PI05_EXTRA_ARGS='--lr_schedule.peak_lr=5e-5 ...'
############################
NGPU=4
MAX_SEG="${MAX_SEG:-25}"

mkdir -p logs

if ! [[ "${SEG}" =~ ^[0-9]+$ ]] || [[ "${SEG}" -lt 1 ]]; then
    echo "Invalid SEG='${SEG}' (need positive integer). Usage: $0 [SEG]" >&2
    exit 1
fi

if [[ "${SEG}" -gt "${MAX_SEG}" ]]; then
    echo "SEG=${SEG} > MAX_SEG=${MAX_SEG}; chain finished, exiting."
    exit 0
fi

echo "========== Job =========="
echo "TRACK=full (pi05 full fine-tune)  SEG=${SEG}/${MAX_SEG}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}  JOB_NAME=${SLURM_JOB_NAME:-}"
echo "HOSTNAME=$(hostname)  PWD=${PWD}"
echo "========================="

############################
# Pre-schedule next segment (survives SIGTERM / pre-timeout)
############################
if [[ "${SEG}" -lt "${MAX_SEG}" ]]; then
    echo "Scheduling follow-up: SEG=$((SEG + 1)) / MAX_SEG=${MAX_SEG}"
    sbatch --dependency=singleton --job-name="${SLURM_JOB_NAME:-pi05-sam-locomani-adjust-full-20000}" "$0" "$((SEG + 1))"
fi

############################
# Graceful exit handler
############################
SRUN_PID=""

graceful_exit() {
    echo "[Graceful Exit] Caught termination signal."

    if [[ -n "${SRUN_PID}" ]] && kill -0 "${SRUN_PID}" 2>/dev/null; then
        echo "Forwarding TERM to srun process ${SRUN_PID} ..."
        kill -TERM "${SRUN_PID}" 2>/dev/null || true
        wait "${SRUN_PID}" || true
    fi

    exit 0
}

trap graceful_exit SIGTERM SIGINT

# shellcheck source=scripts/ckpt_guard.sh
source ./scripts/ckpt_guard.sh

############################
# Main workload
############################
echo "Starting workload..."

set +e
srun \
    --ntasks=1 \
    --time="${SLURM_TIMELIMIT:-04:00:00}" \
    --gpus="${NGPU}" \
    bash -lc '
        set -euo pipefail

        source /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiankais/cache_env.sh
        source /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiankais/Apps/miniforge3/etc/profile.d/conda.sh
        set +u
        conda activate psi0
        cd /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiankais/Programs/hojune/Psi0-hojune
        # pi05 uses a different venv than psi. baselines/pi05/train_pi05.sh
        # activates .venv-openpi internally, so we do not source it here.
        set -u
        export WANDB_API_KEY=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
        export CUDA_HOME="${CONDA_PREFIX}"
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CONDA_PREFIX}/include:${CPATH:-}"
        export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"

        export SAVE_INTERVAL=2500
        PI05_OVERRIDES="--num_train_steps=20000 --lr_schedule.decay_steps=20000 --num_workers=8 --resume --exp_name=pi05-sam-locomani-adjust-full-20000"
        EXTRA="${PI05_EXTRA_ARGS:-}"

        bash baselines/pi05/train_pi05.sh sam_locomani_adjust_pi05 ${PI05_OVERRIDES} ${EXTRA}
    ' &
SRUN_PID=$!

wait "${SRUN_PID}"
EXIT_CODE=$?
set -e

if [[ "${EXIT_CODE}" -eq 0 ]]; then
    echo "Workload exited successfully."
    exit 0
else
    echo "Workload failed with EXIT_CODE=${EXIT_CODE}"
    exit "${EXIT_CODE}"
fi
