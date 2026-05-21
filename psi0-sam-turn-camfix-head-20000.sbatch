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
#SBATCH -J psi0-sam-turn-camfix-head-20000
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# Psi0 18-state head-only fine-tune on the SAM mesh-only baseline universal
# dataset (hojjunekim/sam_turn_camfix_uni). Built by sam2universal.sh on the
# same turn_camfix episode JSONs as wigs, but rendered through
# offline_renderer_baseline.py (no gsplat, ivory ground, raw SAM3D vertex
# colors). This is the no-photorealism A/B against the wigs gsplat dataset.
#
# Head-only EEF track only (matches the wigs psi0 policy). Capped at 20k.
#
# sbatch psi0-sam-turn-camfix-head-20000.sbatch
#
# Arg1 SEG — chain segment index (1..MAX_SEG). Passed through by sbatch requeue.
SEG="${1:-1}"

############################
# User config
#
# Optional tyro extras (appended AFTER baked-in overrides + grad-accum + 20k cap):
#   export SAM_TURN_CAMFIX_EXP='...'                   # --exp (default psi0-sam-turn-camfix-head-20000)
#   export PSI0_FINETUNE_EXTRA_ARGS='...'              # extra tyro args (wins via last-value rule)
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
echo "TRACK=eef (head-only)  SEG=${SEG}/${MAX_SEG}  (18-state, 20k steps)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}  JOB_NAME=${SLURM_JOB_NAME:-}"
echo "HOSTNAME=$(hostname)  PWD=${PWD}"
echo "========================="

############################
# Pre-schedule next segment (survives SIGTERM / pre-timeout)
############################
if [[ "${SEG}" -lt "${MAX_SEG}" ]]; then
    echo "Scheduling follow-up: SEG=$((SEG + 1)) / MAX_SEG=${MAX_SEG}"
    sbatch --dependency=singleton --job-name="${SLURM_JOB_NAME:-psi0-sam-turn-camfix-head-20000}" "$0" "$((SEG + 1))"
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
        source .venv-psi/bin/activate
        set -u
        export WANDB_API_KEY=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
        export CUDA_HOME="${CONDA_PREFIX}"
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CONDA_PREFIX}/include:${CPATH:-}"
        export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"

        EXP="${SAM_TURN_CAMFIX_EXP:-psi0-sam-turn-camfix-head-20000}"

        DEFAULT_GA="--train.gradient_accumulation_steps=1"
        STEPS_CAP="--train.max_training_steps=20000"
        UNIFIED_OVERRIDES="--data.transform.repack.image-keys observation.images.ego_view --data.transform.repack.state-key=observation.state_psi0_18 --data.transform.repack.action-key=action.psi0_18 --data.transform.field.stat-state-key=observation.state_psi0_18 --data.transform.field.stat-action-key=action.psi0_18 --model.odim=18 --train.num_workers=8"

        EXTRA="${PSI0_FINETUNE_EXTRA_ARGS:-}"

        bash scripts/train/psi0/finetune-real-psi0-18act-eef.sh \
            hojjunekim/sam_turn_camfix_uni "${EXP}" \
            ${DEFAULT_GA} ${STEPS_CAP} ${UNIFIED_OVERRIDES} ${EXTRA}
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
