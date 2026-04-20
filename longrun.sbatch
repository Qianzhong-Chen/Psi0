#!/bin/bash
#SBATCH -A coreai_dlalgo_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH -J hojune_longrun
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# sbatch --job-name=psi0_eef longrun.sbatch eef
# sbatch --job-name=psi0_joint longrun.sbatch joint
# sbatch --job-name=psi0_36act longrun.sbatch 36act
# Arg1 TRACK — which finetune:
#   eef    -> psi0-18act-eef-full        + humanoid_18act_15state_eef_psi
#   joint  -> psi0-18act-joint-full      + humanoid_18act_15state_joint_psi
#   36act  -> psi0-36act-joint-aligned-full + humanoid_36act_aligned_joint_psi
#
# Arg2 SEG — chain segment index (1..MAX_SEG). Passed through by sbatch requeue; no state files.
#
# First job (example):  sbatch --job-name=psi0_eef   /path/to/longrun.sbatch eef
#                        sbatch --job-name=psi0_eef   /path/to/longrun.sbatch eef 1
# Three parallel chains: use distinct --job-name so singleton does not block each other.
#
# Resume: finetune scripts use --train.resume_from_checkpoint=latest; LaunchConfig picks
# the latest run under .runs/finetune/ for this --exp (see src/psi/config/config.py).
TRACK="${1:-eef}"
SEG="${2:-1}"
export TRACK

############################
# User config
#
# Optional tyro extras (after task + exp in each *.sh):
#   export PSI0_FINETUNE_EXTRA_ARGS_EEF='...'
#   export PSI0_FINETUNE_EXTRA_ARGS_JOINT='...'
#   export PSI0_FINETUNE_EXTRA_ARGS_36ACT='...'
############################
NGPU=8
MAX_SEG="${MAX_SEG:-25}"

mkdir -p logs

if ! [[ "${SEG}" =~ ^[0-9]+$ ]] || [[ "${SEG}" -lt 1 ]]; then
    echo "Invalid SEG='${SEG}' (need positive integer). Usage: $0 <eef|joint|36act> [SEG]" >&2
    exit 1
fi

if [[ "${SEG}" -gt "${MAX_SEG}" ]]; then
    echo "SEG=${SEG} > MAX_SEG=${MAX_SEG}; chain finished, exiting."
    exit 0
fi

echo "========== Job =========="
echo "TRACK=${TRACK}  SEG=${SEG}/${MAX_SEG}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}  JOB_NAME=${SLURM_JOB_NAME:-}"
echo "HOSTNAME=$(hostname)  PWD=${PWD}"
echo "========================="

############################
# Pre-schedule next segment (same pattern as before: survives SIGTERM / pre-timeout)
############################
if [[ "${SEG}" -lt "${MAX_SEG}" ]]; then
    echo "Scheduling follow-up: TRACK=${TRACK} SEG=$((SEG + 1)) / MAX_SEG=${MAX_SEG}"
    sbatch --dependency=singleton --job-name="${SLURM_JOB_NAME:-hojune_longrun}" "$0" "${TRACK}" "$((SEG + 1))"
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
        # conda activate.d hooks (e.g. cuda-nvcc) reference unset vars; nounset breaks them.
        set +u
        conda activate psi0
        cd /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiankais/Programs/hojune/Psi0-hojune
        source .venv-psi/bin/activate
        set -u
        export WANDB_API_KEY=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
        # Single-quoted bash -lc: do NOT use \${CONDA_PREFIX} — that leaves a literal ${CONDA_PREFIX} in CUDA_HOME.
        export CUDA_HOME="${CONDA_PREFIX}"
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CONDA_PREFIX}/include:${CPATH:-}"
        export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"

        case "${TRACK}" in
            eef)
                EXTRA="${PSI0_FINETUNE_EXTRA_ARGS_EEF:-${PSI0_FINETUNE_EXTRA_ARGS:-}}"
                bash scripts/train/psi0/h100/finetune-real-psi0-18act-eef-full.sh \
                    hojjunekim/humanoid_18act_15state_eef_psi psi0-18act-eef-full ${EXTRA}
                ;;
            joint)
                EXTRA="${PSI0_FINETUNE_EXTRA_ARGS_JOINT:-${PSI0_FINETUNE_EXTRA_ARGS:-}}"
                bash scripts/train/psi0/h100/finetune-real-psi0-18act-joint-full.sh \
                    hojjunekim/humanoid_18act_15state_joint_psi psi0-18act-joint-full ${EXTRA}
                ;;
            36act)
                EXTRA="${PSI0_FINETUNE_EXTRA_ARGS_36ACT:-${PSI0_FINETUNE_EXTRA_ARGS:-}}"
                bash scripts/train/psi0/h100/finetune-real-psi0-36act-joint-aligned-full.sh \
                    hojjunekim/humanoid_36act_aligned_joint_psi psi0-36act-joint-aligned-full ${EXTRA}
                ;;
            *)
                echo "Unknown TRACK=${TRACK}. Use: eef | joint | 36act" >&2
                exit 1
                ;;
        esac
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
