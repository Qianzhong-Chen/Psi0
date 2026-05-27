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
#SBATCH -J grootn16-sam-turn-pure-eef-20000
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# GR00T-N1.6 G1_EEF_UNI fine-tune on the universal-naming SAM mesh-only
# baseline "pure" dataset (hojjunekim/sam_turn_pure_uni). 200-ep single-source
# turn dataset (no 4-quadrant aug), rendered via offline_renderer_baseline.py
# (no gsplat). No-photorealism A/B counterpart to wigs_turn_pure_uni. Same
# 18-state / 18-action psi0 EEF schema, the meta/modality_g1_eef_uni.json
# sidecar split:
#   L_eef_xyz(3) + L_eef_rpy(3) + L_grip(1)
# + R_eef_xyz(3) + R_eef_rpy(3) + R_grip(1)
# + base_vel(3) + height(1)
#
# This sbatch:
#   1. ensures the base model nvidia/GR00T-N1.6-3B is downloaded locally,
#   2. ensures the dataset is downloaded under $HF_LEROBOT_HOME,
#   3. swaps meta/modality.json -> modality_g1_eef_uni.json (gr00t loader
#      hardcodes that filename), and
#   4. launches via baselines/gr00t-n1.6/finetune_gr00t.py with the
#      finetune_sam_turn_pure_uni_eef_gr00t_n16 preset
#      (embodiment_tag G1_EEF_UNI, modality_config_path
#      src/gr00t/gr00t/configs/modality/g1_eef_uni.py).
#
# sbatch grootn16-sam-turn-pure-eef-20000.sbatch
#
# Arg1 SEG — chain segment index (1..MAX_SEG). Passed through by sbatch requeue.
SEG="${1:-1}"

NGPU=4
MAX_SEG="${MAX_SEG:-10}"

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
echo "GR00T-N1.6 G1_EEF_UNI (18/18) finetune  SEG=${SEG}/${MAX_SEG}  (20k steps, sam pure)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}  JOB_NAME=${SLURM_JOB_NAME:-}"
echo "HOSTNAME=$(hostname)  PWD=${PWD}"
echo "========================="

# Pre-schedule next segment (survives SIGTERM / pre-timeout)
if [[ "${SEG}" -lt "${MAX_SEG}" ]]; then
    echo "Scheduling follow-up: SEG=$((SEG + 1)) / MAX_SEG=${MAX_SEG}"
    sbatch --dependency=singleton --job-name="${SLURM_JOB_NAME:-grootn16-sam-turn-pure-eef-20000}" "$0" "$((SEG + 1))"
fi

# Graceful exit handler
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

        # Base model: nvidia/GR00T-N1.6-3B downloaded once to a local dir.
        BASE_MODEL_DIR="./checkpoints/base/GR00T-N1.6-3B"
        if [[ ! -f "${BASE_MODEL_DIR}/config.json" ]]; then
            echo "[setup] downloading nvidia/GR00T-N1.6-3B -> ${BASE_MODEL_DIR}"
            hf download nvidia/GR00T-N1.6-3B --local-dir "${BASE_MODEL_DIR}"
        fi

        # Dataset: hojjunekim/sam_turn_pure_uni. Assumes HF_LEROBOT_HOME
        # set by the 4-block handoff (defaults to $HOME/.cache/huggingface/lerobot).
        DATASET_REPO="hojjunekim/sam_turn_pure_uni"
        DATASET_PATH="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/${DATASET_REPO}"
        if [[ ! -f "${DATASET_PATH}/meta/info.json" ]]; then
            echo "[setup] downloading ${DATASET_REPO} -> ${DATASET_PATH}"
            hf download "${DATASET_REPO}" --repo-type dataset --local-dir "${DATASET_PATH}"
        fi

        # Swap modality.json -> modality_g1_eef_uni.json (idempotent).
        if [[ -f "${DATASET_PATH}/meta/modality_g1_eef_uni.json" ]]; then
            if ! cmp -s "${DATASET_PATH}/meta/modality.json" "${DATASET_PATH}/meta/modality_g1_eef_uni.json"; then
                echo "[setup] swapping meta/modality.json -> modality_g1_eef_uni.json"
                if [[ ! -f "${DATASET_PATH}/meta/modality_sonic.json" ]]; then
                    cp "${DATASET_PATH}/meta/modality.json" "${DATASET_PATH}/meta/modality_sonic.json"
                fi
                cp "${DATASET_PATH}/meta/modality_g1_eef_uni.json" "${DATASET_PATH}/meta/modality.json"
            fi
        else
            echo "[setup] ERROR: no modality_g1_eef_uni.json sidecar at ${DATASET_PATH}/meta/" >&2
            exit 1
        fi

        export DATASET_PATH

        EXTRA="${GR00T_FINETUNE_EXTRA_ARGS:-}"

        python3 baselines/gr00t-n1.6/finetune_gr00t.py \
            --preset finetune_sam_turn_pure_uni_eef_gr00t_n16 \
            --dataset-path "${DATASET_PATH}" \
            ${EXTRA}
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
