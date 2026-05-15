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
#SBATCH -J g1_sonic_orange_plate_turn_18state_longrun
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# 18-STATE variant of longrun_g1_sonic_orange_to_plate_turn_psi0.sbatch.
# Same dataset (hojjunekim/g1_sonic_orange_to_plate_turn_psi0). Built with the
# universal-naming format (GR00T-native columns left intact), so the dataset
# also trains GR00T fine-tune without any reformat. The 18-D state column
# (observation.state_psi0_18) lives side-by-side with the 15-D default
# (observation.state_psi0). psi/ does NOT read modality.json, so every column
# key has to be flipped via tyro.
#
# Overrides injected after the .sh script:
#   --data.transform.repack.image-keys observation.images.ego_view   # video dir
#   --data.transform.repack.state-key=observation.state_psi0_18      # 18-D parquet col
#   --data.transform.repack.action-key=action.psi0_18                # action parquet col
#   --data.transform.field.stat-state-key=observation.state_psi0_18  # 18-D stats
#   --data.transform.field.stat-action-key=action.psi0_18            # action stats
#   --model.odim=18                                                  # state-encoder dim
#
# Missing any column override leaves RealRepackTransform on its old-format
# defaults (states/action/egocentric), producing KeyError or a (1,15) vs (18,)
# broadcast error in normalize_state_func.
#
# sbatch --job-name=eef_18  longrun_g1_sonic_orange_to_plate_turn_psi0_18state.sbatch eef
# sbatch --job-name=full_18 longrun_g1_sonic_orange_to_plate_turn_psi0_18state.sbatch full
#
# Arg1 TRACK — which finetune:
#   eef  -> finetune-real-psi0-18act-eef.sh + exp default: turn-18state-eef-head
#   full -> h100/finetune-real-psi0-18act-eef-full.sh + exp default: turn-18state-eef-full
#   Override: PSI0_G1_TURN_18STATE_EXP_EEF / PSI0_G1_TURN_18STATE_EXP_FULL
#
# Arg2 SEG — chain segment index (1..MAX_SEG).
#
# Parallel chains: use distinct --job-name so singleton does not block each other.
# Resume: latest ckpt is scoped per --exp (eef vs full use different prefixes).
#
# Hardware: 4xH100. Grad-accum=2 baked in => effective batch = 64*2*4 = 512.
# Override MAX_SEG to extend the chain (default 25): `MAX_SEG=100 sbatch ...`.
TRACK="${1:-eef}"
SEG="${2:-1}"
export TRACK

############################
# User config
#
# Optional tyro extras (appended AFTER the 18-state overrides + grad-accum):
#   export PSI0_G1_TURN_18STATE_EXP_EEF='...'   # --exp for eef  (default turn-18state-eef-head)
#   export PSI0_G1_TURN_18STATE_EXP_FULL='...'  # --exp for full (default turn-18state-eef-full)
#   export PSI0_FINETUNE_EXTRA_ARGS_EEF='...'
#   export PSI0_FINETUNE_EXTRA_ARGS_FULL='...'
#   export PSI0_FINETUNE_EXTRA_ARGS='...'       # fallback for both tracks
# User-supplied extras come AFTER the baked-in overrides, so they win (tyro
# takes the last value).
############################
NGPU=4
MAX_SEG="${MAX_SEG:-25}"

mkdir -p logs

if ! [[ "${SEG}" =~ ^[0-9]+$ ]] || [[ "${SEG}" -lt 1 ]]; then
    echo "Invalid SEG='${SEG}' (need positive integer). Usage: $0 <eef|full> [SEG]" >&2
    exit 1
fi

if [[ "${SEG}" -gt "${MAX_SEG}" ]]; then
    echo "SEG=${SEG} > MAX_SEG=${MAX_SEG}; chain finished, exiting."
    exit 0
fi

echo "========== Job =========="
echo "TRACK=${TRACK}  SEG=${SEG}/${MAX_SEG}  (18-state)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}  JOB_NAME=${SLURM_JOB_NAME:-}"
echo "HOSTNAME=$(hostname)  PWD=${PWD}"
echo "========================="

############################
# Pre-schedule next segment (survives SIGTERM / pre-timeout)
############################
if [[ "${SEG}" -lt "${MAX_SEG}" ]]; then
    echo "Scheduling follow-up: TRACK=${TRACK} SEG=$((SEG + 1)) / MAX_SEG=${MAX_SEG}"
    sbatch --dependency=singleton --job-name="${SLURM_JOB_NAME:-g1_sonic_orange_plate_turn_18state_longrun}" "$0" "${TRACK}" "$((SEG + 1))"
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

        # Distinct --exp per track -> .runs/finetune/<exp>.* ; namespace 18state separately
        # so it does not collide with the 15-state turn-18act-eef-* runs.
        G1_TURN_EXP_EEF="${PSI0_G1_TURN_18STATE_EXP_EEF:-turn-18state-eef-head}"
        G1_TURN_EXP_FULL="${PSI0_G1_TURN_18STATE_EXP_FULL:-turn-18state-eef-full}"

        # 4xH100 grad-accum default to preserve effective batch 512.
        DEFAULT_GA="--train.gradient_accumulation_steps=2"

        # Universal-naming + 18-state overrides: parquet uses GR00T-native column
        # names, video dir is observation.images.ego_view/, and we read the 18-D
        # state column (observation.state_psi0_18). psi/ does NOT read
        # modality.json so every column key must be flipped. odim=18 sets the
        # state-encoder input dim. Tyro last-value rule means user EXTRA still
        # wins if they pass a conflicting flag.
        UNIFIED_OVERRIDES="--data.transform.repack.image-keys observation.images.ego_view --data.transform.repack.state-key=observation.state_psi0_18 --data.transform.repack.action-key=action.psi0_18 --data.transform.field.stat-state-key=observation.state_psi0_18 --data.transform.field.stat-action-key=action.psi0_18 --model.odim=18"

        case "${TRACK}" in
            eef)
                EXTRA="${PSI0_FINETUNE_EXTRA_ARGS_EEF:-${PSI0_FINETUNE_EXTRA_ARGS:-}}"
                bash scripts/train/psi0/finetune-real-psi0-18act-eef.sh \
                    hojjunekim/g1_sonic_orange_to_plate_turn_psi0 "${G1_TURN_EXP_EEF}" \
                    ${DEFAULT_GA} ${UNIFIED_OVERRIDES} ${EXTRA}
                ;;
            full)
                EXTRA="${PSI0_FINETUNE_EXTRA_ARGS_FULL:-${PSI0_FINETUNE_EXTRA_ARGS:-}}"
                bash scripts/train/psi0/h100/finetune-real-psi0-18act-eef-full.sh \
                    hojjunekim/g1_sonic_orange_to_plate_turn_psi0 "${G1_TURN_EXP_FULL}" \
                    ${DEFAULT_GA} ${UNIFIED_OVERRIDES} ${EXTRA}
                ;;
            *)
                echo "Unknown TRACK=${TRACK}. Use: eef | full" >&2
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
