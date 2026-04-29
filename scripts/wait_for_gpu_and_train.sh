#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/data/zyx/SLP34K_v2}"
OCR_DIR="${OCR_DIR:-${PROJECT_ROOT}/ocr_training}"
CONDA_SH="${CONDA_SH:-/mnt/data/zyx/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-slpr_ocr}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_COUNT="${GPU_COUNT:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/train/logs/v2_m02}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/wait_for_gpu_and_train.sh [options] [-- hydra_overrides...]

Options:
  --threshold-gb N      Minimum free memory per selected GPU. Default: 40
  --poll-seconds N     Poll interval in seconds. Default: 60
  --gpu-count N        Number of GPUs to reserve. Default: 1
  --conda-env NAME     Conda environment. Default: slpr_ocr
  --log-dir PATH       Training log directory.
  --dry-run            Print the training command when GPUs are available, then exit.
  -h, --help           Show this help.

Default training command:
  cd ocr_training
  CUDA_VISIBLE_DEVICES=<selected> python train.py model=slp_mdiff trainer.gpus=<gpu-count> +trainer.accumulate_grad_batches=1

Any arguments after "--" are appended as Hydra overrides.
EOF
}

HYDRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threshold-gb)
      MIN_FREE_GB="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --gpu-count)
      GPU_COUNT="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      HYDRA_OVERRIDES+=("$@")
      break
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

if ! [[ "${MIN_FREE_GB}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --threshold-gb must be an integer." >&2
  exit 2
fi
if ! [[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --poll-seconds must be an integer." >&2
  exit 2
fi
if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "ERROR: --gpu-count must be a positive integer." >&2
  exit 2
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "ERROR: conda profile script not found: ${CONDA_SH}" >&2
  exit 2
fi
if [[ ! -d "${OCR_DIR}" ]]; then
  echo "ERROR: OCR directory not found: ${OCR_DIR}" >&2
  exit 2
fi

MIN_FREE_MIB=$((MIN_FREE_GB * 1024))

find_available_gpus() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F ', ' -v min_free="${MIN_FREE_MIB}" -v need="${GPU_COUNT}" '
        $2 >= min_free {
          selected = selected (selected == "" ? "" : ",") $1
          count += 1
        }
        count == need {
          print selected
          exit 0
        }
      '
}

echo "Waiting for ${GPU_COUNT} GPU(s) with at least ${MIN_FREE_GB}GB free each."
echo "Polling every ${POLL_SECONDS}s. Training logs will go to ${LOG_DIR}."

SELECTED_GPUS=""
while [[ -z "${SELECTED_GPUS}" ]]; do
  SELECTED_GPUS="$(find_available_gpus || true)"
  if [[ -n "${SELECTED_GPUS}" ]]; then
    break
  fi
  date '+%Y-%m-%d %H:%M:%S'
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu \
    --format=csv,noheader,nounits || true
  sleep "${POLL_SECONDS}"
done

mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/slp_mdiff_train_${RUN_TS}.log"

TRAIN_CMD=(
  python train.py
  model=slp_mdiff
  "trainer.gpus=${GPU_COUNT}"
  +trainer.accumulate_grad_batches=1
)
TRAIN_CMD+=("${HYDRA_OVERRIDES[@]}")

echo "Selected GPU(s): ${SELECTED_GPUS}"
echo "Log file: ${LOG_FILE}"
printf 'Training command: CUDA_VISIBLE_DEVICES=%s ' "${SELECTED_GPUS}"
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${OCR_DIR}"
CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}" "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
