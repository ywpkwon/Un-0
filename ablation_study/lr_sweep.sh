#!/bin/bash
# CIFAR-10 class-conditional LR sweep launcher for Un0.
# Drives the LR grid into the argparse-based un0/train_cifar10.py.
#
# Each LR runs as its own single-GPU process (no DDP). Logs go to a
# per-run dir under --output-root. W&B project + run name match
# the local dir name so checkpoints, logs, and W&B all align by name.

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [options]

LR grid (log-spaced; one LR per --gpus entry):
  --min-lr VALUE          Smallest LR. Default: 5e-4.
  --max-lr VALUE          Largest LR. Default: 5e-3.
  --gpus LIST             Comma-separated GPU ids, or "all". Default: all.
  --sweep-arg FLAG        train_cifar10.py flag the LR grid drives. Default: --lr.

Training (passes through to un0/train_cifar10.py):
  --seed VALUE            Random seed. Default: 42.
  --epochs VALUE          Training epochs. Optional (uses default 2400).
  --batch-size VALUE      Batch size. Optional (uses default 2048).

Periodic FID (logged to W&B every N epochs):
  --fid-every-epochs N    If >0, log FID every N epochs. Default: 0 (disabled).
  --fid-num-samples N     Class-balanced sample count per FID eval. Default: 50000.
  --fid-batch-size N      Inference batch size for FID sample generation. Default: 256.

Naming + output:
  --project NAME          W&B project + run-name prefix. Default: cifar10.
  --output-root PATH      Root for run dirs. Default: outputs/cifar10.

Other:
  --no-kill               Don't kill existing python processes on selected GPUs.
  --no-stream-logs        Write to log files only; don't tee to terminal.
  --dry-run               Print commands without launching.
  --wait                  Block until all launched runs finish (for batch use).
  --override "ARG VALUE"  Append extra arg(s) verbatim to train_cifar10.py (repeatable).
  -h, --help              Show this help.

Examples:
  # 8-LR sweep across all GPUs:
  $0 --gpus all --min-lr 5e-4 --max-lr 5e-3 --project cifar10_sweep

  # Single-GPU sanity at default LR with shorter epochs:
  $0 --gpus 0 --min-lr 5e-4 --max-lr 5e-4 --epochs 200 \\
     --override "--wandb-project myproj"

  # Dry-run to inspect commands without launching:
  $0 --gpus 0,1 --min-lr 0.001 --max-lr 0.003 --dry-run
EOF
}

# ---------- Defaults ----------
MIN_LR="5e-4"
MAX_LR="5e-3"
SWEEP_ARG="--lr"
GPUS_CSV="all"
SEED="42"
EPOCHS=""
BATCH_SIZE=""

FID_EVERY_EPOCHS=""
FID_NUM_SAMPLES=""
FID_BATCH_SIZE=""

PROJECT="cifar10"
OUTPUT_ROOT="outputs/cifar10"

KILL_FIRST=1
STREAM_LOGS=1
DRY_RUN=0
WAIT=0
PIDS=()
EXTRA_OVERRIDES=()

# ---------- Parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-lr)        MIN_LR="$2"; shift 2 ;;
    --max-lr)        MAX_LR="$2"; shift 2 ;;
    --sweep-arg)     SWEEP_ARG="$2"; shift 2 ;;
    --gpus)          GPUS_CSV="$2"; shift 2 ;;
    --seed)          SEED="$2"; shift 2 ;;
    --epochs)        EPOCHS="$2"; shift 2 ;;
    --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
    --fid-every-epochs)   FID_EVERY_EPOCHS="$2"; shift 2 ;;
    --fid-num-samples)    FID_NUM_SAMPLES="$2"; shift 2 ;;
    --fid-batch-size)     FID_BATCH_SIZE="$2"; shift 2 ;;
    --project)       PROJECT="$2"; shift 2 ;;
    --output-root)   OUTPUT_ROOT="$2"; shift 2 ;;
    --no-kill)       KILL_FIRST=0; shift ;;
    --no-stream-logs) STREAM_LOGS=0; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --wait)          WAIT=1; shift ;;
    --override)      EXTRA_OVERRIDES+=("$2"); shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# ---------- Resolve repo dir (parent of ablation_study/) ----------
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

# ---------- Resolve GPUs ----------
if [[ "${GPUS_CSV}" == "all" ]]; then
  GPUS=()
  while IFS= read -r line; do
    [[ -n "${line}" ]] && GPUS+=("${line}")
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
else
  IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
fi
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs selected." >&2
  exit 2
fi

# ---------- Optionally kill existing python on selected GPUs ----------
if [[ "${KILL_FIRST}" -eq 1 && "${DRY_RUN}" -eq 0 ]]; then
  echo "Killing existing python processes on GPUs ${GPUS[*]}..."
  for gpu in "${GPUS[@]}"; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr '\n' ' ')
    if [[ -n "${pids// }" ]]; then
      echo "  gpu ${gpu}: killing ${pids}"
      # shellcheck disable=SC2086
      kill -9 ${pids} 2>/dev/null || true
    fi
  done
  sleep 5
fi

# ---------- Generate log-spaced LR grid ----------
LRS=()
while IFS= read -r lr; do
  [[ -n "${lr}" ]] && LRS+=("${lr}")
done < <(python3 - "${MIN_LR}" "${MAX_LR}" "${#GPUS[@]}" <<'PY'
import math, sys
min_lr, max_lr = float(sys.argv[1]), float(sys.argv[2])
n = int(sys.argv[3])
if min_lr <= 0 or max_lr <= 0:
    raise SystemExit("LRs must be positive.")
if max_lr < min_lr:
    raise SystemExit("--max-lr must be >= --min-lr.")
if n == 1:
    print(f"{math.sqrt(min_lr * max_lr):.8g}")
else:
    log_min, log_max = math.log(min_lr), math.log(max_lr)
    for i in range(n):
        t = i / (n - 1)
        print(f"{math.exp(log_min + t * (log_max - log_min)):.8g}")
PY
)

# ---------- Header ----------
echo "Sweep config:"
echo "  GPUs:           ${GPUS[*]}"
echo "  sweep_arg:      ${SWEEP_ARG}"
echo "  LRs:            ${LRS[*]}"
echo "  seed:           ${SEED}"
echo "  project:        ${PROJECT}"
echo "  output_root:    ${OUTPUT_ROOT}"
[[ -n "${EPOCHS}" ]]       && echo "  epochs:         ${EPOCHS}"
[[ -n "${BATCH_SIZE}" ]]   && echo "  batch_size:     ${BATCH_SIZE}"
[[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]] && echo "  extra:          ${EXTRA_OVERRIDES[*]}"
echo

mkdir -p "${OUTPUT_ROOT}"

# ---------- Launch loop ----------
for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  lr="${LRS[$index]}"
  lr_tag="${lr//./p}"
  lr_tag="${lr_tag//-/_}"

  # Run name: ${PROJECT}_lr${lr}_seed${seed}. Set --project to distinguish
  # different runs.
  run_name="${PROJECT}_lr${lr_tag}_seed${SEED}"
  ckpt_dir="${OUTPUT_ROOT}/${run_name}"
  log_file="${ckpt_dir}/train.log"

  if [[ "${DRY_RUN}" -eq 0 ]]; then
    mkdir -p "${ckpt_dir}"
  fi

  cmd=(
    uv run python un0/train_cifar10.py
    --seed "${SEED}"
    "${SWEEP_ARG}" "${lr}"
    --checkpoint-dir "${ckpt_dir}"
    --wandb-project "${PROJECT}"
    --wandb-name "${run_name}"
  )
  [[ -n "${EPOCHS}" ]]       && cmd+=(--epochs "${EPOCHS}")
  [[ -n "${BATCH_SIZE}" ]]   && cmd+=(--batch-size "${BATCH_SIZE}")
  [[ -n "${FID_EVERY_EPOCHS}" ]]  && cmd+=(--fid-every-epochs "${FID_EVERY_EPOCHS}")
  [[ -n "${FID_NUM_SAMPLES}" ]]   && cmd+=(--fid-num-samples "${FID_NUM_SAMPLES}")
  [[ -n "${FID_BATCH_SIZE}" ]]    && cmd+=(--fid-batch-size "${FID_BATCH_SIZE}")

  for override in "${EXTRA_OVERRIDES[@]}"; do
    # shellcheck disable=SC2206
    cmd+=( ${override} )
  done

  echo "GPU ${gpu} lr=${lr} → ${run_name}"
  printf '  CUDA_VISIBLE_DEVICES=%s PYTHONUNBUFFERED=1' "${gpu}"
  printf ' %q' "${cmd[@]}"
  if [[ "${STREAM_LOGS}" -eq 1 ]]; then
    printf ' > >(tee %q | sed -u %q) 2>&1 &\n' \
      "${log_file}" "s/^/[gpu${gpu} lr=${lr}] /"
  else
    printf ' > %q 2>&1 &\n' "${log_file}"
  fi

  if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ "${STREAM_LOGS}" -eq 1 ]]; then
      log_prefix="[gpu${gpu} lr=${lr}] "
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${cmd[@]}" \
        > >(tee "${log_file}" | sed -u "s/^/${log_prefix}/") 2>&1 &
    else
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${cmd[@]}" \
        > "${log_file}" 2>&1 &
    fi
    PIDS+=("$!")
  fi
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo
  echo "Launched ${#GPUS[@]} jobs."
  echo "Watch logs:    tail -F ${OUTPUT_ROOT}/*/train.log"
  echo "Status check:  nvidia-smi"
  if [[ "${WAIT}" -eq 1 ]]; then
    echo "Waiting for ${#PIDS[@]} runs to finish..."
    wait
    echo "All runs finished."
  fi
fi
