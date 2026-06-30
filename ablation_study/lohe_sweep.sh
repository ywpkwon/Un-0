#!/usr/bin/env bash
# CIFAR-10 fixed-anchor Lohe sweep launcher.
#
# Reuses lr_sweep.sh for the inner LR grid, and runs outer grids over Lohe
# anchor count, oscillator sphere dimension, and class dropout.

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [options]

Outer Lohe grid:
  --lohe-dims "2 4"          Space-separated oscillator dimensions. Default: "2 4".
  --anchors "4 8 16"         Space-separated fixed-anchor counts. Default: "4 8 16".
  --dropouts "0.0 0.1"       Space-separated class dropout probs. Default: "0.0 0.1".
  --n-oscillators N          Main free oscillators. Default: 4096.
  --spatial-decoder          Treat Lohe oscillators as spatial tokens.
  --lohe-decoder-grid N      Grid side length for --spatial-decoder.
  --lohe-latent-dim N        Per-token latent dim for sample-dependent equilibrium. Default: 0.
  --lohe-latent-class-dim N  Class embedding dim for latent Lohe. Default: 64.
  --lohe-latent-pos-dim N    Position embedding dim for latent Lohe. Default: 32.
  --lohe-latent-hidden-dim N Hidden width for latent Lohe drive MLP. Default: 512.
  --lohe-latent-scale V      Latent drive delta scale. Default: 0.3.

Inner LR sweep:
  --min-lr V                 Default: 5e-4.
  --max-lr V                 Default: 5e-3.
  --gpus LIST                Comma-separated GPU ids, or "all". Default: all.
  --epochs N                 Epochs per LR run. Optional; train_cifar10.py default if unset.
  --batch-size N             Batch size. Optional; train_cifar10.py default if unset.

Periodic FID:
  --fid-every-epochs N       If >0, log FID to W&B every N epochs. Default: disabled.
  --fid-num-samples N        Class-balanced sample count per FID eval.
  --fid-batch-size N         Inference batch size for FID sample generation.

Naming + output:
  --project NAME             W&B project. Default: cifar10_lohe.
  --output-root PATH         Default: outputs/cifar10_lohe.

Other:
  --no-kill                  Don't kill existing python processes on selected GPUs.
  --dry-run                  Print commands without launching.
  -h, --help                 Show this help.
EOF
}

LOHE_DIMS="2 4"
ANCHORS="4 8 16"
DROPOUTS="0.0 0.1"
N_OSCILLATORS="4096"
MIN_LR="5e-4"
MAX_LR="5e-3"
GPUS="all"
EPOCHS=""
BATCH_SIZE=""
FID_EVERY_EPOCHS=""
FID_NUM_SAMPLES=""
FID_BATCH_SIZE=""
PROJECT="cifar10_lohe"
OUTPUT_ROOT="outputs/cifar10_lohe"
KILL_FLAG=""
DRY_RUN_FLAG=""
SPATIAL_DECODER=0
LOHE_DECODER_GRID=""
LOHE_LATENT_DIM="0"
LOHE_LATENT_CLASS_DIM="64"
LOHE_LATENT_POS_DIM="32"
LOHE_LATENT_HIDDEN_DIM="512"
LOHE_LATENT_SCALE="0.3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lohe-dims)     LOHE_DIMS="$2"; shift 2 ;;
    --anchors)       ANCHORS="$2"; shift 2 ;;
    --dropouts)      DROPOUTS="$2"; shift 2 ;;
    --n-oscillators) N_OSCILLATORS="$2"; shift 2 ;;
    --spatial-decoder) SPATIAL_DECODER=1; shift ;;
    --lohe-decoder-grid) LOHE_DECODER_GRID="$2"; shift 2 ;;
    --lohe-latent-dim) LOHE_LATENT_DIM="$2"; shift 2 ;;
    --lohe-latent-class-dim) LOHE_LATENT_CLASS_DIM="$2"; shift 2 ;;
    --lohe-latent-pos-dim) LOHE_LATENT_POS_DIM="$2"; shift 2 ;;
    --lohe-latent-hidden-dim) LOHE_LATENT_HIDDEN_DIM="$2"; shift 2 ;;
    --lohe-latent-scale) LOHE_LATENT_SCALE="$2"; shift 2 ;;
    --min-lr)        MIN_LR="$2"; shift 2 ;;
    --max-lr)        MAX_LR="$2"; shift 2 ;;
    --gpus)          GPUS="$2"; shift 2 ;;
    --epochs)        EPOCHS="$2"; shift 2 ;;
    --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
    --fid-every-epochs) FID_EVERY_EPOCHS="$2"; shift 2 ;;
    --fid-num-samples)  FID_NUM_SAMPLES="$2"; shift 2 ;;
    --fid-batch-size)   FID_BATCH_SIZE="$2"; shift 2 ;;
    --project)       PROJECT="$2"; shift 2 ;;
    --output-root)   OUTPUT_ROOT="$2"; shift 2 ;;
    --no-kill)       KILL_FLAG="--no-kill"; shift ;;
    --dry-run)       DRY_RUN_FLAG="--dry-run"; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

for dim in ${LOHE_DIMS}; do
  for anchors in ${ANCHORS}; do
    for dropout in ${DROPOUTS}; do
      tag="lohe_d${dim}_a${anchors}_drop${dropout}"
      if [[ "${SPATIAL_DECODER}" -eq 1 ]]; then
        tag="lohe_spatial_d${dim}_a${anchors}_drop${dropout}"
        [[ -n "${LOHE_DECODER_GRID}" ]] && tag="lohe_spatial_g${LOHE_DECODER_GRID}_d${dim}_a${anchors}_drop${dropout}"
      fi
      if [[ "${LOHE_LATENT_DIM}" != "0" ]]; then
        tag="${tag}_z${LOHE_LATENT_DIM}"
      fi
      tag="${tag//./p}"
      sweep_root="${OUTPUT_ROOT}/${tag}"
      override=(
        --dynamics lohe_fixed
        --lohe-dim "${dim}"
        --n-conditional-oscillators "${anchors}"
        --n-oscillators "${N_OSCILLATORS}"
        --class-dropout-prob "${dropout}"
        --num-steps 1
      )
      if [[ "${SPATIAL_DECODER}" -eq 1 ]]; then
        override+=(--lohe-spatial-decoder)
        [[ -n "${LOHE_DECODER_GRID}" ]] && override+=(--lohe-decoder-grid "${LOHE_DECODER_GRID}")
      fi
      if [[ "${LOHE_LATENT_DIM}" != "0" ]]; then
        override+=(
          --lohe-latent-dim "${LOHE_LATENT_DIM}"
          --lohe-latent-class-dim "${LOHE_LATENT_CLASS_DIM}"
          --lohe-latent-pos-dim "${LOHE_LATENT_POS_DIM}"
          --lohe-latent-hidden-dim "${LOHE_LATENT_HIDDEN_DIM}"
          --lohe-latent-scale "${LOHE_LATENT_SCALE}"
        )
      fi
      printf -v override_string '%q ' "${override[@]}"

      cmd=(
        bash ablation_study/lr_sweep.sh
        --gpus "${GPUS}"
        --wait
        ${KILL_FLAG}
        --min-lr "${MIN_LR}"
        --max-lr "${MAX_LR}"
        --project "${PROJECT}"
        --output-root "${sweep_root}"
        --override "${override_string}"
        --override "--wandb-group ${tag}"
      )
      [[ -n "${EPOCHS}" ]] && cmd+=(--epochs "${EPOCHS}")
      [[ -n "${BATCH_SIZE}" ]] && cmd+=(--batch-size "${BATCH_SIZE}")
      [[ -n "${FID_EVERY_EPOCHS}" ]] && cmd+=(--fid-every-epochs "${FID_EVERY_EPOCHS}")
      [[ -n "${FID_NUM_SAMPLES}" ]] && cmd+=(--fid-num-samples "${FID_NUM_SAMPLES}")
      [[ -n "${FID_BATCH_SIZE}" ]] && cmd+=(--fid-batch-size "${FID_BATCH_SIZE}")
      [[ -n "${DRY_RUN_FLAG}" ]] && cmd+=("${DRY_RUN_FLAG}")

      echo "[lohe_sweep] ${tag}: ${cmd[*]}"
      "${cmd[@]}"
    done
  done
done
