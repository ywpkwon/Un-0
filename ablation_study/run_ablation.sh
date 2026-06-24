#!/usr/bin/env bash
# Single-node dynamics ablation pipeline for Un0.
#
# One recipe: AdamW + dense coupling, sweeping --lr across the 8 dynamics
# experiments. Phase 1: for each ablation, run a short LR sweep (one LR per GPU
# via ablation_study/lr_sweep.sh) and rank the LRs by FID. Phase 2: run the
# best LR per ablation at full length as a single parallel wave (one ablation
# per GPU). Everything runs on one 8-GPU H200 host.
#
# This is a multi-hour run; launch it under tmux or nohup so an SSH
# disconnect doesn't kill it, e.g.:
#   nohup ablation_study/run_ablation.sh > ablation.log 2>&1 &
#
# Usage:
#   ablation_study/run_ablation.sh [options]
#
# Options:
#   --wandb-project NAME       Default: un0-ablations.
#   --epochs-short N           Sweep epochs per ablation. Default: 400.
#   --epochs-long N            Best-LR run epochs. Default: 1200.
#   --eval-num-samples N       FID samples for sweep ranking. Default: 50000.
#   --fid-final-num-samples N  FID samples for the long runs. Default: 50000.
#   --min-lr V / --max-lr V    Override the LR sweep range.
#   --output-root PATH         Run-dir root. Default: outputs.
#   --no-sync                  Skip `uv sync` (devbox already synced).
#   --dry-run                  Print the planned commands; don't launch.
#   -h, --help                 Show this help.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
  sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

WANDB_PROJECT="${WANDB_PROJECT:-un0-ablations}"
EPOCHS_SHORT="${EPOCHS_SHORT:-400}"
EPOCHS_LONG="${EPOCHS_LONG:-1200}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-50000}"
FID_FINAL_NUM_SAMPLES="${FID_FINAL_NUM_SAMPLES:-50000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
MIN_LR_OVERRIDE="${MIN_LR:-}"
MAX_LR_OVERRIDE="${MAX_LR:-}"
DO_SYNC=1
DRY_RUN="${DRY_RUN:-0}"

# Internal knobs, env-var only (not part of the reader-facing flag interface).
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
FID_EVERY_EPOCHS="${FID_EVERY_EPOCHS:-200}"
SKIP_TOPOLOGY_CHECK="${SKIP_TOPOLOGY_CHECK:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wandb-project)         WANDB_PROJECT="$2"; shift 2 ;;
    --epochs-short)          EPOCHS_SHORT="$2"; shift 2 ;;
    --epochs-long)           EPOCHS_LONG="$2"; shift 2 ;;
    --eval-num-samples)      EVAL_NUM_SAMPLES="$2"; shift 2 ;;
    --fid-final-num-samples) FID_FINAL_NUM_SAMPLES="$2"; shift 2 ;;
    --min-lr)                MIN_LR_OVERRIDE="$2"; shift 2 ;;
    --max-lr)                MAX_LR_OVERRIDE="$2"; shift 2 ;;
    --output-root)           OUTPUT_ROOT="$2"; shift 2 ;;
    --no-sync)               DO_SYNC=0; shift ;;
    --dry-run)               DRY_RUN=1; shift ;;
    -h|--help)               usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

NGPU=8
GROUP="dynamics"
GROUP_ROOT="${OUTPUT_ROOT}/${GROUP}"

# Bottom-to-top figure order; the dynamics ablations. More experiments than
# GPUs is fine — Phase 1 and Phase 2 both run in waves of NGPU.
ABLATIONS=(
  decoder_only_raw
  decoder_only
  reservoir_euler1
  reservoir_euler10
  trained_euler1
  trained_euler2
  trained_euler5
  trained_euler10
)

declare -A BEST_LR

log() { printf '[run_ablation] %s\n' "$*"; }

# Fail fast unless this host exposes NGPU full (non-MIG) H200s on one node.
check_topology() {
  if [[ "${SKIP_TOPOLOGY_CHECK}" == "1" || "${DRY_RUN}" == "1" ]]; then
    log "Skipping topology guard (SKIP_TOPOLOGY_CHECK/DRY_RUN set)."
    return 0
  fi
  local list n_gpu n_mig n_h200
  list="$(nvidia-smi -L)"
  n_gpu="$(grep -c '^GPU ' <<<"${list}" || true)"
  n_mig="$(grep -c 'MIG ' <<<"${list}" || true)"
  n_h200="$(grep -c 'H200' <<<"${list}" || true)"
  if [[ "${n_gpu}" -ne "${NGPU}" || "${n_mig}" -ne 0 || "${n_h200}" -lt "${NGPU}" ]]; then
    echo "Topology guard failed: need ${NGPU} full non-MIG H200s on one host." >&2
    echo "  found: ${n_gpu} GPUs, ${n_mig} MIG devices, ${n_h200} H200s" >&2
    echo "nvidia-smi -L reported:" >&2
    echo "${list}" >&2
    echo "(set SKIP_TOPOLOGY_CHECK=1 to bypass for testing.)" >&2
    exit 3
  fi
  log "Topology OK: ${n_gpu} H200s, no MIG devices."
}

# Map an ablation name to integrator flags.
set_ablation_flags() {
  case "$1" in
    decoder_only_raw)  STEP_FLAGS="--num-steps 0 --encoding raw" ;;
    decoder_only)      STEP_FLAGS="--num-steps 0" ;;
    reservoir_euler1)  STEP_FLAGS="--num-steps 1 --solver euler --freeze-dynamics" ;;
    reservoir_euler10) STEP_FLAGS="--num-steps 10 --solver euler --freeze-dynamics" ;;
    trained_euler1)    STEP_FLAGS="--num-steps 1 --solver euler" ;;
    trained_euler2)    STEP_FLAGS="--num-steps 2 --solver euler" ;;
    trained_euler5)    STEP_FLAGS="--num-steps 5 --solver euler" ;;
    trained_euler10)   STEP_FLAGS="--num-steps 10 --solver euler" ;;
    *) echo "unknown ABLATION=$1" >&2; exit 1 ;;
  esac
}

# Phase 1, per ablation: 8-LR sweep, one LR per GPU, block until all finish.
run_sweep_for_ablation() {
  local abl="$1" sweep_root="$2"
  set_ablation_flags "${abl}"
  local SWEEP_ARG="--lr"
  local MIN_LR="${MIN_LR_OVERRIDE:-5e-4}"
  local MAX_LR="${MAX_LR_OVERRIDE:-5e-3}"
  log "Sweep ${abl}: ${SWEEP_ARG} in [${MIN_LR}, ${MAX_LR}] @ ${EPOCHS_SHORT} epochs"

  local -a cmd=(
    bash "$(dirname "${BASH_SOURCE[0]}")/lr_sweep.sh"
    --gpus all --wait --no-kill
    --sweep-arg "${SWEEP_ARG}"
    --min-lr "${MIN_LR}" --max-lr "${MAX_LR}"
    --epochs "${EPOCHS_SHORT}"
    --seed "${SEED}"
    --project "${WANDB_PROJECT}"
    --output-root "${sweep_root}"
    --override "${STEP_FLAGS}"
    --override "--wandb-group ${GROUP}_${abl}_sweep"
  )
  [[ -n "${BATCH_SIZE}" ]] && cmd+=(--batch-size "${BATCH_SIZE}")
  [[ "${DRY_RUN}" == "1" ]] && cmd+=(--dry-run)

  log "+ ${cmd[*]}"
  "${cmd[@]}"
}

# Compute FID for every completed sweep run, one per GPU.
run_eval_wave() {
  local sweep_root="$1"
  local -a dirs=() pids=()
  local d gpu=0
  for d in "${sweep_root}"/*/; do
    [[ -f "${d}final.pt" ]] && dirs+=("${d}")
  done
  if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "No final.pt under ${sweep_root}; sweep produced no checkpoints." >&2
    return 1
  fi
  log "Evaluating ${#dirs[@]} checkpoints (num_samples=${EVAL_NUM_SAMPLES})"
  for d in "${dirs[@]}"; do
    log "  GPU ${gpu}: eval ${d}"
    CUDA_VISIBLE_DEVICES="${gpu}" uv run python un0/eval.py \
      --checkpoint "${d}final.pt" \
      --num-samples "${EVAL_NUM_SAMPLES}" \
      --batch-size "${EVAL_BATCH_SIZE}" \
      --seed "${SEED}" \
      --output "${d}fid.json" > "${d}eval.log" 2>&1 &
    pids+=("$!")
    gpu=$(( (gpu + 1) % NGPU ))
    if [[ ${#pids[@]} -eq ${NGPU} ]]; then
      wait "${pids[@]}" || true
      pids=()
    fi
  done
  if [[ ${#pids[@]} -gt 0 ]]; then
    wait "${pids[@]}" || true
  fi
}

# Pick the lowest-FID run and echo its swept LR. cfg_key is the
# train_cifar10.py config key the sweep drove (lr).
select_best_lr() {
  local sweep_root="$1" cfg_key="$2"
  uv run python - "${sweep_root}" "${cfg_key}" <<'PY'
import glob
import json
import os
import sys

import torch

sweep_root, cfg_key = sys.argv[1], sys.argv[2]
best = None
for fid_path in sorted(glob.glob(os.path.join(sweep_root, "*", "fid.json"))):
    run_dir = os.path.dirname(fid_path)
    ckpt = os.path.join(run_dir, "final.pt")
    if not os.path.exists(ckpt):
        continue
    try:
        fid = float(json.load(open(fid_path))["fid"])
    except (OSError, KeyError, ValueError):
        continue
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    lr = (state.get("config") or {}).get(cfg_key)
    if lr is None:
        continue
    if best is None or fid < best[0]:
        best = (fid, float(lr), run_dir)
if best is None:
    sys.exit(f"no rankable runs under {sweep_root}")
print(f"{best[1]:.8g}")
PY
}

# Phase 2: best-LR configs at full length, one ablation per GPU. More
# experiments than GPUs is fine: they run in waves of NGPU, blocking between
# waves, so the experiment count is not tied to the GPU count.
run_long_wave() {
  local -a pids=()
  local gpu=0 abl best run_dir run_name SWEEP_ARG="--lr"
  for abl in "${ABLATIONS[@]}"; do
    set_ablation_flags "${abl}"
    best="${BEST_LR[${abl}]}"
    run_dir="${GROUP_ROOT}/${abl}/final"
    run_name="${GROUP}_${abl}_final_lr${best}"
    run_name="${run_name//./p}"
    [[ "${DRY_RUN}" != "1" ]] && mkdir -p "${run_dir}"

    local -a cmd=(
      uv run python un0/train_cifar10.py
      --seed "${SEED}"
      "${SWEEP_ARG}" "${best}"
      --epochs "${EPOCHS_LONG}"
      --checkpoint-dir "${run_dir}"
      --wandb-project "${WANDB_PROJECT}"
      --wandb-name "${run_name}"
      --wandb-group "${GROUP}_final"
      --fid-every-epochs "${FID_EVERY_EPOCHS}"
      --fid-num-samples "${FID_FINAL_NUM_SAMPLES}"
    )
    [[ -n "${BATCH_SIZE}" ]] && cmd+=(--batch-size "${BATCH_SIZE}")
    # shellcheck disable=SC2206
    cmd+=(${STEP_FLAGS})

    log "GPU ${gpu}: long run ${abl} ${SWEEP_ARG}=${best} -> ${run_name}"
    log "+ CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      local prefix="[gpu${gpu} ${abl}] "
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${cmd[@]}" \
        > >(tee "${run_dir}/train.log" | sed -u "s/^/${prefix}/") 2>&1 &
      pids+=("$!")
    fi
    gpu=$(( (gpu + 1) % NGPU ))
    if [[ ${#pids[@]} -eq ${NGPU} ]]; then
      wait "${pids[@]}" || true
      pids=()
    fi
  done
  if [[ "${DRY_RUN}" != "1" && ${#pids[@]} -gt 0 ]]; then
    log "Waiting for ${#pids[@]} long runs to finish..."
    wait "${pids[@]}" || true
    log "All long runs finished."
  fi
}

write_best_lr_summary() {
  local out="${GROUP_ROOT}/best_lr.json"
  {
    printf '{\n'
    local first=1 abl
    for abl in "${ABLATIONS[@]}"; do
      [[ ${first} -eq 1 ]] || printf ',\n'
      first=0
      printf '  "%s": "%s"' "${abl}" "${BEST_LR[${abl}]}"
    done
    printf '\n}\n'
  } > "${out}"
  log "Wrote best-LR summary: ${out}"
}

main() {
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WARNING: WANDB_API_KEY is not set; W&B logging will fail at wandb.init." >&2
    echo "         Set WANDB_MODE=offline for a keyless local smoke test." >&2
  fi

  if [[ "${DO_SYNC}" -eq 1 && "${DRY_RUN}" != "1" ]]; then
    echo "+ uv sync --group dev --group logging --group eval"
    uv sync --group dev --group logging --group eval
  fi

  check_topology
  mkdir -p "${GROUP_ROOT}"
  log "Group=${GROUP}  output_root=${GROUP_ROOT}  short=${EPOCHS_SHORT}ep  long=${EPOCHS_LONG}ep"

  # ----- Phase 1: short LR sweep + FID selection, per ablation -----
  local abl sweep_root cfg_key="lr"
  for abl in "${ABLATIONS[@]}"; do
    sweep_root="${GROUP_ROOT}/${abl}/sweep"
    run_sweep_for_ablation "${abl}" "${sweep_root}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      BEST_LR["${abl}"]="DRYRUN"
      log "[dry-run] would eval sweep + select best LR for ${abl}"
      continue
    fi
    run_eval_wave "${sweep_root}"
    BEST_LR["${abl}"]="$(select_best_lr "${sweep_root}" "${cfg_key}")"
    log "Ablation ${abl}: best --lr=${BEST_LR[${abl}]}"
  done

  write_best_lr_summary

  # ----- Phase 2: best-LR long runs, one parallel wave -----
  run_long_wave
  log "Pipeline complete for group '${GROUP}'."
}

main "$@"
