#!/usr/bin/env bash
set -euo pipefail

# Reproduce the target-scale placement/decode sweep on the H200 cluster.
#
# Typical cluster usage:
#   cd ~/class_projects/GPU-Optimized-Engrams
#   bash scripts/run_cluster_placement_sweep.sh
#
# Useful overrides:
#   DECODE_LENGTHS="64 128" bash scripts/run_cluster_placement_sweep.sh
#   DEVICE_GROUPS="0,1,2,3" bash scripts/run_cluster_placement_sweep.sh
#   OUTPUT=results/my_sweep.json bash scripts/run_cluster_placement_sweep.sh
#   SKIP_GIT_PULL=1 bash scripts/run_cluster_placement_sweep.sh

BRANCH="${BRANCH:-engrams-baseline-benchmarking}"
PRESET="${PRESET:-target_40b_approx}"
GROUP_SIZES="${GROUP_SIZES:-4 8}"
DECODE_LENGTHS="${DECODE_LENGTHS:-64 128 256}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PROMPT_LENGTH="${PROMPT_LENGTH:-8}"
TRIALS="${TRIALS:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-120000}"
MAX_GPU_UTIL_PERCENT="${MAX_GPU_UTIL_PERCENT:-10}"
OUTPUT="${OUTPUT:-results/cluster_placement_sweep_target_40b.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_PATH="${VENV_PATH:-.venv/bin/activate}"

if [[ "${SKIP_GIT_PULL:-0}" != "1" ]]; then
  git fetch origin
  git checkout "${BRANCH}"
  git pull --ff-only origin "${BRANCH}"
fi

if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cmd=(
  "${PYTHON_BIN}" scripts/sweep_cluster_placements.py
  --preset "${PRESET}"
  --decode-lengths ${DECODE_LENGTHS}
  --dtype "${DTYPE}"
  --batch-size "${BATCH_SIZE}"
  --prompt-length "${PROMPT_LENGTH}"
  --trials "${TRIALS}"
  --min-free-mib "${MIN_FREE_MIB}"
  --max-gpu-util-percent "${MAX_GPU_UTIL_PERCENT}"
  --output "${OUTPUT}"
)

if [[ -n "${DEVICE_GROUPS:-}" ]]; then
  cmd+=(--device-groups ${DEVICE_GROUPS})
else
  cmd+=(--group-sizes ${GROUP_SIZES})
fi

if [[ "${ALLOW_NON_CONTIGUOUS:-0}" == "1" ]]; then
  cmd+=(--allow-non-contiguous)
fi

if [[ "${FAIL_FAST:-0}" == "1" ]]; then
  cmd+=(--fail-fast)
fi

echo "Running placement sweep:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"

echo
echo "Wrote sweep output to ${OUTPUT}"
