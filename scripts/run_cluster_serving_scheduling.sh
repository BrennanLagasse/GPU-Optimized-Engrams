#!/usr/bin/env bash
set -euo pipefail

# Reproduce the scheduling-oriented 100-request serving benchmark on the H200
# cluster. The default uses two 4-GPU model-parallel replicas so all 8 GPUs are
# active via data parallel request scheduling.
#
# Typical cluster usage:
#   cd ~/class_projects/GPU-Optimized-Engrams
#   bash scripts/run_cluster_serving_scheduling.sh
#
# Useful overrides:
#   BATCH_SIZE=4 bash scripts/run_cluster_serving_scheduling.sh
#   MODEL_IMPL=naive POLICY=random OUTPUT=results/serving_scheduling_target_40b_naive_random.json bash scripts/run_cluster_serving_scheduling.sh
#   MODEL_IMPL=cached_full_engram POLICY=random OUTPUT=results/serving_scheduling_target_40b_cached_full_random.json bash scripts/run_cluster_serving_scheduling.sh
#   DECODE_MODE=compact POLICY=longest_input_first OUTPUT=results/serving_scheduling_target_40b_compact.json bash scripts/run_cluster_serving_scheduling.sh
#   POLICY=fifo bash scripts/run_cluster_serving_scheduling.sh
#   POLICY=longest_output_first OUTPUT=results/serving_scheduling_target_40b_oracle_output.json bash scripts/run_cluster_serving_scheduling.sh
#   REPLICA_ASSIGNMENT=greedy_prefill bash scripts/run_cluster_serving_scheduling.sh
#   DEVICE_GROUPS="0,1,2,3 4,5,6,7" bash scripts/run_cluster_serving_scheduling.sh

BRANCH="${BRANCH:-engrams-baseline-benchmarking}"
PRESET="${PRESET:-target_40b_approx}"
DEVICE_GROUPS="${DEVICE_GROUPS:-0,1,2,3 4,5,6,7}"
DTYPE="${DTYPE:-bfloat16}"
MODEL_IMPL="${MODEL_IMPL:-optimized_cached}"
BATCH_SIZE="${BATCH_SIZE:-8}"
POLICY="${POLICY:-longest_input_first}"
REPLICA_ASSIGNMENT="${REPLICA_ASSIGNMENT:-round_robin}"
DECODE_MODE="${DECODE_MODE:-static}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
MEAN_INPUT_TOKENS="${MEAN_INPUT_TOKENS:-128}"
MEAN_OUTPUT_TOKENS="${MEAN_OUTPUT_TOKENS:-128}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-1024}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-1024}"
SEED="${SEED:-0}"
OUTPUT="${OUTPUT:-results/serving_scheduling_target_40b.json}"
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
  "${PYTHON_BIN}" scripts/benchmark_serving.py
  --preset "${PRESET}"
  --device-groups ${DEVICE_GROUPS}
  --dtype "${DTYPE}"
  --model-impl "${MODEL_IMPL}"
  --batch-size "${BATCH_SIZE}"
  --policy "${POLICY}"
  --replica-assignment "${REPLICA_ASSIGNMENT}"
  --decode-mode "${DECODE_MODE}"
  --num-requests "${NUM_REQUESTS}"
  --mean-input-tokens "${MEAN_INPUT_TOKENS}"
  --mean-output-tokens "${MEAN_OUTPUT_TOKENS}"
  --max-input-tokens "${MAX_INPUT_TOKENS}"
  --max-output-tokens "${MAX_OUTPUT_TOKENS}"
  --seed "${SEED}"
  --output "${OUTPUT}"
)

echo "Running serving scheduling benchmark:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"

echo
echo "Wrote serving benchmark output to ${OUTPUT}"
