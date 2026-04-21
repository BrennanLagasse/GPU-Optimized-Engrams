#!/usr/bin/env bash
set -euo pipefail

# Active-row compaction ablation. Finished rows are removed from the decode
# loop once their observed generation length is reached.
#
# Launch under nohup on the cluster:
#   nohup bash scripts/run_cluster_compact_serving_ablation.sh > logs/compact_serving_ablation_40b.nohup.log 2>&1 &

BRANCH="${BRANCH:-engrams-baseline-benchmarking}"
DEVICE_GROUPS="${DEVICE_GROUPS:-0,1,2,3 4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PRESET="${PRESET:-target_40b_approx}"

git fetch origin
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}"

mkdir -p logs results

run_case() {
  local name="$1"
  local model_impl="$2"
  local policy="$3"
  local output="results/${name}.json"
  local log="logs/${name}.log"

  if [[ -f "${output}" && "${FORCE:-0}" != "1" ]]; then
    echo "Skipping ${name}; ${output} already exists."
    return
  fi

  echo
  echo "=== ${name} ==="
  echo "MODEL_IMPL=${model_impl} POLICY=${policy} DECODE_MODE=compact"
  MODEL_IMPL="${model_impl}" \
    POLICY="${policy}" \
    REPLICA_ASSIGNMENT="round_robin" \
    DECODE_MODE="compact" \
    BATCH_SIZE="${BATCH_SIZE}" \
    DEVICE_GROUPS="${DEVICE_GROUPS}" \
    PRESET="${PRESET}" \
    OUTPUT="${output}" \
    bash scripts/run_cluster_serving_scheduling.sh 2>&1 | tee "${log}"
}

run_case serving_ablation_40b_naive_random_compact_rr naive random
run_case serving_ablation_40b_optimized_input_known_compact_rr optimized_cached longest_input_first
run_case serving_ablation_40b_optimized_oracle_compact_rr optimized_cached longest_output_first

echo
echo "Compact serving ablation complete."
