#!/usr/bin/env bash
set -euo pipefail

# Runs the cached Engram-path ablation matrix:
# - cached_full_engram keeps KV-cached serving but uses full cached Engram local mixing
# - optimized_cached uses the exact cached step-kernel Engram path
#
# Launch under nohup on the cluster:
#   nohup bash scripts/run_cluster_cached_engram_ablation_matrix.sh > logs/cached_engram_ablation_40b.nohup.log 2>&1 &

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
  local replica_assignment="$4"
  local output="results/${name}.json"
  local log="logs/${name}.log"

  if [[ -f "${output}" && "${FORCE:-0}" != "1" ]]; then
    echo "Skipping ${name}; ${output} already exists."
    return
  fi

  echo
  echo "=== ${name} ==="
  echo "MODEL_IMPL=${model_impl} POLICY=${policy} REPLICA_ASSIGNMENT=${replica_assignment}"
  MODEL_IMPL="${model_impl}" \
    POLICY="${policy}" \
    REPLICA_ASSIGNMENT="${replica_assignment}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    DEVICE_GROUPS="${DEVICE_GROUPS}" \
    PRESET="${PRESET}" \
    OUTPUT="${output}" \
    bash scripts/run_cluster_serving_scheduling.sh 2>&1 | tee "${log}"
}

run_case serving_ablation_40b_cached_full_random_rr cached_full_engram random round_robin
run_case serving_ablation_40b_cached_full_input_known_rr cached_full_engram longest_input_first round_robin
run_case serving_ablation_40b_cached_full_oracle_rr cached_full_engram longest_output_first round_robin

echo
echo "Cached Engram ablation matrix complete."
