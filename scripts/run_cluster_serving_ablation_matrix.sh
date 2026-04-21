#!/usr/bin/env bash
set -euo pipefail

# Sequential cluster serving ablation runner. Designed for the browser terminal:
# launch this under nohup so long naive cases survive console disconnects.

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

run_case serving_ablation_40b_optimized_random_rr optimized_cached random round_robin
run_case serving_ablation_40b_optimized_input_known_greedy optimized_cached longest_input_first greedy_prefill
run_case serving_ablation_40b_optimized_oracle_greedy optimized_cached longest_output_first greedy_oracle
run_case serving_ablation_40b_naive_input_known_rr naive longest_input_first round_robin
run_case serving_ablation_40b_naive_oracle_rr naive longest_output_first round_robin

echo
echo "Ablation matrix complete."
python scripts/report_serving_ablation.py --output results/serving_ablation_matrix_report.md
