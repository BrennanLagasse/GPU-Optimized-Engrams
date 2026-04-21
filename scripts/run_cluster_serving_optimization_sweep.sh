#!/usr/bin/env bash
set -euo pipefail

# Run deployable serving/scheduling variants on the H200 cluster. This sweep is
# intentionally focused on changes that are compatible with the current shared
# batch-position KV cache.

BRANCH="${BRANCH:-engrams-baseline-benchmarking}"
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

mkdir -p results logs

export PRESET="${PRESET:-target_40b_approx}"
export DEVICE_GROUPS="${DEVICE_GROUPS:-0,1,2,3 4,5,6,7}"
export DTYPE="${DTYPE:-bfloat16}"
export NUM_REQUESTS="${NUM_REQUESTS:-100}"
export MEAN_INPUT_TOKENS="${MEAN_INPUT_TOKENS:-128}"
export MEAN_OUTPUT_TOKENS="${MEAN_OUTPUT_TOKENS:-128}"
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-1024}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-1024}"
export SEED="${SEED:-0}"
export PYTHON_BIN
export SKIP_GIT_PULL=1

run_case() {
  local name="$1"
  shift
  local output="results/${name}.json"
  if [[ -f "${output}" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "Skipping ${name}; ${output} already exists"
    return
  fi
  echo
  echo "=== ${name} ==="
  env OUTPUT="${output}" "$@" bash scripts/run_cluster_serving_scheduling.sh
}

# Batch-size sweep for the current best realistic policy.
run_case serving_opt_sweep_optimized_input_static_b1 MODEL_IMPL=optimized_cached POLICY=longest_input_first DECODE_MODE=static BATCH_SIZE=1 REPLICA_ASSIGNMENT=round_robin
run_case serving_opt_sweep_optimized_input_static_b2 MODEL_IMPL=optimized_cached POLICY=longest_input_first DECODE_MODE=static BATCH_SIZE=2 REPLICA_ASSIGNMENT=round_robin
run_case serving_opt_sweep_optimized_input_static_b4 MODEL_IMPL=optimized_cached POLICY=longest_input_first DECODE_MODE=static BATCH_SIZE=4 REPLICA_ASSIGNMENT=round_robin
run_case serving_opt_sweep_optimized_input_static_b16 MODEL_IMPL=optimized_cached POLICY=longest_input_first DECODE_MODE=static BATCH_SIZE=16 REPLICA_ASSIGNMENT=round_robin

# Input-bucketed randomization tests whether keeping coarse input buckets while
# decorrelating within-bucket decode lengths improves makespan.
run_case serving_opt_sweep_optimized_bucketed_static_b8 MODEL_IMPL=optimized_cached POLICY=input_bucketed_random DECODE_MODE=static BATCH_SIZE=8 REPLICA_ASSIGNMENT=round_robin
run_case serving_opt_sweep_optimized_bucketed_static_b16 MODEL_IMPL=optimized_cached POLICY=input_bucketed_random DECODE_MODE=static BATCH_SIZE=16 REPLICA_ASSIGNMENT=round_robin

# Compact variants at smaller/larger batch sizes. Previous B=8 compact was
# slower than static, so this checks whether the tradeoff changes with batch.
run_case serving_opt_sweep_optimized_input_compact_b4 MODEL_IMPL=optimized_cached POLICY=longest_input_first DECODE_MODE=compact BATCH_SIZE=4 REPLICA_ASSIGNMENT=round_robin
run_case serving_opt_sweep_optimized_input_compact_b16 MODEL_IMPL=optimized_cached POLICY=longest_input_first DECODE_MODE=compact BATCH_SIZE=16 REPLICA_ASSIGNMENT=round_robin

"${PYTHON_BIN}" scripts/report_serving_optimization_sweep.py \
  --output results/serving_optimization_sweep_report_2026-04-21.md

echo
echo "Wrote results/serving_optimization_sweep_report_2026-04-21.md"
