# Continuous Refill Cluster Report

Timestamp: 2026-04-22 13:10 EDT

## Setup

The continuous-refill serving implementation was benchmarked on the TP cluster after pushing commit `268b8f4` to `engrams-baseline-benchmarking`.

Hardware:

- host: `gpu003`
- GPUs: `8 x NVIDIA H200`
- GPU memory: `143,771 MiB` per GPU as reported by `nvidia-smi`
- execution layout: two 4-GPU model-parallel replicas
- replica 0 devices: `0,1,2,3`
- replica 1 devices: `4,5,6,7`

Model/workload:

- preset: `target_40b_approx`
- approximate model size: `39,978,323,440` parameters, about `39.98B`
- dtype: `bfloat16`
- requests: `100`
- input-length distribution: deterministic long tail, mean `128`, max `1024`
- output-length distribution: deterministic long tail, mean `128`, max `1024`
- requested output tokens: `12,800`

## Commands

Realistic continuous B16:

```bash
MODEL_IMPL=optimized_cached \
POLICY=longest_input_first \
DECODE_MODE=continuous \
BATCH_SIZE=16 \
REPLICA_ASSIGNMENT=round_robin \
OUTPUT=results/serving_opt_sweep_optimized_input_continuous_b16.json \
SKIP_GIT_PULL=1 \
bash scripts/run_cluster_serving_scheduling.sh
```

Realistic continuous B8:

```bash
MODEL_IMPL=optimized_cached \
POLICY=longest_input_first \
DECODE_MODE=continuous \
BATCH_SIZE=8 \
REPLICA_ASSIGNMENT=round_robin \
OUTPUT=results/serving_opt_sweep_optimized_input_continuous_b8.json \
SKIP_GIT_PULL=1 \
bash scripts/run_cluster_serving_scheduling.sh
```

Oracle continuous B16:

```bash
MODEL_IMPL=optimized_cached \
POLICY=longest_output_first \
DECODE_MODE=continuous \
BATCH_SIZE=16 \
REPLICA_ASSIGNMENT=round_robin \
OUTPUT=results/serving_opt_sweep_optimized_oracle_continuous_b16.json \
SKIP_GIT_PULL=1 \
bash scripts/run_cluster_serving_scheduling.sh
```

## Results

| Case | Policy | Batch/replica | Serving seconds | Requested output tok/s | Speedup vs naive static | Delta vs prior best |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| realistic continuous B8 | `longest_input_first` | 8 | `167.398` | `76.464` | `29.17x` | `2.51% slower` |
| realistic continuous B16 | `longest_input_first` | 16 | `164.889` | `77.628` | `29.61x` | `0.97% slower` |
| oracle continuous B16 | `longest_output_first` | 16 | `216.314` | `59.173` | `22.57x` | `32.46% slower` |

Baselines:

- naive random static baseline: `4882.758s`
- prior best realistic measured case: `optimized_cached + longest_input_first + compact + BATCH_SIZE=16`
- prior best realistic measured time: `163.306s`

## Interpretation

The continuous-refill implementation is now benchmarkable at target scale, but it does not beat the prior best fixed-workload result. The best continuous result was B16 at `164.889s`, which is `0.97%` slower than B16 compact.

The likely reason is that this fixed 100-request closed workload already has enough batching to keep the GPUs busy, while the implemented continuous path pays extra overhead:

- exact per-request prefill instead of padded batch prefill
- slot-indexed KV reads/writes
- slot-indexed Engram short-conv state
- Python-level scheduler bookkeeping

The oracle result being worse is also informative. Sorting by output length is not deployable, and for this workload it appears to harm input-length locality enough that any output-length grouping benefit is lost.

## Current State

Continuous refill is implemented and measured. It is useful as an architecture primitive and a correctness base for future serving work, but the project should continue presenting B16 compact as the best measured realistic serving result unless a later optimization improves continuous mode.

Next optimization targets, if continuing:

- profile continuous mode to separate per-request prefill overhead from slot-indexed decode overhead
- batch admissions for multiple new requests at once instead of prefilling one slot at a time
- reduce Python scheduler overhead in the continuous loop
- investigate paged/cache-block metadata only if memory fragmentation or slot indexing becomes a measurable bottleneck
