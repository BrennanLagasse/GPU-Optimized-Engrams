# Serving Ablation Matrix Report

Timestamp: 2026-04-21 17:58 EDT

Cluster hardware: `gpu003`, `8 x NVIDIA H200`.

Model/workload:
- Preset: `target_40b_approx`, approximately 39.98B parameters.
- Requests: 100 deterministic heterogeneous requests.
- Input lengths: mean 128, max 1024, long-tailed synthetic distribution.
- Output lengths: mean 128, max 1024, long-tailed synthetic distribution.
- Total requested output: 12,800 tokens.
- Execution: two 4-GPU replicas, `DEVICE_GROUPS="0,1,2,3 4,5,6,7"`, `BATCH_SIZE=8` per replica.
- Metric below: serving wall seconds excluding model load and worker startup.

## Cluster Commands

```bash
nohup bash scripts/run_cluster_serving_ablation_matrix.sh > logs/serving_ablation_matrix_40b.nohup.log 2>&1 &
```

The ablation runner invoked these cases:

```bash
MODEL_IMPL=optimized_cached POLICY=random REPLICA_ASSIGNMENT=round_robin
MODEL_IMPL=optimized_cached POLICY=longest_input_first REPLICA_ASSIGNMENT=greedy_prefill
MODEL_IMPL=optimized_cached POLICY=longest_output_first REPLICA_ASSIGNMENT=greedy_oracle
MODEL_IMPL=naive POLICY=longest_input_first REPLICA_ASSIGNMENT=round_robin
MODEL_IMPL=naive POLICY=longest_output_first REPLICA_ASSIGNMENT=round_robin
```

Previously completed baseline cases used:

```bash
MODEL_IMPL=naive POLICY=random REPLICA_ASSIGNMENT=round_robin
MODEL_IMPL=optimized_cached POLICY=longest_input_first REPLICA_ASSIGNMENT=round_robin
MODEL_IMPL=optimized_cached POLICY=longest_output_first REPLICA_ASSIGNMENT=round_robin
```

## Results

| Case | Model | Scheduler | Replica assignment | Serving seconds | Serving tok/s | Prefill padding | Decode padding |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `naive_random` | naive | random | round_robin | 4882.758 | 2.621 | 3.2459x | 3.4050x |
| `naive_input` | naive | longest_input_first | round_robin | 3629.573 | 3.527 | 1.3300x | 3.3278x |
| `naive_oracle` | naive | longest_output_first | round_robin | 3271.340 | 3.913 | 3.3256x | 1.3325x |
| `opt_random` | optimized_cached | random | round_robin | 192.714 | 66.420 | 3.2459x | 3.4050x |
| `opt_input_rr` | optimized_cached | longest_input_first | round_robin | 165.990 | 77.113 | 1.3300x | 3.3278x |
| `opt_input_greedy` | optimized_cached | longest_input_first | greedy_prefill | 168.879 | 75.794 | 1.3300x | 3.3278x |
| `opt_oracle_rr` | optimized_cached | longest_output_first | round_robin | 119.568 | 107.052 | 3.3256x | 1.3325x |
| `opt_oracle_greedy` | optimized_cached | longest_output_first | greedy_oracle | 109.873 | 116.498 | 3.3256x | 1.3325x |

## Attribution

| Comparison | Serving seconds | Speedup | Serving-time reduction |
| --- | ---: | ---: | ---: |
| Realistic scheduler only on naive: `naive_random -> naive_input` | 4882.758 -> 3629.573 | 1.35x | 25.67% |
| Oracle scheduler only on naive: `naive_random -> naive_oracle` | 4882.758 -> 3271.340 | 1.49x | 33.00% |
| Optimized model only under random scheduling: `naive_random -> opt_random` | 4882.758 -> 192.714 | 25.34x | 96.05% |
| Realistic scheduler on optimized model: `opt_random -> opt_input_rr` | 192.714 -> 165.990 | 1.16x | 13.87% |
| Full realistic bundle: `naive_random -> opt_input_rr` | 4882.758 -> 165.990 | 29.42x | 96.60% |
| Oracle scheduler on optimized model: `opt_random -> opt_oracle_rr` | 192.714 -> 119.568 | 1.61x | 37.96% |
| Oracle scheduler plus oracle replica assignment: `opt_random -> opt_oracle_greedy` | 192.714 -> 109.873 | 1.75x | 42.99% |

## Optimization Findings

- The dominant contribution in the current benchmark is model-path optimization, not scheduling alone. Under identical random scheduling, optimized cached inference is `25.34x` faster than naive full-context inference.
- Realistic input-known scheduling still matters. On the naive model, longest-input-first reduces serving time by `25.67%`; on the optimized model, it reduces serving time by `13.87%`.
- Oracle output-length scheduling gives a larger upper-bound scheduler gain because it reduces decode padding instead of prefill padding. On the optimized model, oracle scheduling reduces serving time by `37.96%` relative to optimized random.
- The attempted realistic `greedy_prefill` replica assignment did not improve makespan for this workload. It was `1.74%` slower than round-robin after input-known scheduling, because decode imbalance dominated the final batch-replica critical path.
- The oracle `greedy_oracle` replica assignment did help, reducing optimized oracle serving time from `119.568s` to `109.873s`, an additional `8.11%` reduction. This is not deployable directly because it uses true output lengths.

## Interpretation

For the realistic serving setup, the current best deployable result remains `optimized_cached + longest_input_first + round_robin`, not `greedy_prefill`. The full realistic bundle is `29.42x` faster than `naive + random`, but ablations show this is mostly due to cached optimized inference. Scheduler improvements are measurable and useful, but the present static scheduler cannot close the remaining gap to oracle without output-length information or a more dynamic runtime such as continuous batching / active-row compaction.
