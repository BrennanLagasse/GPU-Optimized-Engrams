# Cached Engram Serving Ablation Report

Timestamp: 2026-04-21 14:17 EDT

Cluster hardware: `gpu003`, `8 x NVIDIA H200`.

Model/workload:
- Preset: `target_40b_approx`, approximately 39.98B parameters.
- Requests: 100 deterministic heterogeneous requests.
- Input lengths: mean 128, max 1024.
- Output lengths: mean 128, max 1024.
- Total requested output: 12,800 tokens.
- Execution: two 4-GPU replicas, `DEVICE_GROUPS="0,1,2,3 4,5,6,7"`, `BATCH_SIZE=8` per replica.
- Metric below: serving wall seconds excluding model load and worker startup.

## Purpose

This ablation separates generic cached serving from the Engram-specific exact cached local-mixing step path.

Implementations:
- `naive`: full-context recompute baseline.
- `cached_full_engram`: same KV-cached static serving loop as `optimized_cached`, but cached Engram local mixing uses the full path.
- `optimized_cached`: KV-cached static serving loop with exact cached Engram `step_kernel` local mixing.

## Cluster Command

```bash
nohup bash scripts/run_cluster_cached_engram_ablation_matrix.sh > logs/cached_engram_ablation_40b.nohup.log 2>&1 &
```

## Results

| Case | Serving seconds | Total seconds | Serving tok/s | Total tok/s |
| --- | ---: | ---: | ---: | ---: |
| `naive_random` | 4882.758 | 5007.115 | 2.621 | 2.556 |
| `cached_full_random` | 191.678 | 306.684 | 66.779 | 41.737 |
| `optimized_random` | 192.714 | 317.465 | 66.420 | 40.319 |
| `naive_input` | 3629.573 | 3759.268 | 3.527 | 3.405 |
| `cached_full_input` | 167.351 | 281.733 | 76.486 | 45.433 |
| `optimized_input` | 165.990 | 280.487 | 77.113 | 45.635 |
| `naive_oracle` | 3271.340 | 3386.128 | 3.913 | 3.780 |
| `cached_full_oracle` | 120.373 | 235.387 | 106.336 | 54.379 |
| `optimized_oracle` | 119.568 | 244.181 | 107.052 | 52.420 |

## Attribution

| Comparison | Serving seconds | Speedup | Serving-time reduction |
| --- | ---: | ---: | ---: |
| Generic cached serving under random scheduling: `naive_random -> cached_full_random` | 4882.758 -> 191.678 | 25.47x | 96.07% |
| Engram step-kernel under random scheduling: `cached_full_random -> optimized_random` | 191.678 -> 192.714 | 0.99x | -0.54% |
| Generic cached serving under input-known scheduling: `naive_input -> cached_full_input` | 3629.573 -> 167.351 | 21.69x | 95.39% |
| Engram step-kernel under input-known scheduling: `cached_full_input -> optimized_input` | 167.351 -> 165.990 | 1.01x | 0.81% |
| Generic cached serving under oracle scheduling: `naive_oracle -> cached_full_oracle` | 3271.340 -> 120.373 | 27.18x | 96.32% |
| Engram step-kernel under oracle scheduling: `cached_full_oracle -> optimized_oracle` | 120.373 -> 119.568 | 1.01x | 0.67% |

## Interpretation

In this 40B batched serving workload, the large model-path gain is almost entirely explained by generic KV-cached serving versus full-context recompute. The exact cached Engram `step_kernel` optimization is important in smaller cached Engram microbenchmarks, but it is not a major contributor to end-to-end 40B batched serving time here. Its measured effect is within about `-0.54%` to `+0.81%`, which is small enough to treat as near-noise for this workload.

The best deployable result remains `optimized_cached + longest_input_first + round_robin`, but `cached_full_engram + longest_input_first` is very close. This means further serving gains should likely come from dynamic batching / active-row compaction / better request scheduling rather than additional Engram local-mixing micro-optimizations.
