# Compact Serving Ablation

Timestamp: 2026-04-21 15:49 EDT

Branch: `engrams-baseline-benchmarking`

## Scope

This report compares static padded decode against active-row compaction on the 40B serving workload.

Hardware and workload:
- Cluster node: `gpu003`
- GPUs: `8 x NVIDIA H200`
- Model preset: `target_40b_approx`
- Approximate model size: `39.98B` parameters
- Serving workload: 100 deterministic heterogeneous requests
- Input lengths: mean 128, max 1024
- Output lengths: mean 128, max 1024
- Total requested output tokens: 12,800
- Execution layout: two 4-GPU replicas, `DEVICE_GROUPS="0,1,2,3 4,5,6,7"`
- Batch size: 8 per replica, effective concurrent batch size 16
- Replica assignment: round-robin
- Primary metric: serving wall seconds excluding model load and worker startup

## Implementation

`decode_mode=static` keeps every batch row alive until the longest output in that batch completes. This wastes decode work after shorter requests have already finished.

`decode_mode=compact` removes rows online after they complete generation, then compacts model caches with the surviving row indices. This is realistic because a serving system observes completion only after the request emits EOS or reaches its max-token limit. It does not require knowing output lengths before generation starts.

This differs from oracle scheduling. `longest_output_first` sorts by true output length before generation, which cannot be used in a realistic serving scenario because actual output lengths are unknown until generation is complete.

## Results

| Case | Model | Policy | Decode mode | Serving seconds | Total seconds | Serving tok/s | Total tok/s | Executed decode tokens |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `naive_random_static` | `naive` | `random` | `static` | 4882.758 | 5007.115 | 2.621 | 2.556 | 43,584 |
| `naive_random_compact` | `naive` | `random` | `compact` | 1079.463 | 1194.636 | 11.858 | 10.715 | 12,800 |
| `optimized_input_static` | `optimized_cached` | `longest_input_first` | `static` | 165.990 | 280.487 | 77.113 | 45.635 | 42,596 |
| `optimized_input_compact` | `optimized_cached` | `longest_input_first` | `compact` | 182.102 | 296.789 | 70.290 | 43.128 | 12,800 |
| `optimized_oracle_static` | `optimized_cached` | `longest_output_first` | `static` | 119.568 | 244.181 | 107.052 | 52.420 | 17,056 |
| `optimized_oracle_compact` | `optimized_cached` | `longest_output_first` | `compact` | 173.919 | 291.409 | 73.761 | 43.925 | 12,800 |

## Static vs Compact

| Comparison | Serving seconds | Speedup | Serving-time reduction | Serving tok/s | Executed decode tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `naive random` | 4882.758 -> 1079.463 | 4.52x | 77.89% | 2.621 -> 11.858 | 43,584 -> 12,800 |
| `optimized input-known` | 165.990 -> 182.102 | 0.91x | -9.71% | 77.113 -> 70.290 | 42,596 -> 12,800 |
| `optimized oracle` | 119.568 -> 173.919 | 0.69x | -45.45% | 107.052 -> 73.761 | 17,056 -> 12,800 |

## Interpretation

Active-row compaction works as intended: all compact cases execute exactly the requested 12,800 output-token steps instead of padded decode steps. It substantially improves the naive full-context baseline because static padding plus recompute is extremely wasteful.

For the optimized cached model, compact mode was slower despite reducing executed decode tokens. The most likely explanation is that per-step row filtering, cache `index_select`, and shrinking tensor shapes add synchronization and memory-movement overhead that outweighs saved compute at this batch size and model-parallel layout.

Current best deployable result remains:
- `MODEL_IMPL=optimized_cached`
- `POLICY=longest_input_first`
- `DECODE_MODE=static`
- Serving time: `165.990s`
- Serving throughput: `77.113 requested output tok/s`
- Speedup over `naive + random + static`: `29.42x`
- Serving-time reduction over `naive + random + static`: `96.60%`

Best compact deployable comparison:
- `naive + random + compact`: `1079.463s`
- `optimized_cached + longest_input_first + compact`: `182.102s`
- Speedup: `5.93x`
- Serving-time reduction: `83.13%`

## Practical Conclusion

Keep active-row compaction available as an ablation and correctness feature, but do not make it the default optimized serving mode based on this run. For the current 40B benchmark, static optimized cached decode with input-known scheduling is faster.
