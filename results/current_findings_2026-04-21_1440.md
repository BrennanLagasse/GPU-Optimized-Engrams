# Current Findings

Timestamp: 2026-04-21 14:40 EDT

Branch: `engrams-baseline-benchmarking`

## Scope

This report summarizes the current 40B serving/scheduling findings after the completed scheduler and cached-Engram ablation runs.

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
- Primary metric: serving wall seconds excluding model load and worker startup

## Implementations

`naive`:
- Full-context recompute for every generated token.
- No KV-cache incremental decode.
- Used as the intentionally simple baseline.

`cached_full_engram`:
- Uses the same KV-cached incremental serving loop as the optimized path.
- Forces cached Engram local mixing through the full path.
- Purpose: isolate generic KV-cache / incremental-decode gains from Engram-specific cached-step optimization.

`optimized_cached`:
- Uses KV-cached incremental serving.
- Uses the exact cached Engram `step_kernel` local-mixing path.
- This is the optimized implementation used for the main serving result.

Schedulers:
- `random`: naive random request order.
- `longest_input_first`: realistic input-known scheduler. It does not use output lengths.
- `longest_output_first`: oracle scheduler. It uses true output lengths and is not deployable, but provides an upper bound.

The oracle scheduler cannot be used in a realistic serving scenario because the server does not know actual output lengths before generation starts. It only learns that a request is complete after that request emits an end condition, such as EOS or a configured max-token limit. Therefore, oracle results should be read as an upper-bound diagnostic for decode-length heterogeneity, not as a deployable scheduling policy.

## Main Serving Results

| Case | Serving seconds | Serving tok/s |
| --- | ---: | ---: |
| `naive + random` | 4882.758 | 2.621 |
| `naive + longest_input_first` | 3629.573 | 3.527 |
| `naive + oracle` | 3271.340 | 3.913 |
| `cached_full_engram + random` | 191.678 | 66.779 |
| `cached_full_engram + longest_input_first` | 167.351 | 76.486 |
| `cached_full_engram + oracle` | 120.373 | 106.336 |
| `optimized_cached + random` | 192.714 | 66.420 |
| `optimized_cached + longest_input_first` | 165.990 | 77.113 |
| `optimized_cached + oracle` | 119.568 | 107.052 |

## Attribution

### Generic KV-cache / incremental decode

| Comparison | Speedup | Serving-time reduction |
| --- | ---: | ---: |
| `naive + random -> cached_full_engram + random` | 25.47x | 96.07% |
| `naive + longest_input_first -> cached_full_engram + longest_input_first` | 21.69x | 95.39% |
| `naive + oracle -> cached_full_engram + oracle` | 27.18x | 96.32% |

The dominant 40B batched-serving speedup comes from moving from full-context recompute to KV-cached incremental decode.

### Engram-specific cached `step_kernel`

| Comparison | Speedup | Serving-time reduction |
| --- | ---: | ---: |
| `cached_full_engram + random -> optimized_cached + random` | 0.99x | -0.54% |
| `cached_full_engram + longest_input_first -> optimized_cached + longest_input_first` | 1.01x | 0.81% |
| `cached_full_engram + oracle -> optimized_cached + oracle` | 1.01x | 0.67% |

At this target-scale batched-serving level, the Engram `step_kernel` is effectively noise-level relative to `cached_full_engram`. It remains useful in smaller cached Engram microbenchmarks, but it is not responsible for the large end-to-end 40B serving gain.

### Scheduling

| Comparison | Speedup | Serving-time reduction |
| --- | ---: | ---: |
| `naive + random -> naive + longest_input_first` | 1.35x | 25.67% |
| `optimized_cached + random -> optimized_cached + longest_input_first` | 1.16x | 13.87% |
| `optimized_cached + random -> optimized_cached + oracle` | 1.61x | 37.96% |

The realistic scheduler improves serving time by reducing known input/prefill padding. The oracle scheduler is much stronger because it sorts by output length and reduces decode padding, but it uses information unavailable in real serving.

## Optimization Attempts

`compact` active-row decode:
- Removes completed batch rows online after completion is observed, then compacts model caches with the surviving row indices.
- This is realistic because a server knows when a request finishes, even though it does not know that true output length before generation starts.
- Result for `naive + random`: `4882.758s -> 1079.463s`, a `4.52x` speedup, because executed decode work dropped from `43,584` padded steps to the requested `12,800` steps.
- Result for `optimized_cached + longest_input_first`: `165.990s -> 182.102s`, a `9.71%` slowdown despite reducing executed decode steps from `42,596` to `12,800`.
- Result for `optimized_cached + oracle`: `119.568s -> 173.919s`, a `45.45%` slowdown.
- Conclusion: active-row compaction is a useful ablation and substantially improves the naive padded baseline, but for the current 40B optimized model-parallel serving path its per-step filtering/cache-compaction overhead outweighs the saved decode compute.

`greedy_prefill` replica assignment:
- Intended to balance known padded prefill work across the two 4-GPU replicas.
- Result: `optimized_cached + longest_input_first + greedy_prefill` took `168.879s`, versus `165.990s` for round-robin.
- Conclusion: this realistic optimization did not help. It was `1.74%` slower because decode imbalance dominated the makespan.

`greedy_oracle` replica assignment:
- Uses true padded prefill plus decode work to balance replicas.
- Result: `optimized_cached + oracle + greedy_oracle` took `109.873s`, versus `119.568s` for oracle round-robin.
- Conclusion: oracle balancing helps by `8.11%`, but it is not deployable because it uses true output lengths.

## Interpretation

The earlier `29.42x` full-bundle speedup is real, but it should not be described as an Engram-kernel speedup. The corrected interpretation is:

> For a roughly 40B model serving 100 heterogeneous requests on 8 H200s, moving from naive full-context recompute to KV-cached incremental decode explains nearly all of the large model-path speedup. Realistic input-known scheduling gives an additional measurable gain, while oracle output-length scheduling shows remaining scheduling headroom.

This also explains the difference from batch-size-1 experiments. At batch size 1, there is little padding waste and less repeated batched recomputation, so the optimized-vs-naive gap was much smaller. In the batched heterogeneous workload, static batches with long-tailed lengths make full-context recompute extremely expensive, so KV caching dominates.

## Current Best Deployable Configuration

Best realistic result:
- `MODEL_IMPL=optimized_cached`
- `POLICY=longest_input_first`
- `REPLICA_ASSIGNMENT=round_robin`
- Serving time: `165.990s`
- Serving throughput: `77.113 requested output tok/s`
- Speedup over `naive + random`: `29.42x`
- Serving-time reduction over `naive + random`: `96.60%`

`cached_full_engram + longest_input_first` is extremely close at `167.351s`, so the practical target-scale serving win should be attributed to cached incremental decoding plus scheduler choice, not to the Engram `step_kernel`.

## Recommended Next Work

- Treat active-row compaction as implemented and measured; keep it available, but do not use it as the default optimized path unless a future implementation avoids the per-step compaction overhead.
- Test continuous batching with larger request counts and arrival processes, not just a fixed 100-request batch.
- Keep Engram `step_kernel` in the optimized path because it is exact and helps smaller microbenchmarks, but do not present it as the driver of 40B serving speedup.
