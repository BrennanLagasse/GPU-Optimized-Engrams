# Serving Optimization Sweep

Timestamp: 2026-04-21 16:55 EDT

Branch: `engrams-baseline-benchmarking`

## Scope

This report summarizes the follow-up optimization attempts after the 40B serving attribution pass.

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
- Primary metric: serving wall seconds excluding model load and worker startup

## Experiments Tried

Implemented and measured:
- Batch-size sweep for `optimized_cached + longest_input_first + static`
- Active-row compaction at batch sizes 4 and 16
- Coarse input-bucket randomization via `input_bucketed_random`, which uses known input lengths only

Investigated but not implemented as a measured GPU path:
- True continuous batching / refill-on-completion
- Decode active masking without physical compaction
- Hybrid prefill/decode scheduling
- Tensor parallel execution

The main blocker for true continuous batching is architectural: the current attention cache tracks one shared cache position per batch. A serving-style row refill needs per-row cache positions or paged/cache-block metadata so new requests can be admitted into completed rows without resetting the whole batch cache. Without that, a "continuous" implementation would either be incorrect for variable-length rows or would need to reset/recompute too much state to be a fair optimization.

## Measured Results

Baseline comparisons:
- Previous best deployable result: `optimized_cached + longest_input_first + static + batch_size=8` at `165.990s`, `77.113 tok/s`
- Naive static baseline: `naive + random + static + batch_size=8` at `4882.758s`, `2.621 tok/s`

| Case | Policy | Decode | Batch/replica | Serving seconds | Total seconds | Serving tok/s | Total tok/s | Prefill overhead | Decode overhead | Executed decode tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `optimized_input_compact_b16_repeat1` | `longest_input_first` | `compact` | 16 | 163.306 | n/a | 78.381 | n/a | 1.882 | 4.661 | 12,800 |
| `optimized_input_compact_b16` | `longest_input_first` | `compact` | 16 | 164.075 | 279.717 | 78.013 | 45.761 | 1.882 | 4.661 | 12,800 |
| `previous_best_input_static_b8` | `longest_input_first` | `static` | 8 | 165.990 | 280.487 | 77.113 | 45.635 | 1.330 | 3.328 | 42,596 |
| `optimized_input_static_b4` | `longest_input_first` | `static` | 4 | 176.799 | 291.696 | 72.399 | 43.881 | 1.116 | 2.238 | 28,652 |
| `optimized_input_static_b16` | `longest_input_first` | `static` | 16 | 183.841 | 309.762 | 69.625 | 41.322 | 1.882 | 4.661 | 59,660 |
| `optimized_bucketed_static_b8` | `input_bucketed_random` | `static` | 8 | 191.878 | 306.765 | 66.709 | 41.726 | 1.622 | 3.315 | 42,432 |
| `optimized_input_compact_b4` | `longest_input_first` | `compact` | 4 | 227.471 | 342.661 | 56.271 | 37.355 | 1.116 | 2.238 | 12,800 |
| `optimized_bucketed_static_b16` | `input_bucketed_random` | `static` | 16 | 231.194 | 346.170 | 55.365 | 36.976 | 2.182 | 4.683 | 59,944 |
| `optimized_input_static_b2` | `longest_input_first` | `static` | 2 | 231.256 | 345.747 | 55.350 | 37.021 | 1.035 | 1.508 | 19,300 |
| `optimized_input_static_b1` | `longest_input_first` | `static` | 1 | 253.385 | 367.835 | 50.516 | 34.798 | 1.000 | 1.000 | 12,800 |

## Interpretation

The only measured improvement was `batch_size=16 + compact`. It improved serving time from `165.990s` to `164.075s` on the first run and `163.306s` on the repeat. Relative to the previous best:
- First B16 compact run: `1.012x` speedup, `1.15%` serving-time reduction
- Repeat B16 compact run: `1.016x` speedup, `1.62%` serving-time reduction

This is a small gain, but it is directionally consistent across two runs. The likely reason B16 compact helps while B8 compact did not is that larger static batches create more decode padding waste; compaction has more work to remove, enough to slightly outweigh its cache-indexing overhead.

Batch size 1 removed all padding but was much slower because it gave up batched GPU throughput. Batch sizes 2 and 4 reduced padding but still underperformed B8/B16 because lower batch utilization dominated. Static B16 was worse because decode padding grew too much without compaction.

`input_bucketed_random` did not help. Coarse input buckets increased prefill padding versus exact `longest_input_first` and did not reduce decode imbalance enough to compensate.

## Current Best Deployable Configuration

Best measured deployable result:
- `MODEL_IMPL=optimized_cached`
- `POLICY=longest_input_first`
- `DECODE_MODE=compact`
- `BATCH_SIZE=16`
- `REPLICA_ASSIGNMENT=round_robin`
- Serving time: `163.306s` on repeat, `164.075s` on first run
- Serving throughput: about `78 requested output tok/s`

Speedup versus `naive + random + static + batch_size=8`:
- Using repeat time `163.306s`: `29.90x`
- Serving-time reduction: `96.66%`

This improves the previous best (`165.990s`) modestly, but the improvement is small enough that it should be presented as an incremental scheduling/batching improvement rather than a major new result.

## Remaining Optimization Directions

- Implement true continuous batching only after refactoring the KV cache to support per-row cache positions or paged cache blocks.
- If staying within the current cache design, test additional B16 compact repeats and perhaps `BATCH_SIZE=12` if the benchmark code is extended to support non-power-of-two operational choices.
- Tensor parallelism remains a larger future direction; it could improve per-token latency and GPU utilization, but it is outside this static-batch scheduler patch.
