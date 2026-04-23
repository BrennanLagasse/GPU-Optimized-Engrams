# Focused Batch Size Sweep Report

Timestamp: 2026-04-22 21:55 EDT

## Summary

We ran a focused non-power-of-two batch-size sweep after B32 and B64 bracketed the likely optimum for the realistic serving benchmark. The best measured configuration is now:

| Model | Scheduler | Replica assignment | Decoder | Batch/replica | Serving seconds | Tok/s | Speedup vs naive | Time reduction vs naive |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 48 | `110.087` | `116.272` | `44.35x` | `97.75%` |

This improves the previous B32 greedy-prefill compact best from `131.728s` to `110.087s`, a `16.43%` serving-time reduction. It improves the original naive random static B8 baseline from `4882.758s` to `110.087s`, a `44.35x` speedup.

## Methodology

Hardware and model:

| Field | Value |
| --- | --- |
| Node | `gpu003` |
| GPUs | `8 x NVIDIA H200` |
| Execution layout | two 4-GPU model-parallel replicas |
| Model preset | `target_40b_approx` |
| Approx params | `39.98B` |
| dtype | `bfloat16` |
| Engram layers | `[1, 15]` under 0-indexed layer numbering |

Workload:

| Field | Value |
| --- | ---: |
| Requests | `100` |
| Mean input tokens | `128` |
| Mean output tokens | `128` |
| Max input tokens | `1024` |
| Max output tokens | `1024` |
| Requested output tokens | `12,800` |

Input and output lengths are independent deterministic long-tailed draws, except request `0` is forced to have both max input and max output length. This preserves the realistic serving assumption that input length is known before scheduling while output length is unknown until generation completes.

All focused cases used:

| Category | Setting |
| --- | --- |
| Model implementation | `optimized_cached` |
| Scheduler | `longest_input_first` |
| Replica assignment | `greedy_prefill` |
| Decoder | `compact` |

## Results

Baseline for relative changes:

| Baseline | Seconds |
| --- | ---: |
| Naive random static B8 | `4882.758` |
| Previous B32 greedy compact best | `131.728` |

| Model | Scheduler | Replica assignment | Decoder | Batch/replica | Serving seconds | Tok/s | Speedup vs naive | Time reduction vs naive | Change vs B32 greedy compact |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 24 | `167.809` | `76.279` | `29.10x` | `96.56%` | `27.39% slower` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 28 | `163.321` | `78.373` | `29.90x` | `96.66%` | `23.98% slower` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 32 | `131.728` | `97.170` | `37.07x` | `97.30%` | baseline |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 36 | `125.194` | `102.242` | `39.00x` | `97.44%` | `4.96% faster` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 40 | `121.943` | `104.967` | `40.04x` | `97.50%` | `7.43% faster` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 48 | `110.087` | `116.272` | `44.35x` | `97.75%` | `16.43% faster` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 56 | `119.096` | `107.476` | `41.00x` | `97.56%` | `9.59% faster` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 64 | `132.517` | `96.591` | `36.85x` | `97.29%` | `0.60% slower` |

## Interpretation

B48 appears to be the best measured point for this closed 100-request benchmark. Smaller non-power-of-two values, B24 and B28, do not help. B36 and B40 improve over B32, B48 improves further, and B56/B64 regress. That pattern suggests the improvement is a real batch-size tradeoff rather than a one-off timing artifact.

The likely mechanism is that B48 gives better GPU utilization than B32 while still keeping enough batches to use both replicas and enough active-row compaction benefit to avoid the worst static decode padding. B64 gets close but appears to lose enough replica balance and/or compaction efficiency to fall behind B48. B128 was not run because the 100-request workload would collapse into one effective batch, leaving one 4-GPU replica idle.

## Command

Best measured configuration:

```bash
MODEL_IMPL=optimized_cached \
POLICY=longest_input_first \
REPLICA_ASSIGNMENT=greedy_prefill \
DECODE_MODE=compact \
BATCH_SIZE=48 \
OUTPUT=results/serving_opt_sweep_optimized_input_compact_b48_greedy_prefill.json \
bash scripts/run_cluster_serving_scheduling.sh
```

Focused sweep template:

```bash
for batch in 24 28 36 40 48 56; do
  MODEL_IMPL=optimized_cached \
  POLICY=longest_input_first \
  REPLICA_ASSIGNMENT=greedy_prefill \
  DECODE_MODE=compact \
  BATCH_SIZE="$batch" \
  OUTPUT="results/serving_opt_sweep_optimized_input_compact_b${batch}_greedy_prefill.json" \
  bash scripts/run_cluster_serving_scheduling.sh
done
```

## Related Reports

- [best_serving_results_report_2026-04-22.md](best_serving_results_report_2026-04-22.md)
- [large_batch_sweep_report_2026-04-22.md](large_batch_sweep_report_2026-04-22.md)
- [prefill_decode_stage_costs_b32_greedy_2026-04-22.md](prefill_decode_stage_costs_b32_greedy_2026-04-22.md)
