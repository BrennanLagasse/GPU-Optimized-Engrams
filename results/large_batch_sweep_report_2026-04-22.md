# Large Batch And Stage-Cost Sweep Report

Timestamp: 2026-04-22 21:55 EDT

## Summary

The best measured realistic serving result is now:

| Model | Scheduler | Replica assignment | Decoder | Batch/replica | Serving seconds | Tok/s | Speedup vs naive | Time reduction vs naive | Change vs prior best B16 compact |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 48 | `110.087` | `116.272` | `44.35x` | `97.75%` | `32.59% faster` |

The prior best before the focused non-power-of-two sweep was `optimized_cached + longest_input_first + greedy_prefill + compact + B32` at `131.728s`. The new B48 greedy-prefill compact result improves that by `16.43%`.

The original naive baseline remains `naive + random + round_robin + static + B8` at `4882.758s`, so the new best end-to-end speedup is:

```text
4882.758s / 110.087s = 44.35x
```

## Methodology

Hardware and model:

| Field | Value |
| --- | --- |
| Node | `gpu003` |
| GPUs | `8 x NVIDIA H200`, `143,771 MiB` each |
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

Input and output lengths were generated from independent deterministic long-tailed draws, except request `0` was forced to have both max input and max output length. This preserves the realistic constraint that input lengths are known before scheduling, while output lengths are not known until generation completes.

## Batch Sweep Results

Baseline for percent changes in this table:

| Baseline | Seconds |
| --- | ---: |
| Naive random static B8 | `4882.758` |
| Prior best B16 compact | `163.306` |

| Model | Scheduler | Replica assignment | Decoder | Batch/replica | Serving seconds | Tok/s | Speedup vs naive | Time reduction vs naive | Change vs prior best B16 compact |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `optimized_cached` | `longest_input_first` | `round_robin` | `static` | 8 | `165.990` | `77.113` | `29.42x` | `96.60%` | `1.64% slower` |
| `optimized_cached` | `longest_input_first` | `round_robin` | `compact` | 16 | `163.306` | `78.381` | `29.90x` | `96.66%` | baseline |
| `optimized_cached` | `longest_input_first` | `round_robin` | `static` | 32 | `359.340` | `35.621` | `13.59x` | `92.64%` | `120.04% slower` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `static` | 32 | `243.960` | `52.468` | `20.01x` | `95.00%` | `49.39% slower` |
| `optimized_cached` | `longest_input_first` | `round_robin` | `compact` | 32 | `177.720` | `72.023` | `27.47x` | `96.36%` | `8.83% slower` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 32 | `131.728` | `97.170` | `37.07x` | `97.30%` | `19.34% faster` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 36 | `125.194` | `102.242` | `39.00x` | `97.44%` | `23.34% faster` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 40 | `121.943` | `104.967` | `40.04x` | `97.50%` | `25.33% faster` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 48 | `110.087` | `116.272` | `44.35x` | `97.75%` | `32.59% faster` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 56 | `119.096` | `107.476` | `41.00x` | `97.56%` | `27.07% faster` |
| `optimized_cached` | `longest_input_first` | `round_robin` | `continuous` | 32 | `186.199` | `68.744` | `26.22x` | `96.19%` | `14.02% slower` |
| `optimized_cached` | `longest_input_first` | `round_robin` | `compact` | 64 | `132.731` | `96.436` | `36.79x` | `97.28%` | `18.72% faster` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 64 | `132.517` | `96.591` | `36.85x` | `97.29%` | `18.85% faster` |
| `optimized_cached` | `longest_input_first` | `round_robin` | `continuous` | 64 | `156.372` | `81.856` | `31.23x` | `96.80%` | `4.25% faster` |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `static` | 64 | `446.126` | `28.691` | `10.94x` | `90.86%` | `173.18% slower` |

Interpretation:

- The focused non-power-of-two sweep found that B48 compact with greedy-prefill is the measured optimum among the tested realistic configurations.
- B56 and B64 compact are slower than B48 compact, so increasing batch size past B48 decreased serving speed for this workload.
- B128 was not run because B64 already decreased and `100` requests with `BATCH_SIZE=128` would collapse into a single effective batch, leaving one replica idle.
- Static large-batch decode is poor because decode padding and replica imbalance dominate.
- Greedy-prefill matters more at B32 than it did at B8 because there are fewer, larger batches; one bad batch assignment can dominate makespan.

## Focused Non-Power-Of-Two Sweep

After B32 and B64 bracketed the likely optimum, we ran a focused sweep of non-power-of-two compact greedy-prefill batch sizes. This directly tested whether a value between B32 and B64 could better trade off GPU utilization, prefill padding, decode compaction overhead, and replica balance.

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

Interpretation:

- B48 is the best measured point in the focused sweep.
- B36 and B40 show that the gain is not just noise at one value; performance improves as batch size increases past B32, peaks at B48, and then falls at B56/B64.
- B24/B28 are near the old B16 compact result and are worse than B32, which suggests the main benefit is not simply using any non-power-of-two value. The useful region is the neighborhood where larger batches improve utilization without creating too much remaining imbalance or compaction overhead.

## Prefill/Decode Disaggregation Experiment

The earlier architecture simulation suggested prefill/decode disaggregation could be promising, so the benchmark now records measured per-batch `prefill_seconds` and `decode_seconds` for optimized static/compact serving. This lets us run a measured stage-cost experiment using the actual H200 model path.

Source result:

| Model | Scheduler | Replica assignment | Decoder | Batch/replica | Serving seconds |
| --- | --- | --- | --- | ---: | ---: |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 32 | `131.728` |

Measured stage-cost estimate:

| Metric | Seconds | Relative to measured serving time |
| --- | ---: | ---: |
| Sum of measured prefill stages | `9.574` | `0.07x` |
| Sum of measured decode stages | `220.153` | `1.67x` |
| Estimated staged pipeline makespan | `227.832` | `1.73x` |

| Baseline | Candidate | Baseline seconds | Candidate seconds | Speedup | Serving-time reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| measured current serving | staged prefill/decode estimate | `131.728` | `227.832` | `0.58x` | `-72.96%` |

Interpretation:

- For this closed 100-request workload, the measured stage-cost experiment does not support prefill/decode disaggregation as an immediate win.
- The workload is decode-dominated after B32 compact scheduling; measured decode stage time is much larger than measured prefill stage time.
- The staged estimate also gives up the data-parallel benefit of two full serving replicas, so it is slower despite stage separation.
- A production prefill/decode implementation could still matter for arrival-process workloads, TTFT, or workloads with much heavier prefill, but it is not the best next optimization for this fixed closed-batch benchmark.

## Commands

Best measured configuration:

The single command below is sufficient to reproduce the reported best result. You do not need to run any separate setup block beforehand for the documented default settings.

```bash
MODEL_IMPL=optimized_cached \
POLICY=longest_input_first \
REPLICA_ASSIGNMENT=greedy_prefill \
DECODE_MODE=compact \
BATCH_SIZE=48 \
OUTPUT=results/serving_opt_sweep_optimized_input_compact_b48_greedy_prefill.json \
bash scripts/run_cluster_serving_scheduling.sh
```

Focused non-power-of-two sweep command template:

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

These commands rely on the same internal defaults from `scripts/run_cluster_serving_scheduling.sh` that are documented in [best_serving_results_report_2026-04-22.md](best_serving_results_report_2026-04-22.md). You only need to set additional environment variables if you want to override those defaults explicitly.

Stage-cost report:

```bash
python scripts/report_prefill_decode_stage_costs.py \
  results/serving_opt_sweep_optimized_input_compact_b32_greedy_prefill.json \
  --output results/prefill_decode_stage_costs_b32_greedy_2026-04-22.md
```

## Related Reports

- [cached_engram_ablation_report_2026-04-21.md](cached_engram_ablation_report_2026-04-21.md)
- [proposal_checklist.md](proposal_checklist.md)
- [paper_metrics_summary.md](paper_metrics_summary.md)
- [best_serving_results_report_2026-04-22.md](best_serving_results_report_2026-04-22.md)
- [focused_batch_size_sweep_report_2026-04-22.md](focused_batch_size_sweep_report_2026-04-22.md)
- [prefill_decode_stage_costs_b32_greedy_2026-04-22.md](prefill_decode_stage_costs_b32_greedy_2026-04-22.md)
