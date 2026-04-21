# Serving Scheduling Benchmark Report

Timestamp: 2026-04-20 22:20 EDT

## Configuration

- Cluster node observed through browser console: `gpu003`
- Hardware: `8 x H200`
- Model preset: `target_40b_approx`
- Approximate parameter count: `39,978,323,440` (`~39.98B`)
- Serving workload: `100` requests
- Input tokens: mean `128`, max `1024`, deterministic long-tailed workload
- Output tokens: mean `128`, max `1024`, deterministic long-tailed workload
- Total requested input tokens: `12,800`
- Total requested output tokens: `12,800`
- Execution layout: two data-parallel replicas, each using a 4-GPU model-parallel group
- Device groups: `0,1,2,3` and `4,5,6,7`
- Batch size per replica: `8`
- Effective concurrent batch size: `16`
- dtype: `bfloat16`

## Commands

Naive baseline:

```bash
MODEL_IMPL=naive \
POLICY=random \
BATCH_SIZE=8 \
DEVICE_GROUPS="0,1,2,3 4,5,6,7" \
OUTPUT=results/serving_scheduling_target_40b_naive_model_random_b8.json \
bash scripts/run_cluster_serving_scheduling.sh
```

Optimized input-known scheduler:

```bash
MODEL_IMPL=optimized_cached \
POLICY=longest_input_first \
BATCH_SIZE=8 \
DEVICE_GROUPS="0,1,2,3 4,5,6,7" \
OUTPUT=results/serving_scheduling_target_40b_optimized_input_known_b8.json \
bash scripts/run_cluster_serving_scheduling.sh
```

Oracle scheduler curiosity run:

```bash
MODEL_IMPL=optimized_cached \
POLICY=longest_output_first \
BATCH_SIZE=8 \
DEVICE_GROUPS="0,1,2,3 4,5,6,7" \
OUTPUT=results/serving_scheduling_target_40b_optimized_oracle_b8.json \
bash scripts/run_cluster_serving_scheduling.sh
```

## Results

| Run | Model | Scheduler | Policy | Serving wall, excl. load (s) | Total wall, incl. load (s) | Serving requested tok/s | Total requested tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Naive baseline | `naive` | `naive_random` | `random` | `4882.76` | `5007.12` | `2.62` | `2.56` |
| Optimized realistic | `optimized_cached` | `input_known` | `longest_input_first` | `165.99` | `280.49` | `77.11` | `45.63` |
| Optimized oracle | `optimized_cached` | `oracle` | `longest_output_first` | `119.57` | `244.18` | `107.05` | `52.44` |

## Improvement

- Optimized input-known vs naive/random, excluding model load: `29.42x` speedup and `96.60%` lower serving time.
- Optimized input-known vs naive/random, including model load: `17.85x` speedup and `94.40%` lower total wall time.
- Oracle vs naive/random, excluding model load: `40.84x` speedup and `97.55%` lower serving time.
- Oracle vs optimized input-known, excluding model load: `1.39x` speedup and `27.97%` lower serving time.
- Oracle vs optimized input-known, including model load: `1.15x` speedup and `12.94%` lower total wall time.

## Schedule Padding

Naive random schedule:

```text
prefill_padding_overhead = 3.2459375
decode_padding_overhead = 3.405
padded_prefill_tokens = 41548
padded_decode_tokens = 43584
```

Optimized input-known schedule:

```text
prefill_padding_overhead = 1.33
decode_padding_overhead = 3.3278125
padded_prefill_tokens = 17024
padded_decode_tokens = 42596
```

Optimized oracle schedule:

```text
prefill_padding_overhead = 3.325625
decode_padding_overhead = 1.3325
padded_prefill_tokens = 42568
padded_decode_tokens = 17056
```

## Interpretation

The proposal-relevant comparison is the naive baseline versus the optimized realistic run. That compares the naive model plus random data-parallel scheduler against the optimized cached model plus input-known scheduler. The optimized run is materially faster because it combines cached decode with a scheduler that reduces prefill padding using only known input lengths.

The oracle run is not a deployable scheduler because it sorts by output length, which a real system does not know before generation. It is still useful as an upper-bound measurement: most of its additional win comes from reducing decode padding from `3.33x` to `1.33x`.

The optimized input-known policy remains the right target for optimization. The oracle result suggests that decode-length heterogeneity is still a large source of scheduling waste, but using true output lengths is not realistic.
