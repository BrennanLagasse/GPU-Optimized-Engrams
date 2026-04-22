# Best Serving Results Report

Timestamp: 2026-04-22 13:23 EDT

## Executive Summary

Our best measured realistic result is a `29.90x` speedup over the naive baseline on a 100-request heterogeneous serving workload for the `target_40b_approx` model preset.

Best configuration:

- model implementation: `optimized_cached`
- scheduler: `longest_input_first`
- decode mode: `compact`
- batch size per replica: `16`
- execution layout: two 4-GPU replicas across `8 x NVIDIA H200`
- serving time excluding model load and worker startup: `163.306s`
- requested output throughput: `78.381 tokens/s`
- requested output tokens served: `12,800`

Baseline:

- model implementation: `naive`
- scheduler: `random`
- decode mode: `static`
- batch size per replica: `8`
- serving time excluding model load and worker startup: `4882.758s`
- requested output throughput: `2.621 tokens/s`

The strongest result is not from one optimization alone. The dominant gain comes from moving from naive full-context recomputation to cached optimized inference. Input-aware scheduling and B16 active-row compaction provide additional measured gains.

## Project Scope

This project evaluates GPU-oriented inference and serving optimizations for an Engram/mHC-style transformer implementation. The current benchmark is not a production online serving benchmark with arrivals over time. It is a closed 100-request batch designed to stress heterogeneous request scheduling while keeping the benchmark reproducible.

The project scope is purely inference speed. The goal is not to improve model quality, task accuracy, or downstream performance. For this reason, random weights are sufficient for the speed experiments as long as the naive and optimized implementations have matching behavior under the same weights and inputs.

The goal of this phase was to answer:

- Can the optimized implementation match the naive implementation's outputs?
- Can the optimized serving path run at target scale on the cluster?
- How much speedup do we obtain on a 40B-class model?
- Which parts of the speedup come from model-path optimization versus scheduling?
- Which realistic scheduler improvements help, and which only help in oracle settings?

Out of scope for the best-result claim:

- quality evaluation of generated text
- training or fine-tuning
- production arrival-process serving
- tensor-parallel kernels beyond the current model-parallel layout
- claiming oracle output-length scheduling as deployable

## Hardware And Model Scale

Hardware:

- cluster node: `gpu003`
- GPUs: `8 x NVIDIA H200`
- reported memory: `143,771 MiB` per H200
- execution layout: two model-parallel replicas
- replica 0: `cuda:0,cuda:1,cuda:2,cuda:3`
- replica 1: `cuda:4,cuda:5,cuda:6,cuda:7`

Model:

- preset: `target_40b_approx`
- approximate parameter count: `39,978,323,440`, about `39.98B`
- dtype: `bfloat16`
- Engram layer placement: `[1, 15]`, meaning the second and sixteenth transformer layers under 0-indexed layer numbering
- rough compute framing: a dense transformer forward pass is commonly approximated as about `2 x parameter_count` multiply-add FLOPs per generated token, so this model is roughly `80 GFLOP/token` before attention/KV-cache and Engram-specific overheads. This is a sizing heuristic, not a profiler-derived FLOP count.

## Workload Methodology

The workload contains `100` deterministic synthetic requests.

Input lengths:

- long-tailed distribution
- mean `128` tokens
- max `1024` tokens
- at least one request reaches the max length

Output lengths:

- long-tailed distribution
- mean `128` tokens
- max `1024` tokens
- at least one request reaches the max length

Input and output lengths were generated from separate deterministic long-tailed draws, so they are independent except for one deliberate stress-test constraint: request `0` is forced to have both the maximum input length and the maximum output length. This guarantees that the workload contains at least one worst-case request with `1024` input tokens and `1024` output tokens.

Total requested output tokens:

- `12,800`

Why synthetic long-tailed lengths:

- Real serving traffic is heterogeneous.
- Some requests are short, while a small number of requests are much longer.
- Static batches suffer when short requests are grouped with long requests.
- The deterministic generator makes repeated experiments comparable.

Important realism constraint:

- Input lengths are known before scheduling.
- Output lengths are not known before generation completes.
- Therefore `longest_input_first` is a realistic policy, but `longest_output_first` is an oracle diagnostic only.

## Metric Methodology

Primary metric:

- serving wall seconds excluding model load and worker startup

Why exclude model load and startup:

- The benchmark compares serving-loop/model/scheduler behavior.
- Model load time is largely fixed setup overhead and can vary with cache state, filesystem state, and process startup.
- Excluding load gives a clearer comparison of throughput once serving is active.

Secondary metrics:

- requested output tokens per second
- executed decode tokens
- prefill padding overhead
- decode padding overhead
- total wall seconds including worker startup, when available

The headline `29.90x` is computed from serving time. The numerator is the naive baseline serving time: `naive + random + static + BATCH_SIZE=8`, measured at `4882.758s`. The denominator is the best realistic optimized serving time: `optimized_cached + longest_input_first + compact + BATCH_SIZE=16`, measured at `163.306s`.

```text
4882.758s / 163.306s = 29.90x
```

## Reproduction Command

Best measured realistic configuration:

```bash
MODEL_IMPL=optimized_cached \
POLICY=longest_input_first \
DECODE_MODE=compact \
BATCH_SIZE=16 \
REPLICA_ASSIGNMENT=round_robin \
OUTPUT=results/serving_opt_sweep_optimized_input_compact_b16_repeat1.json \
bash scripts/run_cluster_serving_scheduling.sh
```

Default cluster layout used by the wrapper:

```bash
DEVICE_GROUPS="0,1,2,3 4,5,6,7"
PRESET=target_40b_approx
DTYPE=bfloat16
NUM_REQUESTS=100
MEAN_INPUT_TOKENS=128
MEAN_OUTPUT_TOKENS=128
MAX_INPUT_TOKENS=1024
MAX_OUTPUT_TOKENS=1024
SEED=0
```

## Best Result Table

| Case | Model | Scheduler | Decode | Batch/replica | Serving seconds | Requested tok/s | Speedup vs naive random static |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Best realistic | `optimized_cached` | `longest_input_first` | `compact` | 16 | `163.306` | `78.381` | `29.90x` |
| Previous best realistic | `optimized_cached` | `longest_input_first` | `static` | 8 | `165.990` | `77.113` | `29.42x` |
| Naive baseline | `naive` | `random` | `static` | 8 | `4882.758` | `2.621` | `1.00x` |

The best result improves the previous best by:

```text
(165.990s - 163.306s) / 165.990s = 1.62% serving-time reduction
```

This is a modest incremental improvement over the already-optimized B8 static result, but the full optimized bundle remains a large improvement over the naive baseline.

## What Each Methodological Choice Means

### `naive`

The naive implementation is the correctness-oriented baseline. It recomputes the model over the full context during generation. It is intentionally not optimized for serving speed.

Purpose:

- establishes a simple baseline
- helps validate optimized implementation parity
- makes the cost of missing KV-cache reuse visible

### `optimized_cached`

The optimized implementation uses cached inference, optimized Engram/mHC paths, and model-parallel placement. In serving, the most important improvement is avoiding full-context recomputation during decoding.

Purpose:

- represents the deployable optimized path
- preserves correctness against the naive implementation in tests
- enables target-scale 40B-class experiments

### `random`

Random scheduling is the intentionally weak baseline. It does not use input length, output length, or any workload structure.

Purpose:

- shows what happens without request-aware scheduling
- gives a conservative baseline for measuring scheduling improvements

### `longest_input_first`

This policy sorts by known input length. It is realistic because a server knows prompt length before prefill.

Purpose:

- reduces prefill padding
- keeps long prompts grouped together
- avoids using unavailable output-length information

### `longest_output_first`

This policy sorts by true output length. It is not deployable in a realistic serving system because the output length is only known after generation finishes.

Purpose:

- upper-bound diagnostic
- helps estimate how much static-batch waste comes from decode-length imbalance
- should not be presented as a realistic production result

### `static` Decode

Static decode keeps every row in a batch alive until the longest request in that batch finishes. This is simple and efficient for fixed tensor shapes, but it performs padded decode work for rows that are already complete.

Purpose:

- stable baseline
- high GPU utilization
- exposes padding waste in heterogeneous workloads

### `compact` Decode

Compact decode removes rows after they complete and compacts model caches with the remaining active row indices. This is realistic because it only uses completion information observed online.

Purpose:

- avoids decode work for completed rows
- executes exactly the requested `12,800` decode-token steps
- tests whether reduced work outweighs cache-indexing and tensor-shape overhead

For B8, compact was slower than static. For B16, compact became the best result because the larger batch created enough padding waste for compaction to pay off.

## Ablations And Attribution

### Main Attribution Matrix

| Comparison | Serving seconds | Speedup | Serving-time reduction | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Naive random static -> optimized random static | `4882.758 -> 192.714` | `25.34x` | `96.05%` | Model-path optimization dominates the total speedup. |
| Optimized random static -> optimized input-known static | `192.714 -> 165.990` | `1.16x` | `13.87%` | Realistic input-aware scheduling matters. |
| Optimized input-known static B8 -> optimized input-known compact B16 | `165.990 -> 163.306` | `1.016x` | `1.62%` | B16 compaction adds a small final gain. |
| Naive random static -> best realistic result | `4882.758 -> 163.306` | `29.90x` | `96.66%` | Full realistic optimized bundle. |

### Scheduler Ablation

At B8 static:

| Case | Serving seconds | Tok/s |
| --- | ---: | ---: |
| `optimized_cached + random` | `192.714` | `66.420` |
| `optimized_cached + longest_input_first` | `165.990` | `77.113` |
| `optimized_cached + longest_output_first` oracle | `119.568` | `107.052` |
| `optimized_cached + longest_output_first + greedy_oracle` oracle | `109.873` | `116.498` |

Interpretation:

- Realistic input-known scheduling improves serving time by `13.87%` on the optimized model.
- Oracle output-length scheduling is much stronger, but not deployable.
- The oracle result shows that output-length imbalance is a real source of waste, but the true output length is unavailable in realistic generation.

### Generic Cache vs Engram-Specific Step Kernel

The cached Engram ablation separated generic KV-cached serving from the Engram-specific cached `step_kernel`.

| Comparison | Serving seconds | Speedup | Serving-time reduction |
| --- | ---: | ---: | ---: |
| Naive input-known -> cached full Engram input-known | `3629.573 -> 167.351` | `21.69x` | `95.39%` |
| Cached full Engram input-known -> optimized cached input-known | `167.351 -> 165.990` | `1.01x` | `0.81%` |

Interpretation:

- Most of the 40B serving gain comes from generic cached serving, not Engram-specific micro-optimization.
- The Engram `step_kernel` is exact and useful, but in this end-to-end batched benchmark its contribution is small.

### Static vs Compact

| Case | Static seconds | Compact seconds | Decode tokens static | Decode tokens compact | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| naive random B8 | `4882.758` | `1079.463` | `43,584` | `12,800` | Compact greatly helps naive. |
| optimized input-known B8 | `165.990` | `182.102` | `42,596` | `12,800` | Compact hurts at B8. |
| optimized input-known B16 | `183.841` | `163.306` | `59,660` | `12,800` | Compact wins at B16. |

Interpretation:

- Compact always reduces executed decode work to the requested token count.
- For optimized cached inference, the overhead of cache compaction can outweigh saved compute at smaller batches.
- At B16, static padding waste grows enough that compaction becomes beneficial.

### Batch Size Sweep

| Case | Decode | Batch/replica | Serving seconds | Tok/s |
| --- | --- | ---: | ---: | ---: |
| optimized input B1 | static | 1 | `253.385` | `50.516` |
| optimized input B2 | static | 2 | `231.256` | `55.350` |
| optimized input B4 | static | 4 | `176.799` | `72.399` |
| optimized input B8 | static | 8 | `165.990` | `77.113` |
| optimized input B16 | static | 16 | `183.841` | `69.625` |
| optimized input B16 | compact | 16 | `163.306` | `78.381` |

Interpretation:

- Very small batches avoid padding but underutilize the GPUs.
- B8 static is a strong utilization/padding tradeoff.
- B16 static over-pads decode.
- B16 compact recovers enough decode waste to become the best measured realistic configuration.

### Continuous Refill

Continuous refill was implemented and measured after the best compact result.

| Case | Serving seconds | Tok/s | Delta vs best compact |
| --- | ---: | ---: | ---: |
| continuous B8, `longest_input_first` | `167.398` | `76.464` | `2.51% slower` |
| continuous B16, `longest_input_first` | `164.889` | `77.628` | `0.97% slower` |
| continuous B16, oracle `longest_output_first` | `216.314` | `59.173` | `32.46% slower` |

Interpretation:

- Continuous refill is now implemented and benchmarkable.
- It did not beat B16 compact on this closed 100-request workload.
- Likely causes are exact per-request prefill, slot-indexed cache overhead, Engram short-conv slot state, and Python scheduler bookkeeping.
- This does not invalidate continuous batching generally; it means this implementation and workload did not improve over the simpler best compact path.

## Assumptions And Limitations

Assumptions:

- Input length is known before scheduling.
- Output length is unknown before generation and should not be used for deployable scheduling.
- The workload is closed: all 100 requests are available at the start.
- Input and output lengths are independently generated except for the single forced max-input/max-output stress request.
- The synthetic long-tailed distribution is a proxy for heterogeneous serving traffic.
- Serving time excluding model load is the correct primary metric for comparing serving loop performance.

Limitations:

- The workload does not model real arrival times, queueing delay, or time-to-first-token.
- Output lengths are deterministic targets, not produced by semantic EOS behavior.
- The benchmark measures throughput/makespan, not quality.
- Random model weights are acceptable for this benchmark because the project is evaluating inference speed and implementation equivalence, not model accuracy.
- The implementation is research code, not a production serving stack.
- Tensor parallelism, paged KV, and production continuous batching are not fully explored.
- Some apparent small differences near 1% should be interpreted cautiously because cluster scheduling and GPU runtime noise can matter at that scale.

## What We Can Claim

Strong claim:

- On the fixed 100-request 40B-class H200 serving benchmark, the best realistic optimized configuration is `29.90x` faster than the naive random static baseline.

More precise claim:

- The best measured realistic configuration is `optimized_cached + longest_input_first + compact + BATCH_SIZE=16`, with `163.306s` serving time and `78.381 requested output tok/s`.
- The benchmark uses random weights because the result is an inference-speed comparison, not a quality or accuracy claim.

Attribution claim:

- The largest contribution comes from cached optimized inference versus naive full-context recompute.
- Realistic input-known scheduling contributes a meaningful additional improvement.
- B16 active-row compaction contributes a small final improvement.
- Engram-specific cached step-kernel optimization is exact but not the dominant end-to-end speedup source at 40B scale.

Do not claim:

- Oracle output-length scheduling is deployable.
- Continuous refill is currently better than compact for this workload.
- The result is a production online serving benchmark.
- The `80 GFLOP/token` sizing heuristic is a measured FLOP profile.

## Recommended Presentation

The clearest way to present the result is:

> We implemented a naive correctness baseline and an optimized cached Engram/mHC serving path, then benchmarked both on a deterministic 100-request long-tailed workload using a 40B-class model across 8 H200 GPUs. The best realistic configuration served 12,800 requested output tokens in 163.306 seconds, achieving a 29.90x speedup over the naive random static baseline. Ablations show that most of the speedup comes from cached optimized inference, with additional gains from input-length-aware scheduling and B16 active-row compaction.

This statement is accurate, strong, and does not overclaim oracle or production-online serving results.
