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

The benchmark has three separate categories of choices:

- model implementation: what code path computes logits
- scheduler: how requests are ordered and assigned to batches/replicas
- decoder/runtime mode: how finished rows are handled during generation

These categories should not be compared as alternatives to each other. For example, `optimized_cached` is a model implementation, while `longest_input_first` is a scheduler. The best result combines one choice from each category.

### Model Implementation Choices

#### `naive`

The naive implementation is the correctness-oriented baseline. It recomputes the model over the full context during generation. It is intentionally not optimized for serving speed.

Purpose:

- establishes a simple baseline
- helps validate optimized implementation parity
- makes the cost of missing KV-cache reuse visible

#### `optimized_cached`

The optimized implementation uses the same high-level model behavior as the naive implementation, but changes the inference runtime. The most important optimization is cached incremental decode: instead of recomputing the whole context at every generated token, attention layers reuse cached keys/values and process only the new token during steady-state decode.

Optimizations included relative to `naive`:

- KV-cached incremental attention for decode.
- `torch.inference_mode()` around generation/benchmark paths to remove autograd overhead.
- Tensorized Engram hash preparation paths, including cached last-token hash preparation during decode.
- Per-device Engram hash preparation for model-parallel placement so Engram blocks receive hashes on the right device.
- Exact cached Engram `ShortConv.forward_step()` path, also called the Engram `step_kernel`, for single-token cached decode.
- Model-parallel execution via `device_map`, with contiguous layer placement across GPU groups.
- Stage-aware execution to avoid redundant device transfers across blocks already on the same device.
- Weighted contiguous placement heuristics that account for Engram-heavy early blocks.
- mHC execution in the optimized model path with the same `hc_mult=4` structure used for the target preset.

The target-scale ablations show that the dominant speedup comes from generic KV-cached incremental decode, not from the Engram-specific `step_kernel`. The Engram-specific cached step is exact and kept in the optimized path, but its end-to-end contribution at 40B serving scale is small.

Purpose:

- represents the deployable optimized path
- preserves correctness against the naive implementation in tests
- enables target-scale 40B-class experiments

Related up-to-date references:

- [cached_engram_ablation_report_2026-04-21.md](/Users/vincentli/Desktop/GPU-Optimized-Engrams/results/cached_engram_ablation_report_2026-04-21.md)
- [proposal_checklist.md](/Users/vincentli/Desktop/GPU-Optimized-Engrams/results/proposal_checklist.md)
- [paper_metrics_summary.md](/Users/vincentli/Desktop/GPU-Optimized-Engrams/results/paper_metrics_summary.md)

### Scheduler Choices

#### `random`

Random scheduling is the intentionally weak baseline. It does not use input length, output length, or any workload structure.

Purpose:

- shows what happens without request-aware scheduling
- gives a conservative baseline for measuring scheduling improvements

#### `longest_input_first`

This policy sorts by known input length. It is realistic because a server knows prompt length before prefill.

Purpose:

- reduces prefill padding
- keeps long prompts grouped together
- avoids using unavailable output-length information

#### `longest_output_first`

This policy sorts by true output length. It is not deployable in a realistic serving system because the output length is only known after generation finishes. It is useful as an oracle diagnostic because it batches responses of similar length together, reducing decode padding in static batches.

Purpose:

- upper-bound diagnostic
- helps estimate how much static-batch waste comes from decode-length imbalance
- should not be presented as a realistic production result

### Decoder/Runtime Choices

#### `static` Decode

Static decode keeps every row in a batch alive until the longest request in that batch finishes. This is simple and efficient for fixed tensor shapes, but it performs padded decode work for rows that are already complete.

Purpose:

- stable baseline
- high GPU utilization
- exposes padding waste in heterogeneous workloads

#### `compact` Decode

Compact decode removes rows after they complete and compacts model caches with the remaining active row indices. This is realistic because it only uses completion information observed online.

Purpose:

- avoids decode work for completed rows
- executes exactly the requested `12,800` decode-token steps
- tests whether reduced work outweighs cache-indexing and tensor-shape overhead

For B8, compact was slower than static. For B16, compact became the best result because the larger batch created enough padding waste for compaction to pay off.

## Ablations And Attribution

### Main Attribution Matrix

| Comparison | Before model | Before scheduler | Before decoder | Before batch | After model | After scheduler | After decoder | After batch | Serving seconds | Speedup | Serving-time reduction | Interpretation |
| --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Model path | `naive` | `random` | `static` | 8 | `optimized_cached` | `random` | `static` | 8 | `4882.758 -> 192.714` | `25.34x` | `96.05%` | Model-path optimization dominates the total speedup. |
| Realistic scheduler | `optimized_cached` | `random` | `static` | 8 | `optimized_cached` | `longest_input_first` | `static` | 8 | `192.714 -> 165.990` | `1.16x` | `13.87%` | Reducing prefill padding gives a clear gain. |
| B16 compaction | `optimized_cached` | `longest_input_first` | `static` | 8 | `optimized_cached` | `longest_input_first` | `compact` | 16 | `165.990 -> 163.306` | `1.016x` | `1.62%` | Larger batch plus compaction adds a small final gain. |
| Full realistic bundle | `naive` | `random` | `static` | 8 | `optimized_cached` | `longest_input_first` | `compact` | 16 | `4882.758 -> 163.306` | `29.90x` | `96.66%` | Combined model, scheduler, and decoder/runtime result. |

### Scheduler Ablation

At B8 static:

| Model | Scheduler | Replica assignment | Decoder | Batch/replica | Serving seconds | Tok/s | Speedup vs optimized random | Serving-time reduction |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `optimized_cached` | `random` | `round_robin` | `static` | 8 | `192.714` | `66.420` | `1.00x` | `0.00%` |
| `optimized_cached` | `longest_input_first` | `round_robin` | `static` | 8 | `165.990` | `77.113` | `1.16x` | `13.87%` |
| `optimized_cached` | `longest_output_first` oracle | `round_robin` | `static` | 8 | `119.568` | `107.052` | `1.61x` | `37.96%` |
| `optimized_cached` | `longest_output_first` oracle | `greedy_oracle` | `static` | 8 | `109.873` | `116.498` | `1.75x` | `42.99%` |

Interpretation:

- Realistic input-known scheduling improves serving time by `13.87%` on the optimized model, which suggests reducing prefill padding is valuable.
- Moving from input-known scheduling to oracle output-length scheduling reduces serving time from `165.990s` to `119.568s`, a further `27.97%` reduction. This suggests reducing decode padding can be similarly important or even larger for this workload.
- Oracle output-length scheduling is stronger because responses of similar length are batched together, which reduces padded decode work.
- The oracle result is not deployable because true output length is unavailable before generation completes.

### Generic Cache vs Engram-Specific Step Kernel

The cached Engram ablation separated generic KV-cached serving from the Engram-specific cached `step_kernel`.

| Comparison | Before model | Scheduler | Decoder | Batch/replica | After model | Serving seconds | Speedup | Serving-time reduction |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| Generic cache, random scheduler | `naive` | `random` | `static` | 8 | `cached_full_engram` | `4882.758 -> 191.678` | `25.47x` | `96.07%` |
| Engram step-kernel, random scheduler | `cached_full_engram` | `random` | `static` | 8 | `optimized_cached` | `191.678 -> 192.714` | `0.99x` | `-0.54%` |
| Generic cache, input-known scheduler | `naive` | `longest_input_first` | `static` | 8 | `cached_full_engram` | `3629.573 -> 167.351` | `21.69x` | `95.39%` |
| Engram step-kernel, input-known scheduler | `cached_full_engram` | `longest_input_first` | `static` | 8 | `optimized_cached` | `167.351 -> 165.990` | `1.01x` | `0.81%` |

Interpretation:

- Most of the 40B serving gain comes from generic cached serving, not Engram-specific micro-optimization.
- The Engram `step_kernel` is exact and useful, but in this end-to-end batched benchmark its contribution is small.
- The apparent discrepancy between `25.34x` in the main attribution matrix and `21.69x` here is due to different scheduler settings. Under random scheduling, generic cached serving gives `25.47x`, matching the `25.34x` optimized random model-path result up to the small Engram step-kernel difference. Under input-known scheduling, the naive baseline is already faster, so the generic-cache ratio is smaller at `21.69x`.

### Static vs Compact

| Model | Scheduler | Batch/replica | Static seconds | Compact seconds | Static decode tokens | Compact decode tokens | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `naive` | `random` | 8 | `4882.758` | `1079.463` | `43,584` | `12,800` | Compact greatly helps naive. |
| `optimized_cached` | `longest_input_first` | 8 | `165.990` | `182.102` | `42,596` | `12,800` | Compact hurts at B8. |
| `optimized_cached` | `longest_input_first` | 16 | `183.841` | `163.306` | `59,660` | `12,800` | Compact wins at B16. |

Interpretation:

- Compact always reduces executed decode work to the requested token count, but reducing executed work does not always reduce serving time.
- For optimized cached inference, the overhead of cache compaction can outweigh saved compute at smaller batches.
- At B16, static padding waste grows enough that compaction becomes beneficial.
- The most likely reason compact is slower at B8 is that optimized cached per-token compute is already relatively cheap. Per-step row filtering, cache `index_select`, dynamic tensor shape changes, and extra synchronization/memory movement across model-parallel devices can cost more than the decode work saved. The naive model still benefits because full-context recomputation is so expensive that removing padded rows dominates the compaction overhead.

### Batch Size Sweep

| Model | Scheduler | Decoder | Batch/replica | Serving seconds | Tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| `optimized_cached` | `longest_input_first` | `static` | 1 | `253.385` | `50.516` |
| `optimized_cached` | `longest_input_first` | `static` | 2 | `231.256` | `55.350` |
| `optimized_cached` | `longest_input_first` | `static` | 4 | `176.799` | `72.399` |
| `optimized_cached` | `longest_input_first` | `static` | 8 | `165.990` | `77.113` |
| `optimized_cached` | `longest_input_first` | `static` | 16 | `183.841` | `69.625` |
| `optimized_cached` | `longest_input_first` | `compact` | 16 | `163.306` | `78.381` |

Interpretation:

- Very small batches avoid padding but underutilize the GPUs.
- B8 static is a strong utilization/padding tradeoff.
- B16 static over-pads decode.
- B16 compact recovers enough decode waste to become the best measured realistic configuration.
- We did not run B32 or larger static/compact sweeps, so the report should not claim that B16 compact is the global optimum over all batch sizes. It is the best measured configuration so far. Larger batches might improve GPU utilization further, but they also increase prefill padding, decode padding, memory pressure, and per-step active-row management cost. A B32/B64 static-vs-compact sweep is a reasonable future experiment if more cluster time is available.

### Continuous Refill

Continuous refill was implemented and measured after the best compact result.

| Model | Scheduler | Decoder | Batch/replica | Serving seconds | Tok/s | Delta vs best compact |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `optimized_cached` | `longest_input_first` | `continuous` | 8 | `167.398` | `76.464` | `2.51% slower` |
| `optimized_cached` | `longest_input_first` | `continuous` | 16 | `164.889` | `77.628` | `0.97% slower` |
| `optimized_cached` | `longest_output_first` oracle | `continuous` | 16 | `216.314` | `59.173` | `32.46% slower` |

Interpretation:

- Continuous refill is now implemented and benchmarkable.
- It did not beat B16 compact on this closed 100-request workload.
- Likely causes are exact per-request prefill, slot-indexed cache overhead, Engram short-conv slot state, and Python scheduler bookkeeping. Unlike compact static batches, continuous refill admits new requests one slot at a time, which reduces padding but gives up some batched prefill efficiency and adds per-request cache-management overhead.
- The oracle continuous result is worse because output-length sorting harms input-length locality and therefore prefill efficiency; in this workload, that cost outweighed the decode-padding benefit.
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
