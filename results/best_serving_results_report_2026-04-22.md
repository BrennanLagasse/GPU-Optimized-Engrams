# Best Serving Results Report

Timestamp: 2026-04-22 21:55 EDT

## Executive Summary

We implemented a naive correctness baseline and an optimized cached Engram/mHC serving path, then benchmarked both on a deterministic 100-request long-tailed workload using a 40B-class model across 8 H200 GPUs. The best realistic configuration served 12,800 requested output tokens in 110.087 seconds, achieving a 44.35x speedup over the naive random static baseline. Ablations show that most of the speedup comes from cached optimized inference, with additional gains from input-length-aware scheduling, B48 active-row compaction, greedy-prefill replica assignment, and focused batch-size tuning.

Our best measured realistic result is a `44.35x` speedup over the naive baseline on a 100-request heterogeneous serving workload for the `target_40b_approx` model preset.

Best configuration:

- model implementation: `optimized_cached`
- scheduler: `longest_input_first`
- replica assignment: `greedy_prefill`
- decode mode: `compact`
- batch size per replica: `48`
- execution layout: two 4-GPU replicas across `8 x NVIDIA H200`
- serving time excluding model load and worker startup: `110.087s`
- requested output throughput: `116.272 tokens/s`
- requested output tokens served: `12,800`

Baseline:

- model implementation: `naive`
- scheduler: `random`
- decode mode: `static`
- batch size per replica: `8`
- serving time excluding model load and worker startup: `4882.758s`
- requested output throughput: `2.621 tokens/s`

The strongest result is not from one optimization alone. The dominant gain comes from moving from naive full-context recomputation to cached optimized inference. Input-aware scheduling, larger compact batches, greedy-prefill replica assignment, and focused batch-size tuning provide additional measured gains.

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

The headline `44.35x` is computed from serving time. The numerator is the naive baseline serving time: `naive + random + round_robin + static + BATCH_SIZE=8`, measured at `4882.758s`. The denominator is the best realistic optimized serving time: `optimized_cached + longest_input_first + greedy_prefill + compact + BATCH_SIZE=48`, measured at `110.087s`.

```text
4882.758s / 110.087s = 44.35x
```

## Reproduction Command

Best measured realistic configuration:

The single command below is sufficient to reproduce the reported result. You do not need to run any setup command beforehand for the documented default settings.

```bash
MODEL_IMPL=optimized_cached \
POLICY=longest_input_first \
REPLICA_ASSIGNMENT=greedy_prefill \
DECODE_MODE=compact \
BATCH_SIZE=48 \
OUTPUT=results/serving_opt_sweep_optimized_input_compact_b48_greedy_prefill.json \
bash scripts/run_cluster_serving_scheduling.sh
```

The next block is included only to document which defaults `scripts/run_cluster_serving_scheduling.sh` uses internally for this reported run. It is not necessary to run this block separately unless you want to override or inspect those settings explicitly.

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

| Case | Model | Scheduler | Replica assignment | Decode | Batch/replica | Serving seconds | Requested tok/s | Speedup vs naive random static | Time reduction vs naive |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `v4` | `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 48 | `110.087` | `116.272` | `44.35x` | `97.75%` |
| `v3` | `optimized_cached` | `longest_input_first` | `greedy_prefill` | `compact` | 32 | `131.728` | `97.170` | `37.07x` | `97.30%` |
| `v2` | `optimized_cached` | `longest_input_first` | `round_robin` | `compact` | 16 | `163.306` | `78.381` | `29.90x` | `96.66%` |
| `v1` | `optimized_cached` | `longest_input_first` | `round_robin` | `static` | 8 | `165.990` | `77.113` | `29.42x` | `96.60%` |
| `baseline` | `naive` | `random` | `round_robin` | `static` | 8 | `4882.758` | `2.621` | `1.00x` | `0.00%` |

## What Each Methodological Choice Means

The benchmark has four separate categories of choices:

- model implementation: what code path computes logits
- scheduler: how requests are ordered and assigned to batches/replicas
- replica assignment: how scheduled batches are mapped onto the two 4-GPU replicas
- decoder/runtime mode: how finished rows are handled during generation

These categories should not be compared as alternatives to each other. For example, `optimized_cached` is a model implementation, while `longest_input_first` is a scheduler. The best result combines one choice from each category.

### Model Implementation Choices

#### `naive`

The naive implementation is the correctness-oriented baseline. It recomputes the model over the full context during generation. It is intentionally not optimized for serving speed.

Purpose:

- establishes a simple baseline
- helps validate optimized implementation parity
- makes the cost of missing KV-cache reuse visible

#### `optimized_cached` (best realistic)

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

- [cached_engram_ablation_report_2026-04-21.md](cached_engram_ablation_report_2026-04-21.md)
- [proposal_checklist.md](proposal_checklist.md)
- [paper_metrics_summary.md](paper_metrics_summary.md)
- [large_batch_sweep_report_2026-04-22.md](large_batch_sweep_report_2026-04-22.md)
- [prefill_decode_stage_costs_b32_greedy_2026-04-22.md](prefill_decode_stage_costs_b32_greedy_2026-04-22.md)

### Scheduler Choices

#### `random`

Random scheduling is the baseline. It does not use input length, output length, or any workload structure.

Purpose:

- shows what happens without request-aware scheduling
- gives a conservative baseline for measuring scheduling improvements

#### `longest_input_first` (best realistic)

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

### Replica Assignment Choices

#### `round_robin`

Round-robin assigns scheduled batches alternately across the two 4-GPU replicas. It is simple, but at large batch sizes a single heavy batch can dominate makespan.

#### `greedy_prefill` (best realistic)

Greedy-prefill assigns batches using known input/prefill work. It is realistic because input lengths are known before serving. In practice, it helps in two ways: it reduces prefill-side imbalance across replicas, and because the assignment cost is based on known padded prefill work, it also tends to reduce unnecessary prefill padding exposure at the replica level. At B32 it materially improved load balance versus round-robin, reducing compact serving time from `177.720s` to `131.728s`, a `25.88%` serving-time reduction relative to B32 round-robin compact.

### Decoder/Runtime Choices

#### `static` Decode

Static decode keeps every row in a batch alive until the longest request in that batch finishes. This is simple and efficient for fixed tensor shapes, but it performs padded decode work for rows that are already complete.

Purpose:

- stable baseline
- high GPU utilization
- exposes padding waste in heterogeneous workloads

#### `compact` Decode (best realistic)

Compact decode removes rows after they complete and compacts model caches with the remaining active row indices. This is realistic because it only uses completion information observed online.

Purpose:

- avoids decode work for completed rows
- executes exactly the requested `12,800` decode-token steps
- tests whether reduced work outweighs cache-indexing and tensor-shape overhead

For B8, compact was slower than static. For B16, compact became better than B8 static, and B32/B48 greedy-prefill compact runs improved further because the larger batches created enough padding waste for compaction to pay off while greedy-prefill reduced replica imbalance.

## Ablations And Attribution

### Main Attribution Matrix

| Comparison | Before model | Before scheduler | Before decoder | Before batch | After model | After scheduler | After decoder | After batch | Serving seconds | Speedup | Serving-time reduction | Interpretation |
| --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Model path | `naive` | `random` | `static` | 8 | `optimized_cached` | `random` | `static` | 8 | `4882.758 -> 192.714` | `25.34x` | `96.05%` | Model-path optimization dominates the total speedup. |
| Realistic scheduler | `optimized_cached` | `random` | `static` | 8 | `optimized_cached` | `longest_input_first` | `static` | 8 | `192.714 -> 165.990` | `1.16x` | `13.87%` | Reducing prefill padding gives a clear gain. |
| B16 compaction | `optimized_cached` | `longest_input_first` | `static` | 8 | `optimized_cached` | `longest_input_first` | `compact` | 16 | `165.990 -> 163.306` | `1.016x` | `1.62%` | Larger batch plus compaction added a small gain over B8 static. |
| B32 greedy compaction | `optimized_cached` | `longest_input_first` | `compact` | 16 | `optimized_cached` | `longest_input_first` | `compact` | 32 | `163.306 -> 131.728` | `1.24x` | `19.34%` | Larger compact batches plus greedy-prefill replica assignment improved makespan. |
| Focused B48 tuning | `optimized_cached` | `longest_input_first` | `compact` | 32 | `optimized_cached` | `longest_input_first` | `compact` | 48 | `131.728 -> 110.087` | `1.20x` | `16.43%` | Non-power-of-two batch tuning found a better utilization/padding tradeoff than B32/B64. |
| Full realistic bundle | `naive` | `random` | `static` | 8 | `optimized_cached` | `longest_input_first` | `compact` | 48 | `4882.758 -> 110.087` | `44.35x` | `97.75%` | Combined model, scheduler, replica assignment, decoder/runtime, and batch-size result. |

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

Footnote:

- The `25.34x` figure in the main attribution matrix comes from `naive -> optimized_cached` under `random + static + B8`, specifically `4882.758s -> 192.714s`.
- The `21.69x` figure in this section comes from `naive -> cached_full_engram` under `longest_input_first + static + B8`, specifically `3629.573s -> 167.351s`.
- These ratios use different scheduler settings, so they are not meant to match exactly.

### Static vs Compact

| Model | Scheduler | Replica assignment | Batch/replica | Static seconds | Compact seconds | Static decode tokens | Compact decode tokens | Compact speedup vs static | Compact time reduction | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `naive` | `random` | `round_robin` | 8 | `4882.758` | `1079.463` | `43,584` | `12,800` | `4.52x` | `77.89%` | Compact greatly helps naive. |
| `optimized_cached` | `longest_input_first` | `round_robin` | 8 | `165.990` | `182.102` | `42,596` | `12,800` | `0.91x` | `-9.71%` | Compact hurts at B8. |
| `optimized_cached` | `longest_input_first` | `round_robin` | 16 | `183.841` | `163.306` | `59,660` | `12,800` | `1.13x` | `11.17%` | Compact wins at B16. |
| `optimized_cached` | `longest_input_first` | `greedy_prefill` | 32 | `243.960` | `131.728` | `50,668` | `12,800` | `1.85x` | `45.97%` | Compact plus B32 greedy-prefill was the best before focused batch tuning. |

Interpretation:

- Compact always reduces executed decode work to the requested token count, but reducing executed work does not always reduce serving time.
- For optimized cached inference, the overhead of cache compaction can outweigh saved compute at smaller batches.
- At B16, static padding waste grows enough that compaction becomes beneficial.
- The most likely reason compact is slower at B8 is that optimized cached per-token compute is already relatively cheap. Per-step row filtering, cache `index_select`, dynamic tensor shape changes, and extra synchronization/memory movement across model-parallel devices can cost more than the decode work saved. The naive model still benefits because full-context recomputation is so expensive that removing padded rows dominates the compaction overhead.

### Batch Size Sweep

Model and scheduler are `optimized_cached + longest_input_first` for every row below.

| Replica assignment | Decoder | Batch/replica | Serving seconds | Tok/s | Speedup vs naive | Time reduction vs naive |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `round_robin` | `static` | 1 | `253.385` | `50.516` | `19.27x` | `94.81%` |
| `round_robin` | `static` | 2 | `231.256` | `55.350` | `21.11x` | `95.26%` |
| `round_robin` | `static` | 4 | `176.799` | `72.399` | `27.62x` | `96.38%` |
| `round_robin` | `static` | 8 | `165.990` | `77.113` | `29.42x` | `96.60%` |
| `round_robin` | `static` | 16 | `183.841` | `69.625` | `26.56x` | `96.23%` |
| `round_robin` | `compact` | 16 | `163.306` | `78.381` | `29.90x` | `96.66%` |
| `round_robin` | `static` | 32 | `359.340` | `35.621` | `13.59x` | `92.64%` |
| `greedy_prefill` | `static` | 32 | `243.960` | `52.468` | `20.01x` | `95.00%` |
| `round_robin` | `compact` | 32 | `177.720` | `72.023` | `27.47x` | `96.36%` |
| `greedy_prefill` | `compact` | 32 | `131.728` | `97.170` | `37.07x` | `97.30%` |
| `greedy_prefill` | `compact` | 36 | `125.194` | `102.242` | `39.00x` | `97.44%` |
| `greedy_prefill` | `compact` | 40 | `121.943` | `104.967` | `40.04x` | `97.50%` |
| `greedy_prefill` | `compact` | 48 | `110.087` | `116.272` | `44.35x` | `97.75%` |
| `greedy_prefill` | `compact` | 56 | `119.096` | `107.476` | `41.00x` | `97.56%` |
| `round_robin` | `compact` | 64 | `132.731` | `96.436` | `36.79x` | `97.28%` |
| `greedy_prefill` | `compact` | 64 | `132.517` | `96.591` | `36.85x` | `97.29%` |
| `greedy_prefill` | `static` | 64 | `446.126` | `28.691` | `10.94x` | `90.86%` |

Interpretation:

- Very small batches avoid padding but underutilize the GPUs.
- B8 static is a strong utilization/padding tradeoff.
- B16 static over-pads decode.
- B16 compact recovers enough decode waste to beat B8 static.
- B48 compact with greedy-prefill is the best measured realistic configuration after considering batch sizes that are not a power of two.
- B128 was not run because B64 already decreased in performance relative to the next largest batch size of 56, and `BATCH_SIZE=128` would collapse the 100-request workload into one effective batch, leaving one replica idle.

### Continuous Refill

Continuous refill was implemented and measured before the focused B48 tuning pass. It did not beat either the earlier B32 compact best or the current B48 compact best.

| Model | Scheduler | Replica assignment | Decoder | Batch/replica | Serving seconds | Tok/s | Speedup vs naive | Time reduction vs naive | Change vs best B48 compact |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `optimized_cached` | `longest_input_first` | `round_robin` | `continuous` | 8 | `167.398` | `76.464` | `29.17x` | `96.57%` | `52.06% slower` |
| `optimized_cached` | `longest_input_first` | `round_robin` | `continuous` | 16 | `164.889` | `77.628` | `29.61x` | `96.62%` | `49.78% slower` |
| `optimized_cached` | `longest_input_first` | `round_robin` | `continuous` | 32 | `186.199` | `68.744` | `26.22x` | `96.19%` | `69.14% slower` |
| `optimized_cached` | `longest_input_first` | `round_robin` | `continuous` | 64 | `156.372` | `81.856` | `31.23x` | `96.80%` | `42.04% slower` |
| `optimized_cached` | `longest_output_first` oracle | `round_robin` | `continuous` | 16 | `216.314` | `59.173` | `22.57x` | `95.57%` | `96.49% slower` |

Interpretation:

- Continuous refill is now implemented and benchmarkable.
- It did not beat B48 compact on this closed 100-request workload.
- Likely causes are exact per-request prefill, slot-indexed cache overhead, Engram short-conv slot state, and Python scheduler bookkeeping. Unlike compact static batches, continuous refill admits new requests one slot at a time, which reduces padding but gives up some batched prefill efficiency and adds per-request cache-management overhead.
- The oracle continuous result is worse because output-length sorting harms input-length locality and therefore prefill efficiency; in this workload, that cost outweighed the decode-padding benefit.
- This does not invalidate continuous batching generally; it means this implementation and workload did not improve over the simpler best compact path.

Footnote:

- We did not run a B48 continuous-refill case. Given that the measured continuous-refill cases at B8/B16/B32/B64 are all substantially slower than B48 compact, a B48 continuous run would be reasonable future work but was not likely to change the headline result.

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

## Prefill/Decode Stage-Cost Experiment

An earlier architecture simulation suggested prefill/decode disaggregation could be promising, so the benchmark now records measured per-batch `prefill_seconds` and `decode_seconds` for optimized static/compact serving.

For the B32 compact greedy-prefill result that was used for the stage-cost probe:

| Metric | Seconds | Relative to measured serving time |
| --- | ---: | ---: |
| Measured serving time | `131.728` | `1.00x` |
| Sum of measured prefill stages | `9.574` | `0.07x` |
| Sum of measured decode stages | `220.153` | `1.67x` |
| Estimated staged prefill/decode pipeline | `227.832` | `1.73x` |

| Baseline | Candidate | Baseline seconds | Candidate seconds | Speedup | Serving-time reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| B32 compact serving | staged prefill/decode estimate | `131.728` | `227.832` | `0.58x` | `-72.96%` |

This measured stage-cost experiment does not support prefill/decode disaggregation as an immediate win for the fixed closed-batch workload. The workload is decode-dominated after compact scheduling, and the staged estimate gives up the data-parallel benefit of two full serving replicas. Disaggregation could still matter for arrival-process workloads, TTFT, or much heavier prefill workloads, but it is not the best next optimization for the current headline benchmark.

## What We Can Claim

Strong claim:

- On the fixed 100-request 40B-class H200 serving benchmark, the best realistic optimized configuration is `44.35x` faster than the naive random static baseline.

More precise claim:

- The best measured realistic configuration is `optimized_cached + longest_input_first + greedy_prefill + compact + BATCH_SIZE=48`, with `110.087s` serving time and `116.272 requested output tok/s`.
- The benchmark uses random weights because the result is an inference-speed comparison, not a quality or accuracy claim.

Attribution claim:

- The largest contribution comes from cached optimized inference versus naive full-context recompute.
- Realistic input-known scheduling contributes a meaningful additional improvement.
- B48 active-row compaction with greedy-prefill replica assignment contributes a material additional improvement over the prior B16 and B32 compact results.

Do not claim:

- Oracle output-length scheduling is deployable.
- Continuous refill is currently better than compact for this workload.
- The result is a production online serving benchmark.
