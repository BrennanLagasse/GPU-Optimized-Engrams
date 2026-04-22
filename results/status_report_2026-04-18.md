# Status Report: GPU-Optimized Engrams

Generated: `2026-04-18 11:17 EDT`

Branch: `engrams-baseline-benchmarking`

Current commit: `e6f556f`

## Executive Summary

The repository now has a working Engram+mHC inference prototype with separate naive and optimized implementations, parity tests, local smoke validation, cluster benchmark tooling, target-scale `~32B` and `~40B` model presets, and paper-style result framing.

The main performance result so far is that the optimized path beats the naive path at the `~40B` target scale, especially as cached decode length grows. The best clean paired `~40B` result is currently:

| Hardware | Decode | Optimized cached | Naive | Improvement |
| --- | ---: | ---: | ---: | ---: |
| `4 x NVIDIA H200` | 64 tokens | `21.85 tok/s` | `17.07 tok/s` | `+28.00%` |

The main remaining bottleneck is cross-device activation transfer in the current one-process model-parallel runtime. More GPUs are not automatically faster: earlier 4-GPU and 8-GPU results were similar because 8 GPUs introduce more transfer boundaries.

## Current Implementation State

Implemented:

- Separate naive baseline in [engrams_naive.py](engrams_naive.py).
- Optimized Engram+mHC implementation in [engrams_kv_moe.py](engrams_kv_moe.py).
- Dense FFN fallback and MoE path.
- mHC-style residual wrapper with `hc_mult=4` support.
- KV-cached optimized decoding.
- Cached Engram `ShortConv.forward_step()` fast path for exact single-token cached decode.
- Cached attention mask fast path.
- Model-parallel `device_map` execution.
- Weighted contiguous placement.
- Stage-aware execution grouping for contiguous same-device blocks.
- Inference-only `torch.inference_mode()` generation/profiling paths.
- Automated cluster placement/decode sweep script and shell wrapper.

Validation coverage:

- Naive/optimized forward parity tests.
- Naive/optimized generation parity tests on overlapping feature sets.
- Cache/no-cache generation parity tests for optimized paths.
- Device-map CPU smoke tests.
- Script syntax and CPU smoke tests for the placement sweep.

Latest known full local test result:

```text
conda run -n ai_infra_env_new pytest -q test_engrams.py
21 passed
```

## Benchmark Tooling

Core scripts:

- [scripts/benchmark_decode.py](scripts/benchmark_decode.py): single benchmark run.
- [scripts/run_target_benchmark_matrix.py](scripts/run_target_benchmark_matrix.py): explicit device-group/decode-length matrix.
- [scripts/sweep_cluster_placements.py](scripts/sweep_cluster_placements.py): automatic GPU availability probe and placement/decode sweep.
- [scripts/run_cluster_placement_sweep.sh](scripts/run_cluster_placement_sweep.sh): one-command cluster wrapper for the placement sweep.
- [scripts/estimate_scale.py](scripts/estimate_scale.py): parameter, memory, and approximate cached decode FLOPs/token estimates.
- [scripts/profile_decode_breakdown.py](scripts/profile_decode_breakdown.py): TTFT vs steady-state decode breakdown.
- [scripts/profile_forward_components.py](scripts/profile_forward_components.py): component-level forward profiling.

Cluster command to reproduce the next placement/decode sweep:

```bash
cd ~/class_projects/GPU-Optimized-Engrams
bash scripts/run_cluster_placement_sweep.sh
```

Equivalent expanded command:

```bash
python scripts/sweep_cluster_placements.py \
  --preset target_40b_approx \
  --group-sizes 4 8 \
  --decode-lengths 64 128 256 \
  --dtype bfloat16 \
  --batch-size 1 \
  --prompt-length 8 \
  --trials 1 \
  --min-free-mib 120000 \
  --max-gpu-util-percent 10 \
  --output results/cluster_placement_sweep_target_40b.json
```

Useful overrides:

```bash
DECODE_LENGTHS="64 128" bash scripts/run_cluster_placement_sweep.sh
DEVICE_GROUPS="0,1,2,3" bash scripts/run_cluster_placement_sweep.sh
OUTPUT=results/my_sweep.json bash scripts/run_cluster_placement_sweep.sh
SKIP_GIT_PULL=1 bash scripts/run_cluster_placement_sweep.sh
```

## Target-Scale Model Framing

The current target-scale runs use random weights, as planned in the proposal, because credible pretrained Engram model weights are not available.

Representative target config:

| Preset | Params | Backbone | Cached decode compute descriptor |
| --- | ---: | --- | ---: |
| `target_40b_approx` | `39.98B` | `d=6656, h=26624, L=52, heads=52, hc_mult=4` | `77.37 GFLOPs/token` at cached context length `64` |

Memory estimate for `target_40b_approx` at bf16 and tensor-parallel-style rank count `8`:

| Metric | Approx value |
| --- | ---: |
| Total parameter memory | `74.47 GiB` |
| Parameter memory per rank | `9.31 GiB` |
| KV cache | `5.28 GiB` |
| Activation estimate | `2.64 GiB` |
| Working-set estimate per rank | `10.30 GiB` |
| Engram table memory | `2.47 GiB` |

These are approximate sizing numbers, not exact allocator telemetry.

## Results So Far

### 40B Target-Scale Throughput

Best clean 4-GPU paired run:

| Hardware | Decode | Optimized cached | Naive | Improvement |
| --- | ---: | ---: | ---: | ---: |
| `4 x H200` | 64 tokens | `21.85 tok/s` | `17.07 tok/s` | `+28.00%` |

Earlier 8-GPU matrix result:

| Hardware | Decode | Optimized cached | Naive | Improvement |
| --- | ---: | ---: | ---: | ---: |
| `8 x H200` | 64 tokens | `21.07 tok/s` | `16.51 tok/s` | `+27.62%` |

Earlier 4-GPU matrix result:

| Hardware | Decode | Optimized cached | Naive | Improvement |
| --- | ---: | ---: | ---: | ---: |
| `4 x H200` | 64 tokens | `21.87 tok/s` | `17.13 tok/s` | `+27.67%` |

Decode-length trend on `8 x H200`:

| Decode | Optimized cached | Naive | Improvement |
| ---: | ---: | ---: | ---: |
| 1 token | `1.02 tok/s` | `1.03 tok/s` | `-0.97%` |
| 8 tokens | `6.34 tok/s` | `6.23 tok/s` | `+1.77%` |
| 16 tokens | `10.06 tok/s` | `9.65 tok/s` | `+4.25%` |
| 32 tokens | `16.67 tok/s` | `14.46 tok/s` | `+15.28%` |
| 64 tokens | `20.85 tok/s` | `16.59 tok/s` | `+25.68%` |

Interpretation:

- Optimized is not meaningfully better for a one-token decode.
- The optimized advantage grows with decode length.
- The win is mainly steady-state cached decode, not time-to-first-token.

### TTFT / Steady-State Breakdown

For the `~40B` target config on an 8-token decode:

| Metric | Optimized cached | Naive |
| --- | ---: | ---: |
| TTFT seconds | `0.9013` | `0.8975` |
| Steady-state avg seconds/token | `0.03967` | `0.04632` |
| Steady-state tok/s | `25.21` | `21.59` |
| End-to-end tok/s | `6.79` | `6.55` |

Interpretation:

- TTFT is essentially unchanged.
- Optimized steady-state decode is about `+16.76%` faster.

### Component Profiling Signal

A 4-GPU optimized cached component profile showed:

- total profiled time: about `0.624 s`
- block activation transfers: about `0.292 s`
- block activation transfer was the largest measured bucket
- block 0 with Engram work was also heavy, about `0.125 s`

Interpretation:

- Cross-device activation movement is the dominant scale bottleneck.
- Small Python/control-flow cleanups help less than reducing transfer boundaries or using a better parallelism strategy.

### Inference-Mode Optimization

The latest inference-only local optimization replaced `torch.no_grad()` with `torch.inference_mode()` in generation/profiling paths.

Local medium dense cached microbenchmark:

| Mode | Throughput |
| --- | ---: |
| `no_grad` loop | `401.67 tok/s` |
| `inference_mode` path | `448.05 tok/s` |

Relative improvement: about `+11.55%`.

Interpretation:

- This is a safe inference-path improvement.
- It reduces framework/autograd overhead.
- It is not expected to solve the 40B multi-GPU transfer bottleneck.

## Current Bottlenecks

Main bottleneck:

- Cross-device activation transfer in the one-process model-parallel runtime.

Secondary issues:

- 8-GPU placement is not automatically faster than 4-GPU placement.
- Cluster contention frequently blocks clean 8-GPU measurements.
- The current runtime does not implement true tensor parallelism, pipeline parallelism, NCCL communication scheduling, or host-memory prefetch.

## Proposal / Paper Alignment

Completed or mostly completed:

- Working Engram+mHC inference baseline.
- Naive vs optimized split.
- Correctness/parity tests.
- Target-scale `~32B` and `~40B` approximate configs.
- H200 benchmark results.
- Throughput, TTFT, steady-state, memory, and FLOPs/token reporting.

Partial or not implemented:

- Training path.
- Distributed tensor/pipeline parallel implementation.
- Host-memory prefetch path for Engram tables.
- Paper-comparable quality evaluation.
- Full paper-style infrastructure stack.

## Recommended Next Steps

1. Run the new cluster placement/decode sweep:

```bash
cd ~/class_projects/GPU-Optimized-Engrams
bash scripts/run_cluster_placement_sweep.sh
```

2. Use `summary_ranked` in the output JSON to identify the best feasible serving configuration.

3. If 4-GPU remains best, optimize around the 4-GPU path rather than trying to force 8-GPU execution.

4. For larger improvements, focus on reducing cross-device activation transfer rather than further small decode-path micro-optimizations.

5. If the project scope expands, the next major engineering phase should be real communication-aware parallelism: tensor parallelism, pipeline parallelism, or explicit overlap/prefetch.

## Current Blocker

As of this report, the browser login can reach Cloudflare, but the remote console has intermittently returned:

```text
Unable to connect to origin. Please confirm that the tunnel is set up correctly and the origin is healthy.
```

This blocks launching the sweep through the browser terminal until the Cloudflare tunnel/origin is healthy again.
