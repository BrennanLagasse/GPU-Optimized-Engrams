# Paper-Style Metrics Summary

This file reframes the current benchmark state in the style most useful for the proposal and future writeups.

## Hardware

- Local smoke hardware: Apple M1 Pro CPU
- Cluster hardware: `gpu003`
- Cluster accelerator inventory: `8 x NVIDIA H200 143771 MiB`
- Cluster torch build used in target-scale runs: `torch 2.11.0+cu128`

## Representative Target-Scale Configs

Derived from [scripts/estimate_scale.py](scripts/estimate_scale.py).

| Preset | Params | Backbone | bf16 param GiB/rank at TP=8 | KV cache GiB | Activation GiB | Cached decode FLOPs/token at ctx=64 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `target_32b_approx` | 31.97B | `d=6144, h=24576, L=48, heads=48, hc_mult=4` | 7.44 | 4.50 | 2.25 | 61.34 GFLOPs |
| `target_40b_approx` | 39.98B | `d=6656, h=26624, L=52, heads=52, hc_mult=4` | 9.31 | 5.28 | 2.64 | 77.37 GFLOPs |

Interpretation:
- the current 40B benchmarked model is not just “a 40B-ish model”; it is roughly a `39.98B`-parameter Engram+mHC configuration with about `77.37` approximate cached forward GFLOPs per generated token at context length `64`
- that gives a reasonable scale descriptor for whether the reported tok/s is strong or weak

## Best Measured Inference Results

### 40B Target Preset

| Scenario | Hardware slice | Optimized tok/s | Naive tok/s | Improvement |
| --- | --- | ---: | ---: | ---: |
| 64-token decode, best measured throughput | `4 x H200` | 21.85 | 17.07 | +28.00% |
| 64-token decode, earlier 8-GPU matrix | `8 x H200` | 21.07 | 16.51 | +27.62% |
| 8-token decode breakdown, end-to-end | `8 x H200` | 6.79 | 6.55 | +3.66% |
| 8-token decode breakdown, steady-state only | `8 x H200` | 25.21 | 21.59 | +16.76% |

Interpretation:
- optimized wins are real at target scale
- the win is primarily a steady-state decode win, not a TTFT win
- the current one-process sharded runtime does not scale meaningfully better on 8 GPUs than on 4 GPUs for this model size, which points to communication overhead

## TTFT and Steady-State Framing

For the current repo, the right paper-style inference framing is:
- `TTFT`: first-token latency
- `steady-state tok/s`: throughput after cache reuse dominates
- `end-to-end tok/s`: average throughput over the whole decode window

On the current 40B target breakdown run:

| Metric | Optimized cached | Naive |
| --- | ---: | ---: |
| TTFT seconds | 0.9013 | 0.8975 |
| Steady-state tok/s | 25.21 | 21.59 |
| End-to-end tok/s | 6.79 | 6.55 |

Interpretation:
- TTFT is effectively unchanged
- the optimization story is therefore “better steady-state cached decode,” not “lower first-token latency”

## Approximate Aggregate Compute Rate

Using the `target_40b_approx` cached-decode estimate of `77.37 GFLOPs/token`:
- best measured 4-GPU optimized throughput (`21.85 tok/s`) corresponds to about `1.69 TFLOPs/s` of approximate forward compute
- earlier 8-GPU optimized 64-token throughput (`21.07 tok/s`) corresponds to about `1.63 TFLOPs/s`
- 8-token steady-state optimized throughput (`25.21 tok/s`) corresponds to about `1.95 TFLOPs/s`

These values are only rough aggregate forward-compute rates. They are not GPU-kernel utilization measurements and should not be compared directly to peak hardware FLOPs without accounting for:
- communication
- memory bandwidth effects
- hash-table lookup behavior
- framework overhead
- underutilization during short decode windows

## Current Bottom Line

What can be claimed cleanly:
- the repo runs approximate `32B` and `40B` Engram+mHC target configs on H200 hardware
- optimized beats naive at target scale
- the gain is most visible in steady-state cached decode
- the main remaining bottleneck is cross-device activation transfer, not the correctness of the Engram/mHC baseline

What cannot yet be claimed cleanly:
- that this implementation matches the Engram paper’s infrastructure-efficiency story
- that the current runtime is close to hardware-optimal
- that the repo reproduces the paper’s full system design or training-time conclusions
