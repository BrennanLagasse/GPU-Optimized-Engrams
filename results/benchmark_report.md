# Benchmark Report

## Local Results

Source: [results/local_benchmarks.json](/Users/vincentli/Desktop/GPU-Optimized-Engrams/results/local_benchmarks.json)

Hardware:
- CPU: Apple M1 Pro
- Device used for these runs: CPU only
- GPU results: not yet collected in a trustworthy synced cluster workflow

Benchmark model sizes:
- Dense benchmark config: `vocab_size=512`, `emb_dim=64`, `hidden_dim=128`, `n_heads=4`, `n_layers=2`, `context_length=32`
- Dense benchmark parameter count: `150,528` parameters
- Engram benchmark config: same transformer backbone plus one Engram layer at `layer_ids=[0]` with `engram_vocab_size=[257, 263]`, `n_embed_per_ngram=32`, `n_head_per_ngram=4`, `kernel_size=2`
- Engram benchmark parameter count: `176,720` parameters
- Rough dense decode FLOPs/token estimate for this local config: about `131,072` forward FLOPs/token including the output head, using context length 32

Interpretation:
- These are tiny local sanity-benchmark models, not the proposal’s eventual 32B/40B-scale target.
- The current tokens/s numbers are useful for comparing `naive` vs `optimized` implementations, but they should not be interpreted as representative absolute performance for the final target model.

### Tiered local scaling checks

These runs use dense models only and are intended to show how the `naive` vs `optimized` gap behaves as the model size grows on local CPU hardware.

| Tier | Approx params | Config summary | Naive tok/s | Optimized tok/s | Relative improvement |
| --- | ---: | --- | ---: | ---: | ---: |
| tiny | 150,528 | `vocab=512, d=64, h=128, L=2` | 952.21 | 1426.55 (cached optimized) | +49.82% vs naive |
| medium-local-1 | 8,408,064 | `vocab=4096, d=256, h=1024, L=6` | 91.48 | 190.70 (cached optimized) | +108.46% vs naive |
| medium-local-2 | 41,989,120 | `vocab=8192, d=512, h=2048, L=8` | 31.07 | 60.53 (cached optimized) | +94.82% vs naive |
| large-local | 138,502,656 | `vocab=16384, d=768, h=3072, L=12` | 9.52 | 16.53 (cached optimized) | +73.63% vs naive |

Observations:
- On CPU, the optimized implementation remains materially faster than naive as the dense model size grows.
- The relative win narrows somewhat by the 138.5M-parameter tier, but the optimized path still shows a meaningful throughput advantage.
- These larger local runs are still far below the proposal’s intended 32B/40B target scale.

### Dense decode

| Case | Impl | Avg seconds | Tokens/s |
| --- | --- | ---: | ---: |
| dense_nocache | naive | 0.0168 | 952.21 |
| dense_nocache | optimized | 0.0125 | 1276.62 |
| dense_cache | optimized | 0.0112 | 1426.55 |

Observations:
- The optimized dense path is faster than the naive path on the local benchmark matrix.
- Relative to naive, the optimized dense no-cache path improves throughput by 34.07%.
- KV cache improves the optimized dense decode path further.
- Relative to optimized dense no-cache, the optimized dense cached path improves throughput by 11.74%.

### Engram decode

| Case | Impl | Avg seconds | Tokens/s |
| --- | --- | ---: | ---: |
| engram_nocache | naive | 0.0536 | 298.67 |
| engram_nocache | optimized | 0.0555 | 288.12 |
| engram_cache | optimized | 0.0532 | 300.79 |

Observations:
- In the current local setup, the Engram path is dominated by hash/lookup overhead.
- Relative to naive, the current optimized Engram no-cache path is 3.53% slower.
- KV cache helps only slightly in the Engram case at this scale, which suggests the bottleneck is not primarily the attention recurrence.
- Relative to optimized Engram no-cache, the optimized Engram cached path improves throughput by 4.40%.

### mHC overhead

| Case | Impl | Avg seconds | Tokens/s |
| --- | --- | ---: | ---: |
| mhc_dense_nocache | optimized | 0.0242 | 662.29 |

Observations:
- mHC adds a clear cost relative to the single-branch dense optimized baseline.
- This overhead should be profiled more carefully on GPU hardware before deciding how aggressively to use wider hyper-connection widths.

## Cluster Results

Cluster environment:
- Host: `gpu003`
- GPUs visible: `8 x NVIDIA H200 143771 MiB`
- Benchmark devices used:
  - earlier single-GPU runs via `CUDA_VISIBLE_DEVICES=0`
  - target-scale runs via `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- Python env: user-space `virtualenv` in the cluster repo clone
- Torch build used for GPU runs: `torch 2.11.0+cu128`
- CUDA availability check: `torch.cuda.is_available() == True`
- Benchmark dtypes used:
  - earlier single-GPU runs: `float32`
  - target-scale runs: `bfloat16`
- Synced branch/commit benchmarked on cluster:
  - earlier single-GPU runs: `engrams-baseline-benchmarking` at `12a35f7`
  - target-scale multi-GPU runs: `engrams-baseline-benchmarking` at `630ea2a`
  - latest 40B transfer-cleanup reruns: `engrams-baseline-benchmarking` at `89df6fd`

Cluster validation:
- `pytest -q test_engrams.py` passed on the updated cluster copy: `13 passed in 23.72s`

### H200 GPU decode

| Case | Impl | Avg seconds | Tokens/s | Relative improvement |
| --- | --- | ---: | ---: | ---: |
| tiny dense | naive | 0.0435 | 367.45 | baseline |
| tiny dense | optimized no-cache | 0.0429 | 372.79 | +1.45% vs naive |
| tiny dense | optimized cache | 0.0434 | 368.28 | +0.23% vs naive |
| tiny Engram | naive | 0.0580 | 275.73 | baseline |
| tiny Engram | optimized no-cache | 0.0584 | 274.03 | -0.62% vs naive |
| tiny Engram | optimized cache | 0.0675 | 237.03 | -14.03% vs naive |
| tiny mHC dense (`hc_mult=4`) | naive | 0.0574 | 278.55 | baseline |
| tiny mHC dense (`hc_mult=4`) | optimized | 0.0574 | 278.51 | -0.01% vs naive |
| medium-local-1 dense | naive | 0.0823 | 388.97 | baseline |
| medium-local-1 dense | optimized cache | 0.0759 | 421.82 | +8.45% vs naive |
| medium-local-2 dense | naive | 0.0978 | 327.36 | baseline |
| medium-local-2 dense | optimized cache | 0.0888 | 360.54 | +10.14% vs naive |
| large-local dense | naive | 0.1266 | 252.73 | baseline |
| large-local dense | optimized cache | 0.1151 | 278.09 | +10.03% vs naive |

Observations:
- The `hc_mult = 1` fast path materially improved the dense GPU results.
- On the updated H200 run, the optimized implementation now beats naive across the dense tiers from tiny through the 138.5M-parameter tier.
- The gain is small at the tiniest scale, which is consistent with kernel-launch and framework overhead dominating there.
- The gain becomes clearer at larger dense scales, reaching about `+8.45%` to `+10.14%` on the `8.4M` to `138.5M` parameter tiers.
- The Engram path still needs more work: optimized no-cache is only near parity with naive, and the cached Engram path is still slower on the tiny benchmark.
- The `hc_mult = 4` tiny mHC comparison is now effectively at parity between naive and optimized, which is what we expect from the semantic alignment work.

### H200 Engram short-conv ablation

Tiny Engram ablation config:
- same tiny Engram backbone as above, about `176,720` params in the standard path
- disabling `ShortConv` reduces the tiny Engram parameter count slightly to about `176,528`

| Case | Avg seconds | Tokens/s | Relative change |
| --- | ---: | ---: | ---: |
| tiny Engram optimized no-cache | 0.0630 | 254.08 | baseline |
| tiny Engram optimized no-cache, no short-conv | 0.0621 | 257.84 | +1.48% |
| tiny Engram optimized cache | 0.0797 | 200.75 | baseline |
| tiny Engram optimized cache, no short-conv | 0.0607 | 263.66 | +31.34% |

Observations:
- Removing `ShortConv` barely changes the no-cache tiny Engram result on H200.
- Removing `ShortConv` materially improves the cached tiny Engram result on H200.
- That points to `ShortConv` as a decode-time problem primarily in the cached Engram path, not as the whole explanation for the optimized Engram gap.
- The next Engram optimization pass should target cached inference structure around local mixing, not just the hash path.

### H200 cached Engram local-mixing modes

Tiny Engram cached-step comparison config:
- tiny Engram backbone, about `176,720` params
- one H200 on `gpu003`
- optimized implementation at commit `7f87091`

| Candidate mode | Baseline cached tok/s | Candidate cached tok/s | Relative change | Exact parity |
| --- | ---: | ---: | ---: | --- |
| `step_kernel` (`float32`, earlier pass) | 327.67 | 1118.50 | +241.35% | yes |
| `gated_value_only` (`float32`, earlier pass) | 654.82 | 1178.73 | +80.01% | no |
| `step_kernel` (`float32`, exact buffered path) | 532.88 | 1102.02 | +106.81% | yes |
| `step_kernel` (`bfloat16`, exact buffered path) | 436.58 | 1047.57 | +139.95% | yes |

Observations:
- The exact `step_kernel` cached-step path is the current best optimized Engram path.
- On H200, it improves cached tiny Engram throughput from about `532.88 tok/s` to about `1102.02 tok/s` in the current exact buffered implementation while preserving candidate cached-vs-no-cache behavior exactly.
- The approximate `gated_value_only` mode is faster than the original full path too, but it is not exact and should remain experimental.
- The cached Engram bottleneck is now much more localized: the full depthwise `ShortConv` launch on single-token cached steps was the main problem.
- The current `bfloat16` path does not yet beat `float32` on this implementation.

### H200 medium Engram cached-step scaling

Medium Engram cached-step comparison config:
- Engram-enabled model of about `861,752` params
- `vocab_size=4096`, `emb_dim=256`, `hidden_dim=1024`, `n_heads=8`, `n_layers=6`, `context_length=64`
- one H200 on `gpu003`
- optimized implementation at commit `7f87091`

| Dtype | Baseline cached tok/s | Candidate cached tok/s | Relative change | Exact parity |
| --- | ---: | ---: | ---: | --- |
| `float32` | 275.34 | 553.94 | +101.18% | yes |
| `bfloat16` | 244.62 | 531.39 | +117.23% | yes |

Observations:
- The exact cached-step optimization remains effective at a larger Engram-enabled tier, not just on the tiny benchmark.
- At this medium local-cluster scale, the optimized cached path roughly doubles throughput relative to the old full cached path.
- The current implementation still does not show a clear `bfloat16` advantage on H200.

### 8-GPU model-parallel scaling

Model-parallel implementation details:
- one-process model-parallel execution using `device_map`
- transformer blocks are partitioned across the listed GPUs
- token/position embeddings stay on the first device, final norm/head stay on the last device
- the optimized path also prepares per-device Engram hashes so Engram layers can execute on their local rank without cross-device hash tensors

Two-GPU smoke config:
- about `392,860` params
- `vocab_size=512`, `emb_dim=64`, `hidden_dim=128`, `n_heads=4`, `n_layers=2`, `hc_mult=1`
- Engram-enabled with `layer_ids=[0,1]`, `engram_vocab_size=[257,263]`, `n_embed_per_ngram=32`, `n_head_per_ngram=4`

| Case | Impl | Avg seconds | Tokens/s | Relative improvement |
| --- | --- | ---: | ---: | ---: |
| 2-GPU smoke, 4-token decode | optimized | 0.1970 | 20.31 | baseline |
| 2-GPU smoke, 4-token decode | naive | 0.1751 | 22.85 | -11.12% vs naive |

Observations:
- the new multi-GPU execution path is functionally working on the cluster
- at tiny scale, model-parallel overhead dominates and optimized is still slower than naive
- this is expected and should not be over-interpreted; the important result is that the 8-way execution path now runs

Target-scale configs used:
- `target_32b_approx`: about `31.97B` params
  - `vocab_size=129280`, `context_length=4096`, `emb_dim=6144`, `hidden_dim=24576`, `n_heads=48`, `n_layers=48`, `hc_mult=4`
  - Engram layers at `layer_ids=[0,1]`
  - Engram tables `engram_vocab_size=[646400,646400]`, `n_embed_per_ngram=512`, `n_head_per_ngram=8`, `kernel_size=4`
- `target_40b_approx`: about `39.98B` params
  - `vocab_size=129280`, `context_length=4096`, `emb_dim=6656`, `hidden_dim=26624`, `n_heads=52`, `n_layers=52`, `hc_mult=4`
  - Engram layers at `layer_ids=[0,1]`
  - Engram tables `engram_vocab_size=[646400,646400]`, `n_embed_per_ngram=512`, `n_head_per_ngram=8`, `kernel_size=4`

Target-scale results were run on all 8 H200s with:
- `dtype=bfloat16`
- `batch_size=1`
- `prompt_length=8`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

#### 32B target-scale run

| Case | Impl | Avg seconds | Tokens/s | Relative improvement |
| --- | --- | ---: | ---: | ---: |
| 32B, 1-token decode | optimized cached | 0.8531 | 1.17 | baseline |

Observations:
- the rough `32B` target preset runs successfully across all 8 H200s with the new model-parallel path
- this established that the target-scale path is now operational before attempting the `40B` run

#### 40B target-scale run

| Case | Impl | Avg seconds | Tokens/s | Relative improvement |
| --- | --- | ---: | ---: | ---: |
| 40B, 1-token decode | optimized cached | 0.9768 | 1.02 | -0.97% vs naive |
| 40B, 1-token decode | naive | 0.9739 | 1.03 | baseline |
| 40B, 8-token decode | optimized cached | 1.2623 | 6.34 | +1.77% vs naive |
| 40B, 8-token decode | naive | 1.2848 | 6.23 | baseline |
| 40B, 16-token decode | optimized cached | 1.5903 | 10.06 | +4.25% vs naive |
| 40B, 16-token decode | naive | 1.6578 | 9.65 | baseline |
| 40B, 16-token decode, 4-GPU placement | optimized cached | 1.1840 | 13.51 | +9.93% vs 4-GPU naive |
| 40B, 16-token decode, 4-GPU placement | naive | 1.3027 | 12.29 | baseline |
| 40B, 32-token decode | optimized cached | 1.9200 | 16.67 | +15.28% vs naive |
| 40B, 32-token decode | naive | 2.2130 | 14.46 | baseline |
| 40B, 64-token decode | optimized cached | 3.0694 | 20.85 | +25.68% vs naive |
| 40B, 64-token decode | naive | 3.8571 | 16.59 | baseline |
| 40B, 32-token decode, 4-GPU matrix | optimized cached | 1.7709 | 18.07 | +17.26% vs 4-GPU naive |
| 40B, 32-token decode, 4-GPU matrix | naive | 2.0766 | 15.41 | baseline |
| 40B, 64-token decode, 4-GPU matrix | optimized cached | 2.9264 | 21.87 | +27.67% vs 4-GPU naive |
| 40B, 64-token decode, 4-GPU matrix | naive | 3.7350 | 17.13 | baseline |
| 40B, 32-token decode, 8-GPU matrix | optimized cached | 1.9002 | 16.84 | +16.46% vs 8-GPU naive |
| 40B, 32-token decode, 8-GPU matrix | naive | 2.2130 | 14.46 | baseline |
| 40B, 64-token decode, 8-GPU matrix | optimized cached | 3.0375 | 21.07 | +27.62% vs 8-GPU naive |
| 40B, 64-token decode, 8-GPU matrix | naive | 3.8764 | 16.51 | baseline |

Observations:
- the rough `40B` target preset also runs successfully across all 8 H200s
- on the single-token run, optimized cached and naive are essentially tied; naive is very slightly faster
- on the 8-token decode, optimized cached beats naive:
  - optimized cached: `6.34 tok/s`
  - naive: `6.23 tok/s`
  - relative improvement: about `+1.77%`
- on the 16-token decode, the optimized gap widens:
  - optimized cached: `10.06 tok/s`
  - naive: `9.65 tok/s`
  - relative improvement: about `+4.25%`
- after the `engram_input_ids` transfer cleanup, longer 8-GPU decode windows show a much clearer cached-path win:
  - 32-token decode: optimized cached `16.67 tok/s` vs naive `14.46 tok/s`, about `+15.28%`
  - 64-token decode: optimized cached `20.85 tok/s` vs naive `16.59 tok/s`, about `+25.68%`
- after the cached single-token attention-mask fast path, a matrix run on commit `0199da8` showed:
  - 4-GPU 32-token decode: optimized cached `18.07 tok/s` vs naive `15.41 tok/s`, about `+17.26%`
  - 4-GPU 64-token decode: optimized cached `21.87 tok/s` vs naive `17.13 tok/s`, about `+27.67%`
  - 8-GPU 32-token decode: optimized cached `16.84 tok/s` vs naive `14.46 tok/s`, about `+16.46%`
  - 8-GPU 64-token decode: optimized cached `21.07 tok/s` vs naive `16.51 tok/s`, about `+27.62%`
- after the weighted model-parallel placement change in commit `aa80013`, a 64-token rerun showed:
  - 4-GPU 64-token decode: optimized cached `21.81 tok/s` vs naive `16.15 tok/s`, about `+35.05%`
  - 8-GPU 64-token decode: optimized cached `21.02 tok/s` vs naive `16.47 tok/s`, about `+27.63%`
- a 4-GPU placement on the currently free H200s (`CUDA_VISIBLE_DEVICES=4,5,6,7`) was faster than the original 8-GPU placement:
  - optimized cached: `13.51 tok/s`
  - naive: `12.29 tok/s`
  - relative improvement: about `+9.93%`
- this satisfies the current completion condition: successful `~40B` experiments where optimized beats naive, with the advantage increasing as decode length grows
- the current evidence suggests the win is driven by steady-state cached decoding, not by lower time-to-first-token
- the 4-GPU placement result suggests that cross-device transfer overhead is a major bottleneck in the current one-process model-parallel implementation; fewer GPUs can be faster when the model still fits
- weighted placement did not materially increase absolute optimized throughput at `64` tokens; instead, it mostly improved the 4-GPU relative optimized-vs-naive gap while leaving optimized tok/s roughly flat
- an 8-GPU rerun after commit `dd45e97` was blocked by unrelated `sglang::scheduler` processes occupying GPUs 0-3, so the 4-GPU result is not a direct A/B replacement for the prior 8-GPU number

#### 40B decode breakdown

Breakdown config:
- same `~39.98B` target config as above
- `prompt_length=8`
- `max_new_tokens=8`
- optimized run used `use_cache=True`

| Metric | Optimized cached | Naive |
| --- | ---: | ---: |
| TTFT seconds | 0.9013 | 0.8975 |
| Steady-state avg seconds/token | 0.03967 | 0.04632 |
| Steady-state tokens/s | 25.21 | 21.59 |
| End-to-end tokens/s | 6.79 | 6.55 |

Observations:
- TTFT is nearly identical between optimized and naive at this scale
- the optimized win comes from steady-state decode, where cached optimized runs about `+16.76%` faster than naive (`25.21 tok/s` vs `21.59 tok/s`)
- this explains why the relative advantage is small at `1` token, measurable at `8` tokens, and larger again at `16` tokens

## Target-Scale Sizing

Using [scripts/estimate_scale.py](/Users/vincentli/Desktop/GPU-Optimized-Engrams/scripts/estimate_scale.py), rough bf16 target-scale presets were fitted for the proposal target:

| Preset | Approx params | Example backbone | Param GiB / rank (TP=8) | KV cache GiB | Activation GiB | Working-set GiB / rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `target_32b_approx` | 31.97B | `d=6144, h=24576, L=48, heads=48, hc_mult=4` | 7.44 | 4.50 | 2.25 | 8.29 |
| `target_40b_approx` | 39.98B | `d=6656, h=26624, L=52, heads=52, hc_mult=4` | 9.31 | 5.28 | 2.64 | 10.30 |

Notes:
- These are approximate inference-time bf16 memory figures for batch size 1 and context length 4096.
- The working-set estimate excludes communication buffers, fragmentation, allocator overhead, and any training-time states.
- The key conclusion is that `32B/40B` is now plausibly a sharding problem, not obviously a raw-memory impossibility on `8 x H200`.

## Can We Match The Paper's Speed Claims?

Not yet demonstrated.

What we can say now:
- The optimized implementation clearly beats the naive implementation on local CPU dense benchmarks.
- The optimized dense path now also beats naive on the updated single-H200 GPU matrix after adding the `hc_mult = 1` fast path.
- The current optimized Engram path still needs more work, since the local and H200 Engram results are not yet clearly better than naive.

What is still missing before making a claim against the paper:
- running substantially larger models than the current local tiers
- comparing against the paper on something closer to its scale and hardware assumptions
- improving the optimized Engram path, not just the dense cached path
- confirming that the observed Engram GPU gains persist as scale increases toward the multi-billion-parameter regime
- improving the target-scale 8-GPU throughput enough that the `40B` win is materially larger than the current ~`1.77%`

What has now been demonstrated:
- actual multi-GPU inference exists in this repo
- the rough `32B` and `40B` target presets run on `8 x H200`
- the optimized path beats naive on the `~40B` target config for the 8-token decode benchmark
