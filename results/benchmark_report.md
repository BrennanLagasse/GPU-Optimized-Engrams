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

Cluster benchmarking is pending successful repo bootstrap and benchmark execution on the class cluster.

## Can We Match The Paper's Speed Claims?

Not yet demonstrated.

What we can say now:
- The optimized implementation clearly beats the naive implementation on local CPU dense benchmarks.
- The current optimized Engram path still needs more work, since the local Engram no-cache case is not yet better than naive.
- No trustworthy GPU or target-scale cluster benchmark has been run yet on the authoritative local code.

What is still missing before making a claim against the paper:
- running the benchmark matrix on actual GPU hardware
- running substantially larger models than the current local tiers
- comparing against the paper on something closer to its scale and hardware assumptions
- confirming whether the optimized Engram path, not just the dense cached path, improves at realistic GPU scale
