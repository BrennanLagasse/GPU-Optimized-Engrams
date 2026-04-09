# Experiment Log

## 2026-04-02 17:25 EDT
- Cluster sync completed to `gpu003` via public GitHub clone into `~/class_projects/GPU-Optimized-Engrams`.
- Cluster hardware confirmed: `8 x NVIDIA H200 143771 MiB`.
- Cluster environment required a user-space Python setup because Conda was unavailable.
- Initial Torch install pulled `2.11.0+cu130`, which failed GPU initialization against the installed driver (`torch.cuda.is_available() == False`).
- Reinstalled Torch from the CUDA 12.8 index, resulting in `torch 2.11.0+cu128` and a working H200 CUDA runtime.
- Cluster validation passed: `pytest -q test_engrams.py` reported `11 passed in 24.49s`.

## 2026-04-02 17:31 EDT
- First single-H200 benchmark matrix collected on commit `1c4f28c`.
- Approximate model sizes used:
  - tiny dense: `150,528` params
  - tiny Engram: `176,720` params
  - medium-local-1 dense: `8.4M` params
  - medium-local-2 dense: `42.0M` params
  - large-local dense: `138.5M` params
- Result summary:
  - local CPU: optimized dense path outperformed naive
  - single-H200 GPU: optimized path underperformed naive across the collected dense and Engram tiers
- Working hypothesis:
  - the current optimized implementation is paying extra overhead from mHC wrappers, cache growth, and host-side Engram hashing that is not offset by any GPU-side savings yet

## 2026-04-02 17:44 EDT
- Added naive mHC support so naive and optimized semantics now align for `hc_mult > 1`.
- Added mHC forward/generation parity tests.
- Local validation after that change: `13 passed in 54.91s`.

## 2026-04-02 18:05 EDT
- Confirmed the mHC paper uses expansion rate `n = 4` for the reported 3B, 9B, and 27B experiments.
- Source reference:
  - mHC paper Appendix Table 5 reports `mHC/HC Expansion Rate n = 4`
  - the main text also states the in-house overhead figure is measured at `n = 4`
- Practical implication for this repo:
  - treat `hc_mult = 4` as the default target configuration for paper-aligned benchmarking
  - keep `hc_mult = 1` only as the degenerate baseline / ablation setting

## 2026-04-02 18:08 EDT
- Implemented an optimized fast path for `hc_mult = 1` in `TransformerBlock.forward`.
- Rationale:
  - when the residual stream width is 1, mHC collapses mathematically to ordinary residual addition
  - the previous code still paid wrapper / routing / aggregation scaffolding overhead in this degenerate case
  - bypassing the wrapper should improve the dense GPU benchmark path and make `hc_mult = 1` a cleaner baseline

## 2026-04-02 18:40 EDT
- Pushed the fast-path and naive-mHC alignment changes to GitHub as commit `1ee7045` and refreshed the cluster clone to that commit.
- Re-ran cluster validation on `gpu003`: `pytest -q test_engrams.py` reported `13 passed in 23.72s`.
- Re-ran the single-H200 benchmark matrix on the updated code.
- Updated single-H200 results:
  - tiny dense (`150,528` params): naive `367.45 tok/s`, optimized no-cache `372.79 tok/s`, optimized cache `368.28 tok/s`
  - tiny Engram (`176,720` params): naive `275.73 tok/s`, optimized no-cache `274.03 tok/s`, optimized cache `237.03 tok/s`
  - tiny mHC (`hc_mult = 4`, about tiny dense-scale backbone): naive `278.55 tok/s`, optimized `278.51 tok/s`
  - medium-local-1 dense (`8.4M` params): naive `388.97 tok/s`, optimized cache `421.82 tok/s`
  - medium-local-2 dense (`42.0M` params): naive `327.36 tok/s`, optimized cache `360.54 tok/s`
  - large-local dense (`138.5M` params): naive `252.73 tok/s`, optimized cache `278.09 tok/s`
- Interpretation:
  - the `hc_mult = 1` fast path fixed the dense GPU regression
  - dense optimized now beats naive on the H200
  - Engram remains the main unresolved GPU bottleneck

## 2026-04-02 18:58 EDT
- Implemented a Torch-native optimized Engram hash path to avoid NumPy-to-Torch conversions during GPU inference.
- Pushed the change as commit `10328c2` and refreshed the cluster clone.
- Re-ran the tiny Engram H200 benchmark (`176,720` params):
  - naive: `279.32 tok/s`
  - optimized no-cache: `256.96 tok/s`
  - optimized cache: `225.21 tok/s`
- Result:
  - the Torch-native hash path did not fix the Engram GPU regression on the tiny benchmark
  - this suggests the remaining bottleneck is not just the CPU/NumPy conversion path
  - next likely targets are the embedding/projection/short-conv path and the cached decode regime itself

## 2026-04-02 19:08 EDT
- Added a single-layer optimized Engram shortcut so the model only precomputes hash tables when multiple Engram layers are present.
- Added `scripts/profile_engram_components.py` to time optimized Engram subcomponents independently:
  - hashing
  - embedding lookup
  - projection/gating
  - short convolution
  - full Engram forward
- Local smoke profile on the tiny Engram config (`176,720` params, CPU) completed successfully.
- Local validation after the shortcut/profiler changes: `13 passed in 45.12s`.

## 2026-04-02 19:16 EDT
- Added a dedicated `hc_mult = 1` fast path inside `ShortConv.forward`.
- Motivation:
  - Engram always uses `ShortConv(..., hc_mult = 1)`
  - the previous implementation still looped over groups, normalized into a list, and concatenated, even when there was only one group
  - the new path removes that extra Python/list/concat work for the Engram case
- Local validation after the change: `13 passed in 62.69s`.

## 2026-04-02 19:23 EDT
- Pulled commit `e595e39` onto the cluster and re-ran the tiny Engram H200 benchmark plus the component profiler.
- Tiny Engram benchmark (`176,720` params) after the single-layer shortcut and `ShortConv` fast path:
  - naive: `277.07 tok/s`
  - optimized no-cache: `255.85 tok/s`
  - optimized cache: `224.34 tok/s`
- Component profile on H200 after the `ShortConv` fast path:
  - hash: `0.000157 s`
  - embedding: `0.000035 s`
  - projection/gating: `0.000102 s`
  - short conv: `0.001661 s`
  - full Engram: `0.000343 s`
- Result:
  - the Engram regression is still present on the tiny H200 benchmark
  - embedding lookup is negligible
  - hashing and projection are small relative to the short-conv measurement
  - the next likely optimization target is not the hash path anymore; it is the Engram compute structure itself, especially the convolution-heavy local path and how it interacts with tiny decode workloads

## 2026-04-03 00:12 EDT
- Added an Engram ablation flag (`use_short_conv`) so the local-mixing `ShortConv` branch can be disabled cleanly for profiling and benchmarking.
- Updated the benchmark and component-profiler scripts to accept `--disable-short-conv`.
- Local smoke checks with `ShortConv` disabled:
  - tiny Engram parameter count dropped from about `176,720` to about `176,528`
  - CPU Engram component profile showed `full_engram_seconds` dropping to about `0.000149 s`
  - local optimized Engram decode without `ShortConv` ran successfully
- Next step:
  - rerun the tiny H200 Engram benchmark with `--disable-short-conv`
  - compare against the standard Engram path to confirm whether the convolution branch is the dominant source of the GPU regression

## 2026-04-03 13:12 EDT
- Pulled commit `12a35f7` onto `gpu003` and ran the tiny Engram short-conv ablation on one H200.
- Tiny Engram benchmark size remained about `176,720` params for the standard path and about `176,528` params with `ShortConv` disabled.
- H200 no-cache ablation results:
  - optimized standard Engram: `254.08 tok/s`
  - optimized Engram with `--disable-short-conv`: `257.84 tok/s`
  - relative change from disabling `ShortConv`: about `+1.48%`
- H200 cache-on ablation results:
  - optimized standard Engram cache: `200.75 tok/s`
  - optimized Engram cache with `--disable-short-conv`: `263.66 tok/s`
  - relative change from disabling `ShortConv`: about `+31.34%`
- H200 component profile with `ShortConv` disabled showed `short_conv_seconds` collapsing to about `5.11e-06 s`.
- Interpretation:
  - `ShortConv` is not the main reason the optimized Engram no-cache path is only near parity with naive on the tiny benchmark
- `ShortConv` is, however, a major contributor to the cached Engram regression on the tiny H200 decode path
- the next Engram optimization pass should focus on cached decode structure and on whether local convolution should be bypassed, fused, or made conditional during inference

## 2026-04-09 11:33 EDT
- Implemented an exact cached-step `ShortConv` fast path in the optimized model.
- Rationale:
  - cached Engram decode already runs single-token steps
  - for `T = 1`, the retained causal depthwise-conv output depends only on the last kernel tap and the current token
  - the old code still launched the full `Conv1d` kernel for that case
- Added `cached_inference_short_conv_mode` to `EngramConfig` with:
  - `full`: always run the full local-mixing path
  - `step_kernel`: use the exact last-tap cached-step fast path
  - `gated_value_only`: experimental cached-step approximation that drops local mixing entirely
- Added `scripts/profile_cached_engram_decode.py` to compare:
  - cached vs no-cache decode throughput
  - exact-vs-approximate cached-step modes
  - cached generation parity and logit drift against the full mode
- Local validation after the change: `16 passed in 62.27s`.
- Local CPU component profile on the tiny Engram config (`176,720` params):
  - `short_conv_seconds`: about `0.001967 s`
  - `short_conv_step_seconds`: about `0.000049 s`
  - `full_engram_seconds`: about `0.002380 s`
  - `full_engram_cached_step_seconds`: about `0.000181 s`
- Local CPU decode profile on the tiny Engram config:
  - baseline `full` cached decode: `244.41 tok/s`
  - exact `step_kernel` cached decode: `1017.88 tok/s`
  - cached improvement for `step_kernel`: about `+316.46%`
  - `step_kernel` cached generation parity: exact (`cached_generation_equal = true`, max cached-step logit delta `0.0`)
  - experimental `gated_value_only` cached decode: `1062.22 tok/s`
  - cached improvement for `gated_value_only`: about `+332.20%`
  - `gated_value_only` cached generation parity: not exact (`cached_generation_equal = false`, max cached-step logit delta about `0.197`)
- Interpretation:
  - `step_kernel` is the right optimized path so far because it preserves cached behavior exactly while removing most of the cached local-mixing overhead
  - `gated_value_only` is faster but is now clearly an approximation, not a drop-in optimized path

## 2026-04-09 16:31 EDT
- Pulled commit `c1a309b` onto `gpu003` and reran the cached Engram decode comparison on one H200.
- Tiny Engram benchmark size remained about `176,720` params.
- H200 cached decode comparison against the original `full` cached local-mixing path:
  - baseline cached throughput: `327.67 tok/s`
  - exact `step_kernel` cached throughput: `1118.50 tok/s`
  - cached improvement for `step_kernel`: about `+241.35%`
  - `step_kernel` parity: exact (`cached_generation_equal = true`, max cached-step logit delta `0.0`)
- H200 approximate cached decode comparison for `gated_value_only`:
  - baseline cached throughput on that run: `654.82 tok/s`
  - candidate cached throughput: `1178.73 tok/s`
  - cached improvement for `gated_value_only`: about `+80.01%`
  - `gated_value_only` parity: not exact (`cached_generation_equal = false`, max cached-step logit delta about `0.351`)
- Additional observation:
  - no-cache candidate throughput also jumped in these profiler runs because the profile script measures end-to-end generation in separate fresh processes, so absolute no-cache numbers should be interpreted less strictly than the cached parity/speed comparison.
- Interpretation:
  - the exact `step_kernel` path fixes the main cached Engram regression on H200 while preserving behavior
  - `gated_value_only` remains an approximation and should stay experimental only

## Next Profiling Targets
- Measure the post-fast-path single-H200 benchmark matrix again and compare against the previous GPU results.
- Profile KV-cache behavior: current cache growth still relies on repeated `torch.cat`.
- Profile Engram hash/lookup overhead separately from attention and FFN.
- Once the optimized path wins on single-GPU runs, move to larger scales (`0.5B`, `1B`, `4B`, then higher) before attempting `32B/40B`.
