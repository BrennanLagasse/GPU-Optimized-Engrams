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

## 2026-04-09 18:01 EDT
- Implemented two additional cached-decode optimizations locally and pushed them as commit `7f87091`:
  - preallocated KV-cache buffers in attention instead of repeated `torch.cat`
  - last-token Engram hashing for cached decode via `hash_last_tensor(...)`
- Corrected the cached Engram local-mixing path to keep exact normalized history, so the `step_kernel` path now matches the candidate model's no-cache generation instead of only being "close".
- Local validation after those changes: `18 passed in 69.52s`.
- Re-ran the H200 cached Engram comparison on `gpu003`.
- Tiny Engram H200 results (`176,720` params):
  - `float32` baseline cached `full`: `532.88 tok/s`
  - `float32` exact `step_kernel`: `1102.02 tok/s`
  - `float32` cached improvement: about `+106.81%`
  - `float32` exactness: `candidate_cached_matches_candidate_no_cache = true`
  - `bfloat16` baseline cached `full`: `436.58 tok/s`
  - `bfloat16` exact `step_kernel`: `1047.57 tok/s`
  - `bfloat16` cached improvement: about `+139.95%`
  - `bfloat16` exactness: `candidate_cached_matches_candidate_no_cache = true`
- Medium Engram H200 results (`861,752` params):
  - `float32` baseline cached `full`: `275.34 tok/s`
  - `float32` exact `step_kernel`: `553.94 tok/s`
  - `float32` cached improvement: about `+101.18%`
  - `float32` exactness: `candidate_cached_matches_candidate_no_cache = true`
  - `bfloat16` baseline cached `full`: `244.62 tok/s`
  - `bfloat16` exact `step_kernel`: `531.39 tok/s`
  - `bfloat16` cached improvement: about `+117.23%`
  - `bfloat16` exactness: `candidate_cached_matches_candidate_no_cache = true`
- Interpretation:
  - the exact cached-step Engram path now scales beyond the tiny toy setting
  - the current `bfloat16` path is not yet outperforming `float32` on this implementation
  - the most obvious remaining work is to push to larger Engram-enabled tiers and then move to multi-GPU execution

## 2026-04-09 18:18 EDT
- Added `scripts/estimate_scale.py` with named presets for:
  - tiny / medium / large Engram-enabled tiers
  - rough `32B` and `40B` target-scale Engram configurations
- Used the estimator to fit rough target-scale backbones that are close to the proposal target:
  - `target_32b_approx`: about `31.97B` params with `emb_dim=6144`, `hidden_dim=24576`, `n_layers=48`, `n_heads=48`, `hc_mult=4`
  - `target_40b_approx`: about `39.98B` params with `emb_dim=6656`, `hidden_dim=26624`, `n_layers=52`, `n_heads=52`, `hc_mult=4`
- Approximate 8-way tensor-parallel bf16 memory for the rough target presets:
  - `32B`:
    - parameter bytes per rank: about `7.44 GiB`
    - KV cache at batch 1 / context 4096: about `4.50 GiB`
    - activation estimate: about `2.25 GiB`
    - working-set estimate per rank: about `8.29 GiB` (excluding optimizer / fragmentation / communication buffers)
  - `40B`:
    - parameter bytes per rank: about `9.31 GiB`
    - KV cache at batch 1 / context 4096: about `5.28 GiB`
    - activation estimate: about `2.64 GiB`
    - working-set estimate per rank: about `10.30 GiB` (excluding optimizer / fragmentation / communication buffers)
- Interpretation:
  - rough memory sizing suggests that a `32B/40B` random-weight inference run is plausible on `8 x H200` hardware if the model is actually sharded
  - this is now a concrete engineering blocker rather than a memory blocker: the repo still lacks any multi-GPU inference implementation (`torch.distributed`, tensor parallelism, or pipeline parallelism), so a real `40B optimized beats naive` run cannot be executed yet

## 2026-04-09 18:10 EDT
- Implemented a first model-parallel inference path in the repo and pushed it as commit `630ea2a`.
- Main implementation changes:
  - added `device_map` normalization and per-block device placement in the optimized model
  - added equivalent `device_map` support to the naive model so target-scale comparisons remain semantically aligned
  - routed token embeddings / output head to the edge devices and partitioned transformer blocks across the listed GPUs
  - added per-device Engram hash preparation for the optimized path so Engram layers receive local hash tensors on their assigned GPU
  - extended `scripts/benchmark_decode.py` and `scripts/profile_cached_engram_decode.py` to accept `--device-map`
  - added a CPU smoke test for the split-device code path
- Local validation after the model-parallel patch: `19 passed in 80.07s`.
- Local two-way CPU device-map smoke benchmark succeeded for both naive and optimized implementations.

## 2026-04-09 18:20 EDT
- Pulled commit `630ea2a` onto `gpu003` and validated the new multi-GPU execution path.
- Two-GPU smoke benchmark on `CUDA_VISIBLE_DEVICES=0,1` with an Engram-enabled config of about `392,860` params:
  - optimized: `20.31 tok/s`
  - naive: `22.85 tok/s`
- Interpretation:
  - the model-parallel execution path is functional
  - at tiny scale, model-parallel overhead still dominates and optimized remains slower than naive
  - this is acceptable as a smoke result because the goal of this pass was to validate the distributed execution path before attempting the target-scale presets

## 2026-04-09 18:24 EDT
- Ran the rough `32B` target preset across all `8 x H200` with:
  - `dtype=bfloat16`
  - `batch_size=1`
  - `prompt_length=8`
  - `max_new_tokens=1`
  - `use_cache=True`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Approximate model size:
  - `31.97B` params
- Result:
  - optimized cached: `1.17 tok/s` (`avg_seconds = 0.8531`)
- Interpretation:
  - the target-scale sharded path is now actually runnable on the cluster
  - this cleared the immediate blocker to attempting the `40B` target preset

## 2026-04-09 18:27 EDT
- Ran the rough `40B` target preset across all `8 x H200` with the same setup, first as a one-token decode.
- Approximate model size:
  - `39.98B` params
- One-token results:
  - optimized cached: `1.02 tok/s` (`avg_seconds = 0.9768`)
  - naive: `1.03 tok/s` (`avg_seconds = 0.9739`)
- Interpretation:
  - the `40B` target preset runs successfully on the full 8-GPU setup
  - for a single generated token, optimized cached and naive are effectively tied, with naive slightly ahead
  - this is not yet the success condition, because the cache has very little opportunity to amortize its setup at a decode length of 1

## 2026-04-09 18:37 EDT
- Re-ran the same `~40B` target-scale comparison with a longer decode:
  - `batch_size=1`
  - `prompt_length=8`
  - `max_new_tokens=8`
  - optimized with `use_cache=True`
  - naive with `use_cache=False`
- Results:
  - optimized cached: `6.34 tok/s` (`avg_seconds = 1.2623`)
  - naive: `6.23 tok/s` (`avg_seconds = 1.2848`)
  - relative improvement for optimized: about `+1.77%`
- Interpretation:
  - this satisfies the current completion condition: a successful `~40B` run on `8 x H200` where optimized beats naive
  - the win is real but small, which means the next optimization phase should focus on increasing the gap rather than merely proving feasibility
  - the result also validates that the optimized cached path becomes more competitive once decode length is long enough for cache reuse to matter

## 2026-04-09 23:06 EDT
- Added `scripts/profile_decode_breakdown.py` and pushed it as commit `e434961`.
- Purpose:
  - separate TTFT from steady-state cached decode at target scale
  - determine whether the current 40B optimized win comes from cache reuse or from lower first-token latency
- Re-ran the `~40B` target config on all `8 x H200` with `prompt_length=8`, `max_new_tokens=8`, `dtype=bfloat16`.
- Breakdown results:
  - optimized cached:
    - TTFT: `0.9013 s`
    - steady-state average seconds/token: `0.03967 s`
    - steady-state throughput: `25.21 tok/s`
    - end-to-end throughput: `6.79 tok/s`
  - naive:
    - TTFT: `0.8975 s`
    - steady-state average seconds/token: `0.04632 s`
    - steady-state throughput: `21.59 tok/s`
    - end-to-end throughput: `6.55 tok/s`
- Interpretation:
  - TTFT is effectively the same between the two implementations at 40B
  - the optimized win comes from a better steady-state decode regime once cache reuse kicks in
  - steady-state optimized is about `+16.76%` faster than naive on the 40B / 8-token run

## 2026-04-09 23:07 EDT
- Extended the target-scale throughput comparison to `max_new_tokens=16` on the same `~40B` config.
- Results:
  - optimized cached: `10.06 tok/s` (`avg_seconds = 1.5903`)
  - naive: `9.65 tok/s` (`avg_seconds = 1.6578`)
  - relative improvement for optimized: about `+4.25%`
- Interpretation:
  - the optimized advantage widens as decode length increases
  - this supports the breakdown result: the repo’s current target-scale gain is a steady-state cached decoding gain rather than a TTFT gain

## 2026-04-10 03:15 EDT
- Implemented a small model-parallel transfer cleanup and pushed it as commit `dd45e97`:
  - the optimized and naive model-parallel paths now move `engram_input_ids` to a block's device only when that block actually has an Engram layer
  - this avoids redundant token-ID transfers on non-Engram blocks
- Local validation after the change: `19 passed in 104.92s`.
- Attempted to rerun the 8-GPU `~40B` / 16-token optimized benchmark after the cleanup.
- External cluster blocker:
  - GPUs 0-3 were occupied by long-running `sglang::scheduler` processes under the shared `yale` account
  - GPU 0 had about `135 GiB` already allocated by that process group
  - I did not kill those processes because the cluster is shared and they may belong to another classmate
- Fallback experiment:
  - ran the same `~40B` / 16-token benchmark on the currently free GPUs 4-7 using `CUDA_VISIBLE_DEVICES=4,5,6,7`
  - because `CUDA_VISIBLE_DEVICES` remaps logical device IDs, the script used `--device-map cuda:0,cuda:1,cuda:2,cuda:3`
- 4-GPU results:
  - optimized cached: `13.51 tok/s` (`avg_seconds = 1.1840`)
  - naive: `12.29 tok/s` (`avg_seconds = 1.3027`)
  - optimized improvement: about `+9.93%`
- Interpretation:
  - the 4-GPU placement is faster than the earlier 8-GPU placement for the same approximate 40B model and decode length
  - the likely reason is that the current one-process block-sharded implementation pays cross-device activation-transfer overhead at every device boundary
  - when the 40B model fits on 4 H200s, fewer device boundaries can beat using all 8 GPUs
  - this is a useful optimization direction: choose the smallest feasible GPU count for inference, or reduce cross-device activation transfer with a better parallelism strategy
- Attempted to continue to a 32-token 4-GPU benchmark, but the browser terminal session became unstable before the command could focus the terminal input.

## 2026-04-10 10:27 EDT
- Re-authenticated to the browser-only cluster console via Playwright after the Cloudflare session expired.
- Operational note:
  - `keyboard.insertText(...)` did not reliably feed the xterm input
  - `keyboard.type(..., {delay: 1})` did work
  - base64-encoding a temporary benchmark shell script and typing a single `python3 -c ... && bash ...` command was reliable enough for long benchmark launches
- Confirmed cluster state before rerunning:
  - host: `gpu003`
  - user: `yale`
  - GPUs 0-7 were effectively free before launch, with only minimal memory in use
- The cluster clone fast-forwarded to `89df6fd` before the 32-token run, bringing in the latest transfer-cleanup documentation.
- Ran the `~40B` target preset across all `8 x H200` with:
  - `dtype=bfloat16`
  - `batch_size=1`
  - `prompt_length=8`
  - `hc_mult=4`
  - Engram layers at `layer_ids=[0,1]`
  - optimized with `use_cache=True`
  - naive with `use_cache=False`
- 32-token results:
  - optimized cached: `16.67 tok/s` (`avg_seconds = 1.9200`)
  - naive: `14.46 tok/s` (`avg_seconds = 2.2130`)
  - relative improvement for optimized: about `+15.28%`
- 64-token results:
  - optimized cached: `20.85 tok/s` (`avg_seconds = 3.0694`)
  - naive: `16.59 tok/s` (`avg_seconds = 3.8571`)
  - relative improvement for optimized: about `+25.68%`
- Interpretation:
  - the target-scale optimized cached path now shows a clear and growing win at longer decode lengths
  - this is consistent with the earlier TTFT/steady-state breakdown: first-token cost is similar, and the optimized path wins once cached steady-state decoding dominates
  - the 64-token 8-GPU result is the strongest current `~40B` evidence: optimized beats naive by about `25.68%`

## 2026-04-10 10:45 EDT
- Implemented the next local optimization/profiling pass before another cluster run:
  - added a cached single-token attention fast path that skips causal mask construction when the current query is already at the cache tail
  - added `scripts/run_target_benchmark_matrix.py` for reproducible placement and decode-length sweeps across naive and optimized implementations
  - added `scripts/profile_forward_components.py` for synchronization-heavy component timing of embedding, Engram hash prep, block execution, transfers, final head, and argmax
- Local validation:
  - `conda run -n ai_infra_env_new pytest -q test_engrams.py`: `19 passed in 162.95s`
  - `python -m py_compile engrams_kv_moe.py scripts/run_target_benchmark_matrix.py scripts/profile_forward_components.py`: passed
  - explicit optimized cache/no-cache generation parity smoke: `True`
  - CPU smoke for `scripts/run_target_benchmark_matrix.py` on the tiny Engram preset: passed with `--device-groups cpu`
  - CPU smoke for `scripts/profile_forward_components.py` on the tiny Engram preset: passed
- Next cluster step:
  - push this commit
  - pull it on `gpu003`
  - run the placement/decode matrix for `~40B` on 4 and 8 H200s, preferably at 32 and 64 generated tokens first
  - run the component profiler on a shorter `~40B` decode if the matrix suggests remaining transfer or block-level bottlenecks

## 2026-04-10 11:06 EDT
- Ran the new target-scale benchmark matrix on `gpu003` at commit `0199da8`.
- Matrix config:
  - `~39.98B` target preset
  - `dtype=bfloat16`
  - `batch_size=1`
  - `prompt_length=8`
  - `hc_mult=4`
  - Engram layers at `layer_ids=[0,1]`
  - 4-GPU placement: `CUDA_VISIBLE_DEVICES=4,5,6,7`
  - 8-GPU placement: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- Matrix results:
  - 4-GPU, 32-token decode: optimized cached `18.07 tok/s` vs naive `15.41 tok/s`, about `+17.26%`
  - 4-GPU, 64-token decode: optimized cached `21.87 tok/s` vs naive `17.13 tok/s`, about `+27.67%`
  - 8-GPU, 32-token decode: optimized cached `16.84 tok/s` vs naive `14.46 tok/s`, about `+16.46%`
  - 8-GPU, 64-token decode: optimized cached `21.07 tok/s` vs naive `16.51 tok/s`, about `+27.62%`
- Ran a synchronization-heavy component profile on the 4-GPU optimized cached path with `max_new_tokens=2`.
- Component profile highlights:
  - model size: `39,978,501,600` params
  - total profiled time: about `0.624 s`
  - block activation transfers: about `0.292 s`, the largest measured bucket
  - block 0 with Engram work: about `0.125 s`
  - next heaviest plain blocks were around `0.028 s`
- Interpretation:
  - the cached single-token attention-mask skip appears beneficial at scale: the 8-GPU 64-token run improved from `20.85 tok/s` to `21.07 tok/s`, and the 4-GPU 64-token run improved from `21.87 tok/s` vs the prior 16-token baseline
  - the component profile reinforces that cross-device activation movement is now the main target-scale bottleneck for this one-process model-parallel implementation
  - the next safe implementation tweak is to make explicit tensor transfers non-blocking in both optimized and naive paths, then rerun the 64-token placement comparison

## 2026-04-10 11:12 EDT
- Implemented explicit non-blocking tensor moves for model-parallel data movement:
  - input token movement to the first model device
  - Engram input/hash movement to block devices
  - hidden-state movement between block-device partitions
  - final hidden-state movement to the output-head device
- Applied the transfer primitive to both optimized and naive implementations so benchmark comparisons remain fair.
- Updated `scripts/profile_forward_components.py` to measure the same transfer primitive used by the implementation.
- Local validation:
  - `python -m py_compile engrams_kv_moe.py engrams_naive.py scripts/profile_forward_components.py scripts/run_target_benchmark_matrix.py`: passed
  - CPU profiler smoke: passed
  - `conda run -n ai_infra_env_new pytest -q test_engrams.py`: `19 passed in 207.67s`
- Next cluster step:
  - push the non-blocking transfer commit
  - rerun at least the `~40B` 64-token 4-GPU and 8-GPU comparison

## Next Profiling Targets
- Increase the target-scale decode-length benchmark matrix beyond `max_new_tokens=16` to map where the cached optimized gap saturates.
- Reduce model-parallel overhead:
  - measure per-device idle time
  - reduce cross-device transfers around embeddings / output head where possible
- Compare 4-GPU vs 8-GPU placement systematically when the full cluster is free.
- Improve target-scale cached Engram execution further, especially for longer decode windows.
- Add more systematic target-scale reporting:
  - TTFT
  - steady-state tok/s
  - per-rank memory
  - decode-length scaling curves
