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

## Next Profiling Targets
- Measure the post-fast-path single-H200 benchmark matrix again and compare against the previous GPU results.
- Profile KV-cache behavior: current cache growth still relies on repeated `torch.cat`.
- Profile Engram hash/lookup overhead separately from attention and FFN.
- Once the optimized path wins on single-GPU runs, move to larger scales (`0.5B`, `1B`, `4B`, then higher) before attempting `32B/40B`.
