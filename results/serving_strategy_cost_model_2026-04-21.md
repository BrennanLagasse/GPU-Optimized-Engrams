# Serving Strategy Cost-Model Search

Timestamp: 2026-04-21 17:10 EDT

This is a planning report, not a measured GPU benchmark. It uses measured 40B serving points to fit a simple runtime model, then estimates which scheduler families are worth implementing.

## Calibration

- Measured static/compact cases are preserved exactly when available.
- Unmeasured strategies use a conservative heuristic calibrated to the B8/B16 40B runs.
- Intercept seconds: `128.000000`
- Prefill token seconds: `0.001200000`
- Decode token seconds: `0.000450000`
- Small-batch penalty seconds: `100.000000`

## Top Estimates

| Rank | Strategy | Family | Estimated seconds | Prefill tokens | Decode tokens | Notes |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | `continuous_refill_longest_input_first_slots4_chunk64` | `continuous_refill` | 156.995 | 15552 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 2 | `continuous_refill_longest_input_first_slots8_chunk64` | `continuous_refill` | 156.995 | 15552 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 3 | `continuous_refill_longest_input_first_slots12_chunk64` | `continuous_refill` | 156.995 | 15552 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 4 | `continuous_refill_longest_input_first_slots16_chunk64` | `continuous_refill` | 156.995 | 15552 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 5 | `continuous_refill_longest_input_first_slots24_chunk64` | `continuous_refill` | 156.995 | 15552 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 6 | `continuous_refill_longest_input_first_slots32_chunk64` | `continuous_refill` | 156.995 | 15552 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 7 | `continuous_refill_longest_input_first_slots4_chunk128` | `continuous_refill` | 161.346 | 19072 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 8 | `continuous_refill_longest_input_first_slots8_chunk128` | `continuous_refill` | 161.346 | 19072 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 9 | `continuous_refill_longest_input_first_slots12_chunk128` | `continuous_refill` | 161.346 | 19072 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 10 | `continuous_refill_longest_input_first_slots16_chunk128` | `continuous_refill` | 161.346 | 19072 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 11 | `continuous_refill_longest_input_first_slots24_chunk128` | `continuous_refill` | 161.346 | 19072 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 12 | `continuous_refill_longest_input_first_slots32_chunk128` | `continuous_refill` | 161.346 | 19072 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 13 | `static_longest_input_first_compact_b16` | `static` | 163.306 | 24088 | 12800 | Measured-compatible static batching; compact includes observed cache-indexing overhead multiplier. |
| 14 | `static_longest_input_first_static_b8` | `static` | 165.990 | 17024 | 42596 | Measured-compatible static batching; compact includes observed cache-indexing overhead multiplier. |
| 15 | `static_longest_input_first_compact_b12` | `static` | 170.910 | 20408 | 12800 | Measured-compatible static batching; compact includes observed cache-indexing overhead multiplier. |
| 16 | `continuous_refill_longest_input_first_slots4_chunk256` | `continuous_refill` | 173.528 | 28928 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 17 | `continuous_refill_longest_input_first_slots8_chunk256` | `continuous_refill` | 173.528 | 28928 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 18 | `continuous_refill_longest_input_first_slots12_chunk256` | `continuous_refill` | 173.528 | 28928 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 19 | `continuous_refill_longest_input_first_slots16_chunk256` | `continuous_refill` | 173.528 | 28928 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |
| 20 | `continuous_refill_longest_input_first_slots24_chunk256` | `continuous_refill` | 173.528 | 28928 | 12800 | Idealized per-row/paged KV cache. Requires cache refactor; requested prefill=12800. |

## Interpretation

The model consistently ranks idealized continuous refill / paged-KV variants ahead of the current static-batch path because they keep decode work near requested output tokens without per-step physical cache compaction.

These estimates should be used to choose implementation priorities, not as claimable benchmark results. The next real implementation target would be per-row or paged KV-cache metadata, followed by a continuous batching loop.

This model does not yet distinguish slot count within the idealized continuous-refill family. It should be read as a work-reduction estimate, not a capacity/queuing simulator.
