# Request-Level KV Cache Progress

Timestamp: 2026-04-21 20:15 EDT

## Motivation

The serving architecture reports identified a concrete blocker for true continuous batching, decode microbatching, and prefill/decode disaggregation: the optimized attention cache used one scalar `ptr_current_pos` for the whole batch. That design assumes every row in the batch is at the same decode position. It cannot safely refill one completed request slot while other rows continue decoding.

This change implements the first lower-level primitive needed to remove that blocker: per-row cache positions and per-row logical cache reset.

## Implemented

- `MultiHeadAttention` now tracks `cache_lengths` as a per-row vector in addition to `cache_k` and `cache_v`.
- `MultiHeadAttention.forward(..., cache_positions=...)` accepts one cache position per batch row.
- The per-row cache path writes each row's new keys/values at its own cache offset.
- The attention mask now combines causal masking with row-validity masking, so stale cache entries from a reused slot are invisible.
- `reset_cache_rows(row_indices)` logically clears selected rows by setting their valid cache length to zero.
- `compact_cache(active_indices)` also compacts `cache_lengths`.
- `EngramsModel.forward(..., cache_positions=...)` builds per-row position IDs so reused slots restart at position zero while continuing slots keep their existing position.
- `TransformerBlock` passes `cache_positions` into attention.
- `ShortConv`, `Engram`, `TransformerBlock`, and `EngramsModel` expose row-reset methods so higher-level schedulers can use a common cache-management API.

## Correctness Coverage

Two tests were added:

- `test_attention_per_row_cache_positions_allow_slot_refill`: verifies that a batch slot can be reset and reused while another slot continues, matching full-context attention outputs.
- `test_model_per_row_cache_positions_allow_slot_refill_no_engrams`: verifies the same refill behavior through the full no-Engram model path, comparing cached per-row logits to full-context logits.

Validation commands:

```bash
python -m py_compile engrams_kv_moe.py test_engrams.py
conda run -n ai_infra_env_new pytest -q test_engrams.py -k "per_row_cache_positions or model_per_row_cache_positions or generate_with_and_without_cache_match"
conda run -n ai_infra_env_new pytest -q test_engrams.py
```

Results:

- targeted tests: `4 passed, 19 deselected`
- full local test file: `23 passed in 77.91s`

## Current Limitations

This is a primitive, not a complete serving architecture.

- The Engram-enabled path still needs request-level hash/window state before slot refill can be claimed correct for Engram layers.
- The cache is still a dense per-layer tensor, not a paged allocator.
- There is no active-slot scheduler that admits new requests into completed rows yet.
- There is no KV handoff between prefill and decode workers.
- There is no decode microbatch queue that groups active rows by stage or device.
- The per-row path currently assumes the rows in a call share the same token-step width, which is sufficient for single-token decode refill but not a general ragged prefill API.

## Next Implementation Steps

1. Build a continuous-batching scheduler that maintains slot metadata: request id, input length, generated length, target/max output length, cache position, and active/done state.
2. Use `reset_cache_rows(...)` and `cache_positions` to refill completed slots without flushing the whole batch.
3. Add Engram request-state support so hash preparation and short-conv state are correct for refilled slots.
4. Add a benchmark mode comparing static batching, compact batching, and true refill-on-completion using the same 100-request workload.
5. Only after that, revisit prefill/decode disaggregation and decode microbatching, because both need this request-level cache metadata.
