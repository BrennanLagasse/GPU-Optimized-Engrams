# Continuous Refill Implementation Report

Timestamp: 2026-04-21 20:30 EDT

## Summary

The serving path now has an implemented continuous-refill mode for the optimized cached model. This is the first measured-code implementation of the scheduler family that had previously only existed in simulation reports.

The implementation is intentionally narrower than a production vLLM-style paged KV cache. It uses dense per-layer cache tensors, but each request has explicit cache slot metadata. That makes it possible to refill a completed slot without flushing the other active slots.

## What Changed

### Cache Slot Metadata

`MultiHeadAttention.forward(...)` now accepts:

```python
cache_positions: Tensor[B]
cache_row_indices: Tensor[B]
```

`cache_positions` gives each source row's token position. `cache_row_indices` maps each source row to a persistent cache slot. This is what allows a request in source row 0 to write into cache slot 7, or a decode microbatch containing slots `[2, 0, 5]` to read/write the correct KV rows.

The attention cache now tracks:

- `cache_k`
- `cache_v`
- `cache_lengths`

`cache_lengths` is indexed by persistent cache slot, not just by the current batch row. Resetting a slot sets its valid length to zero; stale KV values can remain allocated because the attention mask hides them.

### Engram ShortConv Slot State

Engram short-conv cache state now also supports `cache_row_indices`. This matters because the optimized cached Engram path stores local convolution history across decode steps. Without slot-indexed short-conv state, refilling non-contiguous slots would mix request histories.

### Continuous Serving Mode

`scripts/benchmark_serving.py` now supports:

```bash
--decode-mode continuous
```

This mode is currently supported for `--model-impl optimized_cached`.

The scheduler:

1. Orders requests by the selected policy.
2. Fills up to `batch_size` persistent slots.
3. Runs exact per-request prefill into each slot.
4. Decodes active slots as a batch.
5. When a request finishes, resets only that slot.
6. Immediately admits the next queued request into the freed slot.

For the realistic scheduler, ordering uses known input length only. Oracle policies such as `longest_output_first` remain diagnostic because output length is not known before generation completes in a real serving system.

## Validation

Commands run locally:

```bash
python -m py_compile engrams_kv_moe.py scripts/benchmark_serving.py scripts/report_serving_optimization_sweep.py test_engrams.py
conda run -n ai_infra_env_new pytest -q test_engrams.py
```

Result:

```text
24 passed in 73.65s
```

Local continuous serving smoke:

```bash
conda run -n ai_infra_env_new python scripts/benchmark_serving.py \
  --preset tiny_engram \
  --device-groups cpu \
  --dtype float32 \
  --model-impl optimized_cached \
  --batch-size 2 \
  --policy longest_input_first \
  --decode-mode continuous \
  --num-requests 5 \
  --mean-input-tokens 8 \
  --mean-output-tokens 6 \
  --max-input-tokens 16 \
  --max-output-tokens 12 \
  --seed 7 \
  --output results/local_continuous_smoke.json
```

Smoke result:

- serving compute time: `0.118027s`
- requested output tokens: `30`
- executed decode tokens: `30`
- model preset: `tiny_engram`
- Engram layers enabled through the preset

## Cluster Command

After pulling the branch on the cluster, run the full optimization sweep:

```bash
FORCE_RERUN=1 bash scripts/run_cluster_serving_optimization_sweep.sh
```

To run only the current best realistic continuous candidate:

```bash
MODEL_IMPL=optimized_cached \
POLICY=longest_input_first \
DECODE_MODE=continuous \
BATCH_SIZE=16 \
REPLICA_ASSIGNMENT=round_robin \
OUTPUT=results/serving_opt_sweep_optimized_input_continuous_b16.json \
bash scripts/run_cluster_serving_scheduling.sh
```

To run the oracle diagnostic:

```bash
MODEL_IMPL=optimized_cached \
POLICY=longest_output_first \
DECODE_MODE=continuous \
BATCH_SIZE=16 \
REPLICA_ASSIGNMENT=round_robin \
OUTPUT=results/serving_opt_sweep_optimized_oracle_continuous_b16.json \
bash scripts/run_cluster_serving_scheduling.sh
```

## Expected Comparison

The key measured comparison is against the current best realistic 40B serving result:

- previous best: `optimized_cached + longest_input_first + compact + BATCH_SIZE=16`
- measured time: `163.306s`

Continuous refill should reduce decode padding and avoid per-step physical compaction. It may or may not win on H200 depending on scheduler overhead, exact prefill overhead, and cache slot indexing overhead. The implementation is now in place so this can be measured rather than simulated.

## Remaining Work

- Run H200 40B continuous B8/B16 benchmarks.
- Compare realistic continuous against static and compact.
- Compare oracle continuous only as an upper-bound diagnostic.
- If continuous underperforms, profile slot-indexed attention and Engram short-conv state updates separately.
- Longer-term: replace dense slot tensors with paged KV/cache-block metadata if memory pressure or refill fragmentation becomes the limiting factor.
