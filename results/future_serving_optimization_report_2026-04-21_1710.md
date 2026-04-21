# Future Serving Optimization Report

Timestamp: 2026-04-21 17:10 EDT

Branch: `engrams-baseline-benchmarking`

## Scope

This report records the follow-up attempts for the larger optimization ideas:
- continuous batching
- paged/per-row KV cache
- decode microbatch scheduling
- prefill/decode disaggregation
- tensor parallelism
- external serving baselines
- cost-model-driven scheduler search

## What Was Implemented

Added `scripts/simulate_serving_strategies.py`.

The script uses the existing deterministic 100-request workload and measured 40B serving results as calibration. It preserves measured static/compact cases when available, then estimates unimplemented scheduler families. These estimates are planning signals only, not benchmark claims.

Generated report:
- [serving_strategy_cost_model_2026-04-21.md](/Users/vincentli/Desktop/GPU-Optimized-Engrams/results/serving_strategy_cost_model_2026-04-21.md)

Added architecture probe:
- [serving_architecture_probe_2026-04-21.md](/Users/vincentli/Desktop/GPU-Optimized-Engrams/results/serving_architecture_probe_2026-04-21.md)

## Cost-Model Result

The top estimated strategy family is idealized continuous refill with a paged/per-row KV cache:
- Estimated time for 64-token prefill chunks: about `156.995s`
- Current best measured deployable result: `163.306s` for `BATCH_SIZE=16`, `DECODE_MODE=compact`
- Estimated improvement if implemented cleanly: about `1.04x`, or roughly `3.9%` serving-time reduction

This is not a large predicted gain, but it is the only remaining path that directly addresses decode waste without per-step physical cache compaction.

## Feasibility Findings

Continuous batching, decode microbatch scheduling, and prefill/decode disaggregation all require the same missing primitive: request-level KV-cache state.

The current attention cache uses:
- `cache_k`: `[batch, cache_position, heads, head_dim]`
- `cache_v`: `[batch, cache_position, heads, head_dim]`
- `ptr_current_pos`: one scalar shared by the full batch

That design supports static batching and compacting surviving rows, but it cannot safely admit a new request into one completed row while other rows continue from longer cached histories.

Required next abstraction:
- per-row cache positions, or
- paged/cache-block metadata, plus
- request-to-cache-slot indirection, plus
- attention masking keyed by per-row lengths

## Ideas Tried

`1. Continuous batching`

Result: not implemented as a GPU benchmark because the current shared-position cache would make row refill incorrect. Cost model says it is still the most promising next family if we first implement per-row/paged KV.

`4. Paged-attention-style cache layout`

Result: identified as the necessary enabling refactor. Not implemented in this pass because it touches attention cache semantics across the model and requires parity tests before benchmarking.

`5. Decode microbatch scheduler`

Result: same blocker as continuous batching. Separating prefill groups from decode microbatches requires moving request-level KV state between groups.

`6. Prefill/decode disaggregation`

Result: not useful as a small patch before request-level KV state exists. It requires KV handoff between workers/GPU groups.

`7. Tensor parallelism`

Result: not implemented in this pass. The repo currently uses contiguous layer placement, which is memory-oriented model parallelism. Tensor parallelism would require sharded linear layers, communication collectives, and new parity/performance tests.

`8. External serving baseline`

Result: checked local availability. `vllm` and `sglang` are not installed locally. I did not install them because the custom Engram architecture cannot be benchmarked fairly in those runtimes without model integration work.

`9. Cost-model-driven scheduler search`

Result: implemented. It suggests idealized continuous refill / paged KV is the only remaining scheduler family likely to beat the current best, and the predicted gain is modest unless arrival-process workloads reveal larger benefits.

## Recommendation

Do not spend more time on static-batch scheduler tweaks unless we need extra repeats for variance. The next meaningful optimization is a per-row or paged KV-cache refactor, followed by true continuous batching.

If the project scope cannot support that refactor, the current measured best should stand:
- `MODEL_IMPL=optimized_cached`
- `POLICY=longest_input_first`
- `DECODE_MODE=compact`
- `BATCH_SIZE=16`
- Serving time: `163.306s` repeat result
- Speedup over naive random static: `29.90x`

## Validation

Commands run:

```bash
python -m py_compile scripts/simulate_serving_strategies.py
python scripts/simulate_serving_strategies.py --output results/serving_strategy_cost_model_2026-04-21.md
conda run -n ai_infra_env_new pytest -q test_serving_scheduler.py
```

Validation result:
- `test_serving_scheduler.py`: `6 passed`
