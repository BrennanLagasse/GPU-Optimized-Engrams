# Serving Architecture Probe

Timestamp: 2026-04-21 17:10 EDT

## Scope

This note records the feasibility probe for larger serving ideas:
- real continuous batching
- paged/per-row KV cache
- decode microbatch scheduling
- prefill/decode disaggregation
- tensor parallelism
- external serving baselines such as vLLM/SGLang
- cost-model-driven scheduler search

## Findings

### Continuous Batching And Paged KV

The current optimized attention cache stores:
- `cache_k`: `[batch, cache_position, heads, head_dim]`
- `cache_v`: `[batch, cache_position, heads, head_dim]`
- `ptr_current_pos`: one scalar shared by the entire batch

This means every row in a batch is assumed to have the same cached sequence length. Static batches and physical compaction work under that assumption because all surviving rows continue from the same shared position.

True refill-on-completion needs new requests to enter individual finished rows while other rows already have longer decode histories. That requires at least one of:
- per-row cache positions
- per-row attention masks
- paged/cache-block metadata
- a request-to-cache-slot indirection table

Without that refactor, continuous batching would either be incorrect or would need to reset/recompute too much state to be a fair optimization.

### Decode Microbatch Scheduling

Decode microbatching is feasible conceptually but not with the current cache as a small patch. Separating prefill batches from decode microbatches requires moving request state from a large prefill batch into smaller decode groups. That is the same underlying problem as continuous batching: the model needs request-level cache metadata rather than one scalar batch position.

### Prefill/Decode Disaggregation

Prefill/decode disaggregation would require:
- separate worker loops for prefill and decode
- a KV handoff format
- request state transfer between GPU groups
- likely paged KV or per-row metadata

It is unlikely to help before continuous batching is available, because current static batches still bind prefill and decode state together.

### Tensor Parallelism

The repo currently uses contiguous layer placement across devices. This is memory-oriented model parallelism, not tensor parallelism. Tensor parallelism would shard large linear projections and attention heads across GPUs within each layer, which could improve per-token latency and GPU utilization.

This remains a larger implementation project. It would require sharded linear layers, collective communication, and careful parity/performance validation. It is not a safe small scheduler patch.

### External Serving Baselines

vLLM/SGLang-style baselines are useful for scheduling comparison, but the custom Engram architecture does not drop into those runtimes directly. A fair external baseline would require either:
- implementing model integration for the Engram architecture, or
- benchmarking a comparable transformer without Engram and clearly labeling it as a scheduler/runtime baseline rather than an Engram model result.

Local availability check:
- `vllm`: not importable in the local environment
- `sglang`: not importable in the local environment

I did not install them locally because that would not by itself produce a fair Engram-serving baseline; integration work is still required.

## Current Practical Next Step

The most promising next engineering path is a minimal paged/per-row KV-cache abstraction for the optimized model. Once the cache can track request-level positions, we can implement true continuous batching and decode microbatching as measured GPU paths.
