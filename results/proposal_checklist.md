# Proposal Checklist Review

Source documents:
- [docs/proposal/main.tex](docs/proposal/main.tex)
- [docs/Engram_paper.pdf](docs/Engram_paper.pdf)
- [docs/mHC- Manifold-Constrained Hyper-Connections.pdf](docs/mHC-%20Manifold-Constrained%20Hyper-Connections.pdf)

Status legend:
- `Complete`: implemented and validated in this repo
- `Partial`: implemented in a narrower or simpler form than the proposal/paper vision
- `Not done`: not implemented in this repo

## Proposal Schedule Items

### 1. Implement Baseline Engrams Architecture

Status: `Complete` for inference, `Partial` overall

What is complete:
- working Engram transformer backbone in [engrams_kv_moe.py](engrams_kv_moe.py)
- separate naive baseline in [engrams_naive.py](engrams_naive.py)
- dense FFN and MoE code paths
- mHC-style residual wrapper with width `hc_mult=4`
- KV-cached optimized decoding
- parity tests across naive/optimized overlap in [test_engrams.py](test_engrams.py)

What is only partial:
- architecture is implemented for inference benchmarking, not full training
- MoE exists functionally but has not been a primary target of target-scale benchmarking
- implementation is a practical research baseline, not a faithful reproduction of the paper’s full training/infrastructure stack

### 2. Profile Engrams Architecture

Status: `Partial`

What is complete:
- parameter and memory estimation in [scripts/estimate_scale.py](scripts/estimate_scale.py)
- target-scale approximate `32B` and `40B` presets
- component profiling in:
  - [scripts/profile_engram_components.py](scripts/profile_engram_components.py)
  - [scripts/profile_forward_components.py](scripts/profile_forward_components.py)
  - [scripts/profile_decode_breakdown.py](scripts/profile_decode_breakdown.py)

What is missing:
- no THOP-based verification
- no backward-pass or training-memory profiling
- no full communication/memory-trace accounting across 8 GPUs

### 3. Parallelize Engrams Architecture for Inference

Status: `Partial`, but sufficient for the baseline deliverable

What is complete:
- one-process model-parallel execution using `device_map`
- 32B and 40B approximate configs run on H200 cluster hardware
- optimized beats naive at target scale for longer decode windows
- placement sweeps and benchmark matrix tooling in [scripts/run_target_benchmark_matrix.py](scripts/run_target_benchmark_matrix.py)

What is missing relative to the proposal:
- no torch.distributed / NCCL tensor parallel implementation
- no real pipeline parallel runtime
- no overlap-optimized communication schedule
- no automated search over several parallelism strategies beyond one-process sharding and placement heuristics

### 4. Reach: Parallelize for Training

Status: `Not done`

Missing:
- backward pass benchmarking
- optimizer-state / gradient memory accounting
- distributed training implementation

## Deliverables Review

### Baseline deliverable: host a 32/40B Engram model on 8 H200s and generate responses

Status: `Partial to Complete`

Why:
- the repo can execute rough `32B` and `40B` Engram+mHC target presets on H200 hardware
- target-scale inference benchmarking works
- random-weight generation works
- however, this is still a benchmarking/research codebase, not a polished hosted serving system

### Reach deliverable: training software

Status: `Not done`

## Evaluation Plan Review

### Before/after naive vs optimized comparison

Status: `Complete`

Evidence:
- explicit naive and optimized implementations
- parity tests
- benchmark report comparing the two across local and cluster tiers

### Throughput as primary inference metric

Status: `Complete`

Evidence:
- tokens/s reported throughout [results/benchmark_report.md](results/benchmark_report.md)

### Latency / TTFT if possible

Status: `Partial`

Evidence:
- TTFT and steady-state split exist for the 40B decode breakdown
- not yet a full latency study across placement choices and decode lengths

### Memory comparison for N-gram table vs rest of model

Status: `Partial`

Evidence:
- Engram table sizing is estimated in [scripts/estimate_scale.py](scripts/estimate_scale.py)
- report includes approximate Engram-table memory
- not yet validated with a full live memory breakdown on cluster

## Engram Paper Alignment

### Core Engram ideas

Status: `Mostly complete`

Implemented:
- 2-gram and 3-gram hash-based lookup path
- early Engram layers
- Engram projections and gating into the residual stream
- cached-step Engram decode optimization
- naive vs optimized comparison requested by the proposal

Not implemented or only partial:
- exact paper-scale training setup
- infrastructure-aware host-memory prefetch path discussed in the paper
- full paper evaluation suite or iso-FLOPs quality comparison

### Paper claim: deterministic lookup enables runtime prefetching with negligible overhead

Status: `Partial`

What exists:
- hash/lookups can be precomputed per layer/device

What is missing:
- no explicit host-memory prefetch implementation
- no communication/runtime system that overlaps lookup prefetch with compute in the paper’s sense

## mHC Paper Alignment

### Core mHC ideas

Status: `Mostly complete for inference`

Implemented:
- multi-branch residual stream with width `hc_mult`
- learned pre/post aggregation
- bistochastic residual mixing via Sinkhorn-style normalization
- width `4` usage, matching the project’s chosen paper-aligned setting

Missing:
- training-stability validation at scale
- paper-style infrastructure optimization for training
- paper-level ablations on depth/scaling behavior

## Bottom Line

What the repo now credibly delivers:
- a working Engram+mHC inference baseline
- a separate naive baseline for fair benchmarking
- target-scale `~32B` and `~40B` approximate inference on H200s
- evidence that optimized beats naive at target scale for longer cached decode

What still separates it from a full proposal/paper completion:
- no training path
- no distributed tensor/pipeline parallel implementation
- no host-memory prefetch system
- no paper-comparable quality evaluation
- no final communication-efficient inference stack beyond the current one-process sharded baseline
