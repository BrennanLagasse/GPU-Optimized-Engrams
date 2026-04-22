# Prefill/Decode Stage Cost Report

This report uses real benchmark batch timings and estimates a staged prefill/decode pipeline. It is not a full KV-handoff implementation; it measures stage costs from the real model path, then models one prefill worker feeding one decode worker with a fixed per-batch handoff cost.

## Source Benchmark

- input JSON: `results/serving_opt_sweep_optimized_input_compact_b32_greedy_prefill.json`
- model: `optimized_cached`
- scheduler: `longest_input_first`
- decoder: `compact`
- batch/replica: `32`
- measured serving seconds: `131.728`
- requested output tokens: `12800`

## Stage Totals

| Metric | Seconds | Relative to measured serving time |
| --- | ---: | ---: |
| Sum of measured prefill stages | `9.574` | `0.07x` |
| Sum of measured decode stages | `220.153` | `1.67x` |
| Estimated staged pipeline makespan | `227.832` | `1.73x` |

## Estimated Disaggregation Result

| Baseline | Candidate | Baseline seconds | Candidate seconds | Speedup | Serving-time reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| measured current serving | staged prefill/decode estimate | `131.728` | `227.832` | `0.58x` | `-72.96%` |

## Interpretation

- This is a measured stage-cost experiment, not a production disaggregated server.
- A full implementation still needs KV-cache export/import or shared paged-cache blocks.
- For this closed 100-request workload, the measured stage-cost estimate is slower because the best current serving path is decode-dominated and uses two full data-parallel replicas.
- Prefill/decode disaggregation may still be useful for arrival-process workloads, TTFT, or heavier prefill workloads, but it should not be presented as a measured win for the current headline benchmark.
