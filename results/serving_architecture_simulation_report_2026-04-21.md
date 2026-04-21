# Serving Architecture Simulation

Timestamp: 2026-04-21 17:30 EDT

This is a discrete-event planning simulation, not a measured GPU benchmark.

Absolute times are scaled so `static_b16_compact` matches the measured H200 repeat result `163.306s`; relative ordering still depends on the simulator assumptions.

This report is self-contained enough to interpret the simulation table. It uses the same 100-request long-tail serving workload described below and compares simulated serving architectures that require paged/per-row KV cache support.

Strategy coverage in this simulation:
- `continuous_*`: idealized continuous batching with refill-on-completion and paged/per-row KV.
- `disagg_*`: separate prefill and decode worker pools with KV handoff assumptions.
- `decode_microbatch_*`: large prefill batches followed by smaller decode groups.
- `static_*`: current static-batch family used as calibration and comparison.

## Workload

- Requests: `100`
- Arrival mode: `bursty`
- Arrival rate: `2.0` requests/s
- Mean input/output tokens: `128` / `128`
- Max input/output tokens: `1024` / `1024`

## Top Strategies

| Rank | Strategy | Family | Makespan s | Speedup vs B16 compact | Mean latency s | P95 latency s | Mean TTFT s | P95 TTFT s | Assumptions |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `continuous_slots32_tp1.15` | `continuous_paged_kv` | 17.824 | 9.16x | 2.316 | 5.379 | 1.202 | 2.419 | Idealized per-row/paged KV continuous batching; no physical cache compaction. |
| 2 | `continuous_slots32_tp1.00` | `continuous_paged_kv` | 18.576 | 8.79x | 2.574 | 6.088 | 1.293 | 2.685 | Idealized per-row/paged KV continuous batching; no physical cache compaction. |
| 3 | `continuous_slots16_tp1.15` | `continuous_paged_kv` | 19.318 | 8.45x | 3.334 | 6.573 | 2.220 | 3.843 | Idealized per-row/paged KV continuous batching; no physical cache compaction. |
| 4 | `continuous_slots16_tp1.00` | `continuous_paged_kv` | 21.058 | 7.76x | 4.025 | 7.672 | 2.744 | 4.761 | Idealized per-row/paged KV continuous batching; no physical cache compaction. |
| 5 | `continuous_slots8_tp1.15` | `continuous_paged_kv` | 34.960 | 4.67x | 11.688 | 20.192 | 10.252 | 18.608 | Idealized per-row/paged KV continuous batching; no physical cache compaction. |
| 6 | `continuous_slots8_tp1.00` | `continuous_paged_kv` | 39.307 | 4.15x | 13.775 | 23.749 | 12.122 | 21.887 | Idealized per-row/paged KV continuous batching; no physical cache compaction. |
| 7 | `disagg_prefill4_decode16_same_device_tp1.00` | `prefill_decode_disagg` | 41.393 | 3.95x | 15.638 | 28.101 | 14.357 | 26.573 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |
| 8 | `disagg_prefill4_decode32_same_device_tp1.00` | `prefill_decode_disagg` | 41.393 | 3.95x | 15.638 | 28.101 | 14.357 | 26.573 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |
| 9 | `disagg_prefill4_decode16_nvlink_tp1.00` | `prefill_decode_disagg` | 41.777 | 3.91x | 15.828 | 28.291 | 14.547 | 26.751 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |
| 10 | `disagg_prefill4_decode32_nvlink_tp1.00` | `prefill_decode_disagg` | 41.777 | 3.91x | 15.828 | 28.291 | 14.547 | 26.751 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |
| 11 | `disagg_prefill4_decode8_same_device_tp1.00` | `prefill_decode_disagg` | 42.456 | 3.85x | 16.010 | 28.312 | 14.358 | 26.573 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |
| 12 | `disagg_prefill4_decode8_nvlink_tp1.00` | `prefill_decode_disagg` | 42.643 | 3.83x | 16.200 | 28.488 | 14.548 | 26.751 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |
| 13 | `disagg_prefill2_decode16_same_device_tp1.00` | `prefill_decode_disagg` | 80.545 | 2.03x | 36.096 | 66.372 | 34.815 | 65.202 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |
| 14 | `disagg_prefill2_decode32_same_device_tp1.00` | `prefill_decode_disagg` | 80.545 | 2.03x | 36.096 | 66.372 | 34.815 | 65.202 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |
| 15 | `disagg_prefill2_decode16_nvlink_tp1.00` | `prefill_decode_disagg` | 80.719 | 2.02x | 36.286 | 66.550 | 35.005 | 65.382 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |
| 16 | `disagg_prefill2_decode32_nvlink_tp1.00` | `prefill_decode_disagg` | 80.719 | 2.02x | 36.286 | 66.550 | 35.005 | 65.382 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |
| 17 | `disagg_prefill2_decode8_same_device_tp1.00` | `prefill_decode_disagg` | 81.762 | 2.00x | 36.467 | 66.641 | 34.815 | 65.202 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |
| 18 | `disagg_prefill2_decode8_nvlink_tp1.00` | `prefill_decode_disagg` | 81.949 | 1.99x | 36.657 | 66.819 | 35.005 | 65.382 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |
| 19 | `static_b8_compact` | `static` | 157.788 | 1.03x | 90.030 | 150.257 | 76.896 | 134.195 | Static batches; compact uses requested decode tokens but still completes per batch. |
| 20 | `disagg_prefill1_decode16_same_device_tp1.00` | `prefill_decode_disagg` | 160.328 | 1.02x | 76.977 | 143.626 | 75.696 | 141.371 | Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state. |

## Interpretation

The simulation supports the intuition that prefill/decode disaggregation can matter when it is paired with per-row/paged KV and continuous decode slots. It improves both makespan and TTFT in the model because prefill can proceed independently instead of waiting behind decode-heavy batches.

The result should not be presented as a benchmark. It is an implementation-priority signal: disaggregation is worth building only after the KV cache can represent request-level state and hand off cache blocks between prefill and decode workers.
