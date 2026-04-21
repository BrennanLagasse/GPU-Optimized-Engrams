#!/usr/bin/env python3
"""Discrete-event simulation for serving architecture ideas.

This is a planning tool, not a GPU benchmark. It models request arrivals,
prefill/decode stage overlap, continuous refill, and rough tensor-parallel
speed assumptions using calibration from the measured 40B H200 runs.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.serving_scheduler import make_static_batches, summarize_schedule
from scripts.serving_workload import ServingRequest, build_serving_requests


@dataclass(frozen=True)
class TimedRequest:
    request: ServingRequest
    arrival_s: float


@dataclass(frozen=True)
class ArchitectureResult:
    name: str
    family: str
    makespan_s: float
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    mean_ttft_s: float
    p95_ttft_s: float
    completed: int
    assumptions: str


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def build_timed_requests(args: argparse.Namespace) -> list[TimedRequest]:
    requests = build_serving_requests(
        count=args.num_requests,
        mean_input=args.mean_input_tokens,
        mean_output=args.mean_output_tokens,
        max_input=args.max_input_tokens,
        max_output=args.max_output_tokens,
        seed=args.seed,
    )
    rng = random.Random(args.seed + 12345)
    arrivals = []
    now = 0.0
    if args.arrival_mode == "closed":
        arrivals = [0.0 for _ in requests]
    elif args.arrival_mode == "poisson":
        for _ in requests:
            now += rng.expovariate(args.arrival_rate)
            arrivals.append(now)
    elif args.arrival_mode == "bursty":
        for idx, _ in enumerate(requests):
            if idx % args.burst_size == 0:
                now += rng.expovariate(args.arrival_rate)
            arrivals.append(now)
    else:
        raise ValueError(f"unknown arrival mode {args.arrival_mode}")
    return [TimedRequest(request=request, arrival_s=arrival) for request, arrival in zip(requests, arrivals)]


def prefill_seconds(tokens: int, *, tp_factor: float = 1.0, disaggregated: bool = False) -> float:
    # Calibrated so total prefill for 12.8k requested tokens is materially
    # smaller than decode but still visible in TTFT.
    overhead = 0.18 if disaggregated else 0.12
    return overhead + tokens * 0.0009 / tp_factor


def decode_seconds(tokens: int, *, batch_size: int, tp_factor: float = 1.0) -> float:
    # Decode benefits from batching but saturates; this captures why B1 was
    # slower while very large padded static batches were also not ideal.
    utilization = min(1.0, max(0.32, math.log2(batch_size + 1) / math.log2(17)))
    return tokens * 0.00185 / (utilization * tp_factor)


def handoff_seconds(tokens: int, *, kv_handoff: str) -> float:
    if kv_handoff == "same_device":
        return 0.02
    if kv_handoff == "nvlink":
        return 0.05 + tokens * 0.00004
    if kv_handoff == "host":
        return 0.25 + tokens * 0.00025
    raise ValueError(f"unknown kv handoff mode: {kv_handoff}")


def latency_result(
    *,
    name: str,
    family: str,
    arrivals: dict[int, float],
    first_token: dict[int, float],
    completions: dict[int, float],
    assumptions: str,
) -> ArchitectureResult:
    latencies = [completions[idx] - arrivals[idx] for idx in sorted(completions)]
    ttfts = [first_token[idx] - arrivals[idx] for idx in sorted(first_token)]
    return ArchitectureResult(
        name=name,
        family=family,
        makespan_s=max(completions.values(), default=0.0) - min(arrivals.values(), default=0.0),
        mean_latency_s=statistics.mean(latencies) if latencies else 0.0,
        p50_latency_s=percentile(latencies, 50),
        p95_latency_s=percentile(latencies, 95),
        p99_latency_s=percentile(latencies, 99),
        mean_ttft_s=statistics.mean(ttfts) if ttfts else 0.0,
        p95_ttft_s=percentile(ttfts, 95),
        completed=len(completions),
        assumptions=assumptions,
    )


def simulate_static(timed: list[TimedRequest], *, batch_size: int, num_replicas: int, compact: bool) -> ArchitectureResult:
    requests = [item.request for item in timed]
    arrivals = {item.request.request_id: item.arrival_s for item in timed}
    batches = make_static_batches(
        requests,
        batch_size=batch_size,
        num_replicas=num_replicas,
        policy="longest_input_first",
        seed=0,
    )
    replica_available = [0.0 for _ in range(num_replicas)]
    first_token = {}
    completions = {}
    for batch in batches:
        ready = max(arrivals[request.request_id] for request in batch.requests)
        start = max(replica_available[batch.replica_id], ready)
        prefill = prefill_seconds(batch.padded_prefill_tokens)
        decode_tokens = batch.requested_output_tokens if compact else batch.padded_decode_tokens
        decode = decode_seconds(decode_tokens, batch_size=batch.batch_size)
        done = start + prefill + decode
        for request in batch.requests:
            first_token[request.request_id] = start + prefill
            completions[request.request_id] = done
        replica_available[batch.replica_id] = done
    return latency_result(
        name=f"static_b{batch_size}_{'compact' if compact else 'padded'}",
        family="static",
        arrivals=arrivals,
        first_token=first_token,
        completions=completions,
        assumptions="Static batches; compact uses requested decode tokens but still completes per batch.",
    )


def simulate_continuous(timed: list[TimedRequest], *, slots: int, tp_factor: float) -> ArchitectureResult:
    arrivals = {item.request.request_id: item.arrival_s for item in timed}
    slot_heap = [0.0 for _ in range(slots)]
    heapq.heapify(slot_heap)
    first_token = {}
    completions = {}
    for item in sorted(timed, key=lambda value: (value.arrival_s, -value.request.input_length)):
        slot_ready = heapq.heappop(slot_heap)
        start = max(slot_ready, item.arrival_s)
        prefill = prefill_seconds(item.request.input_length, tp_factor=tp_factor)
        decode = decode_seconds(item.request.output_length, batch_size=slots, tp_factor=tp_factor)
        first_token[item.request.request_id] = start + prefill
        done = start + prefill + decode
        completions[item.request.request_id] = done
        heapq.heappush(slot_heap, done)
    return latency_result(
        name=f"continuous_slots{slots}_tp{tp_factor:.2f}",
        family="continuous_paged_kv",
        arrivals=arrivals,
        first_token=first_token,
        completions=completions,
        assumptions="Idealized per-row/paged KV continuous batching; no physical cache compaction.",
    )


def simulate_disaggregated(
    timed: list[TimedRequest],
    *,
    prefill_workers: int,
    decode_slots: int,
    kv_handoff: str,
    tp_factor: float,
) -> ArchitectureResult:
    arrivals = {item.request.request_id: item.arrival_s for item in timed}
    prefill_heap = [0.0 for _ in range(prefill_workers)]
    decode_heap = [0.0 for _ in range(decode_slots)]
    heapq.heapify(prefill_heap)
    heapq.heapify(decode_heap)
    first_token = {}
    completions = {}
    for item in sorted(timed, key=lambda value: (value.arrival_s, -value.request.input_length)):
        prefill_ready = heapq.heappop(prefill_heap)
        prefill_start = max(prefill_ready, item.arrival_s)
        prefill_done = prefill_start + prefill_seconds(
            item.request.input_length,
            tp_factor=tp_factor,
            disaggregated=True,
        )
        heapq.heappush(prefill_heap, prefill_done)

        decode_ready = heapq.heappop(decode_heap)
        handoff = handoff_seconds(item.request.input_length, kv_handoff=kv_handoff)
        decode_start = max(decode_ready, prefill_done + handoff)
        first_token[item.request.request_id] = decode_start
        done = decode_start + decode_seconds(item.request.output_length, batch_size=decode_slots, tp_factor=tp_factor)
        completions[item.request.request_id] = done
        heapq.heappush(decode_heap, done)
    return latency_result(
        name=f"disagg_prefill{prefill_workers}_decode{decode_slots}_{kv_handoff}_tp{tp_factor:.2f}",
        family="prefill_decode_disagg",
        arrivals=arrivals,
        first_token=first_token,
        completions=completions,
        assumptions="Separate prefill/decode pools with KV handoff; assumes paged/per-row KV state.",
    )


def simulate_decode_microbatch(timed: list[TimedRequest], *, prefill_batch: int, decode_batch: int) -> ArchitectureResult:
    arrivals = {item.request.request_id: item.arrival_s for item in timed}
    first_token = {}
    completions = {}
    now = 0.0
    queue = sorted(timed, key=lambda value: (value.arrival_s, -value.request.input_length))
    cursor = 0
    decode_ready: list[TimedRequest] = []
    while cursor < len(queue) or decode_ready:
        if not decode_ready:
            now = max(now, queue[cursor].arrival_s)
        prefill_group = []
        while cursor < len(queue) and len(prefill_group) < prefill_batch and queue[cursor].arrival_s <= now:
            prefill_group.append(queue[cursor])
            cursor += 1
        if prefill_group:
            padded_prefill = len(prefill_group) * max(item.request.input_length for item in prefill_group)
            now += prefill_seconds(padded_prefill)
            decode_ready.extend(prefill_group)
            for item in prefill_group:
                first_token[item.request.request_id] = now
        decode_ready.sort(key=lambda item: (item.request.output_length, item.request.request_id), reverse=True)
        decode_group = decode_ready[:decode_batch]
        decode_ready = decode_ready[decode_batch:]
        if decode_group:
            padded_decode = len(decode_group) * max(item.request.output_length for item in decode_group)
            now += decode_seconds(padded_decode, batch_size=len(decode_group))
            for item in decode_group:
                completions[item.request.request_id] = now
    return latency_result(
        name=f"decode_microbatch_prefill{prefill_batch}_decode{decode_batch}",
        family="decode_microbatch",
        arrivals=arrivals,
        first_token=first_token,
        completions=completions,
        assumptions="Two-phase model: large prefill batches then smaller decode batches. Requires KV regrouping.",
    )


def result_to_dict(result: ArchitectureResult) -> dict[str, float | int | str]:
    return {
        "name": result.name,
        "family": result.family,
        "makespan_s": result.makespan_s,
        "mean_latency_s": result.mean_latency_s,
        "p50_latency_s": result.p50_latency_s,
        "p95_latency_s": result.p95_latency_s,
        "p99_latency_s": result.p99_latency_s,
        "mean_ttft_s": result.mean_ttft_s,
        "p95_ttft_s": result.p95_ttft_s,
        "completed": result.completed,
        "assumptions": result.assumptions,
    }


def scale_result(result: ArchitectureResult, factor: float) -> ArchitectureResult:
    return ArchitectureResult(
        name=result.name,
        family=result.family,
        makespan_s=result.makespan_s * factor,
        mean_latency_s=result.mean_latency_s * factor,
        p50_latency_s=result.p50_latency_s * factor,
        p95_latency_s=result.p95_latency_s * factor,
        p99_latency_s=result.p99_latency_s * factor,
        mean_ttft_s=result.mean_ttft_s * factor,
        p95_ttft_s=result.p95_ttft_s * factor,
        completed=result.completed,
        assumptions=result.assumptions,
    )


def build_results(args: argparse.Namespace) -> list[ArchitectureResult]:
    timed = build_timed_requests(args)
    results = []
    for batch_size in [8, 16]:
        results.append(simulate_static(timed, batch_size=batch_size, num_replicas=2, compact=False))
        results.append(simulate_static(timed, batch_size=batch_size, num_replicas=2, compact=True))
    for slots in [8, 16, 32]:
        results.append(simulate_continuous(timed, slots=slots, tp_factor=1.0))
        results.append(simulate_continuous(timed, slots=slots, tp_factor=1.15))
    for prefill_workers in [1, 2, 4]:
        for decode_slots in [8, 16, 32]:
            for kv_handoff in ["same_device", "nvlink"]:
                results.append(
                    simulate_disaggregated(
                        timed,
                        prefill_workers=prefill_workers,
                        decode_slots=decode_slots,
                        kv_handoff=kv_handoff,
                        tp_factor=1.0,
                    )
                )
    for prefill_batch in [8, 16, 32]:
        for decode_batch in [2, 4, 8]:
            results.append(simulate_decode_microbatch(timed, prefill_batch=prefill_batch, decode_batch=decode_batch))
    if args.calibrate_to_b16_compact:
        baseline = next(result for result in results if result.name == "static_b16_compact")
        scale = 163.306 / baseline.makespan_s
        results = [scale_result(result, scale) for result in results]
    return results


def write_report(path: Path, args: argparse.Namespace, results: list[ArchitectureResult]) -> None:
    sorted_results = sorted(results, key=lambda item: item.makespan_s)
    baseline = next(item for item in results if item.name == "static_b16_compact")
    lines = [
        "# Serving Architecture Simulation",
        "",
        "Timestamp: 2026-04-21 17:30 EDT",
        "",
        "This is a discrete-event planning simulation, not a measured GPU benchmark.",
        "",
        "Absolute times are scaled so `static_b16_compact` matches the measured H200 repeat result `163.306s`; relative ordering still depends on the simulator assumptions.",
        "",
        "## Workload",
        "",
        f"- Requests: `{args.num_requests}`",
        f"- Arrival mode: `{args.arrival_mode}`",
        f"- Arrival rate: `{args.arrival_rate}` requests/s",
        f"- Mean input/output tokens: `{args.mean_input_tokens}` / `{args.mean_output_tokens}`",
        f"- Max input/output tokens: `{args.max_input_tokens}` / `{args.max_output_tokens}`",
        "",
        "## Top Strategies",
        "",
        "| Rank | Strategy | Family | Makespan s | Speedup vs B16 compact | Mean latency s | P95 latency s | Mean TTFT s | P95 TTFT s | Assumptions |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, result in enumerate(sorted_results[:20], start=1):
        speedup = baseline.makespan_s / result.makespan_s if result.makespan_s else 0.0
        lines.append(
            f"| {rank} | `{result.name}` | `{result.family}` | {result.makespan_s:.3f} | "
            f"{speedup:.2f}x | {result.mean_latency_s:.3f} | {result.p95_latency_s:.3f} | "
            f"{result.mean_ttft_s:.3f} | {result.p95_ttft_s:.3f} | {result.assumptions} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The simulation supports the intuition that prefill/decode disaggregation can matter when it is paired with per-row/paged KV and continuous decode slots. It improves both makespan and TTFT in the model because prefill can proceed independently instead of waiting behind decode-heavy batches.",
            "",
            "The result should not be presented as a benchmark. It is an implementation-priority signal: disaggregation is worth building only after the KV cache can represent request-level state and hand off cache blocks between prefill and decode workers.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--mean-input-tokens", type=int, default=128)
    parser.add_argument("--mean-output-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--arrival-mode", choices=["closed", "poisson", "bursty"], default="bursty")
    parser.add_argument("--arrival-rate", type=float, default=2.0)
    parser.add_argument("--burst-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=Path("results/serving_architecture_simulation_2026-04-21.json"))
    parser.add_argument("--report-output", type=Path, default=Path("results/serving_architecture_simulation_report_2026-04-21.md"))
    parser.add_argument("--calibrate-to-b16-compact", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    results = build_results(args)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps([result_to_dict(result) for result in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.report_output, args, results)
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.report_output}")
    for result in sorted(results, key=lambda item: item.makespan_s)[:8]:
        print(f"{result.makespan_s:8.3f}s  {result.name}")


if __name__ == "__main__":
    main()
