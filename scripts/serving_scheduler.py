from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Literal

from scripts.serving_workload import ServingRequest


SchedulePolicy = Literal[
    "fifo",
    "random",
    "longest_input_first",
    "shortest_input_first",
    "input_bucketed_random",
    "longest_output_first",
    "longest_total_first",
]

ORACLE_POLICIES = {"longest_output_first", "longest_total_first"}
ReplicaAssignment = Literal["round_robin", "greedy_prefill", "greedy_oracle"]


@dataclass(frozen=True)
class ScheduledBatch:
    batch_id: int
    replica_id: int
    requests: tuple[ServingRequest, ...]

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    @property
    def max_input_length(self) -> int:
        return max(request.input_length for request in self.requests)

    @property
    def max_output_length(self) -> int:
        return max(request.output_length for request in self.requests)

    @property
    def requested_input_tokens(self) -> int:
        return sum(request.input_length for request in self.requests)

    @property
    def requested_output_tokens(self) -> int:
        return sum(request.output_length for request in self.requests)

    @property
    def padded_prefill_tokens(self) -> int:
        return self.batch_size * self.max_input_length

    @property
    def padded_decode_tokens(self) -> int:
        return self.batch_size * self.max_output_length


def order_requests(
    requests: Iterable[ServingRequest],
    policy: SchedulePolicy,
    *,
    seed: int = 0,
) -> list[ServingRequest]:
    items = list(requests)
    if policy == "fifo":
        return items
    if policy == "random":
        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        return shuffled
    if policy == "longest_input_first":
        return sorted(items, key=lambda item: (item.input_length, item.request_id), reverse=True)
    if policy == "shortest_input_first":
        return sorted(items, key=lambda item: (item.input_length, item.request_id))
    if policy == "input_bucketed_random":
        rng = random.Random(seed)
        buckets: dict[int, list[ServingRequest]] = {}
        for item in items:
            bucket = (item.input_length - 1) // 128
            buckets.setdefault(bucket, []).append(item)
        ordered = []
        for bucket in sorted(buckets, reverse=True):
            members = buckets[bucket]
            rng.shuffle(members)
            ordered.extend(members)
        return ordered
    if policy == "longest_output_first":
        return sorted(items, key=lambda item: (item.output_length, item.input_length), reverse=True)
    if policy == "longest_total_first":
        return sorted(
            items,
            key=lambda item: (item.input_length + item.output_length, item.output_length),
            reverse=True,
        )
    raise ValueError(f"unknown schedule policy: {policy}")


def make_static_batches(
    requests: Iterable[ServingRequest],
    *,
    batch_size: int,
    num_replicas: int = 1,
    policy: SchedulePolicy = "longest_input_first",
    replica_assignment: ReplicaAssignment = "round_robin",
    seed: int = 0,
) -> list[ScheduledBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_replicas <= 0:
        raise ValueError("num_replicas must be positive")

    ordered = order_requests(requests, policy, seed=seed)
    chunks = [
        tuple(ordered[start : start + batch_size])
        for start in range(0, len(ordered), batch_size)
    ]
    replica_loads = [0 for _ in range(num_replicas)]
    batches = []
    for batch_id, chunk in enumerate(chunks):
        if replica_assignment == "round_robin":
            replica_id = batch_id % num_replicas
        else:
            replica_id = min(range(num_replicas), key=lambda idx: (replica_loads[idx], idx))
        batch = ScheduledBatch(
            batch_id=batch_id,
            replica_id=replica_id,
            requests=chunk,
        )
        if replica_assignment == "round_robin":
            pass
        elif replica_assignment == "greedy_prefill":
            replica_loads[replica_id] += batch.padded_prefill_tokens
        elif replica_assignment == "greedy_oracle":
            replica_loads[replica_id] += batch.padded_prefill_tokens + batch.padded_decode_tokens
        else:
            raise ValueError(f"unknown replica assignment: {replica_assignment}")
        batches.append(batch)
    return batches


def summarize_schedule(batches: Iterable[ScheduledBatch]) -> dict[str, float | int]:
    items = list(batches)
    if not items:
        return {
            "num_batches": 0,
            "requested_input_tokens": 0,
            "requested_output_tokens": 0,
            "padded_prefill_tokens": 0,
            "padded_decode_tokens": 0,
            "prefill_padding_overhead": 0.0,
            "decode_padding_overhead": 0.0,
        }

    requested_input = sum(batch.requested_input_tokens for batch in items)
    requested_output = sum(batch.requested_output_tokens for batch in items)
    padded_prefill = sum(batch.padded_prefill_tokens for batch in items)
    padded_decode = sum(batch.padded_decode_tokens for batch in items)
    return {
        "num_batches": len(items),
        "requested_input_tokens": requested_input,
        "requested_output_tokens": requested_output,
        "padded_prefill_tokens": padded_prefill,
        "padded_decode_tokens": padded_decode,
        "prefill_padding_overhead": (padded_prefill / requested_input) if requested_input else 0.0,
        "decode_padding_overhead": (padded_decode / requested_output) if requested_output else 0.0,
    }
