from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from scripts.serving_workload import ServingRequest


SchedulePolicy = Literal["fifo", "longest_output_first", "longest_total_first"]


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
) -> list[ServingRequest]:
    items = list(requests)
    if policy == "fifo":
        return items
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
    policy: SchedulePolicy = "longest_output_first",
) -> list[ScheduledBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_replicas <= 0:
        raise ValueError("num_replicas must be positive")

    ordered = order_requests(requests, policy)
    batches = []
    for batch_id, start in enumerate(range(0, len(ordered), batch_size)):
        chunk = tuple(ordered[start : start + batch_size])
        batches.append(
            ScheduledBatch(
                batch_id=batch_id,
                replica_id=batch_id % num_replicas,
                requests=chunk,
            )
        )
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
