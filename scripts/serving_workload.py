from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ServingRequest:
    request_id: int
    input_length: int
    output_length: int


def _scaled_long_tail_lengths(
    *,
    count: int,
    mean: int,
    max_length: int,
    seed: int,
    force_max: bool,
) -> list[int]:
    if count <= 0:
        return []
    if mean <= 0:
        raise ValueError("mean must be positive")
    if max_length < mean:
        raise ValueError("max_length must be >= mean")

    rng = random.Random(seed)
    raw = [rng.paretovariate(1.35) for _ in range(count)]
    start = 1 if force_max else 0

    def build(scale: float) -> list[int]:
        lengths = [min(max_length, max(1, int(round(value * scale)))) for value in raw]
        if force_max:
            lengths[0] = max_length
        return lengths

    low = 0.01
    high = float(mean)
    for _ in range(80):
        trial = build(high)
        if sum(trial[start:]) / max(count - start, 1) >= mean:
            break
        high *= 2.0

    target_total = mean * count
    if force_max:
        target_total -= max_length
        target_count = count - 1
    else:
        target_count = count
    target_mean = max(1.0, target_total / max(target_count, 1))

    for _ in range(80):
        mid = (low + high) / 2.0
        trial = build(mid)
        avg = sum(trial[start:]) / max(count - start, 1)
        if avg < target_mean:
            low = mid
        else:
            high = mid

    lengths = build(high)

    # Deterministically nudge non-forced entries so the requested mean is exact
    # unless clipping makes that impossible.
    delta = mean * count - sum(lengths)
    indices = list(range(start, count))
    rng.shuffle(indices)
    cursor = 0
    while delta != 0 and indices:
        idx = indices[cursor % len(indices)]
        if delta > 0 and lengths[idx] < max_length:
            lengths[idx] += 1
            delta -= 1
        elif delta < 0 and lengths[idx] > 1:
            lengths[idx] -= 1
            delta += 1
        cursor += 1
        if cursor > len(indices) * max_length * 2:
            break

    return lengths


def build_serving_requests(
    *,
    count: int = 100,
    mean_input: int = 128,
    mean_output: int = 128,
    max_input: int = 1024,
    max_output: int = 1024,
    seed: int = 0,
) -> list[ServingRequest]:
    input_lengths = _scaled_long_tail_lengths(
        count=count,
        mean=mean_input,
        max_length=max_input,
        seed=seed,
        force_max=True,
    )
    output_lengths = _scaled_long_tail_lengths(
        count=count,
        mean=mean_output,
        max_length=max_output,
        seed=seed + 10_000,
        force_max=True,
    )
    return [
        ServingRequest(
            request_id=idx,
            input_length=input_lengths[idx],
            output_length=output_lengths[idx],
        )
        for idx in range(count)
    ]


def summarize_requests(requests: Iterable[ServingRequest]) -> dict[str, float | int]:
    items = list(requests)
    if not items:
        return {
            "num_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "mean_input_tokens": 0.0,
            "mean_output_tokens": 0.0,
            "max_input_tokens": 0,
            "max_output_tokens": 0,
        }
    total_input = sum(item.input_length for item in items)
    total_output = sum(item.output_length for item in items)
    return {
        "num_requests": len(items),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "mean_input_tokens": total_input / len(items),
        "mean_output_tokens": total_output / len(items),
        "max_input_tokens": max(item.input_length for item in items),
        "max_output_tokens": max(item.output_length for item in items),
    }


def write_requests(path: Path, requests: Iterable[ServingRequest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(request) for request in requests]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_requests(path: Path) -> list[ServingRequest]:
    payload = json.loads(path.read_text())
    return [ServingRequest(**item) for item in payload]
