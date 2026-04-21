#!/usr/bin/env python3
"""Summarize serving ablation JSON outputs into a Markdown report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Case:
    label: str
    path: Path


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def serving_seconds(data: dict) -> float:
    return float(data.get("serving_wall_seconds_excluding_model_load", data.get("wall_seconds", 0.0)))


def total_seconds(data: dict) -> float:
    return float(data.get("wall_seconds", 0.0))


def throughput(data: dict) -> float:
    return float(data.get("serving_requested_output_tokens_per_second", 0.0))


def pct_lower(new_seconds: float, old_seconds: float) -> float:
    if old_seconds == 0:
        return 0.0
    return (old_seconds - new_seconds) / old_seconds * 100.0


def speedup(new_seconds: float, old_seconds: float) -> float:
    if new_seconds == 0:
        return 0.0
    return old_seconds / new_seconds


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def case_row(label: str, data: dict | None) -> str:
    if data is None:
        return f"| {label} | missing | - | - | - | - | - | - | - |"
    schedule = data.get("schedule_summary", {})
    return (
        f"| {label} "
        f"| {data.get('model_impl', '-')} "
        f"| {data.get('scheduler_impl', '-')} "
        f"| {data.get('policy', '-')} "
        f"| {data.get('replica_assignment', '-')} "
        f"| {fmt(serving_seconds(data))} "
        f"| {fmt(throughput(data))} "
        f"| {fmt(float(schedule.get('prefill_padding_overhead', 0.0)), 3)}x "
        f"| {fmt(float(schedule.get('decode_padding_overhead', 0.0)), 3)}x |"
    )


def comparison_row(label: str, old: dict | None, new: dict | None) -> str:
    if old is None or new is None:
        return f"| {label} | missing | - | - | - |"
    old_s = serving_seconds(old)
    new_s = serving_seconds(new)
    old_t = throughput(old)
    new_t = throughput(new)
    throughput_gain = ((new_t / old_t) - 1.0) * 100.0 if old_t else 0.0
    return (
        f"| {label} "
        f"| {fmt(speedup(new_s, old_s))}x "
        f"| {fmt(pct_lower(new_s, old_s))}% "
        f"| {fmt(old_t)} -> {fmt(new_t)} "
        f"| {fmt(throughput_gain)}% |"
    )


def write_report(cases: Iterable[Case], output: Path) -> None:
    loaded = {case.label: load_json(case.path) for case in cases}
    lines = [
        "# Serving Ablation Matrix",
        "",
        "This report compares model implementation, scheduling policy, and replica-assignment effects for the 100-request target-scale serving workload.",
        "",
        "## Cases",
        "",
        "| Case | Model | Scheduler | Policy | Replica assignment | Serving seconds | Requested output tok/s | Prefill padding | Decode padding |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, data in loaded.items():
        lines.append(case_row(label, data))

    lines.extend(
        [
            "",
            "## Attribution",
            "",
            "| Comparison | Speedup | Serving-time reduction | Throughput | Throughput gain |",
            "| --- | ---: | ---: | ---: | ---: |",
            comparison_row(
                "Realistic scheduler only on naive model",
                loaded["naive + random"],
                loaded["naive + input-known"],
            ),
            comparison_row(
                "Optimized model only under random scheduling",
                loaded["naive + random"],
                loaded["optimized + random"],
            ),
            comparison_row(
                "Realistic scheduler on optimized model",
                loaded["optimized + random"],
                loaded["optimized + input-known"],
            ),
            comparison_row(
                "Full realistic bundle",
                loaded["naive + random"],
                loaded["optimized + input-known"],
            ),
            comparison_row(
                "Oracle scheduler on naive model",
                loaded["naive + random"],
                loaded["naive + oracle"],
            ),
            comparison_row(
                "Oracle scheduler on optimized model",
                loaded["optimized + random"],
                loaded["optimized + oracle"],
            ),
            comparison_row(
                "Oracle replica balancing after oracle scheduling",
                loaded["optimized + oracle rr"],
                loaded["optimized + oracle"],
            ),
            comparison_row(
                "Greedy prefill replica assignment after input-known scheduling",
                loaded["optimized + input-known rr"],
                loaded["optimized + input-known"],
            ),
            "",
            "## Notes",
            "",
            "- `input-known` uses input lengths only and is the realistic scheduling policy in this benchmark.",
            "- `oracle` uses output lengths and should be treated as an upper-bound diagnostic, not a deployable policy.",
            "- Serving seconds exclude model-load and worker-startup overhead, so they are the right metric for scheduler/model attribution.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/serving_ablation_matrix_report.md"))
    args = parser.parse_args()

    root = args.results_dir
    cases = [
        Case("naive + random", root / "serving_scheduling_target_40b_naive_model_random_b8.json"),
        Case("naive + input-known", root / "serving_ablation_40b_naive_input_known_rr.json"),
        Case("naive + oracle", root / "serving_ablation_40b_naive_oracle_rr.json"),
        Case("optimized + random", root / "serving_ablation_40b_optimized_random_rr.json"),
        Case("optimized + input-known rr", root / "serving_scheduling_target_40b_optimized_input_known_b8.json"),
        Case("optimized + input-known", root / "serving_ablation_40b_optimized_input_known_greedy.json"),
        Case("optimized + oracle rr", root / "serving_scheduling_target_40b_optimized_oracle_b8.json"),
        Case("optimized + oracle", root / "serving_ablation_40b_optimized_oracle_greedy.json"),
    ]
    write_report(cases, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
