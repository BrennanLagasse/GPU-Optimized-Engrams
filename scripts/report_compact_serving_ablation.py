#!/usr/bin/env python3
"""Summarize static-vs-compact serving ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def serving_seconds(data: dict) -> float:
    return float(data["serving_wall_seconds_excluding_model_load"])


def serving_tps(data: dict) -> float:
    return float(data["serving_requested_output_tokens_per_second"])


def executed_decode_tokens(data: dict) -> int:
    total = sum(int(worker.get("executed_decode_tokens", 0)) for worker in data.get("worker_results", []))
    if total:
        return total
    return int(data.get("schedule_summary", {}).get("padded_decode_tokens", 0))


def comparison(label: str, static: dict, compact: dict) -> str:
    old = serving_seconds(static)
    new = serving_seconds(compact)
    return (
        f"| {label} | {old:.3f} -> {new:.3f} | {old / new:.2f}x | "
        f"{(old - new) / old * 100.0:.2f}% | "
        f"{serving_tps(static):.3f} -> {serving_tps(compact):.3f} | "
        f"{executed_decode_tokens(static)} -> {executed_decode_tokens(compact)} |"
    )


def case_row(label: str, data: dict) -> str:
    return (
        f"| {label} | {data.get('model_impl')} | {data.get('policy')} | "
        f"{data.get('decode_mode', 'static')} | {serving_seconds(data):.3f} | "
        f"{serving_tps(data):.3f} | {executed_decode_tokens(data)} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/compact_serving_ablation_report.md"))
    args = parser.parse_args()

    root = args.results_dir
    cases = {
        "naive_random_static": root / "serving_scheduling_target_40b_naive_model_random_b8.json",
        "naive_random_compact": root / "serving_ablation_40b_naive_random_compact_rr.json",
        "optimized_input_static": root / "serving_scheduling_target_40b_optimized_input_known_b8.json",
        "optimized_input_compact": root / "serving_ablation_40b_optimized_input_known_compact_rr.json",
        "optimized_oracle_static": root / "serving_scheduling_target_40b_optimized_oracle_b8.json",
        "optimized_oracle_compact": root / "serving_ablation_40b_optimized_oracle_compact_rr.json",
    }
    loaded = {name: load(path) for name, path in cases.items()}

    lines = [
        "# Compact Serving Ablation",
        "",
        "This report compares static padded decode against active-row compaction. Compact decode removes a row only after that row has completed generation, so it does not require knowing output lengths before serving starts.",
        "",
        "## Cases",
        "",
        "| Case | Model | Policy | Decode mode | Serving seconds | Serving tok/s | Executed decode tokens |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for name, data in loaded.items():
        lines.append(case_row(name, data))

    lines.extend(
        [
            "",
            "## Static vs Compact",
            "",
            "| Comparison | Serving seconds | Speedup | Serving-time reduction | Serving tok/s | Executed decode tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            comparison("naive random", loaded["naive_random_static"], loaded["naive_random_compact"]),
            comparison("optimized input-known", loaded["optimized_input_static"], loaded["optimized_input_compact"]),
            comparison("optimized oracle", loaded["optimized_oracle_static"], loaded["optimized_oracle_compact"]),
            "",
            "## Notes",
            "",
            "- Static decode keeps completed rows in a batch until the longest row finishes.",
            "- Compact decode removes completed rows online after completion is observed, which is realistic because a server knows when a request has emitted EOS or reached its generation limit.",
            "- Compact decode is different from oracle scheduling: oracle scheduling sorts by output length before generation, which a real server cannot know.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
