#!/usr/bin/env python3
"""Summarize cached full-Engram vs optimized step-kernel serving ablations."""

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


def pct_reduction(old: float, new: float) -> float:
    return (old - new) / old * 100.0


def row(label: str, data: dict) -> str:
    summary = data.get("schedule_summary", {})
    return (
        f"| {label} | {data.get('model_impl')} | {data.get('policy')} | "
        f"{data.get('replica_assignment')} | {serving_seconds(data):.3f} | "
        f"{serving_tps(data):.3f} | {summary.get('prefill_padding_overhead', 0.0):.4f}x | "
        f"{summary.get('decode_padding_overhead', 0.0):.4f}x |"
    )


def comparison(label: str, baseline: dict, candidate: dict) -> str:
    old = serving_seconds(baseline)
    new = serving_seconds(candidate)
    return (
        f"| {label} | {old:.3f} -> {new:.3f} | {old / new:.2f}x | "
        f"{pct_reduction(old, new):.2f}% | {serving_tps(baseline):.3f} -> {serving_tps(candidate):.3f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/cached_engram_ablation_report.md"))
    args = parser.parse_args()

    root = args.results_dir
    cases = {
        "naive_random": root / "serving_scheduling_target_40b_naive_model_random_b8.json",
        "cached_full_random": root / "serving_ablation_40b_cached_full_random_rr.json",
        "optimized_random": root / "serving_ablation_40b_optimized_random_rr.json",
        "naive_input": root / "serving_ablation_40b_naive_input_known_rr.json",
        "cached_full_input": root / "serving_ablation_40b_cached_full_input_known_rr.json",
        "optimized_input": root / "serving_scheduling_target_40b_optimized_input_known_b8.json",
        "naive_oracle": root / "serving_ablation_40b_naive_oracle_rr.json",
        "cached_full_oracle": root / "serving_ablation_40b_cached_full_oracle_rr.json",
        "optimized_oracle": root / "serving_scheduling_target_40b_optimized_oracle_b8.json",
    }
    loaded = {name: load(path) for name, path in cases.items()}

    lines = [
        "# Cached Engram Serving Ablation",
        "",
        "This report separates generic cached serving from the Engram-specific exact cached step-kernel path.",
        "",
        "## Cases",
        "",
        "| Case | Model | Policy | Replica assignment | Serving seconds | Serving tok/s | Prefill padding | Decode padding |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, data in loaded.items():
        lines.append(row(name, data))

    lines.extend(
        [
            "",
            "## Attribution",
            "",
            "| Comparison | Serving seconds | Speedup | Serving-time reduction | Serving tok/s |",
            "| --- | ---: | ---: | ---: | ---: |",
            comparison("Generic cached serving under random scheduling", loaded["naive_random"], loaded["cached_full_random"]),
            comparison("Engram step-kernel under random scheduling", loaded["cached_full_random"], loaded["optimized_random"]),
            comparison("Generic cached serving under input-known scheduling", loaded["naive_input"], loaded["cached_full_input"]),
            comparison("Engram step-kernel under input-known scheduling", loaded["cached_full_input"], loaded["optimized_input"]),
            comparison("Generic cached serving under oracle scheduling", loaded["naive_oracle"], loaded["cached_full_oracle"]),
            comparison("Engram step-kernel under oracle scheduling", loaded["cached_full_oracle"], loaded["optimized_oracle"]),
            "",
            "## Notes",
            "",
            "- `cached_full_engram` keeps the same KV-cached serving loop as `optimized_cached` but forces cached Engram local mixing through the full path.",
            "- `optimized_cached` uses the exact cached Engram step-kernel path.",
            "- `naive` remains the full-context recompute baseline.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
