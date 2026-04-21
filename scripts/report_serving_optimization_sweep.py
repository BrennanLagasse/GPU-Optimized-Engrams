#!/usr/bin/env python3
"""Summarize serving optimization sweep outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def serving_seconds(data: dict) -> float:
    return float(data.get("serving_wall_seconds_excluding_model_load", 0.0))


def serving_tps(data: dict) -> float:
    return float(data.get("serving_requested_output_tokens_per_second", 0.0))


def executed_decode_tokens(data: dict) -> int:
    total = sum(int(worker.get("executed_decode_tokens", 0)) for worker in data.get("worker_results", []))
    if total:
        return total
    return int(data.get("schedule_summary", {}).get("padded_decode_tokens", 0))


def summarize_file(path: Path) -> dict:
    data = load(path)
    if data.get("mode") != "coordinator":
        raise ValueError(f"{path} is not a coordinator benchmark output")
    summary = data.get("schedule_summary", {})
    return {
        "path": path,
        "case": path.stem,
        "model_impl": data.get("model_impl", ""),
        "policy": data.get("policy", ""),
        "replica_assignment": data.get("replica_assignment", ""),
        "decode_mode": data.get("decode_mode", "static"),
        "batch_size": data.get("batch_size_per_replica", 0),
        "serving_seconds": serving_seconds(data),
        "serving_tps": serving_tps(data),
        "total_seconds": float(data.get("wall_seconds", 0.0)),
        "total_tps": float(data.get("requested_output_tokens_per_second", 0.0)),
        "prefill_overhead": float(summary.get("prefill_padding_overhead", 0.0)),
        "decode_overhead": float(summary.get("decode_padding_overhead", 0.0)),
        "executed_decode_tokens": executed_decode_tokens(data),
    }


def row(item: dict, baseline_seconds: float) -> str:
    seconds = item["serving_seconds"]
    speedup = baseline_seconds / seconds if seconds else 0.0
    reduction = (baseline_seconds - seconds) / baseline_seconds * 100.0 if baseline_seconds else 0.0
    return (
        f"| `{item['case']}` | `{item['model_impl']}` | `{item['policy']}` | "
        f"`{item['decode_mode']}` | {item['batch_size']} | {seconds:.3f} | "
        f"{item['serving_tps']:.3f} | {speedup:.2f}x | {reduction:.2f}% | "
        f"{item['prefill_overhead']:.3f} | {item['decode_overhead']:.3f} | "
        f"{item['executed_decode_tokens']} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--pattern", default="serving_opt_sweep_*.json")
    parser.add_argument("--baseline", default="serving_scheduling_target_40b_naive_model_random_b8.json")
    parser.add_argument("--output", type=Path, default=Path("results/serving_optimization_sweep_report.md"))
    args = parser.parse_args()

    files = sorted(args.results_dir.glob(args.pattern))
    items = []
    for path in files:
        try:
            items.append(summarize_file(path))
        except ValueError:
            continue
    baseline_path = args.results_dir / args.baseline
    baseline = load(baseline_path)
    baseline_seconds = serving_seconds(baseline)

    best = min(items, key=lambda item: item["serving_seconds"], default=None)
    lines = [
        "# Serving Optimization Sweep",
        "",
        "This report summarizes measured 40B serving runs for deployable scheduling and batching variants.",
        "",
        f"Baseline: `{args.baseline}` at `{baseline_seconds:.3f}s` serving wall time.",
        "",
        "## Results",
        "",
        "| Case | Model | Policy | Decode | Batch/replica | Serving seconds | Serving tok/s | Speedup vs naive static | Serving-time reduction | Prefill overhead | Decode overhead | Executed decode tokens |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(items, key=lambda value: (value["serving_seconds"], value["case"])):
        lines.append(row(item, baseline_seconds))

    lines.extend(["", "## Interpretation", ""])
    if best:
        lines.append(
            f"Best measured sweep case: `{best['case']}` with `{best['serving_seconds']:.3f}s` "
            f"serving time and `{best['serving_tps']:.3f}` requested output tok/s."
        )
    lines.append(
        "Continuous batching with true row-level refill is not included as a measured model run because "
        "the current attention cache tracks one shared cache position per batch. A production continuous "
        "batching implementation would need per-row cache positions or cache-page metadata before it can "
        "safely admit new requests into completed rows without resetting the whole batch."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
