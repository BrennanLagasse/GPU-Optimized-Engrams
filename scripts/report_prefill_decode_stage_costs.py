#!/usr/bin/env python3
"""Estimate prefill/decode disaggregation from measured batch stage timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct_reduction(before: float, after: float) -> float:
    return (before - after) / before * 100.0 if before else 0.0


def speedup(before: float, after: float) -> float:
    return before / after if after else 0.0


def load_batches(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    batches: list[dict] = []
    for worker in data.get("worker_results", []):
        for batch in worker.get("batches", []):
            if "prefill_seconds" not in batch or "decode_seconds" not in batch:
                raise SystemExit(
                    f"{path} does not contain per-batch prefill/decode timings. "
                    "Rerun the benchmark after the stage-timing patch."
                )
            batches.append(batch)
    return data, sorted(batches, key=lambda item: (item["batch_id"], item["replica_id"]))


def estimate_pipeline(batches: list[dict], *, handoff_seconds: float) -> dict[str, float]:
    prefill_ready = 0.0
    decode_ready = 0.0
    first_decode_start = None
    for batch in batches:
        prefill_done = prefill_ready + float(batch["prefill_seconds"])
        decode_start = max(decode_ready, prefill_done + handoff_seconds)
        decode_done = decode_start + float(batch["decode_seconds"])
        if first_decode_start is None:
            first_decode_start = decode_start
        prefill_ready = prefill_done
        decode_ready = decode_done
    return {
        "prefill_serial_seconds": prefill_ready,
        "decode_serial_seconds": sum(float(batch["decode_seconds"]) for batch in batches),
        "pipeline_makespan_seconds": decode_ready,
        "first_decode_start_seconds": first_decode_start or 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/prefill_decode_stage_costs_report.md"))
    parser.add_argument("--handoff-seconds", type=float, default=0.05)
    args = parser.parse_args()

    data, batches = load_batches(args.input)
    measured = float(data["serving_wall_seconds_excluding_model_load"])
    pipeline = estimate_pipeline(batches, handoff_seconds=args.handoff_seconds)
    total_prefill = sum(float(batch["prefill_seconds"]) for batch in batches)
    total_decode = sum(float(batch["decode_seconds"]) for batch in batches)
    requested_output = int(data["requested_output_tokens"])
    estimated = pipeline["pipeline_makespan_seconds"]

    lines = [
        "# Prefill/Decode Stage Cost Report",
        "",
        "This report uses real benchmark batch timings and estimates a staged prefill/decode pipeline. "
        "It is not a full KV-handoff implementation; it measures stage costs from the real model path, "
        "then models one prefill worker feeding one decode worker with a fixed per-batch handoff cost.",
        "",
        "## Source Benchmark",
        "",
        f"- input JSON: `{args.input}`",
        f"- model: `{data['model_impl']}`",
        f"- scheduler: `{data['policy']}`",
        f"- decoder: `{data['decode_mode']}`",
        f"- batch/replica: `{data['batch_size_per_replica']}`",
        f"- measured serving seconds: `{measured:.3f}`",
        f"- requested output tokens: `{requested_output}`",
        "",
        "## Stage Totals",
        "",
        "| Metric | Seconds | Relative to measured serving time |",
        "| --- | ---: | ---: |",
        f"| Sum of measured prefill stages | `{total_prefill:.3f}` | `{total_prefill / measured:.2f}x` |",
        f"| Sum of measured decode stages | `{total_decode:.3f}` | `{total_decode / measured:.2f}x` |",
        f"| Estimated staged pipeline makespan | `{estimated:.3f}` | `{estimated / measured:.2f}x` |",
        "",
        "## Estimated Disaggregation Result",
        "",
        "| Baseline | Candidate | Baseline seconds | Candidate seconds | Speedup | Serving-time reduction |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        (
            f"| measured current serving | staged prefill/decode estimate | `{measured:.3f}` | "
            f"`{estimated:.3f}` | `{speedup(measured, estimated):.2f}x` | "
            f"`{pct_reduction(measured, estimated):.2f}%` |"
        ),
        "",
        "## Interpretation",
        "",
        "- This is a measured stage-cost experiment, not a production disaggregated server.",
        "- A full implementation still needs KV-cache export/import or shared paged-cache blocks.",
        "- If the staged estimate is slower than the measured baseline, the current workload is decode-bound or the stage split loses too much data-parallel capacity.",
        "- If the staged estimate is faster, the next implementation target should be real KV handoff and overlapped prefill/decode workers.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
