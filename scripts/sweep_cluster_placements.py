import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_target_benchmark_matrix import (
    PRESETS,
    parse_device_group,
    run_case,
    summarize_pair,
)
from scripts.estimate_scale import estimate_model_params


def query_gpus():
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []

    gpus = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, used_mib, total_mib, util_percent = parts
        gpus.append(
            {
                "index": int(index),
                "memory_used_mib": int(used_mib),
                "memory_total_mib": int(total_mib),
                "memory_free_mib": int(total_mib) - int(used_mib),
                "utilization_gpu_percent": int(util_percent),
            }
        )
    return gpus


def parse_manual_groups(values):
    return [parse_device_group(value) for value in values]


def contiguous_groups(indices, size):
    return [
        indices[start : start + size]
        for start in range(0, len(indices) - size + 1)
        if indices[start : start + size] == list(range(indices[start], indices[start] + size))
    ]


def candidate_groups(args):
    if args.device_groups:
        return parse_manual_groups(args.device_groups)

    gpus = query_gpus()
    free = [
        gpu["index"]
        for gpu in gpus
        if gpu["memory_free_mib"] >= args.min_free_mib
        and gpu["utilization_gpu_percent"] <= args.max_gpu_util_percent
    ]
    free.sort()

    groups = []
    for size in args.group_sizes:
        if size <= 0:
            continue
        if not args.allow_non_contiguous:
            raw_groups = contiguous_groups(free, size)
        else:
            raw_groups = itertools.combinations(free, size)
        for group in raw_groups:
            groups.append(parse_device_group(",".join(str(index) for index in group)))
    return groups


def run_sweep(args):
    preset = PRESETS[args.preset]
    groups = candidate_groups(args)
    results = []
    summary = []

    for group in groups:
        for decode_length in args.decode_lengths:
            for impl in args.impls:
                result = run_case(args, preset, group, decode_length, impl)
                print(json.dumps(result, sort_keys=True), flush=True)
                results.append(result)
                if result["returncode"] != 0 and not args.continue_on_error:
                    raise SystemExit(result["returncode"])

            paired = summarize_pair(results, group, decode_length)
            if paired is not None:
                summary.append(paired)

    summary.sort(
        key=lambda item: (
            item["optimized_tokens_per_second"],
            item["optimized_improvement_percent"],
        ),
        reverse=True,
    )
    output = {
        "preset": args.preset,
        "model_size_params_approx": estimate_model_params(preset)["total_params_approx"],
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "trials": args.trials,
        "gpu_probe": query_gpus(),
        "candidate_groups": groups,
        "results": results,
        "summary_ranked": summary,
        "best": summary[0] if summary else None,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="target_40b_approx")
    parser.add_argument("--decode-lengths", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--device-groups", nargs="+")
    parser.add_argument("--group-sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--min-free-mib", type=int, default=120_000)
    parser.add_argument("--max-gpu-util-percent", type=int, default=10)
    parser.add_argument("--allow-non-contiguous", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--fail-fast", dest="continue_on_error", action="store_false")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--impls", nargs="+", choices=["optimized", "naive"], default=["optimized", "naive"])
    parser.add_argument("--output", type=Path, default=Path("results/placement_sweep.json"))
    args = parser.parse_args()

    run_sweep(args)


if __name__ == "__main__":
    main()
