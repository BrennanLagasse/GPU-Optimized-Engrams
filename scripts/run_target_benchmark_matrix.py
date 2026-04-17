import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.estimate_scale import PRESETS, estimate_model_params


RESULT_RE = {
    "implementation": re.compile(r"^implementation=(?P<value>.+)$"),
    "avg_seconds": re.compile(r"^avg_seconds=(?P<value>[0-9.]+)$"),
    "tokens_per_second": re.compile(r"^tokens_per_second=(?P<value>[0-9.]+)$"),
    "use_cache": re.compile(r"^use_cache=(?P<value>.+)$"),
}


def parse_device_group(value):
    if value.lower() == "cpu":
        return {
            "physical": "cpu",
            "logical_device_map": "",
            "device": "cpu",
            "num_devices": 0,
        }
    physical = [entry.strip() for entry in value.split(",") if entry.strip()]
    if not physical:
        raise argparse.ArgumentTypeError("device groups must contain at least one device")
    logical_map = ",".join(f"cuda:{idx}" for idx in range(len(physical)))
    return {
        "physical": ",".join(physical),
        "logical_device_map": logical_map,
        "device": "cuda",
        "num_devices": len(physical),
    }


def benchmark_args(preset, device, device_map, decode_length, impl, dtype, batch_size, prompt_length, trials):
    use_cache = impl == "optimized"
    command = [
        sys.executable,
        "scripts/benchmark_decode.py",
        "--impl",
        impl,
        "--device-map",
        device_map,
        "--device",
        device,
        "--dtype",
        dtype,
        "--batch-size",
        str(batch_size),
        "--prompt-length",
        str(prompt_length),
        "--max-new-tokens",
        str(decode_length),
        "--trials",
        str(trials),
        "--vocab-size",
        str(preset["vocab_size"]),
        "--context-length",
        str(preset["context_length"]),
        "--emb-dim",
        str(preset["emb_dim"]),
        "--hidden-dim",
        str(preset["hidden_dim"]),
        "--n-heads",
        str(preset["n_heads"]),
        "--n-layers",
        str(preset["n_layers"]),
        "--hc-mult",
        str(preset["hc_mult"]),
        "--layer-ids",
        *[str(layer_id) for layer_id in preset["layer_ids"]],
        "--engram-vocab-size",
        *[str(size) for size in preset["engram_vocab_size"]],
        "--max-ngram-size",
        str(preset["max_ngram_size"]),
        "--n-embed-per-ngram",
        str(preset["n_embed_per_ngram"]),
        "--n-head-per-ngram",
        str(preset["n_head_per_ngram"]),
        "--kernel-size",
        str(preset["kernel_size"]),
    ]
    if use_cache:
        command.append("--use-cache")
    if not preset.get("use_short_conv", True):
        command.append("--disable-short-conv")
    return command


def parse_benchmark_output(stdout):
    parsed = {}
    for line in stdout.splitlines():
        for key, pattern in RESULT_RE.items():
            match = pattern.match(line.strip())
            if match:
                value = match.group("value")
                parsed[key] = float(value) if key in {"avg_seconds", "tokens_per_second"} else value
    missing = {"implementation", "avg_seconds", "tokens_per_second"} - set(parsed)
    if missing:
        raise RuntimeError(f"benchmark output missing fields {sorted(missing)}:\n{stdout}")
    return parsed


def run_case(args, preset, group, decode_length, impl):
    env = os.environ.copy()
    if group["physical"] == "cpu":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = group["physical"]
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = benchmark_args(
        preset=preset,
        device=group["device"],
        device_map=group["logical_device_map"],
        decode_length=decode_length,
        impl=impl,
        dtype=args.dtype,
        batch_size=args.batch_size,
        prompt_length=args.prompt_length,
        trials=args.trials,
    )
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    result = {
        "physical_devices": group["physical"],
        "device_map": group["logical_device_map"],
        "num_devices": group["num_devices"],
        "decode_length": decode_length,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        return result
    result.update(parse_benchmark_output(completed.stdout))
    return result


def summarize_pair(results, group, decode_length):
    opt = next(
        (
            item
            for item in results
            if item["num_devices"] == group["num_devices"]
            and item["physical_devices"] == group["physical"]
            and item["decode_length"] == decode_length
            and item.get("implementation") == "optimized"
            and item["returncode"] == 0
        ),
        None,
    )
    naive = next(
        (
            item
            for item in results
            if item["num_devices"] == group["num_devices"]
            and item["physical_devices"] == group["physical"]
            and item["decode_length"] == decode_length
            and item.get("implementation") == "naive"
            and item["returncode"] == 0
        ),
        None,
    )
    if opt is None or naive is None:
        return None
    return {
        "physical_devices": group["physical"],
        "num_devices": group["num_devices"],
        "decode_length": decode_length,
        "optimized_tokens_per_second": opt["tokens_per_second"],
        "naive_tokens_per_second": naive["tokens_per_second"],
        "optimized_improvement_percent": (
            (opt["tokens_per_second"] / naive["tokens_per_second"]) - 1.0
        )
        * 100.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="target_40b_approx")
    parser.add_argument("--device-groups", nargs="+", type=parse_device_group, required=True)
    parser.add_argument("--decode-lengths", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--impls", nargs="+", choices=["optimized", "naive"], default=["optimized", "naive"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    params = estimate_model_params(preset)
    results = []
    for group in args.device_groups:
        for decode_length in args.decode_lengths:
            for impl in args.impls:
                result = run_case(args, preset, group, decode_length, impl)
                print(json.dumps(result, sort_keys=True))
                results.append(result)
                if result["returncode"] != 0:
                    raise SystemExit(result["returncode"])

    summary = [
        summary
        for group in args.device_groups
        for decode_length in args.decode_lengths
        if (summary := summarize_pair(results, group, decode_length)) is not None
    ]
    output = {
        "preset": args.preset,
        "model_size_params_approx": params["total_params_approx"],
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "trials": args.trials,
        "results": results,
        "summary": summary,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
