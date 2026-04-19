from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engrams_kv_moe import EngramConfig, EngramsModel, engram_cfg
from scripts.estimate_scale import PRESETS, estimate_model_params
from scripts.run_target_benchmark_matrix import parse_device_group
from scripts.serving_scheduler import ScheduledBatch, make_static_batches, summarize_schedule
from scripts.serving_workload import (
    ServingRequest,
    build_serving_requests,
    read_requests,
    summarize_requests,
    write_requests,
)


def config_from_preset(preset: dict, device_map: list[str]) -> dict:
    return {
        "vocab_size": preset["vocab_size"],
        "context_length": preset["context_length"],
        "emb_dim": preset["emb_dim"],
        "hidden_dim": preset["hidden_dim"],
        "n_heads": preset["n_heads"],
        "n_layers": preset["n_layers"],
        "drop_rate": 0.0,
        "qkv_bias": preset["qkv_bias"],
        "num_experts": preset["num_experts"],
        "num_experts_per_tok": preset["num_experts_per_tok"],
        "hc_mult": preset["hc_mult"],
        "layer_ids": preset["layer_ids"],
        "device_map": device_map,
        "engrams_cfg": EngramConfig(
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            engram_vocab_size=preset["engram_vocab_size"],
            max_ngram_size=preset["max_ngram_size"],
            n_embed_per_ngram=preset["n_embed_per_ngram"],
            n_head_per_ngram=preset["n_head_per_ngram"],
            layer_ids=preset["layer_ids"],
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
            kernel_size=preset["kernel_size"],
            use_short_conv=preset.get("use_short_conv", True),
            cached_inference_short_conv_mode="step_kernel",
        ),
    }


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def random_padded_batch(
    requests: tuple[ServingRequest, ...],
    *,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    max_len = max(request.input_length for request in requests)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    input_ids = torch.zeros((len(requests), max_len), device=device, dtype=torch.long)
    for row, request in enumerate(requests):
        values = torch.randint(
            3,
            vocab_size,
            (request.input_length,),
            device=device,
            dtype=torch.long,
            generator=generator,
        )
        input_ids[row, max_len - request.input_length :] = values
    return input_ids


def serve_static_batch(
    model: EngramsModel,
    batch: ScheduledBatch,
    *,
    config: dict,
    device: torch.device,
    seed: int,
) -> dict[str, float | int]:
    input_ids = random_padded_batch(
        batch.requests,
        vocab_size=config["vocab_size"],
        device=device,
        seed=seed + batch.batch_id,
    )
    max_output = batch.max_output_length
    model.reset_cache()
    sync_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(
            input_ids,
            use_cache=True,
            position_offset=0,
            engram_input_ids=input_ids,
        )
        next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
        engram_window = torch.cat(
            [input_ids[:, -max(config["engrams_cfg"].max_ngram_size - 1, 1) :], next_idx],
            dim=1,
        )
        generated = 1
        while generated < max_output:
            # For static batching, finished rows remain present until the batch's
            # longest request completes. This intentionally measures scheduling
            # waste from heterogeneous response lengths.
            logits = model(
                next_idx,
                use_cache=True,
                position_offset=input_ids.shape[1] + generated - 1,
                engram_input_ids=engram_window,
            )
            next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
            engram_window = torch.cat([engram_window, next_idx], dim=1)[
                :, -config["engrams_cfg"].max_ngram_size :
            ]
            generated += 1
    sync_device(device)
    seconds = time.perf_counter() - start
    return {
        "batch_id": batch.batch_id,
        "replica_id": batch.replica_id,
        "batch_size": batch.batch_size,
        "max_input_length": batch.max_input_length,
        "max_output_length": batch.max_output_length,
        "requested_input_tokens": batch.requested_input_tokens,
        "requested_output_tokens": batch.requested_output_tokens,
        "padded_prefill_tokens": batch.padded_prefill_tokens,
        "padded_decode_tokens": batch.padded_decode_tokens,
        "seconds": seconds,
    }


def run_worker(args: argparse.Namespace) -> dict:
    preset = PRESETS[args.preset]
    group = parse_device_group(args.device_group)
    device_map = [entry.strip() for entry in group["logical_device_map"].split(",") if entry.strip()]
    device = torch.device(device_map[0] if device_map else group["device"])
    dtype = getattr(torch, args.dtype)
    requests = read_requests(args.request_file)
    requests_by_id = {request.request_id: request for request in requests}
    if args.batch_file:
        planned_batches = json.loads(args.batch_file.read_text())
        batches = [
            ScheduledBatch(
                batch_id=item["batch_id"],
                replica_id=args.replica_id,
                requests=tuple(requests_by_id[request_id] for request_id in item["request_ids"]),
            )
            for item in planned_batches
        ]
        selected = [request for batch in batches for request in batch.requests]
    else:
        selected_ids = {int(value) for value in args.request_ids.split(",") if value}
        selected = [request for request in requests if request.request_id in selected_ids]
        batches = make_static_batches(
            selected,
            batch_size=args.batch_size,
            num_replicas=1,
            policy=args.policy,
        )
    config = config_from_preset(preset, device_map)
    torch.manual_seed(args.seed)
    model = EngramsModel(config)
    if device_map:
        model.apply_device_map(dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()

    batch_results = [
        serve_static_batch(model, batch, config=config, device=device, seed=args.seed)
        for batch in batches
    ]
    total_seconds = sum(item["seconds"] for item in batch_results)
    requested_output = sum(item["requested_output_tokens"] for item in batch_results)
    result = {
        "mode": "worker",
        "preset": args.preset,
        "physical_devices": group["physical"],
        "device_map": group["logical_device_map"],
        "replica_id": args.replica_id,
        "num_requests": len(selected),
        "batch_size": args.batch_size,
        "policy": args.policy,
        "worker_compute_seconds": total_seconds,
        "requested_output_tokens": requested_output,
        "requested_output_tokens_per_second": requested_output / total_seconds if total_seconds else 0.0,
        "batches": batch_results,
    }
    if args.worker_output:
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def split_batches_by_replica(batches: list[ScheduledBatch], num_replicas: int) -> list[list[ScheduledBatch]]:
    batches_by_replica = [[] for _ in range(num_replicas)]
    for batch in batches:
        batches_by_replica[batch.replica_id].append(batch)
    return batches_by_replica


def run_coordinator(args: argparse.Namespace) -> dict:
    requests = build_serving_requests(
        count=args.num_requests,
        mean_input=args.mean_input_tokens,
        mean_output=args.mean_output_tokens,
        max_input=args.max_input_tokens,
        max_output=args.max_output_tokens,
        seed=args.seed,
    )
    groups = [parse_device_group(value) for value in args.device_groups]
    batches = make_static_batches(
        requests,
        batch_size=args.batch_size,
        num_replicas=len(groups),
        policy=args.policy,
    )
    request_file = args.output.with_suffix(".requests.json") if args.output else Path("results/serving_requests.json")
    write_requests(request_file, requests)
    batches_by_replica = split_batches_by_replica(batches, len(groups))

    processes = []
    worker_outputs = []
    start = time.perf_counter()
    for replica_id, (group, replica_batches) in enumerate(zip(groups, batches_by_replica)):
        worker_output = (
            args.output.with_suffix(f".replica{replica_id}.json")
            if args.output
            else Path(f"results/serving_replica{replica_id}.json")
        )
        batch_file = (
            args.output.with_suffix(f".replica{replica_id}.batches.json")
            if args.output
            else Path(f"results/serving_replica{replica_id}.batches.json")
        )
        batch_file.parent.mkdir(parents=True, exist_ok=True)
        batch_file.write_text(
            json.dumps(
                [
                    {
                        "batch_id": batch.batch_id,
                        "request_ids": [request.request_id for request in batch.requests],
                    }
                    for batch in replica_batches
                ],
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        worker_outputs.append(worker_output)
        env = os.environ.copy()
        if group["physical"] == "cpu":
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = group["physical"]
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        command = [
            sys.executable,
            "scripts/benchmark_serving.py",
            "--worker",
            "--preset",
            args.preset,
            "--device-group",
            group["physical"],
            "--dtype",
            args.dtype,
            "--batch-size",
            str(args.batch_size),
            "--policy",
            args.policy,
            "--seed",
            str(args.seed),
            "--request-file",
            str(request_file),
            "--batch-file",
            str(batch_file),
            "--replica-id",
            str(replica_id),
            "--worker-output",
            str(worker_output),
        ]
        processes.append(
            {
                "replica_id": replica_id,
                "group": group,
                "command": command,
                "process": subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ),
            }
        )

    worker_results = []
    raw_results = []
    failed = False
    for entry in processes:
        stdout, stderr = entry["process"].communicate()
        raw = {
            "replica_id": entry["replica_id"],
            "physical_devices": entry["group"]["physical"],
            "command": entry["command"],
            "returncode": entry["process"].returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        raw_results.append(raw)
        if entry["process"].returncode != 0:
            failed = True
        elif worker_outputs[entry["replica_id"]].exists():
            worker_results.append(json.loads(worker_outputs[entry["replica_id"]].read_text()))
    wall_seconds = time.perf_counter() - start

    requested_output = sum(request.output_length for request in requests)
    serving_wall_seconds = max(
        (worker["worker_compute_seconds"] for worker in worker_results),
        default=0.0,
    )
    output = {
        "mode": "coordinator",
        "preset": args.preset,
        "model_size_params_approx": estimate_model_params(PRESETS[args.preset])["total_params_approx"],
        "dtype": args.dtype,
        "batch_size_per_replica": args.batch_size,
        "num_replicas": len(groups),
        "effective_concurrent_batch_size": args.batch_size * len(groups),
        "policy": args.policy,
        "request_summary": summarize_requests(requests),
        "schedule_summary": summarize_schedule(batches),
        "wall_seconds": wall_seconds,
        "serving_wall_seconds_excluding_model_load": serving_wall_seconds,
        "requested_output_tokens": requested_output,
        "requested_output_tokens_per_second": requested_output / wall_seconds if wall_seconds else 0.0,
        "serving_requested_output_tokens_per_second": (
            requested_output / serving_wall_seconds if serving_wall_seconds else 0.0
        ),
        "request_file": str(request_file),
        "worker_results": worker_results,
        "raw_worker_results": raw_results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    if failed:
        raise SystemExit(1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="target_40b_approx")
    parser.add_argument("--device-groups", nargs="+", default=["0,1,2,3", "4,5,6,7"])
    parser.add_argument("--device-group", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--policy", choices=["fifo", "longest_output_first", "longest_total_first"], default="longest_output_first")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--mean-input-tokens", type=int, default=128)
    parser.add_argument("--mean-output-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/serving_scheduling_target_40b.json"))
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--batch-file", type=Path)
    parser.add_argument("--request-ids", default="")
    parser.add_argument("--replica-id", type=int, default=0)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()

    if args.worker:
        if args.request_file is None:
            raise SystemExit("--worker requires --request-file")
        run_worker(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
