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
from engrams_naive import NaiveEngramsModel
from scripts.estimate_scale import PRESETS, estimate_model_params
from scripts.run_target_benchmark_matrix import parse_device_group
from scripts.serving_scheduler import (
    ORACLE_POLICIES,
    ScheduledBatch,
    make_static_batches,
    order_requests,
    summarize_schedule,
)
from scripts.serving_workload import (
    ServingRequest,
    build_serving_requests,
    read_requests,
    summarize_requests,
    write_requests,
)


def scheduler_impl(policy: str) -> str:
    if policy == "random":
        return "naive_random"
    if policy in ORACLE_POLICIES:
        return "oracle"
    return "input_known"


def config_from_preset(preset: dict, device_map: list[str], *, cached_short_conv_mode: str = "step_kernel") -> dict:
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
            cached_inference_short_conv_mode=cached_short_conv_mode,
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


def random_request_input(
    request: ServingRequest,
    *,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + request.request_id)
    return torch.randint(
        3,
        vocab_size,
        (1, request.input_length),
        device=device,
        dtype=torch.long,
        generator=generator,
    )


def pad_engram_windows(windows: list[torch.Tensor], *, pad_id: int = 0) -> torch.Tensor:
    max_len = max(window.shape[1] for window in windows)
    padded = []
    for window in windows:
        if window.shape[1] == max_len:
            padded.append(window)
            continue
        pad = torch.full(
            (window.shape[0], max_len - window.shape[1]),
            pad_id,
            device=window.device,
            dtype=window.dtype,
        )
        padded.append(torch.cat([pad, window], dim=1))
    return torch.cat(padded, dim=0)


def serve_static_batch_optimized(
    model: EngramsModel,
    batch: ScheduledBatch,
    *,
    config: dict,
    device: torch.device,
    seed: int,
    decode_mode: str,
) -> dict[str, float | int]:
    input_ids = random_padded_batch(
        batch.requests,
        vocab_size=config["vocab_size"],
        device=device,
        seed=seed + batch.batch_id,
    )
    max_output = batch.max_output_length
    output_lengths = torch.tensor(
        [request.output_length for request in batch.requests],
        device=input_ids.device,
        dtype=torch.long,
    )
    model.reset_cache()
    sync_device(device)
    start = time.perf_counter()
    executed_decode_tokens = 0
    with torch.inference_mode():
        logits = model(
            input_ids,
            use_cache=True,
            position_offset=0,
            engram_input_ids=input_ids,
        )
        next_idx = logits[:, -1].argmax(dim=-1, keepdim=True).to(device=input_ids.device)
        executed_decode_tokens += int(next_idx.shape[0])
        engram_window = torch.cat(
            [input_ids[:, -max(config["engrams_cfg"].max_ngram_size - 1, 1) :], next_idx],
            dim=1,
        )
        generated = 1
        if decode_mode == "compact":
            active = torch.nonzero(output_lengths > generated, as_tuple=False).flatten()
            if active.numel() == 0:
                generated = max_output
            else:
                model.compact_cache(active)
                next_idx = next_idx.index_select(0, active)
                engram_window = engram_window.index_select(0, active)
                output_lengths = output_lengths.index_select(0, active)
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
            next_idx = logits[:, -1].argmax(dim=-1, keepdim=True).to(device=input_ids.device)
            executed_decode_tokens += int(next_idx.shape[0])
            engram_window = torch.cat([engram_window, next_idx], dim=1)[
                :, -config["engrams_cfg"].max_ngram_size :
            ]
            generated += 1
            if decode_mode == "compact":
                active = torch.nonzero(output_lengths > generated, as_tuple=False).flatten()
                if active.numel() == 0:
                    break
                model.compact_cache(active)
                next_idx = next_idx.index_select(0, active)
                engram_window = engram_window.index_select(0, active)
                output_lengths = output_lengths.index_select(0, active)
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
        "executed_decode_tokens": executed_decode_tokens,
        "seconds": seconds,
    }


def serve_continuous_optimized(
    model: EngramsModel,
    requests: list[ServingRequest],
    *,
    config: dict,
    device: torch.device,
    seed: int,
    capacity: int,
    policy: str,
    replica_id: int,
) -> dict[str, float | int | str]:
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    ordered = order_requests(requests, policy, seed=seed)
    queue_idx = 0
    slots: list[dict | None] = [None for _ in range(capacity)]
    max_ngram = config["engrams_cfg"].max_ngram_size
    pad_id = config["engrams_cfg"].pad_id
    requested_input = sum(request.input_length for request in ordered)
    requested_output = sum(request.output_length for request in ordered)
    executed_decode_tokens = 0
    prefill_tokens = 0
    completed = 0

    def admit_request(slot: int, request: ServingRequest) -> None:
        nonlocal executed_decode_tokens, prefill_tokens, completed
        model.reset_cache_rows(torch.tensor([slot], device=device, dtype=torch.long))
        input_ids = random_request_input(
            request,
            vocab_size=config["vocab_size"],
            device=device,
            seed=seed,
        )
        logits = model(
            input_ids,
            use_cache=True,
            cache_positions=torch.tensor([0], device=device, dtype=torch.long),
            cache_row_indices=torch.tensor([slot], device=device, dtype=torch.long),
            engram_input_ids=input_ids,
        )
        next_idx = logits[:, -1].argmax(dim=-1, keepdim=True).to(device=input_ids.device)
        executed_decode_tokens += 1
        prefill_tokens += request.input_length
        if request.output_length <= 1:
            completed += 1
            model.reset_cache_rows(torch.tensor([slot], device=device, dtype=torch.long))
            slots[slot] = None
            return
        engram_window = torch.cat(
            [input_ids[:, -max(max_ngram - 1, 1) :], next_idx],
            dim=1,
        )[:, -max_ngram:]
        slots[slot] = {
            "request": request,
            "next_idx": next_idx,
            "engram_window": engram_window,
            "generated": 1,
        }

    def refill_free_slots() -> None:
        nonlocal queue_idx
        for slot in range(capacity):
            while slots[slot] is None and queue_idx < len(ordered):
                request = ordered[queue_idx]
                queue_idx += 1
                admit_request(slot, request)

    model.reset_cache()
    sync_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        refill_free_slots()
        while completed < len(ordered):
            active_slots = [slot for slot, state in enumerate(slots) if state is not None]
            if not active_slots:
                refill_free_slots()
                continue
            next_batch = torch.cat([slots[slot]["next_idx"] for slot in active_slots], dim=0)
            cache_positions = torch.tensor(
                [
                    slots[slot]["request"].input_length + slots[slot]["generated"] - 1
                    for slot in active_slots
                ],
                device=device,
                dtype=torch.long,
            )
            cache_row_indices = torch.tensor(active_slots, device=device, dtype=torch.long)
            engram_input_ids = pad_engram_windows(
                [slots[slot]["engram_window"] for slot in active_slots],
                pad_id=pad_id,
            )
            logits = model(
                next_batch,
                use_cache=True,
                cache_positions=cache_positions,
                cache_row_indices=cache_row_indices,
                engram_input_ids=engram_input_ids,
            )
            new_next = logits[:, -1].argmax(dim=-1, keepdim=True).to(device=next_batch.device)
            executed_decode_tokens += len(active_slots)

            for row, slot in enumerate(active_slots):
                state = slots[slot]
                state["generated"] += 1
                request = state["request"]
                if state["generated"] >= request.output_length:
                    completed += 1
                    model.reset_cache_rows(torch.tensor([slot], device=device, dtype=torch.long))
                    slots[slot] = None
                    continue
                state["next_idx"] = new_next[row : row + 1]
                state["engram_window"] = torch.cat(
                    [state["engram_window"], state["next_idx"]],
                    dim=1,
                )[:, -max_ngram:]
            refill_free_slots()
    sync_device(device)
    seconds = time.perf_counter() - start
    return {
        "batch_id": -1,
        "replica_id": replica_id,
        "batch_size": min(capacity, len(ordered)),
        "max_input_length": max((request.input_length for request in ordered), default=0),
        "max_output_length": max((request.output_length for request in ordered), default=0),
        "requested_input_tokens": requested_input,
        "requested_output_tokens": requested_output,
        "padded_prefill_tokens": prefill_tokens,
        "padded_decode_tokens": requested_output,
        "executed_decode_tokens": executed_decode_tokens,
        "seconds": seconds,
        "scheduler_runtime": "continuous_refill",
    }


def serve_static_batch_naive(
    model: NaiveEngramsModel,
    batch: ScheduledBatch,
    *,
    config: dict,
    device: torch.device,
    seed: int,
    decode_mode: str,
) -> dict[str, float | int]:
    input_ids = random_padded_batch(
        batch.requests,
        vocab_size=config["vocab_size"],
        device=device,
        seed=seed + batch.batch_id,
    )
    max_output = batch.max_output_length
    context_len = config["context_length"]
    output_lengths = torch.tensor(
        [request.output_length for request in batch.requests],
        device=input_ids.device,
        dtype=torch.long,
    )
    output = torch.empty(
        (input_ids.shape[0], input_ids.shape[1] + max_output),
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    output[:, : input_ids.shape[1]] = input_ids
    current_len = input_ids.shape[1]
    sync_device(device)
    start = time.perf_counter()
    executed_decode_tokens = 0
    with torch.inference_mode():
        for generated in range(max_output):
            context_start = max(0, current_len - context_len)
            window = output[:, context_start:current_len]
            logits = model(
                window,
                use_cache=False,
                position_offset=context_start,
                engram_input_ids=window,
            )
            next_idx = logits[:, -1].argmax(dim=-1, keepdim=True).to(device=output.device)
            executed_decode_tokens += int(next_idx.shape[0])
            output[:, current_len : current_len + 1] = next_idx
            current_len += 1
            if decode_mode == "compact":
                generated_count = generated + 1
                active = torch.nonzero(output_lengths > generated_count, as_tuple=False).flatten()
                if active.numel() == 0:
                    break
                output = output.index_select(0, active)
                output_lengths = output_lengths.index_select(0, active)
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
        "executed_decode_tokens": executed_decode_tokens,
        "seconds": seconds,
    }


def build_model(args: argparse.Namespace, config: dict, device: torch.device, dtype: torch.dtype):
    model_cls = EngramsModel if args.model_impl != "naive" else NaiveEngramsModel
    model = model_cls(config)
    device_map = config.get("device_map") or []
    if device_map:
        model.apply_device_map(dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


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
            replica_assignment=args.replica_assignment,
            seed=args.seed,
        )
    cached_short_conv_mode = "full" if args.model_impl == "cached_full_engram" else "step_kernel"
    config = config_from_preset(preset, device_map, cached_short_conv_mode=cached_short_conv_mode)
    torch.manual_seed(args.seed)
    model = build_model(args, config=config, device=device, dtype=dtype)
    if args.decode_mode == "continuous":
        if args.model_impl == "naive":
            raise SystemExit("continuous decode requires the optimized cached model")
        batch_results = [
            serve_continuous_optimized(
                model,
                selected,
                config=config,
                device=device,
                seed=args.seed,
                capacity=args.batch_size,
                policy=args.policy,
                replica_id=args.replica_id,
            )
        ]
    else:
        serve_batch = serve_static_batch_naive if args.model_impl == "naive" else serve_static_batch_optimized
        batch_results = [
            serve_batch(model, batch, config=config, device=device, seed=args.seed, decode_mode=args.decode_mode)
            for batch in batches
        ]
    total_seconds = sum(item["seconds"] for item in batch_results)
    requested_output = sum(item["requested_output_tokens"] for item in batch_results)
    executed_decode = sum(item["executed_decode_tokens"] for item in batch_results)
    result = {
        "mode": "worker",
        "preset": args.preset,
        "physical_devices": group["physical"],
        "device_map": group["logical_device_map"],
        "replica_id": args.replica_id,
        "num_requests": len(selected),
        "batch_size": args.batch_size,
        "model_impl": args.model_impl,
        "scheduler_impl": scheduler_impl(args.policy),
        "replica_assignment": args.replica_assignment,
        "decode_mode": args.decode_mode,
        "policy": args.policy,
        "policy_uses_output_lengths": args.policy in ORACLE_POLICIES,
        "worker_compute_seconds": total_seconds,
        "requested_output_tokens": requested_output,
        "executed_decode_tokens": executed_decode,
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
        replica_assignment=args.replica_assignment,
        seed=args.seed,
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
            "--model-impl",
            args.model_impl,
            "--batch-size",
            str(args.batch_size),
            "--policy",
            args.policy,
            "--replica-assignment",
            args.replica_assignment,
            "--decode-mode",
            args.decode_mode,
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
        "model_impl": args.model_impl,
        "scheduler_impl": scheduler_impl(args.policy),
        "replica_assignment": args.replica_assignment,
        "decode_mode": args.decode_mode,
        "batch_size_per_replica": args.batch_size,
        "num_replicas": len(groups),
        "effective_concurrent_batch_size": args.batch_size * len(groups),
        "policy": args.policy,
        "policy_uses_output_lengths": args.policy in ORACLE_POLICIES,
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
    parser.add_argument(
        "--model-impl",
        choices=["optimized_cached", "cached_full_engram", "naive"],
        default="optimized_cached",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--policy",
        choices=[
            "fifo",
            "random",
            "longest_input_first",
            "shortest_input_first",
            "input_bucketed_random",
            "longest_output_first",
            "longest_total_first",
        ],
        default="longest_input_first",
    )
    parser.add_argument(
        "--replica-assignment",
        choices=["round_robin", "greedy_prefill", "greedy_oracle"],
        default="round_robin",
    )
    parser.add_argument("--decode-mode", choices=["static", "compact", "continuous"], default="static")
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
