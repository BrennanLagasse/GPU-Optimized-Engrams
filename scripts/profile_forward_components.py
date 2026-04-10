import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engrams_kv_moe import EngramConfig, EngramsModel, engram_cfg, move_tensor_to_device
from engrams_naive import NaiveEngramsModel


def build_config(args):
    return {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "hidden_dim": args.hidden_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "hc_mult": args.hc_mult,
        "layer_ids": args.layer_ids,
        "device_map": args.device_map,
        "engrams_cfg": EngramConfig(
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            engram_vocab_size=args.engram_vocab_size,
            max_ngram_size=args.max_ngram_size,
            n_embed_per_ngram=args.n_embed_per_ngram,
            n_head_per_ngram=args.n_head_per_ngram,
            layer_ids=args.layer_ids,
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
            kernel_size=args.kernel_size,
            use_short_conv=not args.disable_short_conv,
            cached_inference_short_conv_mode=args.cached_inference_short_conv_mode,
        ),
    }


def sync_all():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(stats, key, fn):
    sync_all()
    start = time.perf_counter()
    result = fn()
    sync_all()
    stats[key] += time.perf_counter() - start
    return result


def build_model(args, device, dtype):
    config = build_config(args)
    model_cls = EngramsModel if args.impl == "optimized" else NaiveEngramsModel
    model = model_cls(config)
    if args.device_map:
        model.apply_device_map(dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, config


def profiled_forward(model, input_ids, use_cache, position_offset, engram_input_ids):
    stats = defaultdict(float)
    block_stats = []
    if hasattr(model, "block_device_map") and model.block_device_map:
        input_ids = timed(stats, "input_id_transfer_seconds", lambda: move_tensor_to_device(input_ids, model.input_device))
        input_device = model.input_device
    else:
        input_device = input_ids.device

    _, seq_len = input_ids.shape
    pos_ids = timed(
        stats,
        "position_ids_seconds",
        lambda: torch.arange(
            start=position_offset,
            end=position_offset + seq_len,
            device=input_device,
            dtype=torch.long,
        ),
    )
    token_embeds = timed(stats, "token_embedding_seconds", lambda: model.token_embed(input_ids))
    pos_embeds = timed(stats, "position_embedding_seconds", lambda: model.pos_embed(pos_ids))
    x = timed(stats, "embedding_add_dropout_seconds", lambda: model.drop_emb(token_embeds + pos_embeds))

    if model.config["hc_mult"] > 1:
        x = timed(
            stats,
            "hc_expand_seconds",
            lambda: x.unsqueeze(2).expand(-1, -1, model.config["hc_mult"], -1).contiguous(),
        )

    if engram_input_ids is None:
        engram_input_ids = input_ids
    engram_hashes = None
    if isinstance(model, EngramsModel):
        if model.block_device_map:
            engram_hashes = timed(
                stats,
                "engram_hash_prep_seconds",
                lambda: model._prepare_engram_hashes(engram_input_ids, use_cache),
            )
        elif model.num_engram_layers > 1:
            first_engram = next(block.engram for block in model.transformer_blocks if block.engram is not None)
            if torch.is_tensor(engram_input_ids):
                hash_fn = (
                    first_engram.hash_mapping.hash_last_tensor
                    if use_cache and input_ids.shape[1] == 1
                    else first_engram.hash_mapping.hash_tensor
                )
                engram_hashes = timed(stats, "engram_hash_prep_seconds", lambda: hash_fn(engram_input_ids))
            else:
                engram_hashes = timed(
                    stats, "engram_hash_prep_seconds", lambda: first_engram.hash_mapping.hash(engram_input_ids)
                )

    for idx, block in enumerate(model.transformer_blocks):
        block_device = None
        if model.block_device_map:
            block_device = torch.device(model.block_device_map[idx])
            if x.device != block_device:
                x = timed(
                    stats,
                    "block_activation_transfer_seconds",
                    lambda block_device=block_device: move_tensor_to_device(x, block_device),
                )

        if model.block_device_map:
            block_input_ids = (
                timed(
                    stats,
                    "engram_id_transfer_seconds",
                    lambda block_device=block_device: move_tensor_to_device(engram_input_ids, block_device),
                )
                if block.engram is not None and torch.is_tensor(engram_input_ids)
                else engram_input_ids if block.engram is not None else None
            )
            local_hashes = engram_hashes.get(str(block_device)) if isinstance(engram_hashes, dict) else None
        else:
            block_input_ids = engram_input_ids if block.engram is not None else None
            local_hashes = engram_hashes

        block_key = f"block_{idx:03d}_seconds"
        if isinstance(model, EngramsModel):
            x = timed(
                stats,
                block_key,
                lambda block=block, block_input_ids=block_input_ids, local_hashes=local_hashes: block(
                    input_ids=block_input_ids,
                    x=x,
                    use_cache=use_cache,
                    engram_hashes=local_hashes,
                ),
            )
        else:
            x = timed(
                stats,
                block_key,
                lambda block=block, block_input_ids=block_input_ids: block(block_input_ids, x),
            )
        block_stats.append(
            {
                "block_index": idx,
                "device": str(block_device or x.device),
                "has_engram": block.engram is not None,
                "seconds": stats[block_key],
            }
        )

    if x.dim() == 4:
        x = timed(stats, "hc_reduce_seconds", lambda: x.mean(dim=2))
    if model.block_device_map and x.device != model.output_device:
        x = timed(stats, "output_activation_transfer_seconds", lambda: move_tensor_to_device(x, model.output_device))
    x = timed(stats, "final_norm_seconds", lambda: model.final_norm(x))
    logits = timed(stats, "out_head_seconds", lambda: model.out_head(x))
    stats["total_profiled_seconds"] = sum(value for key, value in stats.items() if key != "total_profiled_seconds")
    return logits, stats, block_stats


def profile_decode(args, model, config, input_ids):
    if hasattr(model, "reset_cache"):
        model.reset_cache()
    out = torch.empty(
        input_ids.shape[0],
        input_ids.shape[1] + args.max_new_tokens,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    out[:, : input_ids.shape[1]] = input_ids
    current_len = input_ids.shape[1]
    aggregate = defaultdict(float)
    per_step = []
    block_totals = defaultdict(float)

    with torch.no_grad():
        for step_idx in range(args.max_new_tokens):
            if args.impl == "optimized" and args.use_cache:
                if step_idx == 0:
                    idx_cond = out[:, :current_len]
                    position_offset = 0
                    engram_input_ids = idx_cond
                else:
                    idx_cond = out[:, current_len - 1 : current_len]
                    position_offset = current_len - 1
                    engram_start = max(0, current_len - config["engrams_cfg"].max_ngram_size)
                    engram_input_ids = out[:, engram_start:current_len]
            else:
                context_start = max(0, current_len - config["context_length"])
                idx_cond = out[:, context_start:current_len]
                position_offset = context_start
                engram_input_ids = idx_cond

            logits, stats, block_stats = profiled_forward(
                model=model,
                input_ids=idx_cond,
                use_cache=args.impl == "optimized" and args.use_cache,
                position_offset=position_offset,
                engram_input_ids=engram_input_ids,
            )
            next_idx = timed(stats, "argmax_seconds", lambda: logits[:, -1].argmax(dim=-1))
            out[:, current_len] = next_idx
            current_len += 1
            step_record = {
                "step_index": step_idx,
                "sequence_length": idx_cond.shape[1],
                **dict(stats),
            }
            per_step.append(step_record)
            for key, value in stats.items():
                aggregate[key] += value
            for item in block_stats:
                block_totals[item["block_index"]] += item["seconds"]

    aggregate = dict(sorted(aggregate.items()))
    block_summary = [
        {
            "block_index": idx,
            "seconds": seconds,
            "percent_of_profiled_total": (
                seconds / aggregate["total_profiled_seconds"] * 100.0
                if aggregate.get("total_profiled_seconds", 0.0)
                else 0.0
            ),
        }
        for idx, seconds in sorted(block_totals.items())
    ]
    return {
        "aggregate_seconds": aggregate,
        "block_summary": block_summary,
        "per_step": per_step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["optimized", "naive"], default="optimized")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device-map", type=str, default="")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--hc-mult", type=int, default=1)
    parser.add_argument("--layer-ids", type=int, nargs="*", default=[])
    parser.add_argument("--engram-vocab-size", type=int, nargs="*", default=[257, 263])
    parser.add_argument("--max-ngram-size", type=int, default=3)
    parser.add_argument("--n-embed-per-ngram", type=int, default=32)
    parser.add_argument("--n-head-per-ngram", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--disable-short-conv", action="store_true")
    parser.add_argument(
        "--cached-inference-short-conv-mode",
        choices=["full", "step_kernel", "gated_value_only"],
        default="step_kernel",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    args.device_map = [entry.strip() for entry in args.device_map.split(",") if entry.strip()]
    device = torch.device(args.device_map[0] if args.device_map else args.device)
    dtype = getattr(torch, args.dtype)
    model, config = build_model(args, device, dtype)
    input_ids = torch.randint(
        0,
        config["vocab_size"],
        (args.batch_size, args.prompt_length),
        device=device,
        dtype=torch.long,
    )
    result = profile_decode(args, model, config, input_ids)
    result.update(
        {
            "implementation": args.impl,
            "use_cache": bool(args.use_cache and args.impl == "optimized"),
            "model_size_params": sum(p.numel() for p in model.parameters()),
            "prompt_length": args.prompt_length,
            "max_new_tokens": args.max_new_tokens,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
