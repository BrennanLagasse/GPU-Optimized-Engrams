import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engrams_kv_moe import EngramConfig, EngramsModel, engram_cfg
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


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(args, device, dtype):
    config = build_config(args)
    model_cls = EngramsModel if args.impl == "optimized" else NaiveEngramsModel
    model = model_cls(config)
    if args.device_map:
        model.apply_device_map(dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    return model, config


def profile_decode(model, config, input_ids, impl, use_cache):
    model.eval()
    if hasattr(model, "reset_cache"):
        model.reset_cache()

    batch_size, base_len = input_ids.shape
    total_len = base_len + config["max_new_tokens"]
    out = torch.empty(batch_size, total_len, dtype=input_ids.dtype, device=input_ids.device)
    out[:, :base_len] = input_ids
    current_len = base_len
    context_len = config["context_length"]
    step_seconds = []

    with torch.inference_mode():
        for step_idx in range(config["max_new_tokens"]):
            if impl == "optimized" and use_cache:
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
                context_start = max(0, current_len - context_len)
                idx_cond = out[:, context_start:current_len]
                position_offset = max(0, current_len - context_len)
                engram_input_ids = idx_cond

            sync(input_ids.device)
            start = time.perf_counter()
            logits = model(
                idx_cond,
                use_cache=(impl == "optimized" and use_cache),
                position_offset=position_offset,
                engram_input_ids=engram_input_ids,
            )
            next_idx = logits[:, -1].argmax(dim=-1)
            sync(input_ids.device)
            step_seconds.append(time.perf_counter() - start)

            out[:, current_len] = next_idx
            current_len += 1

    total_seconds = sum(step_seconds)
    total_generated = batch_size * config["max_new_tokens"]
    steady_state_steps = step_seconds[1:]
    return {
        "ttft_seconds": step_seconds[0],
        "steady_state_avg_seconds": (
            sum(steady_state_steps) / len(steady_state_steps) if steady_state_steps else 0.0
        ),
        "steady_state_tokens_per_second": (
            batch_size / (sum(steady_state_steps) / len(steady_state_steps))
            if steady_state_steps
            else 0.0
        ),
        "avg_seconds_per_step": total_seconds / len(step_seconds),
        "tokens_per_second": total_generated / total_seconds,
        "total_seconds": total_seconds,
        "step_seconds": step_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["optimized", "naive"], default="optimized")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device-map", type=str, default="")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
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
    config["max_new_tokens"] = args.max_new_tokens
    input_ids = torch.randint(
        0,
        config["vocab_size"],
        (args.batch_size, args.prompt_length),
        device=device,
        dtype=torch.long,
    )

    result = profile_decode(
        model=model,
        config=config,
        input_ids=input_ids,
        impl=args.impl,
        use_cache=args.use_cache,
    )
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
