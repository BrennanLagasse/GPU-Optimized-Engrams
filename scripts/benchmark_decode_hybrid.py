import argparse
import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engrams_kv_moe_hybrid import EngramConfig, EngramsModel, engram_cfg, generate_text
from engrams_naive import NaiveEngramsModel, generate_text_naive

from tqdm import tqdm


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
        "num_experts": args.num_experts,
        "num_experts_per_tok": args.num_experts_per_tok if args.num_experts > 0 else 0,
        "hc_mult": args.hc_mult,
        "layer_ids": args.layer_ids,
        "device_map": args.device_map,
        "engrams_cfg": EngramConfig(
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            # engram_vocab_size=args.engram_vocab_size,
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
        "offload_lookup": args.offload_lookup,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--impl", choices=["optimized", "naive"], default="optimized")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device-map", type=str, default="")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--use-cache", action="store_true")

    # DeepSeek evaluates in the context of a 27B param model that is not public
    # Defaults are inferred based on Qwen3.6-27B
    parser.add_argument("--vocab-size", type=int, default=129280)
    parser.add_argument("--context-length", type=int, default=262144) # Qwen-27B
    parser.add_argument("--emb-dim", type=int, default=256) # Qwen-27B
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=64) # Qwen3-27B
    parser.add_argument("--num-experts", type=int, default=0)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)

    # Defaults follow Engram-32B model setup
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--layer-ids", type=int, nargs="*", default=[1, 14])
    parser.add_argument("--max-ngram-size", type=int, default=3) 
    parser.add_argument("--n-embed-per-ngram", type=int, default=1280)
    parser.add_argument("--n-head-per-ngram", type=int, default=8) 

    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--disable-short-conv", action="store_true")
    parser.add_argument(
        "--cached-inference-short-conv-mode",
        choices=["full", "step_kernel", "gated_value_only"],
        default="step_kernel",
    )

    # Whether to offload lookup table to GPU
    parser.add_argument(
        "--offload_lookup",
        action="store_true",
        help="True if lookup table is offloaded to CPU",
    )

    args = parser.parse_args()

    torch.manual_seed(0)
    device_map = [entry.strip() for entry in args.device_map.split(",") if entry.strip()]
    args.device_map = device_map
    device = torch.device(device_map[0] if device_map else args.device)
    dtype = getattr(torch, args.dtype)
    config = build_config(args)
    model_cls = EngramsModel if args.impl == "optimized" else NaiveEngramsModel

    print("Initializing model...")
    model = model_cls(config)
    if device_map:
        model.apply_device_map(dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    input_ids = torch.randint(
        0,
        config["vocab_size"],
        (args.batch_size, args.prompt_length),
        device=device,
        dtype=torch.long,
    )

    print("Evaluating...")
    durations = []
    for _ in tqdm(range(args.trials)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        if args.impl == "optimized":
            generate_text(
                model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                context_size=config["context_length"],
                use_cache=args.use_cache,
            )
        else:
            generate_text_naive(
                model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                context_size=config["context_length"],
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)

    avg_duration = sum(durations) / len(durations)
    toks = args.batch_size * args.max_new_tokens
    print(f"implementation={args.impl}")
    print(f"avg_seconds={avg_duration:.4f}")
    print(f"tokens_per_second={toks / avg_duration:.2f}")
    print(f"use_cache={args.use_cache if args.impl == 'optimized' else False}")


if __name__ == "__main__":
    main()
