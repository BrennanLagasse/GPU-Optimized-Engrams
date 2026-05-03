import argparse
import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engrams_kv_moe_hybrid import (
    EngramConfig, 
    EngramsModel,  
    generate_text,
    config_parser
)
from engrams_naive import NaiveEngramsModel, generate_text_naive

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()

    # CLI arguments
    parser.add_argument("--impl", choices=["optimized", "naive"], default="optimized")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device-map", type=str, default="")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument(
        "--offload-lookup",
        action="store_true",
        help="True if lookup table is offloaded to CPU",
    )

    parser.add_argument("--config", type=str, help="Path to config file")
    

    args = parser.parse_args()

    # Load in config from file
    config = config_parser(args.config)
    if args.device_map:
        config.device_map = args.device_map
    config.device_map = [entry.strip() for entry in config.device_map.split(",") if entry.strip()]

    print(f"Device map: {config.device_map}")

    config.offload_lookup = args.offload_lookup

    torch.manual_seed(0)
    device = torch.device(config.device_map[0] if config.device_map else args.device)
    dtype = getattr(torch, args.dtype)
    model_cls = EngramsModel if args.impl == "optimized" else NaiveEngramsModel

    print("Initializing model...")
    model = model_cls(config)
    if config.device_map:
        model.apply_device_map(dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    input_ids = torch.randint(
        0,
        config.vocab_size,
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
                context_size=config.context_length,
                use_cache=args.use_cache,
            )
        else:
            generate_text_naive(
                model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                context_size=config.context_length,
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
