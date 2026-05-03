"""
Check loading in the Engrams Model and get some basic stats.

Performs:
1. Loads model onto devices
2. Determine how much time to precompute all embeddings for the Engram modules
3. Outputs the memory profile and breaks into the lookup table, embedding module, and entire model

"""

import argparse
import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engrams_kv_moe_hybrid import (
    EngramConfig, 
    EngramsModel, 
    config_parser
)
from engrams_naive import NaiveEngramsModel

DTYPE_BYTES = {
    "torch.float32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}

def profile_module(model):
    """ Compute memory use of module. Returns parameter count and memory (in GBs) """

    param_count = sum(p.numel() for p in model.parameters())
    mem_bytes = sum(p.numel() * DTYPE_BYTES[str(p.dtype)] for p in model.parameters()) / 10 ** 9

    return param_count, mem_bytes

def profile_engrams(engram_model: EngramsModel):
    """ Compute total memory overhead for the model """

    total_param_count, total_mem_gbs = profile_module(engram_model)

    table_param_count, table_mem_gbs = 0, 0
    emb_param_count, emb_mem_gbs = 0, 0
    
    for block in engram_model.transformer_blocks:
        if block.engram:
            table_params, table_mem = profile_module(block.engram.multi_head_embedding.embedding)
            emb_params, emb_mem = profile_module(block.engram)
            table_param_count += table_params
            table_mem_gbs += table_mem
            emb_param_count += emb_params
            emb_mem_gbs += emb_mem

    param_ratio = table_param_count / total_param_count * 100
    mem_ratio = table_mem_gbs / total_mem_gbs * 100

    print(f"Model Params: {total_param_count / 10 ** 9}B")
    print(f"Embedding Module Params: {emb_param_count / 10 ** 9}B")
    print(f"Lookup Table Params: {table_param_count / 10 ** 9}B ({param_ratio:.2f}% are allocated to the lookup table)")
    print()
    print(f"Model Memory: {total_mem_gbs} GBs")
    print(f"Embedding Module Memory: {emb_mem_gbs} GBs")
    print(f"Lookup Table Memory: {table_mem_gbs} GBs ({mem_ratio:.2f}% is allocated to the lookup table)")


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

    torch.manual_seed(0)
    device = torch.device(config.device_map[0] if config.device_map else args.device)
    dtype = getattr(torch, args.dtype)
    model_cls = EngramsModel if args.impl == "optimized" else NaiveEngramsModel
    
    model = model_cls(config)
    if config.device_map:
        model.apply_device_map(dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)

    # Construct a random string
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.prompt_length),
        device=device,
        dtype=torch.long,
    )
    print(f"Input size: {input_ids.shape}")

    print("\n", "="*30, "Timing Profile", "="*30)

    # Determine how long it takes to perform just the engram state pred
    start = time.perf_counter()

    # model._prepare_engram_hashes(input_ids, use_cache=True)
    for block in model.transformer_blocks:
        if block.engram:
            block.engram.precompute_embeddings
    duration = time.perf_counter() - start

    avg_duration = duration / args.batch_size
    

    print(f"Device: {device}")

    
    print(f"Total time: {duration}s")
    print(f"Time per input: {avg_duration}s")

    print("\n", "="*30, "Memory Profile", "="*30)
    profile_engrams(model)

    print()


if __name__ == "__main__":
    main()
