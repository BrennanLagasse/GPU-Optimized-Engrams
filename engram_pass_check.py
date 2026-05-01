import argparse
import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engrams_kv_moe import EngramConfig, EngramsModel, engram_cfg, generate_text
from engrams_naive import NaiveEngramsModel, generate_text_naive

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

def compute_engrams_memory(model: EngramsModel):
    """ Compute the size of the Engrams lookup table """

    lookup_table_mem = 0
    lookup_table_params = 0

    for block in model.transformer_blocks:
        if block.engram:

            # Compute memory for the lookup table
            emb = block.engram.multi_head_embedding.embedding
            bytes_per_param = DTYPE_BYTES[str(emb.weight.dtype)]
            num_params = emb.num_embeddings * emb.embedding_dim
            lookup_table_mem += bytes_per_param * num_params
            lookup_table_params += num_params

    total_params = lookup_table_params
    total_mem = lookup_table_mem
    total_mem_gbs = total_mem / 10 ** 9

    print(f"Lookup Table Params: {lookup_table_params}")
    print(f"Total Parameters: {total_params}")
    print(f"Total Memory: {total_mem_gbs} GB")

    return total_mem, total_params

def profile_engrams(engram_model: EngramsModel):
    """ Compute total memory overhead for the model """

    total_param_count, total_mem_gbs = profile_module(engram_model)

    table_param_count, table_mem_gbs = 0, 0
    
    for block in engram_model.transformer_blocks:
        if block.engram:
            params, mem = profile_module(block.engram.multi_head_embedding.embedding)
            table_param_count += params
            table_mem_gbs += mem

    param_ratio = total_param_count / table_param_count * 100
    mem_ratio = table_mem_gbs / total_mem_gbs * 100

    print(f"Model Params: {total_param_count}")
    print(f"Lookup Table Params: {table_param_count} ({param_ratio:.2f}% are allocated to the lookup table)")
    print(f"Model Memory: {total_mem_gbs} GBs")
    print(f"Lookup Table Memory: {table_mem_gbs} GBs ({mem_ratio:.2f}% is allocated to the lookup table)")

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
            # engram_vocab_size=[250, 250],
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["optimized", "naive"], default="optimized")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device-map", type=str, default="")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=0)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--hc-mult", type=int, default=1)
    parser.add_argument("--layer-ids", type=int, nargs="*", default=[])
    # parser.add_argument("--engram-vocab-size", type=int, nargs="*", default=[257, 263])
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
    device_map = [entry.strip() for entry in args.device_map.split(",") if entry.strip()]
    args.device_map = device_map
    device = torch.device(device_map[0] if device_map else args.device)
    dtype = getattr(torch, args.dtype)
    config = build_config(args)
    model_cls = EngramsModel if args.impl == "optimized" else NaiveEngramsModel
    model = model_cls(config)
    if device_map:
        model.apply_device_map(dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)

    # Construct a random string
    input_ids = torch.randint(
        0,
        config["vocab_size"],
        (args.batch_size, args.prompt_length),
        device=device,
        dtype=torch.long,
    )
    print(f"Input size: {input_ids.shape}")

    # Determine how long it takes to perform just the engram state pred
    start = time.perf_counter()
    model._prepare_engram_hashes(input_ids, use_cache=True)
    duration = time.perf_counter() - start

    avg_duration = duration / args.batch_size
    

    print(f"Device: {device}")

    
    print(f"Total time: {duration}")
    print(f"Time per input: {avg_duration}")

    print("\n", "="*30, "\nMemory Profile\n", "="*30)

    compute_engrams_memory(model)
    print()
    profile_engrams(model)


if __name__ == "__main__":
    main()
