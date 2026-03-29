import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engrams_kv_moe import EngramConfig, EngramsModel, engram_cfg
from engrams_naive import NaiveEngramsModel


def build_config(args):
    cfg = {
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
        ),
    }
    return cfg


def parameter_bytes(model):
    return sum(param.numel() * param.element_size() for param in model.parameters())


def buffer_bytes(model):
    return sum(buf.numel() * buf.element_size() for buf in model.buffers())


def engram_table_rows(model):
    total_rows = 0
    for block in model.transformer_blocks:
        if block.engram is None:
            continue
        total_rows += block.engram.multi_head_embedding.embedding.num_embeddings
    return total_rows


def activation_bytes_estimate(cfg, batch_size):
    base_hidden = batch_size * cfg["context_length"] * cfg["emb_dim"] * 4
    if cfg["hc_mult"] > 1:
        base_hidden *= cfg["hc_mult"]
    return base_hidden * max(cfg["n_layers"], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["optimized", "naive"], default="optimized")
    parser.add_argument("--vocab-size", type=int, default=129280)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=3072)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--num-experts", type=int, default=0)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--hc-mult", type=int, default=1)
    parser.add_argument("--layer-ids", type=int, nargs="*", default=[1])
    parser.add_argument("--engram-vocab-size", type=int, nargs="*", default=[129280 * 5, 129280 * 5])
    parser.add_argument("--max-ngram-size", type=int, default=3)
    parser.add_argument("--n-embed-per-ngram", type=int, default=512)
    parser.add_argument("--n-head-per-ngram", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    cfg = build_config(args)
    model_cls = EngramsModel if args.impl == "optimized" else NaiveEngramsModel
    model = model_cls(cfg)

    stats = {
        "implementation": args.impl,
        "parameters": sum(param.numel() for param in model.parameters()),
        "parameter_bytes": parameter_bytes(model),
        "buffer_bytes": buffer_bytes(model),
        "engram_embedding_rows": engram_table_rows(model),
        "activation_bytes_estimate": activation_bytes_estimate(cfg, args.batch_size),
        "num_engram_layers": len(cfg["layer_ids"]),
        "num_experts": cfg["num_experts"],
        "hc_mult": cfg["hc_mult"],
    }
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
