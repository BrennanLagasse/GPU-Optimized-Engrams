import argparse
import json
import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engrams_kv_moe import EngramConfig, EngramsModel, engram_cfg


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
        "layer_ids": [0],
        "engrams_cfg": EngramConfig(
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            engram_vocab_size=args.engram_vocab_size,
            max_ngram_size=args.max_ngram_size,
            n_embed_per_ngram=args.n_embed_per_ngram,
            n_head_per_ngram=args.n_head_per_ngram,
            layer_ids=[0],
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
            kernel_size=args.kernel_size,
        ),
    }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device, trials):
    durations = []
    for _ in range(trials):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        durations.append(time.perf_counter() - start)
    return sum(durations) / len(durations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--hc-mult", type=int, default=1)
    parser.add_argument("--engram-vocab-size", type=int, nargs="*", default=[257, 263])
    parser.add_argument("--max-ngram-size", type=int, default=3)
    parser.add_argument("--n-embed-per-ngram", type=int, default=32)
    parser.add_argument("--n-head-per-ngram", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    model = EngramsModel(build_config(args)).to(device=device, dtype=dtype)
    block = next(block for block in model.transformer_blocks if block.engram is not None)
    engram = block.engram

    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.prompt_length),
        device=device,
        dtype=torch.long,
    )
    hidden_states = torch.randn(
        args.batch_size,
        args.prompt_length,
        args.emb_dim,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        hash_ids = engram.hash_mapping.hash_tensor(input_ids)[engram.layer_id]
        flat_hash_ids = hash_ids[:, -hidden_states.shape[1]:, :]
        embeddings = engram.multi_head_embedding(flat_hash_ids).flatten(start_dim=-2)

        def hash_only():
            engram.hash_mapping.hash_tensor(input_ids)

        def embedding_only():
            engram.multi_head_embedding(flat_hash_ids)

        def proj_gate_only():
            key = engram.key_norm(engram.key_proj(embeddings))
            query = engram.query_norm(hidden_states)
            gate = (key * query).sum(dim=-1) / (engram.model_dim ** 0.5)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            value = gate * engram.value_proj(embeddings)
            return value

        value = proj_gate_only()

        def short_conv_only():
            return engram.short_conv(value.unsqueeze(2))

        def full_engram():
            return engram(hidden_states, input_ids)

        results = {
            "model_size_params": sum(p.numel() for p in model.parameters()),
            "hash_seconds": timed(hash_only, device, args.trials),
            "embedding_seconds": timed(embedding_only, device, args.trials),
            "proj_gate_seconds": timed(proj_gate_only, device, args.trials),
            "short_conv_seconds": timed(short_conv_only, device, args.trials),
            "full_engram_seconds": timed(full_engram, device, args.trials),
        }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
