import argparse
import json
import math


def attention_params(emb_dim, qkv_bias):
    qkv = 3 * emb_dim * emb_dim
    qkv_bias_params = 3 * emb_dim if qkv_bias else 0
    out_proj = emb_dim * emb_dim + emb_dim
    return qkv + qkv_bias_params + out_proj


def dense_ffn_params(emb_dim, hidden_dim):
    return 3 * emb_dim * hidden_dim


def moe_params(emb_dim, hidden_dim, num_experts):
    if num_experts <= 0:
        return dense_ffn_params(emb_dim, hidden_dim)
    return emb_dim * num_experts + num_experts * dense_ffn_params(emb_dim, hidden_dim)


def norm_params(emb_dim, count):
    return count * 2 * emb_dim


def mhc_router_params(width, model_dim):
    if width <= 1:
        return 0
    out_dim = width * width + 2 * width
    return model_dim * out_dim + out_dim


def approx_engram_table_rows(engram_vocab_size, n_head_per_ngram):
    return sum(v * n_head_per_ngram for v in engram_vocab_size)


def engram_layer_params(cfg):
    engram_hidden = (cfg["max_ngram_size"] - 1) * cfg["n_embed_per_ngram"]
    emb_per_head = cfg["n_embed_per_ngram"] // cfg["n_head_per_ngram"]
    table_rows = approx_engram_table_rows(cfg["engram_vocab_size"], cfg["n_head_per_ngram"])
    table_params = table_rows * emb_per_head
    proj_params = 2 * engram_hidden * cfg["emb_dim"] + 2 * cfg["emb_dim"]
    short_conv_params = 0
    if cfg["use_short_conv"]:
        short_conv_params = cfg["emb_dim"] * cfg["kernel_size"] + cfg["emb_dim"]
    mhc_params = mhc_router_params(cfg["hc_mult"], cfg["emb_dim"])
    return {
        "table_rows_approx": table_rows,
        "table_params_approx": table_params,
        "projection_params": proj_params,
        "short_conv_params": short_conv_params,
        "mhc_params": mhc_params,
        "total_params_approx": table_params + proj_params + short_conv_params + mhc_params,
    }


def estimate_model_params(cfg):
    emb_dim = cfg["emb_dim"]
    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]
    n_layers = cfg["n_layers"]
    num_engram_layers = len(cfg["layer_ids"])

    embeddings = vocab_size * emb_dim + context_length * emb_dim
    out_head = emb_dim * vocab_size

    block_base = (
        attention_params(emb_dim, cfg["qkv_bias"])
        + moe_params(emb_dim, cfg["hidden_dim"], cfg["num_experts"])
        + norm_params(emb_dim, 3)
        + 2 * mhc_router_params(cfg["hc_mult"], emb_dim)
    )
    engram = engram_layer_params(cfg)
    total = embeddings + out_head + n_layers * block_base + num_engram_layers * engram["total_params_approx"]

    return {
        "total_params_approx": total,
        "embeddings_params": embeddings,
        "out_head_params": out_head,
        "per_block_base_params": block_base,
        "engram_layer_params_approx": engram,
        "num_engram_layers": num_engram_layers,
    }


def estimate_inference_memory_bytes(cfg, dtype_bytes, batch_size, tensor_parallel):
    params = estimate_model_params(cfg)
    total_param_bytes = params["total_params_approx"] * dtype_bytes
    kv_per_token = 2 * cfg["n_layers"] * cfg["emb_dim"] * dtype_bytes
    kv_cache_bytes = batch_size * cfg["context_length"] * kv_per_token
    activation_bytes = batch_size * cfg["context_length"] * cfg["emb_dim"] * max(cfg["n_layers"], 1) * dtype_bytes
    engram = params["engram_layer_params_approx"]
    engram_table_bytes = engram["table_params_approx"] * params["num_engram_layers"] * dtype_bytes

    return {
        "total_param_bytes_approx": total_param_bytes,
        "param_bytes_per_rank_approx": total_param_bytes / max(tensor_parallel, 1),
        "kv_cache_bytes_approx": kv_cache_bytes,
        "activation_bytes_approx": activation_bytes,
        "engram_table_bytes_approx": engram_table_bytes,
        "working_set_per_rank_bytes_approx": (
            (total_param_bytes / max(tensor_parallel, 1))
            + kv_cache_bytes / max(tensor_parallel, 1)
            + activation_bytes / max(tensor_parallel, 1)
        ),
    }


def dense_decode_flops_per_layer(emb_dim, hidden_dim, seq_len, use_cache=True):
    projection_flops = 8 * emb_dim * emb_dim
    ffn_flops = 6 * emb_dim * hidden_dim
    if use_cache:
        attention_flops = 4 * seq_len * emb_dim
    else:
        attention_flops = 4 * seq_len * seq_len * emb_dim
    return projection_flops + ffn_flops + attention_flops


def mhc_decode_flops_per_router(width, model_dim, sinkhorn_iters=4):
    if width <= 1:
        return 0
    router_out = width * width + 2 * width
    router_linear = 2 * model_dim * router_out
    sinkhorn = 4 * sinkhorn_iters * width * width
    reductions = 4 * width * model_dim + 2 * width * width * model_dim
    return router_linear + sinkhorn + reductions


def engram_decode_flops_per_layer(cfg):
    engram_hidden = (cfg["max_ngram_size"] - 1) * cfg["n_embed_per_ngram"]
    model_dim = cfg["emb_dim"]
    projection_flops = 4 * engram_hidden * model_dim
    gating_flops = 6 * model_dim
    short_conv_flops = 0
    if cfg["use_short_conv"]:
        short_conv_flops = 2 * cfg["kernel_size"] * model_dim
    return projection_flops + gating_flops + short_conv_flops


def estimate_decode_flops_per_token(cfg, seq_len, use_cache=True):
    per_layer_dense = dense_decode_flops_per_layer(
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        seq_len=seq_len,
        use_cache=use_cache,
    )
    router_flops = mhc_decode_flops_per_router(cfg["hc_mult"], cfg["emb_dim"])
    router_count = 2 * cfg["n_layers"] + len(cfg["layer_ids"])
    engram_extra = len(cfg["layer_ids"]) * engram_decode_flops_per_layer(cfg)
    token_embed = 2 * cfg["vocab_size"] * cfg["emb_dim"]
    output_head = 2 * cfg["emb_dim"] * cfg["vocab_size"]
    total = (
        cfg["n_layers"] * per_layer_dense
        + router_count * router_flops
        + engram_extra
        + token_embed
        + output_head
    )
    return {
        "seq_len": seq_len,
        "use_cache": use_cache,
        "forward_flops_per_token_approx": total,
        "forward_gflops_per_token_approx": total / 1e9,
    }


PRESETS = {
    "tiny_engram": {
        "vocab_size": 512,
        "context_length": 32,
        "emb_dim": 64,
        "hidden_dim": 128,
        "n_heads": 4,
        "n_layers": 2,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "hc_mult": 1,
        "layer_ids": [0],
        "engram_vocab_size": [257, 263],
        "max_ngram_size": 3,
        "n_embed_per_ngram": 32,
        "n_head_per_ngram": 4,
        "kernel_size": 2,
        "use_short_conv": True,
        "qkv_bias": False,
    },
    "medium_engram": {
        "vocab_size": 4096,
        "context_length": 64,
        "emb_dim": 256,
        "hidden_dim": 1024,
        "n_heads": 8,
        "n_layers": 6,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "hc_mult": 1,
        "layer_ids": [0],
        "engram_vocab_size": [1021, 1031],
        "max_ngram_size": 3,
        "n_embed_per_ngram": 64,
        "n_head_per_ngram": 4,
        "kernel_size": 2,
        "use_short_conv": True,
        "qkv_bias": False,
    },
    "large_engram": {
        "vocab_size": 32768,
        "context_length": 128,
        "emb_dim": 2048,
        "hidden_dim": 8192,
        "n_heads": 16,
        "n_layers": 24,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "hc_mult": 1,
        "layer_ids": [0, 1],
        "engram_vocab_size": [8191, 8209],
        "max_ngram_size": 3,
        "n_embed_per_ngram": 128,
        "n_head_per_ngram": 8,
        "kernel_size": 2,
        "use_short_conv": True,
        "qkv_bias": False,
    },
    "target_32b_approx": {
        "vocab_size": 129280,
        "context_length": 4096,
        "emb_dim": 6144,
        "hidden_dim": 24576,
        "n_heads": 48,
        "n_layers": 48,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "hc_mult": 4,
        "layer_ids": [0, 1],
        "engram_vocab_size": [129280 * 5, 129280 * 5],
        "max_ngram_size": 3,
        "n_embed_per_ngram": 512,
        "n_head_per_ngram": 8,
        "kernel_size": 4,
        "use_short_conv": True,
        "qkv_bias": False,
    },
    "target_40b_approx": {
        "vocab_size": 129280,
        "context_length": 4096,
        "emb_dim": 6656,
        "hidden_dim": 26624,
        "n_heads": 52,
        "n_layers": 52,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "hc_mult": 4,
        "layer_ids": [0, 1],
        "engram_vocab_size": [129280 * 5, 129280 * 5],
        "max_ngram_size": 3,
        "n_embed_per_ngram": 512,
        "n_head_per_ngram": 8,
        "kernel_size": 4,
        "use_short_conv": True,
        "qkv_bias": False,
    },
}


def bytes_to_gib(value):
    return value / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    args = parser.parse_args()

    cfg = PRESETS[args.preset]
    params = estimate_model_params(cfg)
    mem = estimate_inference_memory_bytes(cfg, args.dtype_bytes, args.batch_size, args.tensor_parallel)

    output = {
        "preset": args.preset,
        "config": cfg,
        "total_params_approx": params["total_params_approx"],
        "total_params_approx_billions": params["total_params_approx"] / 1e9,
        "engram_table_rows_approx": params["engram_layer_params_approx"]["table_rows_approx"],
        "engram_table_params_per_layer_approx": params["engram_layer_params_approx"]["table_params_approx"],
        "param_gib_total_approx": bytes_to_gib(mem["total_param_bytes_approx"]),
        "param_gib_per_rank_approx": bytes_to_gib(mem["param_bytes_per_rank_approx"]),
        "kv_cache_gib_approx": bytes_to_gib(mem["kv_cache_bytes_approx"]),
        "activation_gib_approx": bytes_to_gib(mem["activation_bytes_approx"]),
        "engram_table_gib_approx": bytes_to_gib(mem["engram_table_bytes_approx"]),
        "working_set_per_rank_gib_approx": bytes_to_gib(mem["working_set_per_rank_bytes_approx"]),
        "cached_decode_flops_per_token_context_64_approx": estimate_decode_flops_per_token(
            cfg,
            seq_len=64,
            use_cache=True,
        ),
        "tensor_parallel": args.tensor_parallel,
        "dtype_bytes": args.dtype_bytes,
        "batch_size": args.batch_size,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
