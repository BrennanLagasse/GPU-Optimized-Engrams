import math
import torch
from transformers import AutoTokenizer
from engrams_naive import NaiveEngramsModel, generate_text_naive
from engrams_kv_moe import (
    CompressedTokenizer,
    EngramConfig,
    EngramsModel,
    MultiHeadEmbedding,
    NgramHashMapping,
    ShortConv,
    engram_cfg,
    generate_text,
)

DEFAULT_CONFIG = {
    "vocab_size": 129280,                                   # Vocabulary size (prev 50257)
    "context_length": 256,                                  # Context length
    "emb_dim": 768,                                         # Embedding dimension
    "hidden_dim": 768*4,                                    # Intermediate size
    "n_heads": 12,                                          # Number of attention heads
    "n_layers": 12,                                         # Number of layers
    "drop_rate": 0.0,                                       # Dropout rate
    "qkv_bias": False,                                      # Query-Key-Value bias
    "num_experts": 0,
    "num_experts_per_tok": 0,
    "hc_mult": 1,                                           # Branching factor for HC (> 1 when HC is used)
    "layer_ids": []
}

def test_compressed_tokenizer_setup():
    """ Ensure that the vocab compression works, check how much the vocab is compressed """

    compressed_tokenizer = CompressedTokenizer(engram_cfg.tokenizer_name_or_path)

    vocab_size = len(compressed_tokenizer.tokenizer)
    new_vocab_size = len(compressed_tokenizer)

    print(f"Initial vocab size: {vocab_size}")
    print(f"New vocab size: {new_vocab_size}")

    print(f"Compression ratio {vocab_size / new_vocab_size}")

    assert new_vocab_size <= vocab_size

def test_generation_no_engrams_no_cache():
    """ Ensure basic forward pass of model works (no Engrams or KV-cache) """

    config = DEFAULT_CONFIG.copy()

    text = "Hello, I am"

    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path,
        trust_remote_code=True
    )

    input_ids = tokenizer(text,return_tensors='pt').input_ids

    model = EngramsModel(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    model.to(device, dtype=torch.bfloat16)

    output_ids = generate_text(
        model=model, 
        input_ids=input_ids.to(device), 
        max_new_tokens=128,
        use_cache=False
    )

    decoded_output = tokenizer.decode(output_ids.squeeze(0).tolist())

    assert decoded_output is not None

def test_multi_head_embedding_shape():
    embedding = MultiHeadEmbedding(list_of_N=[7, 11, 13], D=5)
    input_ids = torch.tensor([[[0, 1, 2], [3, 4, 5]]], dtype=torch.long)

    output = embedding(input_ids)

    assert output.shape == (1, 2, 3, 5)

def test_ngram_hash_mapping_shape_and_range():
    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path,
        trust_remote_code=True,
    )
    input_ids = tokenizer("hello world", return_tensors="pt").input_ids

    mapping = NgramHashMapping(
        engram_vocab_size=[17, 19],
        max_ngram_size=3,
        n_embed_per_ngram=16,
        n_head_per_ngram=2,
        layer_ids=[0],
        tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
        pad_id=engram_cfg.pad_id,
        seed=0,
    )

    hashes = mapping.hash(input_ids)[0]

    assert hashes.shape == (1, input_ids.shape[1], 4)
    assert hashes.dtype.kind == "i"
    assert hashes.min() >= 0


def test_ngram_hash_mapping_last_token_matches_full_hash():
    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path,
        trust_remote_code=True,
    )
    input_ids = tokenizer("hello world again", return_tensors="pt").input_ids

    mapping = NgramHashMapping(
        engram_vocab_size=[17, 19],
        max_ngram_size=3,
        n_embed_per_ngram=16,
        n_head_per_ngram=2,
        layer_ids=[0],
        tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
        pad_id=engram_cfg.pad_id,
        seed=0,
    )

    full_hashes = mapping.hash_tensor(input_ids)[0][:, -1:, :]
    last_hashes = mapping.hash_last_tensor(input_ids)[0]

    assert torch.equal(full_hashes, last_hashes)

def test_forward_with_small_engrams_config():
    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path,
        trust_remote_code=True,
    )
    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids

    small_engrams_cfg = EngramConfig(
        tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
        engram_vocab_size=[17, 19],
        max_ngram_size=3,
        n_embed_per_ngram=16,
        n_head_per_ngram=4,
        layer_ids=[0],
        pad_id=engram_cfg.pad_id,
        seed=0,
        kernel_size=2,
    )
    config = DEFAULT_CONFIG.copy()
    config.update({
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 1,
        "layer_ids": [0],
        "engrams_cfg": small_engrams_cfg,
    })

    model = EngramsModel(config)
    logits = model(input_ids)

    assert logits.shape == (1, input_ids.shape[1], config["vocab_size"])

def test_forward_with_small_moe_config():
    config = DEFAULT_CONFIG.copy()
    config.update({
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 1,
        "num_experts": 2,
        "num_experts_per_tok": 1,
    })

    model = EngramsModel(config)
    input_ids = torch.randint(0, config["vocab_size"], (1, 6), dtype=torch.long)
    logits = model(input_ids)

    assert logits.shape == (1, 6, config["vocab_size"])

def test_forward_with_small_mhc_config():
    config = DEFAULT_CONFIG.copy()
    config.update({
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 1,
        "hc_mult": 3,
    })

    model = EngramsModel(config)
    input_ids = torch.randint(0, config["vocab_size"], (1, 5), dtype=torch.long)
    logits = model(input_ids)

    assert logits.shape == (1, 5, config["vocab_size"])

def test_generate_with_and_without_cache_match():
    torch.manual_seed(0)
    config = DEFAULT_CONFIG.copy()
    config.update({
        "vocab_size": 256,
        "context_length": 32,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "layer_ids": [],
    })

    model = EngramsModel(config)
    input_ids = torch.randint(0, config["vocab_size"], (1, 6), dtype=torch.long)

    no_cache = generate_text(model, input_ids, max_new_tokens=4, context_size=32, use_cache=False)
    with_cache = generate_text(model, input_ids, max_new_tokens=4, context_size=32, use_cache=True)

    assert torch.equal(no_cache, with_cache)


def test_generate_with_and_without_cache_match_with_two_engram_layers():
    torch.manual_seed(0)
    config = DEFAULT_CONFIG.copy()
    config.update({
        "vocab_size": 256,
        "context_length": 32,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "layer_ids": [0, 1],
        "engrams_cfg": _small_engrams_cfg(layer_ids=[0, 1]),
    })

    model = EngramsModel(config)
    input_ids = torch.randint(0, config["vocab_size"], (1, 6), dtype=torch.long)

    no_cache = generate_text(model, input_ids, max_new_tokens=4, context_size=32, use_cache=False)
    with_cache = generate_text(model, input_ids, max_new_tokens=4, context_size=32, use_cache=True)

    assert torch.equal(no_cache, with_cache)

def _load_shared_weights(dst_model, src_model):
    dst_state = dst_model.state_dict()
    src_state = src_model.state_dict()
    shared = {
        key: value
        for key, value in src_state.items()
        if key in dst_state and dst_state[key].shape == value.shape
    }
    missing, unexpected = dst_model.load_state_dict(shared, strict=False)
    assert not unexpected
    return missing


def _small_engrams_cfg(**overrides):
    cfg = EngramConfig(
        tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
        engram_vocab_size=[17, 19],
        max_ngram_size=3,
        n_embed_per_ngram=16,
        n_head_per_ngram=4,
        layer_ids=[0],
        pad_id=engram_cfg.pad_id,
        seed=0,
        kernel_size=2,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg

def test_naive_and_optimized_forward_match_without_engrams():
    torch.manual_seed(0)
    config = DEFAULT_CONFIG.copy()
    config.update({
        "vocab_size": 129280,
        "context_length": 16,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "layer_ids": [],
    })

    optimized = EngramsModel(config)
    naive = NaiveEngramsModel(config)
    _load_shared_weights(naive, optimized)

    input_ids = torch.randint(0, config["vocab_size"], (1, 6), dtype=torch.long)
    optimized_logits = optimized(input_ids, use_cache=False)
    naive_logits = naive(input_ids, use_cache=False)

    assert torch.allclose(optimized_logits, naive_logits, atol=1e-5, rtol=1e-5)

def test_naive_and_optimized_forward_match_with_small_engrams():
    torch.manual_seed(0)
    small_engrams_cfg = _small_engrams_cfg()
    config = DEFAULT_CONFIG.copy()
    config.update({
        "vocab_size": 129280,
        "context_length": 16,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 1,
        "layer_ids": [0],
        "engrams_cfg": small_engrams_cfg,
    })

    optimized = EngramsModel(config)
    naive = NaiveEngramsModel(config)
    _load_shared_weights(naive, optimized)

    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path,
        trust_remote_code=True,
    )
    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
    optimized_logits = optimized(input_ids, use_cache=False, engram_input_ids=input_ids)
    naive_logits = naive(input_ids, use_cache=False, engram_input_ids=input_ids)

    assert torch.allclose(optimized_logits, naive_logits, atol=1e-5, rtol=1e-5)


def test_short_conv_step_matches_full_conv_for_single_token():
    torch.manual_seed(0)
    short_conv = ShortConv(hidden_size=8, kernel_size=3, dilation=2, hc_mult=1)
    x = torch.randn(2, 1, 1, 8)

    full = short_conv(x)
    step = short_conv.forward_step(x)

    assert torch.allclose(full, step, atol=1e-6, rtol=1e-6)


def test_cached_step_kernel_mode_matches_no_cache_generation():
    torch.manual_seed(0)
    step_cfg = DEFAULT_CONFIG.copy()
    step_cfg.update({
        "vocab_size": 256,
        "context_length": 16,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 1,
        "layer_ids": [0],
        "engrams_cfg": _small_engrams_cfg(cached_inference_short_conv_mode="step_kernel"),
    })

    step_model = EngramsModel(step_cfg)
    input_ids = torch.randint(0, step_cfg["vocab_size"], (1, 6), dtype=torch.long)
    no_cache_out = generate_text(step_model, input_ids, max_new_tokens=4, context_size=16, use_cache=False)
    step_out = generate_text(step_model, input_ids, max_new_tokens=4, context_size=16, use_cache=True)

    assert torch.equal(no_cache_out, step_out)


def test_gated_value_only_cached_mode_has_bounded_logit_drift():
    torch.manual_seed(0)
    full_cfg = DEFAULT_CONFIG.copy()
    full_cfg.update({
        "vocab_size": 256,
        "context_length": 16,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 1,
        "layer_ids": [0],
        "engrams_cfg": _small_engrams_cfg(cached_inference_short_conv_mode="full"),
    })
    value_only_cfg = DEFAULT_CONFIG.copy()
    value_only_cfg.update({
        "vocab_size": 256,
        "context_length": 16,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 1,
        "layer_ids": [0],
        "engrams_cfg": _small_engrams_cfg(cached_inference_short_conv_mode="gated_value_only"),
    })

    full_model = EngramsModel(full_cfg)
    value_only_model = EngramsModel(value_only_cfg)
    _load_shared_weights(value_only_model, full_model)

    input_ids = torch.randint(0, full_cfg["vocab_size"], (1, 6), dtype=torch.long)

    full_model.reset_cache()
    value_only_model.reset_cache()
    with torch.no_grad():
        _ = full_model(input_ids, use_cache=True, position_offset=0, engram_input_ids=input_ids)
        _ = value_only_model(input_ids, use_cache=True, position_offset=0, engram_input_ids=input_ids)
        step_input = input_ids[:, -1:]
        engram_window = input_ids[:, -full_cfg["engrams_cfg"].max_ngram_size + 1 :]
        full_logits = full_model(
            step_input,
            use_cache=True,
            position_offset=input_ids.shape[1] - 1,
            engram_input_ids=engram_window,
        )
        value_only_logits = value_only_model(
            step_input,
            use_cache=True,
            position_offset=input_ids.shape[1] - 1,
            engram_input_ids=engram_window,
        )

    max_abs_delta = (full_logits - value_only_logits).abs().max().item()
    assert math.isfinite(max_abs_delta)
    assert max_abs_delta < 1.0

def test_naive_and_optimized_forward_match_with_mhc():
    torch.manual_seed(0)
    config = DEFAULT_CONFIG.copy()
    config.update({
        "vocab_size": 512,
        "context_length": 16,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "layer_ids": [],
        "hc_mult": 3,
    })

    optimized = EngramsModel(config)
    naive = NaiveEngramsModel(config)
    _load_shared_weights(naive, optimized)

    input_ids = torch.randint(0, config["vocab_size"], (1, 6), dtype=torch.long)
    optimized_logits = optimized(input_ids, use_cache=False)
    naive_logits = naive(input_ids, use_cache=False)

    assert torch.allclose(optimized_logits, naive_logits, atol=1e-5, rtol=1e-5)

def test_naive_and_optimized_generation_match_without_engrams():
    torch.manual_seed(0)
    config = DEFAULT_CONFIG.copy()
    config.update({
        "vocab_size": 256,
        "context_length": 16,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "layer_ids": [],
    })

    optimized = EngramsModel(config)
    naive = NaiveEngramsModel(config)
    _load_shared_weights(naive, optimized)

    input_ids = torch.randint(0, config["vocab_size"], (1, 6), dtype=torch.long)
    optimized_out = generate_text(optimized, input_ids, max_new_tokens=4, context_size=16, use_cache=False)
    naive_out = generate_text_naive(naive, input_ids, max_new_tokens=4, context_size=16)

    assert torch.equal(optimized_out, naive_out)

def test_naive_and_optimized_generation_match_with_mhc():
    torch.manual_seed(0)
    config = DEFAULT_CONFIG.copy()
    config.update({
        "vocab_size": 256,
        "context_length": 16,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "layer_ids": [],
        "hc_mult": 3,
    })

    optimized = EngramsModel(config)
    naive = NaiveEngramsModel(config)
    _load_shared_weights(naive, optimized)

    input_ids = torch.randint(0, config["vocab_size"], (1, 6), dtype=torch.long)
    optimized_out = generate_text(optimized, input_ids, max_new_tokens=4, context_size=16, use_cache=False)
    naive_out = generate_text_naive(naive, input_ids, max_new_tokens=4, context_size=16)

    assert torch.equal(optimized_out, naive_out)


def test_device_map_cpu_smoke_matches_single_device():
    torch.manual_seed(0)
    config = DEFAULT_CONFIG.copy()
    config.update({
        "vocab_size": 256,
        "context_length": 16,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "layer_ids": [0, 1],
        "engrams_cfg": _small_engrams_cfg(layer_ids=[0, 1]),
    })

    mapped_config = dict(config)
    mapped_config["device_map"] = ["cpu", "cpu"]

    optimized = EngramsModel(config)
    optimized_mapped = EngramsModel(mapped_config)
    _load_shared_weights(optimized_mapped, optimized)

    naive = NaiveEngramsModel(config)
    naive_mapped = NaiveEngramsModel(mapped_config)
    _load_shared_weights(naive, optimized)
    _load_shared_weights(naive_mapped, optimized)

    input_ids = torch.randint(0, config["vocab_size"], (1, 6), dtype=torch.long)

    optimized_logits = optimized(input_ids, use_cache=False, engram_input_ids=input_ids)
    optimized_mapped_logits = optimized_mapped(input_ids, use_cache=False, engram_input_ids=input_ids)
    naive_logits = naive(input_ids, use_cache=False, engram_input_ids=input_ids)
    naive_mapped_logits = naive_mapped(input_ids, use_cache=False, engram_input_ids=input_ids)

    assert torch.allclose(optimized_logits, optimized_mapped_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(naive_logits, naive_mapped_logits, atol=1e-5, rtol=1e-5)




def main():
    test_compressed_tokenizer_setup()

if __name__ == '__main__':
    main()
