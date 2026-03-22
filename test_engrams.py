import torch
from transformers import AutoTokenizer
from engrams_kv_moe import CompressedTokenizer, EngramsModel, engram_cfg, generate_text

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




def main():
    test_compressed_tokenizer_setup()

if __name__ == '__main__':
    main()