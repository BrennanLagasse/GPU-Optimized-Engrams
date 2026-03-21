from engrams_kv_moe import *

def test_compressed_tokenizer_setup():
    compressed_tokenizer = CompressedTokenizer(engram_cfg.tokenizer_name_or_path)

    vocab_size = len(compressed_tokenizer.tokenizer)
    new_vocab_size = len(compressed_tokenizer)

    print(f"Initial vocab size: {vocab_size}")
    print(f"New vocab size: {new_vocab_size}")

    print(f"Compression ratio {vocab_size / new_vocab_size}")

    assert new_vocab_size <= vocab_size


def main():
    test_compressed_tokenizer_setup()

if __name__ == '__main__':
    main()