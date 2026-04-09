import argparse
import json
import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engrams_kv_moe import EngramConfig, EngramsModel, engram_cfg, generate_text


def build_config(args, cached_mode):
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
            use_short_conv=not args.disable_short_conv,
            cached_inference_short_conv_mode=cached_mode,
        ),
    }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_generation(model, input_ids, max_new_tokens, context_length, use_cache, device, trials):
    durations = []
    output = None
    for _ in range(trials):
        sync(device)
        start = time.perf_counter()
        output = generate_text(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            context_size=context_length,
            use_cache=use_cache,
        )
        sync(device)
        durations.append(time.perf_counter() - start)
    avg_seconds = sum(durations) / len(durations)
    toks = input_ids.shape[0] * max_new_tokens
    return {
        "avg_seconds": avg_seconds,
        "tokens_per_second": toks / avg_seconds,
        "output": output,
    }


def copy_shared_weights(dst_model, src_model):
    dst_state = dst_model.state_dict()
    src_state = src_model.state_dict()
    shared = {
        key: value
        for key, value in src_state.items()
        if key in dst_state and dst_state[key].shape == value.shape
    }
    missing, unexpected = dst_model.load_state_dict(shared, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading weights: {unexpected}")
    return missing


def cached_step_logits(model, prompt):
    model.reset_cache()
    with torch.no_grad():
        _ = model(
            prompt,
            use_cache=True,
            position_offset=0,
            engram_input_ids=prompt,
        )
        step_input = prompt[:, -1:]
        engram_window = prompt[:, -model.config["engrams_cfg"].max_ngram_size + 1 :]
        return model(
            step_input,
            use_cache=True,
            position_offset=prompt.shape[1] - 1,
            engram_input_ids=engram_window,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--trials", type=int, default=3)
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
    parser.add_argument("--disable-short-conv", action="store_true")
    parser.add_argument(
        "--candidate-mode",
        choices=["step_kernel", "gated_value_only"],
        default="step_kernel",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    baseline = EngramsModel(build_config(args, "full")).to(device=device, dtype=dtype)
    candidate = EngramsModel(build_config(args, args.candidate_mode)).to(device=device, dtype=dtype)
    copy_shared_weights(candidate, baseline)

    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.prompt_length),
        device=device,
        dtype=torch.long,
    )

    with torch.no_grad():
        baseline_nocache = timed_generation(
            baseline, input_ids, args.max_new_tokens, args.context_length, False, device, args.trials
        )
        baseline_cache = timed_generation(
            baseline, input_ids, args.max_new_tokens, args.context_length, True, device, args.trials
        )
        candidate_nocache = timed_generation(
            candidate, input_ids, args.max_new_tokens, args.context_length, False, device, args.trials
        )
        candidate_cache = timed_generation(
            candidate, input_ids, args.max_new_tokens, args.context_length, True, device, args.trials
        )

        baseline_step_logits = cached_step_logits(baseline, input_ids)
        candidate_step_logits = cached_step_logits(candidate, input_ids)

    results = {
        "baseline_mode": "full",
        "candidate_mode": args.candidate_mode,
        "model_size_params": sum(p.numel() for p in baseline.parameters()),
        "baseline_no_cache_tokens_per_second": baseline_nocache["tokens_per_second"],
        "baseline_cache_tokens_per_second": baseline_cache["tokens_per_second"],
        "candidate_no_cache_tokens_per_second": candidate_nocache["tokens_per_second"],
        "candidate_cache_tokens_per_second": candidate_cache["tokens_per_second"],
        "candidate_cache_improvement_percent": (
            (candidate_cache["tokens_per_second"] / baseline_cache["tokens_per_second"]) - 1.0
        ) * 100.0,
        "cached_generation_equal": torch.equal(baseline_cache["output"], candidate_cache["output"]),
        "no_cache_generation_equal": torch.equal(baseline_nocache["output"], candidate_nocache["output"]),
        "cached_step_max_abs_logit_delta": (
            baseline_step_logits - candidate_step_logits
        ).abs().max().item(),
        "cached_step_mean_abs_logit_delta": (
            baseline_step_logits - candidate_step_logits
        ).abs().mean().item(),
    }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
