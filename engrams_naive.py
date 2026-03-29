"""
Naive implementation for Engrams.

This file keeps a straightforward, correctness-first implementation separate
from the optimized baseline in `engrams_kv_moe.py` so before/after
benchmarking is well-defined.
"""

import math

import torch
import torch.nn as nn

from engrams_kv_moe import (
    DenseFeedForward,
    EngramConfig,
    LayerNorm,
    MoEFeedForward,
    MultiHeadAttention,
    NgramHashMapping,
    MultiHeadEmbedding,
    ShortConv,
    engram_cfg,
)


class NaiveEngram(nn.Module):
    """
    Straightforward Engram block.

    Differences from the optimized path:
    - recomputes hashes every forward pass
    - does not precompute hashes across layers
    - does not use mHC
    - keeps the residual stream as a single branch
    """

    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.model_dim = config["emb_dim"]
        self.engram_cfg = config.get("engrams_cfg", engram_cfg)

        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=self.engram_cfg.engram_vocab_size,
            max_ngram_size=self.engram_cfg.max_ngram_size,
            n_embed_per_ngram=self.engram_cfg.n_embed_per_ngram,
            n_head_per_ngram=self.engram_cfg.n_head_per_ngram,
            layer_ids=self.engram_cfg.layer_ids,
            tokenizer_name_or_path=self.engram_cfg.tokenizer_name_or_path,
            pad_id=self.engram_cfg.pad_id,
            seed=self.engram_cfg.seed,
        )
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=[x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D=self.engram_cfg.n_embed_per_ngram // self.engram_cfg.n_head_per_ngram,
        )
        self.short_conv = ShortConv(
            hidden_size=self.model_dim,
            kernel_size=self.engram_cfg.kernel_size,
            dilation=self.engram_cfg.max_ngram_size,
            hc_mult=1,
        )

        engram_hidden_size = (self.engram_cfg.max_ngram_size - 1) * self.engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, self.model_dim)
        self.key_proj = nn.Linear(engram_hidden_size, self.model_dim)
        self.key_norm = nn.RMSNorm(self.model_dim)
        self.query_norm = nn.RMSNorm(self.model_dim)

    def forward(self, hidden_states, input_ids):
        device = hidden_states.device
        hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id]).to(device)
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        key = self.key_norm(self.key_proj(embeddings))
        query = self.query_norm(hidden_states)
        gate = (key * query).sum(dim=-1) / math.sqrt(self.model_dim)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid().unsqueeze(-1)
        value = gate * self.value_proj(embeddings)
        return value + self.short_conv(value.unsqueeze(2)).squeeze(2)


class NaiveTransformerBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        if config["num_experts"] > 0:
            self.ff = MoEFeedForward(config)
        else:
            self.ff = DenseFeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.norm3 = LayerNorm(config["emb_dim"])
        self.engram = None
        if layer_id in config["layer_ids"]:
            self.engram = NaiveEngram(config=config, layer_id=layer_id)

    def forward(self, input_ids, x):
        if self.engram is not None:
            x = x + self.engram(self.norm1(x), input_ids)
        x = x + self.attn(self.norm2(x), use_cache=False)
        x = x + self.ff(self.norm3(x))
        return x


class NaiveEngramsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embed = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.ModuleList(
            [NaiveTransformerBlock(config, layer_id=i) for i in range(config["n_layers"])]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, input_ids, use_cache=False, position_offset=0, engram_input_ids=None):
        if use_cache:
            raise NotImplementedError("Naive model intentionally does not support KV cache")
        _, seq_len = input_ids.shape
        pos_ids = torch.arange(
            start=position_offset,
            end=position_offset + seq_len,
            device=input_ids.device,
            dtype=torch.long,
        )
        x = self.drop_emb(self.token_embed(input_ids) + self.pos_embed(pos_ids))
        if engram_input_ids is None:
            engram_input_ids = input_ids
        for block in self.transformer_blocks:
            x = block(engram_input_ids, x)
        x = self.final_norm(x)
        return self.out_head(x)

    def reset_cache(self):
        for block in self.transformer_blocks:
            block.attn.reset_cache()


def generate_text_naive(model, input_ids, max_new_tokens, context_size=None):
    model.eval()
    context_len = context_size or model.config["context_length"]
    batch_size, base_len = input_ids.shape
    total_len = base_len + max_new_tokens
    current_len = base_len

    out = torch.empty(batch_size, total_len, dtype=input_ids.dtype, device=input_ids.device)
    out[:, :base_len] = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_start = max(0, current_len - context_len)
            window = out[:, context_start:current_len]
            logits = model(
                window,
                use_cache=False,
                position_offset=context_start,
                engram_input_ids=window,
            )
            next_idx = logits[:, -1].argmax(dim=-1)
            out[:, current_len] = next_idx
            current_len += 1

    return out
