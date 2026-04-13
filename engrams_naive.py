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
    build_execution_stages,
    engram_cfg,
    move_tensor_to_device,
    normalize_device_map,
)


class NaiveManifoldConstrainedHyperConnection(nn.Module):
    """
    Straightforward mHC implementation used to keep the naive path
    semantically aligned with the optimized model.
    """

    def __init__(self, width, model_dim, sinkhorn_iters=4):
        super().__init__()
        self.width = width
        self.model_dim = model_dim
        self.sinkhorn_iters = sinkhorn_iters

        if self.width > 1:
            self.router = nn.Linear(model_dim, width * width + 2 * width, bias=True)
            diag_bias = torch.full((width, width), -2.0)
            diag_bias.fill_diagonal_(2.0)
            self.register_buffer("residual_bias", diag_bias)
        else:
            self.router = None
            self.register_buffer("residual_bias", torch.ones(1, 1))

    def _sinkhorn_project(self, logits):
        weights = torch.exp(logits)
        for _ in range(self.sinkhorn_iters):
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            weights = weights / weights.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        return weights

    def forward(self, x, op):
        squeeze_output = False
        if x.dim() == 3:
            x = x.unsqueeze(2)
            squeeze_output = True

        if self.width == 1:
            aggregated = x[:, :, 0, :]
            updated = x + op(aggregated).unsqueeze(2)
            return updated.squeeze(2) if squeeze_output else updated

        routing_input = x.mean(dim=2)
        raw = self.router(routing_input)
        pre_logits = raw[..., :self.width]
        post_logits = raw[..., self.width:2 * self.width]
        res_logits = raw[..., 2 * self.width:].view(*x.shape[:2], self.width, self.width)
        res_logits = res_logits + self.residual_bias.to(device=x.device, dtype=x.dtype)

        pre = torch.softmax(pre_logits, dim=-1)
        post = torch.softmax(post_logits, dim=-1)
        residual = self._sinkhorn_project(res_logits)

        aggregated = (x * pre.unsqueeze(-1)).sum(dim=2)
        op_out = op(aggregated)
        mixed_residual = torch.einsum("btij,btjd->btid", residual, x)
        updated = mixed_residual + post.unsqueeze(-1) * op_out.unsqueeze(2)

        return updated.squeeze(2) if squeeze_output else updated


class NaiveEngram(nn.Module):
    """
    Straightforward Engram block.

    Differences from the optimized path:
    - recomputes hashes every forward pass
    - does not precompute hashes across layers
    - does not precompute hashes across layers
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
        self.hc_mult = config["hc_mult"]
        self.attn = MultiHeadAttention(config)
        if config["num_experts"] > 0:
            self.ff = MoEFeedForward(config)
        else:
            self.ff = DenseFeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.norm3 = LayerNorm(config["emb_dim"])
        self.hc_attn = NaiveManifoldConstrainedHyperConnection(config["hc_mult"], config["emb_dim"])
        self.hc_ff = NaiveManifoldConstrainedHyperConnection(config["hc_mult"], config["emb_dim"])
        self.engram = None
        if layer_id in config["layer_ids"]:
            self.engram = NaiveEngram(config=config, layer_id=layer_id)
            self.hc_engram = NaiveManifoldConstrainedHyperConnection(config["hc_mult"], config["emb_dim"])
        else:
            self.hc_engram = None

    def forward(self, input_ids, x):
        if self.engram is not None:
            x = self.hc_engram(x, lambda agg: self.engram(self.norm1(agg), input_ids))
        x = self.hc_attn(x, lambda agg: self.attn(self.norm2(agg), use_cache=False))
        x = self.hc_ff(x, lambda agg: self.ff(self.norm3(agg)))
        return x


class NaiveEngramsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_device_map = normalize_device_map(
            config.get("device_map"),
            config["n_layers"],
            layer_ids=config.get("layer_ids"),
            hc_mult=config.get("hc_mult", 1),
        )
        self.execution_stages = build_execution_stages(
            self.block_device_map,
            engram_layer_ids=config.get("layer_ids"),
        )
        self.input_device = torch.device(self.block_device_map[0]) if self.block_device_map else None
        self.output_device = torch.device(self.block_device_map[-1]) if self.block_device_map else None
        self.token_embed = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embed = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.ModuleList(
            [NaiveTransformerBlock(config, layer_id=i) for i in range(config["n_layers"])]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        if self.block_device_map:
            self.apply_device_map()

    def apply_device_map(self, dtype=None):
        if not self.block_device_map:
            return self

        embed_device = torch.device(self.block_device_map[0])
        head_device = torch.device(self.block_device_map[-1])
        self.token_embed.to(device=embed_device, dtype=dtype)
        self.pos_embed.to(device=embed_device, dtype=dtype)
        self.drop_emb.to(device=embed_device)
        for block, device_str in zip(self.transformer_blocks, self.block_device_map):
            block.to(device=torch.device(device_str), dtype=dtype)
        self.final_norm.to(device=head_device, dtype=dtype)
        self.out_head.to(device=head_device, dtype=dtype)
        self.input_device = embed_device
        self.output_device = head_device
        return self

    def forward(self, input_ids, use_cache=False, position_offset=0, engram_input_ids=None):
        if use_cache:
            raise NotImplementedError("Naive model intentionally does not support KV cache")
        if self.block_device_map:
            input_ids = move_tensor_to_device(input_ids, self.input_device)
            input_device = self.input_device
        else:
            input_device = input_ids.device
        _, seq_len = input_ids.shape
        pos_ids = torch.arange(
            start=position_offset,
            end=position_offset + seq_len,
            device=input_device,
            dtype=torch.long,
        )
        x = self.drop_emb(self.token_embed(input_ids) + self.pos_embed(pos_ids))
        if self.config["hc_mult"] > 1:
            x = x.unsqueeze(2).expand(-1, -1, self.config["hc_mult"], -1).contiguous()
        if engram_input_ids is None:
            engram_input_ids = input_ids
        if self.execution_stages:
            stage_local_inputs = {}
            if torch.is_tensor(engram_input_ids):
                for stage in self.execution_stages:
                    if stage["has_engram"]:
                        stage_local_inputs[stage["device"]] = move_tensor_to_device(
                            engram_input_ids,
                            stage["device"],
                        )

            for stage in self.execution_stages:
                stage_device = torch.device(stage["device"])
                if x.device != stage_device:
                    x = move_tensor_to_device(x, stage_device)
                stage_input_ids = stage_local_inputs.get(stage["device"], engram_input_ids)

                for idx in range(stage["start"], stage["end"]):
                    block = self.transformer_blocks[idx]
                    x = block(stage_input_ids if block.engram is not None else None, x)
        else:
            for block in self.transformer_blocks:
                x = block(engram_input_ids if block.engram is not None else None, x)
        if x.dim() == 4:
            x = x.mean(dim=2)
        if self.block_device_map and x.device != self.output_device:
            x = move_tensor_to_device(x, self.output_device)
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

    with torch.inference_mode():
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
