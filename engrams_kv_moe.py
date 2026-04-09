"""
Engrams components follow by implementation by DeepSeek

Uses the Engrams Architecture with MHA and MoE

Pre-norm residual units (to be replaced by mHC in later version)

"""

from typing import List
from dataclasses import dataclass, field
import math
import argparse

from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex 

@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    use_short_conv: bool = True
    cached_inference_short_conv_mode: str = "step_kernel"
    
engram_cfg = EngramConfig()

class CompressedTokenizer:
    """
    Compressed tokenizer maps tokens with approx. semantically equivalent tokens to the same IDs
    to reduce overall alphabet size.
    (Conditional Memory via Scalable Lookup, Section 2.2, Tokenizer Compression)
    """

    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        """ Return the number of tokens in the compressed alphabet """
        return self.num_new_token
    
    def _build_lookup_table(self):
        """ Construct surjective map of initial token IDs to new token IDs where tokens that agree 
        up to normalization are mapped to the same IDs """

        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        """ Convert ids from initial alphabet to ids from compressed alphabet """

        arr = np.asarray(input_ids.cpu(), dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        return self._compress(input_ids)
            
class ShortConv(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps) 
            for _ in range(hc_mult)
        ])
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,L,HC_MULT,D)
        Output: (B,L,HC_MULT,D)
        """
        B, T, G, C = x.shape
        
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        if G == 1:
            chunk = self.norms[0](x[:, :, 0, :])
            y_bct = self.conv(chunk.transpose(1, 2))[..., :T]
            if self.activation:
                y_bct = self.act_fn(y_bct)
            return y_bct.transpose(1, 2).unsqueeze(2).contiguous()

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        
        return y

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exact fast path for a single decode-time token.

        When T == 1, the retained causal output only depends on the last
        convolution tap and the current token, so we can avoid launching the
        full depthwise convolution kernel.
        """
        B, T, G, C = x.shape
        if T != 1:
            raise ValueError("forward_step expects a single-token input")
        if G != self.hc_mult:
            raise ValueError(f"Input groups {G} != hc_mult {self.hc_mult}")

        if G == 1:
            chunk = self.norms[0](x[:, :, 0, :])
            tap = self.conv.weight[:, 0, -1].view(1, 1, C)
            y = chunk * tap
            if self.activation:
                y = self.act_fn(y)
            return y.unsqueeze(2).contiguous()

        normed_chunks = []
        for i in range(G):
            normed_chunks.append(self.norms[i](x[:, :, i, :]))

        x_norm = torch.cat(normed_chunks, dim=-1)
        tap = self.conv.weight[:, 0, -1].view(1, 1, G * C)
        y = x_norm * tap
        if self.activation:
            y = self.act_fn(y)
        return y.view(B, T, G, C).contiguous()
    
def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class NgramHashMapping:
    """ 
    NGramHashMapping maps tokens to latent embeddings using hash embeddings.

    Hash embeddings are computed by creating k tables of embeddings and k hash functions
    that map the input ids to corresponding embedded entries. Embeddings are computed as a weighted sum
    across the k embeddings in the table.

    (Hash Embeddings for Efficient Word Representations, Conditional Memory via Scalable Lookup)
    """

    def __init__(
        self, 
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,  
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        # Compute the random seeds for the hash functions for each layer
        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()
        self.lookup_table_torch = torch.from_numpy(self.compressed_tokenizer.lookup_table.copy()).long()
        self._torch_cache = {}

    def calculate_vocab_size_across_layers(self):
        """ Returns a dictionary indexed by layer id with entries of dim (m-1, h) """

        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        """
        Maps a list of input ids in a given alphabet to the hashes for the component n-grams.
        Output is in the form (batch, num_tokens, max_ngram_size - 1). Each hash index corresponds
        with a embedding table unique to each layer and head.

        See Conditional Memory via Scalable Lookup, Section 2.2, Multi-Head Hashing

        Return
            all_hashes (np.ndarray): (b, t, (m-1)*h), type=int

        (Where m is max_engrams_size, h is num_heads_for_this_engram)
        """

        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0: 
                return x
            
            # TODO: Review the choice of the constant non-zero padding value
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)] # (k, b, t)

        all_hashes = []
        
        # Seperately hash n-grams for all choices n
        for n in range(2, self.max_ngram_size + 1):

            # Compute mix, a XOR sum of labels of all tokens in n-gram starting
            # and each index multiplied by random weights
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0]) # (b, t)
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])

            # Determine number of heads for engram and size of vocab for each
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            # For each head compute the hash by modding mix by size of head alphabet
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False)) 
                
        # Note: all_hashes is ((m-1)*h, b, t), so out is (b, t, (m-1)*h)
        
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        """ 
        Takes sequence of ids of input tokens and returns the hash for all layers/n-grams/heads

        Return
            hash_ids (dict): (layers, b, t, (m-1)*h), dtype=int64

        (Where m is max_engrams_size, h is num_heads_for_this_engram)
        """

        # Re-encode input with compressed alphabet
        input_ids = self.compressed_tokenizer(input_ids)
        
        # Hash n-grams
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

    def _get_torch_constants(self, device):
        key = str(device)
        if key not in self._torch_cache:
            layer_multipliers = {
                layer_id: torch.tensor(multipliers, device=device, dtype=torch.long)
                for layer_id, multipliers in self.layer_multipliers.items()
            }
            vocab_sizes = {
                layer_id: [
                    torch.tensor(head_vocab_sizes, device=device, dtype=torch.long)
                    for head_vocab_sizes in self.vocab_size_across_layers[layer_id]
                ]
                for layer_id in self.layer_ids
            }
            self._torch_cache[key] = {
                "lookup_table": self.lookup_table_torch.to(device=device),
                "layer_multipliers": layer_multipliers,
                "vocab_sizes": vocab_sizes,
            }
        return self._torch_cache[key]

    def hash_tensor(self, input_ids: torch.Tensor):
        """
        Torch-native hash path used by the optimized implementation to avoid
        CPU/NumPy round-trips during GPU inference.
        """
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        consts = self._get_torch_constants(input_ids.device)
        compressed = consts["lookup_table"][input_ids]
        B, T = compressed.shape

        base_shifts = []
        for k in range(self.max_ngram_size):
            if k == 0:
                shifted = compressed
            else:
                shifted = F.pad(compressed, (k, 0), value=self.pad_id)[:, :T]
            base_shifts.append(shifted)

        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            multipliers = consts["layer_multipliers"][layer_id]
            all_hashes = []
            for n in range(2, self.max_ngram_size + 1):
                n_gram_index = n - 2
                mix = base_shifts[0] * multipliers[0]
                for k in range(1, n):
                    mix = torch.bitwise_xor(mix, base_shifts[k] * multipliers[k])

                head_vocab_sizes = consts["vocab_sizes"][layer_id][n_gram_index]
                for j in range(self.n_head_per_ngram):
                    all_hashes.append(torch.remainder(mix, head_vocab_sizes[j]))

            hash_ids_for_all_layers[layer_id] = torch.stack(all_hashes, dim=2)

        return hash_ids_for_all_layers

class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        
        # Compute the offset for each n-gram size, head combo when tables are all concatenated
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        # Represent the embeddings associated with all n-gram sizes, heads in one table
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Map the hash for each n-gram to the learned embedding.

        Params:
            input_ids (tensor): (b, t, (m-1)*h), dtype=int - a list of hashes for n-grams

        Return:
            output (tensor): (b, t, (m-1)*h, d) - a list of corresponding embedding vectors

        (Where b is batch size, t is the number of input tokens, m is max_engram_size, 
        d is engrams_emb_dim)
        """

        # Compute indices of embeddings corresponding with each hash (given all tables are concatenated)
        shifted_input_ids = input_ids + self.offsets
        
        # Index the embedding table
        output = self.embedding(shifted_input_ids)
        
        return output

class DenseFeedForward(nn.Module):
    """
    Dense gated MLP fallback used when MoE is disabled.
    Mirrors the expert MLP structure so the dense and sparse paths stay aligned.
    """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))

class ManifoldConstrainedHyperConnection(nn.Module):
    """
    Lightweight mHC-style residual wrapper.

    For each token position, compute:
        x_{l+1} = H_res x_l + H_post^T f(H_pre x_l)
    where H_res is projected onto the bistochastic manifold with Sinkhorn
    normalization and H_pre / H_post are simplex weights.
    """

    def __init__(self, width: int, model_dim: int, sinkhorn_iters: int = 4):
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

    def _sinkhorn_project(self, logits: torch.Tensor) -> torch.Tensor:
        weights = torch.exp(logits)
        for _ in range(self.sinkhorn_iters):
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            weights = weights / weights.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        return weights

    def _compute_mixing_weights(self, x: torch.Tensor):
        if self.width == 1:
            shape = x.shape[:2] + (1,)
            ones = torch.ones(shape, device=x.device, dtype=x.dtype)
            residual = torch.ones(x.shape[:2] + (1, 1), device=x.device, dtype=x.dtype)
            return ones, ones, residual

        routing_input = x.mean(dim=2)
        raw = self.router(routing_input)
        split = self.width
        pre_logits = raw[..., :split]
        post_logits = raw[..., split:2 * split]
        res_logits = raw[..., 2 * split:].view(*x.shape[:2], self.width, self.width)
        res_logits = res_logits + self.residual_bias.to(device=x.device, dtype=x.dtype)

        pre = torch.softmax(pre_logits, dim=-1)
        post = torch.softmax(post_logits, dim=-1)
        residual = self._sinkhorn_project(res_logits)
        return pre, post, residual

    def forward(self, x: torch.Tensor, op):
        squeeze_output = False
        if x.dim() == 3:
            x = x.unsqueeze(2)
            squeeze_output = True

        pre, post, residual = self._compute_mixing_weights(x)
        aggregated = (x * pre.unsqueeze(-1)).sum(dim=2)
        op_out = op(aggregated)
        mixed_residual = torch.einsum("btij,btjd->btid", residual, x)
        updated = mixed_residual + post.unsqueeze(-1) * op_out.unsqueeze(2)

        if squeeze_output:
            return updated.squeeze(2)
        return updated
    
class Engram(nn.Module):
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
        ) if self.engram_cfg.use_short_conv else None

        engram_hidden_size = (self.engram_cfg.max_ngram_size - 1) * self.engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, self.model_dim)
        self.key_proj = nn.Linear(engram_hidden_size, self.model_dim)
        self.key_norm = nn.RMSNorm(self.model_dim)
        self.query_norm = nn.RMSNorm(self.model_dim)
    
    def forward(self, hidden_states, input_ids, precomputed_hashes=None, use_cache=False):
        """
        hidden_states: [B, L, D]
        input_ids: [B, L]
        """
        if hidden_states.dim() != 3:
            raise ValueError("Engram expects aggregated hidden states of shape [B, L, D]")

        device = hidden_states.device

        # Retrieve the hashing of all n-grams associated with the given layer (b, t, (m-1)*h)
        if precomputed_hashes is None:
            if torch.is_tensor(input_ids):
                hash_values = self.hash_mapping.hash_tensor(input_ids)[self.layer_id]
            else:
                hash_values = self.hash_mapping.hash(input_ids)[self.layer_id]
        else:
            hash_values = precomputed_hashes[self.layer_id]

        if torch.is_tensor(hash_values):
            hash_input_ids = hash_values[:, -hidden_states.shape[1]:, :].to(device=device, dtype=torch.long)
        else:
            hash_input_ids = torch.from_numpy(hash_values[:, -hidden_states.shape[1]:, :]).to(device)

        # Retrieve embeddings corresponding with hashes (b, t, (m-1)*h, d)
        embeddings = self.multi_head_embedding(hash_input_ids)
        
        # Concatenate all embeddings for n-grams starting at each token (b, t, (m-1)*h*d)
        embeddings = embeddings.flatten(start_dim=-2)

        key = self.key_norm(self.key_proj(embeddings))
        query = self.query_norm(hidden_states)
        gate = (key * query).sum(dim=-1) / math.sqrt(self.model_dim)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid().unsqueeze(-1)

        value = gate * self.value_proj(embeddings)
        if self.short_conv is None:
            return value

        short_conv_input = value.unsqueeze(2)
        short_conv_mode = self.engram_cfg.cached_inference_short_conv_mode

        if use_cache and hidden_states.shape[1] == 1 and short_conv_mode != "full":
            if short_conv_mode == "step_kernel":
                short_conv_out = self.short_conv.forward_step(short_conv_input)
            elif short_conv_mode == "gated_value_only":
                short_conv_out = torch.zeros_like(short_conv_input)
            else:
                raise ValueError(
                    "cached_inference_short_conv_mode must be one of "
                    "{'full', 'step_kernel', 'gated_value_only'}"
                )
        else:
            short_conv_out = self.short_conv(short_conv_input)

        output = short_conv_input + short_conv_out
        return output.squeeze(2)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.emb_dim = config["emb_dim"]
        self.num_heads = config["n_heads"]
        self.qkv_bias = config["qkv_bias"]

        assert self.emb_dim % self.num_heads == 0, "d_out must be divisible by num_heads"

        self.head_dim = self.emb_dim // self.num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(self.emb_dim, self.emb_dim, bias=self.qkv_bias)
        self.W_key = nn.Linear(self.emb_dim, self.emb_dim, bias=self.qkv_bias)
        self.W_value = nn.Linear(self.emb_dim, self.emb_dim, bias=self.qkv_bias)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(config["drop_rate"])

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        b, num_tokens, _ = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # causal mask
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        if use_cache:
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q
        else:
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
        mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.emb_dim)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0

class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.num_experts = cfg["num_experts"]
        self.emb_dim = cfg["emb_dim"]

        if self.num_experts <= 0:
            raise ValueError("MoEFeedForward requires num_experts > 0")
        if self.num_experts_per_tok <= 0:
            raise ValueError("MoEFeedForward requires num_experts_per_tok > 0")

        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False)
        self.fc1 = nn.ModuleList(
            [
                nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
                for _ in range(self.num_experts)
            ]
        )
        self.fc2 = nn.ModuleList(
            [
                nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
                for _ in range(self.num_experts)
            ]
        )
        self.fc3 = nn.ModuleList(
            [
                nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x):
        """
        Params
            x: (batch, seq_len, emb_dim)
        """

        scores = self.gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1) # (b, seq_len, num_experts_per_tok)
        topk_probs = torch.softmax(topk_scores, dim=-1)
        
        batch, seq_len, _ = x.shape
        x_flat = x.reshape(batch * seq_len, self.emb_dim) 
        out_flat = torch.zeros(batch * seq_len, self.emb_dim, device=x.device, dtype=x.dtype)

        # Find all the different experts used on at least one token
        topk_indices_flat = topk_indices.reshape(batch * seq_len, self.num_experts_per_tok)
        topk_probs_flat = topk_probs.reshape(batch * seq_len, self.num_experts_per_tok)

        used_experts = torch.unique(topk_indices_flat)

        # Compute the output for each expert, masking the portion of the input to which it does not apply
        for expert in used_experts:
            expert_id = int(expert.item())

            # Determine if a given token uses an expert
            mask = topk_indices_flat == expert_id # (b*t, k)
            if not mask.any():
                continue
            token_mask = mask.any(dim=-1) # (b*t)

            # Get the indices of tokens using the expert
            select_idx = token_mask.nonzero(as_tuple=False).squeeze(-1) # [b*t]

            # Compute the forward pass for the tokens using the expert
            expert_in = x_flat.index_select(0, select_idx)
            expert_hidden = F.silu(self.fc1[expert_id](expert_in)) * self.fc2[expert_id](expert_in)
            expert_out = self.fc3[expert_id](expert_hidden)

            # Determine probability assigned to each head
            mask_selected = mask[select_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(
                topk_probs_flat.index_select(0, select_idx), dim=-1, index=slot_indices
            )

            # Compute output as a weighted sum of outputs by probabilities
            out_flat.index_add_(0, select_idx, expert_out * selected_probs)

        return out_flat.reshape(batch, seq_len, self.emb_dim)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class TransformerBlock(nn.Module):
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
        self.hc_attn = ManifoldConstrainedHyperConnection(config["hc_mult"], config["emb_dim"])
        self.hc_ff = ManifoldConstrainedHyperConnection(config["hc_mult"], config["emb_dim"])
        self.engram = None
        if layer_id in config["layer_ids"]:
            self.engram = Engram(config=config, layer_id=layer_id)
            self.hc_engram = ManifoldConstrainedHyperConnection(config["hc_mult"], config["emb_dim"])
        else:
            self.hc_engram = None
    
    def forward(self, input_ids, x, use_cache=False, engram_hashes=None):
        if self.hc_mult == 1:
            if x.dim() != 3:
                raise ValueError("Expected [B, L, D] hidden states when hc_mult == 1")

            if self.engram is not None:
                x = x + self.engram(
                    hidden_states=self.norm1(x),
                    input_ids=input_ids,
                    precomputed_hashes=engram_hashes,
                    use_cache=use_cache,
                )

            x = x + self.attn(self.norm2(x), use_cache)
            x = x + self.ff(self.norm3(x))
            return x

        # (Engram Layer Only) Engram + Residual Connection
        if self.engram is not None:
            x = self.hc_engram(
                x,
                lambda agg: self.engram(
                    hidden_states=self.norm1(agg),
                    input_ids=input_ids,
                    precomputed_hashes=engram_hashes,
                    use_cache=use_cache,
                ),
            )
        
        # Attention + Residual Connection
        x = self.hc_attn(x, lambda agg: self.attn(self.norm2(agg), use_cache))

        # FFN + Residual Connection
        x = self.hc_ff(x, lambda agg: self.ff(self.norm3(agg)))

        return x

    def reset_cache(self):
        self.attn.reset_cache()
    
class EngramsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.engram_layer_ids = set(config["layer_ids"])

        self.token_embed = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embed = nn.Embedding(config["context_length"], config["emb_dim"]) 
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config, layer_id=id) for id in range(config["n_layers"])]
        )
        self.num_engram_layers = sum(1 for block in self.transformer_blocks if block.engram is not None)
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    
    def forward(self, input_ids, use_cache=False, position_offset=0, engram_input_ids=None):

        # Figure out positional embeddings based on shift
        _, seq_len = input_ids.shape
        pos_ids = torch.arange(
            start=position_offset,
            end=position_offset + seq_len,
            device=input_ids.device,
            dtype=torch.long,
        )

        token_embeds = self.token_embed(input_ids)
        pos_embeds = self.pos_embed(pos_ids)
        x = token_embeds + pos_embeds
        x = self.drop_emb(x)

        if self.config["hc_mult"] > 1:
            x = x.unsqueeze(2).expand(-1, -1, self.config["hc_mult"], -1).contiguous()

        engram_hashes = None
        if engram_input_ids is None:
            engram_input_ids = input_ids
        if self.num_engram_layers > 1:
            first_engram = next(block.engram for block in self.transformer_blocks if block.engram is not None)
            if torch.is_tensor(engram_input_ids):
                engram_hashes = first_engram.hash_mapping.hash_tensor(engram_input_ids)
            else:
                engram_hashes = first_engram.hash_mapping.hash(engram_input_ids)
        
        for block in self.transformer_blocks:
            x = block(
                input_ids=engram_input_ids,
                x=x,
                use_cache=use_cache,
                engram_hashes=engram_hashes,
            )

        if x.dim() == 4:
            x = x.mean(dim=2)

        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

    def reset_cache(self):
        for block in self.transformer_blocks:
            block.reset_cache()

def generate_text(model, input_ids, max_new_tokens, context_size=None, use_cache=True):
    """
    Params:
        model (EngramsModel)
        idx_ids (b, l) - list of ids of the tokens in the idx for each batch
        max_new_tokens (int)
        context_size (int)
        use_cache (bool)
    """

    # TODO: look into CUDA synchronization

    model.eval()
    model.reset_cache()
    model_engrams_cfg = getattr(model, "config", {}).get("engrams_cfg", engram_cfg)

    context_len = context_size or 256
    batch_size, base_len = input_ids.shape
    total_len = base_len + max_new_tokens
    current_len = base_len

    # Create tensor to hold entire output, populate portion associated with idx
    out = torch.empty(batch_size, total_len, dtype=input_ids.dtype, device=input_ids.device)
    out[:, :base_len] = input_ids

    with torch.no_grad():
        if use_cache:
            logits = model(
                input_ids,
                use_cache=True,
                position_offset=0,
                engram_input_ids=input_ids,
            )
            next_idx = logits[:, -1].argmax(dim=-1)
            out[:, current_len] = next_idx
            current_len += 1

            for _ in range(1, max_new_tokens):
                step_input = out[:, current_len - 1:current_len]
                engram_start = max(0, current_len - model_engrams_cfg.max_ngram_size + 1)
                engram_input_ids = out[:, engram_start:current_len]
                logits = model(
                    step_input,
                    use_cache=True,
                    position_offset=current_len - 1,
                    engram_input_ids=engram_input_ids,
                )
                next_idx = logits[:, -1].argmax(dim=-1)
                out[:, current_len] = next_idx
                current_len += 1
        else:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=768*4, help="Intermediate FFN or MoE size.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Disable KV caching during generation.",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=0,
        help="Number of experts. If 0, use dense FFN. If >0, use MoE.",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=2,
        help="Number of routed experts per token when MoE is enabled.",
    )
    args = parser.parse_args()

    text = "Hello, I am"

    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path,
        trust_remote_code=True
    )

    input_ids = tokenizer(text,return_tensors='pt').input_ids

    print(f"input_ids: {input_ids.shape}")

    config = {
        "vocab_size": 129280,                                   # Vocabulary size (prev 50257)
        "context_length": args.max_new_tokens + len(input_ids), # Context length
        "emb_dim": args.emb_dim,                                # Embedding dimension
        "hidden_dim": args.hidden_dim,                          # Intermediate size
        "n_heads": args.n_heads,                                # Number of attention heads
        "n_layers": args.n_layers,                              # Number of layers
        "drop_rate": 0.0,                                       # Dropout rate
        "qkv_bias": False,                                      # Query-Key-Value bias
        "num_experts": args.num_experts,
        "num_experts_per_tok": args.num_experts_per_tok if args.num_experts > 0 else 0,
        "hc_mult": 1,                                           # Branching factor for HC (> 1 when HC is used)
        "layer_ids": [1],                                       # A list of which layers have engram block
        "engrams_cfg": engram_cfg
    }

    model = EngramsModel(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    model.to(device, dtype=torch.bfloat16)

    output_ids = generate_text(
        model=model, 
        input_ids=input_ids.to(device), 
        max_new_tokens=args.max_new_tokens,
        use_cache=False
    )

    decoded_output = tokenizer.decode(output_ids.squeeze(0).tolist())

    print("✅ Forward Complete!")
    print(decoded_output)


if __name__ == '__main__':
    main()
