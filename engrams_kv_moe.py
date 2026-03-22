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
    
class Engram(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()

        self.layer_id = layer_id
        self.hidden_dim = config["hidden_dim"]
        self.hc_mult = config["hc_mult"]

        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size = engram_cfg.max_ngram_size,
            n_embed_per_ngram = engram_cfg.n_embed_per_ngram,
            n_head_per_ngram = engram_cfg.n_head_per_ngram,
            layer_ids = engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id = engram_cfg.pad_id,
            seed = engram_cfg.seed,
        )

        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )

        self.short_conv = ShortConv(
            hidden_size = self.hidden_dim,
            kernel_size = engram_cfg.kernel_size,
            dilation    = engram_cfg.max_ngram_size,
            hc_mult     = self.hc_mult,
        )

        engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, self.hidden_dim)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, self.hidden_dim) for _ in range(self.hc_mult)]
        )

        self.norm1 = nn.ModuleList([nn.RMSNorm(self.hidden_dim) for _ in range(self.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(self.hidden_dim) for _ in range(self.hc_mult)])
    
    def forward(self, hidden_states, input_ids):
        """
        hidden_states: [B, L, HC_MULT, D]
        input_ids: [B, L]
        """

        # TODO: Currently hidden states is [B, L, D] form, this will cause some dimension issues
        if len(hidden_states.shape) == 4:
            pass
        elif len(hidden_states.shape) == 3:
            print("Kludge: adding extra dimensions since mHC not used")
            raise ValueError
        else:
            raise ValueError


        device = hidden_states.device

        # Retrieve the hashing of all n-grams associated with the given layer (b, t, (m-1)*h)
        # TODO: There is some recomputation here, seems quite inefficient
        hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id]).to(device)

        # Retrieve embeddings corresponding with hashes (b, t, (m-1)*h, d)
        embeddings = self.multi_head_embedding(hash_input_ids)
        
        # Concatenate all embeddings for n-grams starting at each token (b, t, (m-1)*h*d)
        embeddings = embeddings.flatten(start_dim=-2)

        gates = []
        for hc_idx in range(self.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:,:,hc_idx,:]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.hidden_dim)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates = torch.stack(gates,dim=2)
        value = gates * self.value_proj(embeddings).unsqueeze(2)
        output = value + self.short_conv(value)

        return output 
    
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
            select_idx = token_mask.non_zero(as_tuple=False).squeeze(-1) # [b*t]

            # Compute the forward pass for the tokens using the expert
            expert_in = x_flat.index_select(0, select_idx)

            expert_out = self.fc1[expert_id](expert_in) * self.fc2[expert_id](expert_in)
            expert_out = torch.nn.SilU(expert_out) # SiLU used for MoE
            expert_out = self.fc3[expert_id](expert_out)

            # Determine probability assigned to each head
            mask_selected = mask[select_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(
                topk_probs_flat.index_select(0, select_idx), dim=-1, index=slot_indices
            )

            # Compute output as a weighted sum of outputs by probabilities
            out_flat.index_add_(0, select_idx, expert_out * selected_probs.unsqueeze(-1))

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
        self.attn = MultiHeadAttention(config)
        self.moe = MoEFeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.norm3 = LayerNorm(config["emb_dim"])
        self.engram = None
        if layer_id in config["layer_ids"]:
            self.engram = Engram(config=config, layer_id=layer_id)
    
    def forward(self, input_ids, x, use_cache=False):

        # Note that all residual units are pre-norm

        if use_cache:
            raise NotImplementedError

        # (Engram Layer Only) Engram + Residual Connection
        if self.engram is not None:
            shortcut = x
            x = self.norm1(x)
            x = self.engram(hidden_states=x, input_ids=input_ids)
            x = x + shortcut
        
        # Attention + Residual Connection
        shortcut = x
        x = self.norm2(x)
        x = self.attn(x, use_cache) 
        x = x + shortcut

        # FFN + Residual Connection
        shortcut = x
        x = self.norm3(x)
        x = self.moe(x)
        x = x + shortcut

        return x
    
class EngramsModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embed = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embed = nn.Embedding(config["context_length"], config["emb_dim"]) 
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config, layer_id=id) for id in range(config["n_layers"])]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

        assert config["hc_mult"] == 1, "Hyper-Connections not yet supported"

    
    def forward(self, input_ids, use_cache=False):

        # Figure out positional embeddings based on shift
        _, seq_len = input_ids.shape
        if use_cache:
            raise NotImplementedError
        else:
            pos_ids = torch.arange(start=0, end=seq_len, device=input_ids.device, dtype=torch.long)

        token_embeds = self.token_embed(input_ids)
        pos_embeds = self.pos_embed(pos_ids)
        x = token_embeds + pos_embeds
        x = self.drop_emb(x)
        
        for block in self.transformer_blocks:
            x = block(input_ids=input_ids, x=x, use_cache=use_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

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

    context_len = context_size or 256
    batch_size, base_len = input_ids.shape
    total_len = base_len + max_new_tokens
    current_len = base_len

    # Create tensor to hold entire output, populate portion associated with idx
    out = torch.empty(batch_size, total_len, dtype=input_ids.dtype, device=input_ids.device)
    out[:, :base_len] = input_ids

    with torch.no_grad():
        if use_cache:
            # TODO: Implement
            raise NotImplementedError
        else:
            for _ in range(max_new_tokens):
                context_start = max(0, current_len - context_len)
                logits = model(input_ids[:, context_start:current_len], use_cache=False)
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