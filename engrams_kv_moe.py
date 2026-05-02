"""
Engrams components follow by implementation by DeepSeek

Uses the Engrams Architecture with MHA and MoE

Pre-norm residual units (to be replaced by mHC in later version)

"""

from typing import List, Optional
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
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*2, 129280*2])
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


def estimate_block_weights(n_layers, layer_ids=None, hc_mult=1):
    engram_layers = set(layer_ids or [])
    weights = []
    for layer_idx in range(n_layers):
        weight = 1.0
        if layer_idx in engram_layers:
            weight += 1.5
        if layer_idx == 0 or layer_idx == n_layers - 1:
            weight += 0.25
        if hc_mult > 1:
            weight += 0.10 * (hc_mult - 1)
        weights.append(weight)
    return weights


def weighted_contiguous_partition(weights, num_buckets):
    if num_buckets <= 0:
        return []
    total_weight = sum(weights)
    if total_weight <= 0:
        return [(idx * num_buckets) // max(len(weights), 1) for idx in range(len(weights))]

    assignments = []
    prefix = 0.0
    for weight in weights:
        midpoint = prefix + 0.5 * weight
        bucket = min(int((midpoint * num_buckets) / total_weight), num_buckets - 1)
        assignments.append(bucket)
        prefix += weight
    if assignments:
        assignments[0] = 0
        assignments[-1] = num_buckets - 1
    for idx in range(1, len(assignments)):
        if assignments[idx] < assignments[idx - 1]:
            assignments[idx] = assignments[idx - 1]
    for idx in range(len(assignments) - 2, -1, -1):
        max_allowed = assignments[idx + 1]
        if assignments[idx] > max_allowed:
            assignments[idx] = max_allowed
    return assignments


def normalize_device_map(
    device_map,
    n_layers,
    layer_ids: Optional[List[int]] = None,
    hc_mult: int = 1,
):
    """ Given the list of available GPUs, parse among available devices """

    if not device_map:
        return None
    devices = [str(device) for device in device_map]
    if len(devices) == n_layers:
        return devices
    if len(devices) > n_layers:
        return devices[:n_layers]

    weights = estimate_block_weights(n_layers, layer_ids=layer_ids, hc_mult=hc_mult)
    buckets = weighted_contiguous_partition(weights, len(devices))
    return [devices[bucket] for bucket in buckets]


def build_execution_stages(block_device_map, engram_layer_ids=None):
    if not block_device_map:
        return []

    engram_layer_ids = set(engram_layer_ids or [])
    stages = []
    start = 0
    current_device = block_device_map[0]
    stage_has_engram = 0 in engram_layer_ids

    for idx in range(1, len(block_device_map)):
        device_str = block_device_map[idx]
        if device_str != current_device:
            stages.append({
                "device": current_device,
                "start": start,
                "end": idx,
                "has_engram": stage_has_engram,
            })
            start = idx
            current_device = device_str
            stage_has_engram = idx in engram_layer_ids
        else:
            stage_has_engram = stage_has_engram or idx in engram_layer_ids

    stages.append({
        "device": current_device,
        "start": start,
        "end": len(block_device_map),
        "has_engram": stage_has_engram,
    })
    return stages

def move_tensor_to_device(tensor, device):
    return tensor.to(device, non_blocking=True)

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
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.history_len = (kernel_size - 1) * dilation
        
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

        self.register_buffer("cache_x_norm", None, persistent=False)

    def _ensure_history_capacity(
        self,
        batch_size: int,
        groups: int,
        channels: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cache_row_indices: torch.Tensor | None = None,
    ) -> None:
        if self.history_len <= 0:
            return
        required_rows = batch_size
        if cache_row_indices is not None and cache_row_indices.numel() > 0:
            required_rows = max(required_rows, int(cache_row_indices.max().item()) + 1)
        needs_realloc = (
            self.cache_x_norm is None
            or self.cache_x_norm.shape[0] < required_rows
            or self.cache_x_norm.shape[1] != self.history_len
            or self.cache_x_norm.shape[2] != groups
            or self.cache_x_norm.shape[3] != channels
            or self.cache_x_norm.device != device
            or self.cache_x_norm.dtype != dtype
        )
        if not needs_realloc:
            return
        new_cache = torch.zeros(
            required_rows,
            self.history_len,
            groups,
            channels,
            device=device,
            dtype=dtype,
        )
        if self.cache_x_norm is not None:
            rows = min(self.cache_x_norm.shape[0], new_cache.shape[0])
            hist = min(self.cache_x_norm.shape[1], new_cache.shape[1])
            new_cache[:rows, -hist:] = self.cache_x_norm[:rows, -hist:].to(device=device, dtype=dtype)
        self.cache_x_norm = new_cache

    def _update_history(self, x_norm: torch.Tensor, cache_row_indices: torch.Tensor | None = None) -> None:
        if self.history_len <= 0:
            return
        B, _, G, C = x_norm.shape
        history = x_norm[:, -self.history_len :, :, :].detach().clone()
        if history.shape[1] < self.history_len:
            pad = torch.zeros(
                B,
                self.history_len - history.shape[1],
                G,
                C,
                device=x_norm.device,
                dtype=x_norm.dtype,
            )
            history = torch.cat([pad, history], dim=1)
        if cache_row_indices is None:
            self.cache_x_norm = history
            return
        cache_row_indices = cache_row_indices.to(device=x_norm.device, dtype=torch.long)
        self._ensure_history_capacity(
            B,
            G,
            C,
            device=x_norm.device,
            dtype=x_norm.dtype,
            cache_row_indices=cache_row_indices,
        )
        self.cache_x_norm[cache_row_indices] = history

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        B, T, G, C = x.shape
        if G == 1:
            return self.norms[0](x[:, :, 0, :]).unsqueeze(2)

        normed_chunks = []
        for i in range(G):
            normed_chunks.append(self.norms[i](x[:, :, i, :]))
        return torch.stack(normed_chunks, dim=2)

    def forward(self, x: torch.Tensor, cache_row_indices: torch.Tensor | None = None) -> torch.Tensor:
        """
        Input:  (B,L,HC_MULT,D)
        Output: (B,L,HC_MULT,D)
        """
        B, T, G, C = x.shape
        
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        x_norm = self._normalize(x)

        if G == 1:
            chunk = x_norm[:, :, 0, :]
            y_bct = self.conv(chunk.transpose(1, 2))[..., :T]
            if self.activation:
                y_bct = self.act_fn(y_bct)
            self._update_history(x_norm, cache_row_indices=cache_row_indices)
            return y_bct.transpose(1, 2).unsqueeze(2).contiguous()

        self._update_history(x_norm, cache_row_indices=cache_row_indices)

        x_norm = x_norm.reshape(B, T, G * C)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        
        return y

    def forward_step(self, x: torch.Tensor, cache_row_indices: torch.Tensor | None = None) -> torch.Tensor:
        """
        Exact cached-decode path for a single token using buffered normalized
        history instead of re-running the convolution over the full sequence.
        """
        B, T, G, C = x.shape
        if T != 1:
            raise ValueError("forward_step expects a single-token input")
        if G != self.hc_mult:
            raise ValueError(f"Input groups {G} != hc_mult {self.hc_mult}")

        x_norm = self._normalize(x)
        if self.history_len > 0:
            if cache_row_indices is None:
                self._ensure_history_capacity(B, G, C, device=x.device, dtype=x.dtype)
                history = self.cache_x_norm[:B]
            else:
                cache_row_indices = cache_row_indices.to(device=x.device, dtype=torch.long)
                self._ensure_history_capacity(
                    B,
                    G,
                    C,
                    device=x.device,
                    dtype=x.dtype,
                    cache_row_indices=cache_row_indices,
                )
                history = self.cache_x_norm.index_select(0, cache_row_indices)
            seq = torch.cat([history, x_norm], dim=1)
        else:
            seq = x_norm

        seq_flat = seq.reshape(B, seq.shape[1], G * C)
        y_bct = self.conv(seq_flat.transpose(1, 2))
        last_index = seq.shape[1] - 1
        y = y_bct[..., last_index:last_index + 1].transpose(1, 2)
        if self.activation:
            y = self.act_fn(y)

        if self.history_len > 0:
            history = seq[:, -self.history_len :, :, :].detach().clone()
            if cache_row_indices is None:
                self.cache_x_norm[:B] = history
            else:
                self.cache_x_norm[cache_row_indices] = history

        return y.view(B, T, G, C).contiguous()

    def reset_cache(self):
        self.cache_x_norm = None

    def reset_cache_rows(self, row_indices: torch.Tensor):
        if self.cache_x_norm is not None:
            indices = row_indices.to(device=self.cache_x_norm.device, dtype=torch.long)
            indices = indices[indices < self.cache_x_norm.shape[0]]
            if indices.numel() > 0:
                self.cache_x_norm[indices] = 0

    def compact_cache(self, active_indices: torch.Tensor):
        if self.cache_x_norm is not None:
            indices = active_indices.to(device=self.cache_x_norm.device, dtype=torch.long)
            self.cache_x_norm = self.cache_x_norm.index_select(0, indices)
    
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
        """ Returns a dictionary indexed by layer id with entries of dim (m-1, h) 
        \n The dictionary for each head has prime size starting with the vocab size and ascending
        \n Here m=max_ngram_size and h=n_head_per_engram
        """

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
        """ Load engrams params and lookup table to device if this has not occured yet (reduces unnecessary .to(device) calls) """

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

    def hash_last_tensor(self, input_ids: torch.Tensor):
        """
        Compute hashes only for the final token position of a short decode window.

        This is used during cached decode, where the Engram block consumes a
        single hidden-state step and only needs the hashes for the newest token.
        """

        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        consts = self._get_torch_constants(input_ids.device)
        compressed = consts["lookup_table"][input_ids]
        B, T = compressed.shape

        tail_tokens = []
        for k in range(self.max_ngram_size):
            if k < T:
                tail_tokens.append(compressed[:, T - 1 - k])
            else:
                tail_tokens.append(
                    torch.full(
                        (B,),
                        self.pad_id,
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                )

        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            multipliers = consts["layer_multipliers"][layer_id]
            all_hashes = []
            for n in range(2, self.max_ngram_size + 1):
                n_gram_index = n - 2
                mix = tail_tokens[0] * multipliers[0]
                for k in range(1, n):
                    mix = torch.bitwise_xor(mix, tail_tokens[k] * multipliers[k])

                head_vocab_sizes = consts["vocab_sizes"][layer_id][n_gram_index]
                for j in range(self.n_head_per_ngram):
                    all_hashes.append(torch.remainder(mix, head_vocab_sizes[j]))

            hash_ids_for_all_layers[layer_id] = torch.stack(all_hashes, dim=1).unsqueeze(1)

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
        # TODO: Possibly split this up if computation remains on GPUs
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
        """
        Params:
            x (tensor): input
            op (function): function performed on aggregated input
        """

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
    
    def forward(
        self,
        hidden_states,
        input_ids,
        precomputed_hashes=None,
        use_cache=False,
        cache_row_indices=None,
    ):
        """
        hidden_states: [B, L, D] - embedding of input sequence
        input_ids: [B, L] - tokenized input sequence
        precomputed_hashes [B, L, (M-1)*H] - hash of input sequence computed in advance
        """
        if hidden_states.dim() != 3:
            raise ValueError("Engram expects aggregated hidden states of shape [B, L, D]")

        device = hidden_states.device

        # Retrieve the hashing of all n-grams associated with the given layer (b, t, (m-1)*h)
        if precomputed_hashes is None:
            if torch.is_tensor(input_ids):
                if use_cache and hidden_states.shape[1] == 1:
                    hash_values = self.hash_mapping.hash_last_tensor(input_ids)[self.layer_id]
                else:
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
                short_conv_out = self.short_conv.forward_step(
                    short_conv_input,
                    cache_row_indices=cache_row_indices,
                )
            elif short_conv_mode == "gated_value_only":
                short_conv_out = torch.zeros_like(short_conv_input)
            else:
                raise ValueError(
                    "cached_inference_short_conv_mode must be one of "
                    "{'full', 'step_kernel', 'gated_value_only'}"
                )
        else:
            short_conv_out = self.short_conv(short_conv_input, cache_row_indices=cache_row_indices)

        output = short_conv_input + short_conv_out
        return output.squeeze(2)

    def reset_cache(self):
        if self.short_conv is not None:
            self.short_conv.reset_cache()

    def reset_cache_rows(self, row_indices: torch.Tensor):
        if self.short_conv is not None:
            self.short_conv.reset_cache_rows(row_indices)

    def compact_cache(self, active_indices: torch.Tensor):
        if self.short_conv is not None:
            self.short_conv.compact_cache(active_indices)
    
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
        self.register_buffer("cache_lengths", None, persistent=False)
        self.ptr_current_pos = 0
        self.cache_capacity = config["context_length"]

    def _ensure_cache_capacity(self, keys_new, values_new, required=None, required_rows=None):
        batch_size, num_tokens, _, _ = keys_new.shape
        if required is None:
            required = self.ptr_current_pos + num_tokens
        if required_rows is None:
            required_rows = batch_size
        capacity = max(self.cache_capacity, required)

        needs_realloc = (
            self.cache_k is None
            or self.cache_v is None
            or self.cache_k.shape[0] < required_rows
            or self.cache_k.shape[1] < required
            or self.cache_k.dtype != keys_new.dtype
            or self.cache_k.device != keys_new.device
        )
        if not needs_realloc:
            return

        new_k = torch.empty(
            required_rows,
            capacity,
            self.num_heads,
            self.head_dim,
            device=keys_new.device,
            dtype=keys_new.dtype,
        )
        new_v = torch.empty_like(new_k)

        if self.cache_k is not None:
            rows = min(self.cache_k.shape[0], new_k.shape[0])
            prefix = min(self.cache_k.shape[1], new_k.shape[1])
            new_k[:rows, :prefix] = self.cache_k[:rows, :prefix]
            new_v[:rows, :prefix] = self.cache_v[:rows, :prefix]

        self.cache_k = new_k
        self.cache_v = new_v

        if self.cache_lengths is None or self.cache_lengths.shape[0] < required_rows:
            new_lengths = torch.zeros(required_rows, device=keys_new.device, dtype=torch.long)
            if self.cache_lengths is not None:
                rows = min(self.cache_lengths.shape[0], new_lengths.shape[0])
                new_lengths[:rows] = self.cache_lengths[:rows].to(device=keys_new.device)
            self.cache_lengths = new_lengths
        else:
            self.cache_lengths = self.cache_lengths.to(device=keys_new.device)

    def _forward_per_row_cache(self, keys_new, values_new, queries, cache_positions, cache_row_indices=None):
        b, num_tokens, _, _ = keys_new.shape
        if cache_positions.shape != (b,):
            raise ValueError(f"cache_positions must have shape ({b},), got {tuple(cache_positions.shape)}")
        if (cache_positions < 0).any():
            raise ValueError("cache_positions must be non-negative")

        cache_positions = cache_positions.to(device=keys_new.device, dtype=torch.long)
        if cache_row_indices is None:
            cache_row_indices = torch.arange(b, device=keys_new.device, dtype=torch.long)
        else:
            cache_row_indices = cache_row_indices.to(device=keys_new.device, dtype=torch.long)
        if cache_row_indices.shape != (b,):
            raise ValueError(f"cache_row_indices must have shape ({b},), got {tuple(cache_row_indices.shape)}")
        if (cache_row_indices < 0).any():
            raise ValueError("cache_row_indices must be non-negative")

        row_ends = cache_positions + num_tokens
        required = int(row_ends.max().item()) if row_ends.numel() else num_tokens
        required_rows = int(cache_row_indices.max().item()) + 1 if cache_row_indices.numel() else b
        self._ensure_cache_capacity(keys_new, values_new, required=required, required_rows=required_rows)

        for row in range(b):
            start = int(cache_positions[row].item())
            end = start + num_tokens
            cache_row = int(cache_row_indices[row].item())
            self.cache_k[cache_row, start:end] = keys_new[row]
            self.cache_v[cache_row, start:end] = values_new[row]
            self.cache_lengths[cache_row] = max(int(self.cache_lengths[cache_row].item()), end)

        selected_lengths = self.cache_lengths.index_select(0, cache_row_indices)
        max_end = int(selected_lengths.max().item()) if selected_lengths.numel() else required
        keys = self.cache_k.index_select(0, cache_row_indices)[:, :max_end].transpose(1, 2)
        values = self.cache_v.index_select(0, cache_row_indices)[:, :max_end].transpose(1, 2)
        queries = queries.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        q_offsets = torch.arange(num_tokens, device=queries.device, dtype=torch.long)
        q_positions = cache_positions.unsqueeze(1) + q_offsets.unsqueeze(0)
        k_positions = torch.arange(max_end, device=queries.device, dtype=torch.long)
        row_lengths = selected_lengths.to(device=queries.device).unsqueeze(1)
        mask_bool = (k_positions.view(1, 1, max_end) > q_positions.unsqueeze(-1)) | (
            k_positions.view(1, 1, max_end) >= row_lengths.unsqueeze(-1)
        )
        attn_scores.masked_fill_(mask_bool.unsqueeze(1), -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.emb_dim)
        return self.out_proj(context_vec)

    def forward(self, x, use_cache=False, cache_positions=None, cache_row_indices=None):
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
            if cache_positions is not None:
                return self._forward_per_row_cache(
                    keys_new,
                    values_new,
                    queries,
                    cache_positions,
                    cache_row_indices=cache_row_indices,
                )
            self._ensure_cache_capacity(keys_new, values_new)
            start = self.ptr_current_pos
            end = start + num_tokens
            self.cache_k[:, start:end] = keys_new
            self.cache_v[:, start:end] = values_new
            keys = self.cache_k[:, :end]
            values = self.cache_v[:, :end]
            if self.cache_lengths is None or self.cache_lengths.shape[0] != b:
                self.cache_lengths = torch.full((b,), end, device=x.device, dtype=torch.long)
            else:
                self.cache_lengths.fill_(end)
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
        skip_causal_mask = False
        if use_cache:
            q_start = self.ptr_current_pos
            self.ptr_current_pos += num_tokens_Q
            skip_causal_mask = num_tokens_Q == 1 and q_start == num_tokens_K - 1
            if not skip_causal_mask:
                q_positions = torch.arange(
                    q_start,
                    q_start + num_tokens_Q,
                    device=device,
                    dtype=torch.long,
                )
        else:
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        if not skip_causal_mask:
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
        self.cache_k, self.cache_v, self.cache_lengths = None, None, None
        self.ptr_current_pos = 0

    def reset_cache_rows(self, row_indices: torch.Tensor):
        if self.cache_lengths is None:
            return
        indices = row_indices.to(device=self.cache_lengths.device, dtype=torch.long)
        indices = indices[indices < self.cache_lengths.shape[0]]
        if indices.numel() > 0:
            self.cache_lengths[indices] = 0

    def compact_cache(self, active_indices: torch.Tensor):
        if self.cache_k is not None:
            indices = active_indices.to(device=self.cache_k.device, dtype=torch.long)
            self.cache_k = self.cache_k.index_select(0, indices)
        if self.cache_v is not None:
            indices = active_indices.to(device=self.cache_v.device, dtype=torch.long)
            self.cache_v = self.cache_v.index_select(0, indices)
        if self.cache_lengths is not None:
            indices = active_indices.to(device=self.cache_lengths.device, dtype=torch.long)
            self.cache_lengths = self.cache_lengths.index_select(0, indices)

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
    
    def forward(
        self,
        input_ids,
        x,
        use_cache=False,
        engram_hashes=None,
        cache_positions=None,
        cache_row_indices=None,
    ):
        # Seperately handle case with trivial engrams expansion
        if self.hc_mult == 1:
            if x.dim() != 3:
                raise ValueError("Expected [B, L, D] hidden states when hc_mult == 1")

            if self.engram is not None:
                x = x + self.engram(
                    hidden_states=self.norm1(x),
                    input_ids=input_ids,
                    precomputed_hashes=engram_hashes,
                    use_cache=use_cache,
                    cache_row_indices=cache_row_indices,
                )

            x = x + self.attn(
                self.norm2(x),
                use_cache,
                cache_positions=cache_positions,
                cache_row_indices=cache_row_indices,
            )
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
                    cache_row_indices=cache_row_indices,
                ),
            )
        
        # Attention + Residual Connection
        x = self.hc_attn(
            x,
            lambda agg: self.attn(
                self.norm2(agg),
                use_cache,
                cache_positions=cache_positions,
                cache_row_indices=cache_row_indices,
            ),
        )

        # FFN + Residual Connection
        x = self.hc_ff(x, lambda agg: self.ff(self.norm3(agg)))

        return x

    def reset_cache(self):
        self.attn.reset_cache()
        if self.engram is not None:
            self.engram.reset_cache()

    def reset_cache_rows(self, row_indices: torch.Tensor):
        self.attn.reset_cache_rows(row_indices)
        if self.engram is not None:
            self.engram.reset_cache_rows(row_indices)

    def compact_cache(self, active_indices: torch.Tensor):
        self.attn.compact_cache(active_indices)
        if self.engram is not None:
            self.engram.compact_cache(active_indices)
    
class EngramsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.engram_layer_ids = set(config["layer_ids"])
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
            [TransformerBlock(config, layer_id=id) for id in range(config["n_layers"])]
        )
        self.num_engram_layers = sum(1 for block in self.transformer_blocks if block.engram is not None)
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

    def _prepare_engram_hashes(self, engram_input_ids, use_cache):
        """ Compute all engram hashes in advance """

        if self.num_engram_layers <= 1:
            return None

        hashes_by_device = {}
        if not torch.is_tensor(engram_input_ids):
            return None

        for block, device_str in zip(self.transformer_blocks, self.block_device_map or []):
            if block.engram is None or device_str in hashes_by_device:
                continue
            local_ids = move_tensor_to_device(engram_input_ids, device_str)
            if use_cache and local_ids.shape[1] <= block.engram.engram_cfg.max_ngram_size:
                hashes_by_device[device_str] = block.engram.hash_mapping.hash_last_tensor(local_ids)
            else:
                hashes_by_device[device_str] = block.engram.hash_mapping.hash_tensor(local_ids)
        return hashes_by_device if hashes_by_device else None

    
    def forward(
        self,
        input_ids,
        use_cache=False,
        position_offset=0,
        engram_input_ids=None,
        cache_positions=None,
        cache_row_indices=None,
    ):
        if self.block_device_map:
            input_device = self.input_device
            input_ids = move_tensor_to_device(input_ids, input_device)
        else:
            input_device = input_ids.device

        # Figure out positional embeddings based on shift
        _, seq_len = input_ids.shape
        if cache_positions is not None:
            cache_positions = cache_positions.to(device=input_device, dtype=torch.long)
            pos_offsets = torch.arange(seq_len, device=input_device, dtype=torch.long)
            pos_ids = cache_positions.unsqueeze(1) + pos_offsets.unsqueeze(0)
        else:
            pos_ids = torch.arange(
                start=position_offset,
                end=position_offset + seq_len,
                device=input_device,
                dtype=torch.long,
            )

        token_embeds = self.token_embed(input_ids)
        pos_embeds = self.pos_embed(pos_ids)
        x = token_embeds + pos_embeds
        x = self.drop_emb(x)

        if self.config["hc_mult"] > 1:
            x = x.unsqueeze(2).expand(-1, -1, self.config["hc_mult"], -1).contiguous()

        # Precompute engrams hashes if possible
        engram_hashes = None
        if engram_input_ids is None:
            engram_input_ids = input_ids
        if self.block_device_map:
            engram_hashes = self._prepare_engram_hashes(engram_input_ids, use_cache)
        elif self.num_engram_layers > 1:
            first_engram = next(block.engram for block in self.transformer_blocks if block.engram is not None)
            if torch.is_tensor(engram_input_ids):
                if use_cache and input_ids.shape[1] == 1:
                    engram_hashes = first_engram.hash_mapping.hash_last_tensor(engram_input_ids)
                else:
                    engram_hashes = first_engram.hash_mapping.hash_tensor(engram_input_ids)
            else:
                engram_hashes = first_engram.hash_mapping.hash(engram_input_ids)
        
        # Pass through all layers
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
                local_hashes = engram_hashes.get(stage["device"]) if isinstance(engram_hashes, dict) else None
                local_cache_positions = (
                    move_tensor_to_device(cache_positions, stage["device"])
                    if cache_positions is not None
                    else None
                )
                local_cache_row_indices = (
                    move_tensor_to_device(cache_row_indices, stage["device"])
                    if cache_row_indices is not None
                    else None
                )

                for idx in range(stage["start"], stage["end"]):
                    block = self.transformer_blocks[idx]
                    x = block(
                        input_ids=stage_input_ids if block.engram is not None else None,
                        x=x,
                        use_cache=use_cache,
                        engram_hashes=local_hashes,
                        cache_positions=local_cache_positions,
                        cache_row_indices=local_cache_row_indices,
                    )
        else:
            for block in self.transformer_blocks:
                x = block(
                    input_ids=engram_input_ids if block.engram is not None else None,
                    x=x,
                    use_cache=use_cache,
                    engram_hashes=engram_hashes,
                    cache_positions=cache_positions,
                    cache_row_indices=cache_row_indices,
                )

        if x.dim() == 4:
            x = x.mean(dim=2)

        if self.block_device_map and x.device != self.output_device:
            x = move_tensor_to_device(x, self.output_device)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

    def reset_cache(self):
        for block in self.transformer_blocks:
            block.reset_cache()

    def reset_cache_rows(self, row_indices: torch.Tensor):
        for block in self.transformer_blocks:
            block.reset_cache_rows(row_indices)

    def compact_cache(self, active_indices: torch.Tensor):
        for block in self.transformer_blocks:
            block.compact_cache(active_indices)

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

    with torch.inference_mode():
        if use_cache:

            # Prefill
            logits = model(
                input_ids,
                use_cache=True,
                position_offset=0,
                engram_input_ids=input_ids,
            )
            next_idx = logits[:, -1].argmax(dim=-1)
            out[:, current_len] = next_idx
            current_len += 1

            # Subsequent token generation
            for _ in range(1, max_new_tokens):
                step_input = out[:, current_len - 1:current_len]
                engram_start = max(0, current_len - model_engrams_cfg.max_ngram_size)
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
