"""
POP (Parallel Observation Prediction) mechanism with standard Attention.

Positional encoding modes (selectable at construction time):
  pos_enc='sinusoidal'  — classic additive sinusoidal embeddings (default)
  pos_enc='rotary'      — RoPE via rotary-embedding-torch (lucidrains)
                          pip install rotary-embedding-torch

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEQUENCE LAYOUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Original sequence (length k*(n+m)):
  [obs_1 | act_1 | obs_2 | act_2 | ... | obs_k | act_k]
   n toks  m toks  n toks  m toks        n toks  m toks

Prediction tokens (k copies of a fixed n-token template P):
  [P_1 | P_2 | ... | P_k]
   n     n         n

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal: for each block i (1-indexed), predict obs_i from everything BEFORE it.

Two-pass efficient implementation:
  1. Context pass   : encode the full original sequence with a causal mask.
                      Store the K and V *projections* of each layer — these
                      are reused directly by the prediction pass so they are
                      never recomputed.
  2. Prediction pass: run k*n prediction tokens, at each layer attending to
                      the cached (K, V) of the context plus their own (K, V)
                      (causal within each copy, isolated across copies).

Attention mask for the prediction pass  [k*n  ×  k*(n+m) + k*n]
  Columns 0 .. k*(n+m)-1   → context (K, V cache)
  Columns k*(n+m) .. end   → prediction tokens (self)
  Copy i, token j:
    Context cols : attend to 0 .. i*(n+m)-1
    Self cols    : attend to copy-i tokens 0 .. j  (lower-triangular block)
    Other copies : never

Positional encoding for prediction tokens:
  Copy i, slot j  →  absolute position i*(n+m) + j  (same as obs_i).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KVCache stores per-layer (K, V) projections and grows incrementally.
Previously processed tokens are never recomputed.

  cache  = model.init_cache(device)           # empty, or pre-fill a context
  logits = model.predict_next(cache)          # predict obs_1 (no context yet)
  obs_1  = sample(logits)                     # your sampling logic here
  act_1  = your_policy(obs_1)
  cache  = model.append_block(cache, obs_1, act_1)

  logits = model.predict_next(cache)          # predict obs_2 from block 1
  obs_2  = sample(logits)
  ...
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, List

# Attention pattern among prediction tokens within the same copy:
#   'causal' — lower-triangular (token j sees 0..j)
#   'full'   — all-to-all (every token sees all n tokens in the copy)
PredSelfMask = Literal["causal", "full"]

import torch
import torch.nn as nn
import torch.nn.functional as F


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mask and position helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_pop_mask(
    k: int,
    n: int,
    m: int,
    device: torch.device,
    pred_self_mask: PredSelfMask = "causal",
) -> torch.Tensor:
    """
    Training-time POP attention mask.

    Returns BoolTensor [k*n, k*(n+m) + k*n]  (True = attend).

    Column regions
    --------------
    [0 .. k*(n+m)-1]   context K/V cache
    [k*(n+m) .. end]   prediction tokens (self)

    pred_self_mask controls the within-copy self-attention pattern:
      'causal' — token j attends to tokens 0..j (lower-triangular)
      'full'   — every token attends to all n tokens in the same copy
    In both cases the context columns are always strictly causal (prefix
    before obs_i) and cross-copy attention is always blocked.
    """
    pred_len = k * n
    orig_len = k * (n + m)
    total_kv = orig_len + pred_len

    if pred_self_mask == "causal":
        self_block = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
    else:  # "full"
        self_block = torch.ones(n, n, dtype=torch.bool, device=device)

    mask = torch.zeros(pred_len, total_kv, dtype=torch.bool, device=device)
    for i in range(k):
        row_s = i * n
        row_e = row_s + n

        # (a) Context prefix: everything strictly before obs_i
        col_e = i * (n + m)
        if col_e > 0:
            mask[row_s:row_e, :col_e] = True

        # (b) Within-copy self-attention
        self_col_s = orig_len + i * n
        mask[row_s:row_e, self_col_s : self_col_s + n] = self_block

    return mask


def build_prediction_positions(k: int, n: int, m: int, device: torch.device) -> torch.Tensor:
    """
    Absolute position index for each of the k*n prediction tokens.
    Copy i, slot j  →  i*(n+m) + j   (same as obs_i in the original sequence).
    Returns LongTensor [k*n].
    """
    return torch.cat([
        torch.arange(i * (n + m), i * (n + m) + n, device=device)
        for i in range(k)
    ])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Inference KV cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class LayerKVCache:
    """
    K and V projections for one transformer layer.

    k, v : [B, H, T, Dh]   T grows as blocks are appended.
    """
    k: torch.Tensor
    v: torch.Tensor

    @property
    def seq_len(self) -> int:
        return self.k.shape[2]

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor) -> "LayerKVCache":
        """Return a new LayerKVCache with new_k/new_v appended along T."""
        return LayerKVCache(
            k=torch.cat([self.k, new_k], dim=2),
            v=torch.cat([self.v, new_v], dim=2),
        )


@dataclass
class KVCache:
    """
    Per-layer KV cache for the full model.

    layers  : list of LayerKVCache, one per transformer layer
    n, m    : tokens per obs / act block

    n_blocks_committed is intentionally absent — it is always inferrable
    as context_len // (n + m) and keeping it as explicit state would risk
    it going stale if a KVCache is constructed manually.
    """
    layers: List[LayerKVCache]
    n: int
    m: int

    @property
    def context_len(self) -> int:
        return self.layers[0].seq_len if self.layers else 0

    @property
    def n_blocks_committed(self) -> int:
        return self.context_len // (self.n + self.m)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Positional encodings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe[positions]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multi-Head Attention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention using torch.nn.functional.scaled_dot_product_attention
    for a fused, memory-efficient kernel (FlashAttention when available).

    Supports both sinusoidal (additive, handled outside) and RoPE (applied
    to Q/K after projection, before SDPA).  RoPE frequencies must be
    pre-computed by the caller (once per forward call) and passed as
    q_freqs / k_freqs — this avoids redundant recomputation across layers.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model
        self.dropout = dropout          # scalar passed to SDPA (not a module)

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] → [B, H, T, Dh]"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, T, Dh] → [B, T, D]"""
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).reshape(B, T, H * Dh)

    def _apply_rope(
        self,
        q: torch.Tensor,       # [B, H, Tq, Dh]
        k: torch.Tensor,       # [B, H, Tk, Dh]
        q_freqs: torch.Tensor, # pre-computed RoPE freqs for Q positions
        k_freqs: torch.Tensor, # pre-computed RoPE freqs for K positions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from rotary_embedding_torch import apply_rotary_emb
        return (
            apply_rotary_emb(q_freqs, q, seq_dim=-2),
            apply_rotary_emb(k_freqs, k, seq_dim=-2),
        )

    def _project(
        self,
        q_in: torch.Tensor,  # [B, Tq, D]
        k_in: torch.Tensor,  # [B, Tk, D]
        v_in: torch.Tensor,  # [B, Tk, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and reshape to [B, H, T, Dh]."""
        return (
            self._split_heads(self.q_proj(q_in)),
            self._split_heads(self.k_proj(k_in)),
            self._split_heads(self.v_proj(v_in)),
        )

    def _attend(
        self,
        Q: torch.Tensor,                          # [B, H, Tq, Dh]
        K: torch.Tensor,                          # [B, H, Tk, Dh]
        V: torch.Tensor,                          # [B, H, Tk, Dh]
        mask: Optional[torch.Tensor],             # [Tq, Tk] or [B, H, Tq, Tk] bool
    ) -> torch.Tensor:
        """
        Fused attention via SDPA.  mask: True = attend, False = block.
        SDPA expects attn_mask in additive form OR bool with True=keep —
        PyTorch >= 2.0 accepts bool directly with the correct semantics.
        dropout_p is only applied during training.
        """
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=dropout_p,
        )
        return self._merge_heads(out)  # [B, Tq, D]

    # ------------------------------------------------------------------
    # Standard forward (used by training pass and context encoding)
    # ------------------------------------------------------------------

    def forward(
        self,
        q_in: torch.Tensor,                        # [B, Tq, D]
        k_in: torch.Tensor,                        # [B, Tk, D]
        v_in: torch.Tensor,                        # [B, Tk, D]
        mask: Optional[torch.Tensor] = None,       # [Tq, Tk] bool, True=attend
        q_freqs: Optional[torch.Tensor] = None,    # pre-computed RoPE freqs for Q
        k_freqs: Optional[torch.Tensor] = None,    # pre-computed RoPE freqs for K
    ) -> torch.Tensor:
        Q, K, V = self._project(q_in, k_in, v_in)
        if q_freqs is not None:
            Q, K = self._apply_rope(Q, K, q_freqs, k_freqs)
        return self.out_proj(self._attend(Q, K, V, mask))

    # ------------------------------------------------------------------
    # Inference forward: projects only new tokens, appends to KV cache.
    # Returns (output, updated LayerKVCache).
    # ------------------------------------------------------------------

    def forward_with_kv_cache(
        self,
        q_in: torch.Tensor,                        # [B, Tq, D]  new tokens
        new_kv_in: torch.Tensor,                   # [B, Tq, D]  new K/V source
        cache: Optional[LayerKVCache],             # existing K/V (None = empty)
        mask: Optional[torch.Tensor] = None,       # [Tq, T_total] bool
        q_freqs: Optional[torch.Tensor] = None,    # pre-computed RoPE freqs for Q
        k_freqs: Optional[torch.Tensor] = None,    # pre-computed RoPE freqs for K (T_total)
    ) -> Tuple[torch.Tensor, LayerKVCache]:
        # Project only the new tokens
        Q     = self._split_heads(self.q_proj(q_in))
        new_K = self._split_heads(self.k_proj(new_kv_in))
        new_V = self._split_heads(self.v_proj(new_kv_in))

        # Append to existing cache to get the full K, V for attention
        if cache is not None:
            K = torch.cat([cache.k, new_K], dim=2)
            V = torch.cat([cache.v, new_V], dim=2)
        else:
            K, V = new_K, new_V

        # Cache stores raw (unrotated) projections so RoPE can be re-applied
        # correctly over the full sequence on each call.
        updated_cache = LayerKVCache(k=K, v=V)

        if q_freqs is not None:
            Q, K = self._apply_rope(Q, K, q_freqs, k_freqs)

        return self.out_proj(self._attend(Q, K, V, mask)), updated_cache


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Transformer layer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,         # pre-computed K/V source (or None = self)
        mask: Optional[torch.Tensor] = None,
        q_freqs: Optional[torch.Tensor] = None,    # pre-computed RoPE freqs for Q
        k_freqs: Optional[torch.Tensor] = None,    # pre-computed RoPE freqs for K
    ) -> torch.Tensor:
        kv = kv if kv is not None else x
        x = self.norm1(x + self.drop(
            self.self_attn(x, kv, kv, mask, q_freqs, k_freqs)
        ))
        return self.norm2(x + self.drop(self.ff(x)))

    def forward_with_kv_cache(
        self,
        x: torch.Tensor,
        cache: Optional[LayerKVCache],
        mask: Optional[torch.Tensor] = None,
        q_freqs: Optional[torch.Tensor] = None,    # pre-computed RoPE freqs for Q
        k_freqs: Optional[torch.Tensor] = None,    # pre-computed RoPE freqs for K
    ) -> Tuple[torch.Tensor, LayerKVCache]:
        attn_out, new_cache = self.self_attn.forward_with_kv_cache(
            q_in=x, new_kv_in=x, cache=cache, mask=mask,
            q_freqs=q_freqs, k_freqs=k_freqs,
        )
        x = self.norm1(x + self.drop(attn_out))
        return self.norm2(x + self.drop(self.ff(x))), new_cache


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POP Transformer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class POPTransformer(nn.Module):
    """
    Parameters
    ----------
    n_obs      : observation tokens per block
    n_act      : action tokens per block
    n_layers   : transformer depth
    d_model    : embedding dimension
    n_heads    : attention heads
    d_ff       : feed-forward hidden dim
    vocab_size : shared vocabulary
    dropout    : dropout rate
    pos_enc         : 'sinusoidal' (additive) or 'rotary' (RoPE)
    pred_self_mask  : attention pattern among prediction tokens within each copy:
                      'causal' (default) — token j sees tokens 0..j
                      'full'             — every token sees all n tokens in the copy
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        vocab_size: int,
        dropout: float = 0.1,
        pos_enc: Literal["sinusoidal", "rotary"] = "sinusoidal",
        pred_self_mask: PredSelfMask = "causal",
    ):
        super().__init__()
        self.n_obs   = n_obs
        self.n_act   = n_act
        self.d_model = d_model
        self.pos_enc = pos_enc
        self.pred_self_mask = pred_self_mask

        self.embed = nn.Embedding(vocab_size, d_model)

        if pos_enc == "sinusoidal":
            self.sinusoidal = SinusoidalPositionalEncoding(d_model)
            rotary_emb = None
        elif pos_enc == "rotary":
            from rotary_embedding_torch import RotaryEmbedding
            rotary_emb = RotaryEmbedding(dim=(d_model // n_heads) // 2)
            self.rotary_emb = rotary_emb
            self.sinusoidal = None
        else:
            raise ValueError(f"pos_enc must be 'sinusoidal' or 'rotary', got {pos_enc!r}")

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.pred_tokens = nn.Parameter(torch.randn(n_obs, d_model) * 0.02)
        self.obs_head    = nn.Linear(d_model, vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Embed tokens and optionally add sinusoidal pos encoding."""
        x = self.embed(tokens)
        if self.pos_enc == "sinusoidal":
            x = x + self.sinusoidal(positions).unsqueeze(0)
        return x

    def _pred_token_embeds(
        self, block_idx: int, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeddings + positions for one copy of the prediction tokens.
        block_idx : 0-indexed block being predicted.
        Returns (x [B, n, D], positions [n]).
        """
        n, m = self.n_obs, self.n_act
        positions = torch.arange(
            block_idx * (n + m), block_idx * (n + m) + n, device=device
        )
        x = self.pred_tokens.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if self.pos_enc == "sinusoidal":
            x = x + self.sinusoidal(positions).unsqueeze(0)
        return x, positions

    def _empty_layer_cache(self, batch_size: int, device: torch.device) -> LayerKVCache:
        H  = self.layers[0].self_attn.n_heads
        Dh = self.layers[0].self_attn.d_head
        empty = torch.zeros(batch_size, H, 0, Dh, device=device)
        return LayerKVCache(k=empty, v=empty)

    def _rope_freqs(
        self,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE frequency tensors once per forward call.
        Shared across all layers — callers must NOT pass seq_len to the
        library so that frequencies are always derived from actual position
        values rather than sequential slot indices (see _apply_rope note).
        """
        return (
            self.rotary_emb(q_positions.float()),
            self.rotary_emb(k_positions.float()),
        )

    # ══════════════════════════════════════════════════════════════════
    # SHARED CORE METHODS  (position-encoded x already prepared by callers)
    # ══════════════════════════════════════════════════════════════════

    def _encode_x_to_kv_cache(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, KVCache]:
        """
        Causally encode pre-embedded [B, T, D] tokens and build a KV cache.
        positions [T] are used for RoPE; ignored for sinusoidal (already baked in).
        Returns (hidden_states [B, T, D], KVCache).
        """
        T, device = x.shape[1], x.device
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        q_freqs, k_freqs = self._rope_freqs(positions, positions) if self.pos_enc == "rotary" else (None, None)

        layer_caches: List[LayerKVCache] = []
        for layer in self.layers:
            x, lc = layer.forward_with_kv_cache(
                x, cache=None, mask=causal, q_freqs=q_freqs, k_freqs=k_freqs,
            )
            layer_caches.append(lc)
        return x, KVCache(layer_caches, n=self.n_obs, m=self.n_act)

    def _run_prediction_pass(
        self, k: int, context_cache: KVCache, device: torch.device
    ) -> torch.Tensor:
        """
        Prediction pass over k copies of the prediction-token template.
        Returns raw latents [B, k*n, D] — callers apply their own head.
        """
        n, m = self.n_obs, self.n_act
        B    = context_cache.layers[0].k.shape[0]

        x = (
            self.pred_tokens
            .unsqueeze(0).expand(k, -1, -1)
            .reshape(1, k * n, self.d_model)
            .expand(B, -1, -1).clone()              # [B, k*n, D]
        )
        pred_pos = build_prediction_positions(k, n, m, device)
        if self.pos_enc == "sinusoidal":
            x = x + self.sinusoidal(pred_pos).unsqueeze(0)

        pop_mask = build_pop_mask(k, n, m, device, self.pred_self_mask)
        orig_pos = torch.arange(k * (n + m), device=device)

        if self.pos_enc == "rotary":
            jk_pos = torch.cat([orig_pos, pred_pos])
            q_freqs, k_freqs = self._rope_freqs(pred_pos, jk_pos)
        else:
            q_freqs = k_freqs = None

        for layer, ctx_cache in zip(self.layers, context_cache.layers):
            # pred tokens attend to context K/V plus their own (causal within copy);
            # the updated cache (pred K/V appended) is discarded — not committed.
            x, _ = layer.forward_with_kv_cache(
                x, cache=ctx_cache, mask=pop_mask, q_freqs=q_freqs, k_freqs=k_freqs,
            )
        return x  # [B, k*n, D]

    def _append_x_to_cache(
        self,
        cache: KVCache,
        x: torch.Tensor,
        new_positions: torch.Tensor,
        all_positions: torch.Tensor,
    ) -> KVCache:
        """
        Append pre-embedded [B, T_new, D] tokens to the cache with a causal mask.
        Returns a new KVCache; the original is not mutated.
        """
        T_ctx, T_new = cache.context_len, x.shape[1]
        device = x.device
        mask = torch.cat([
            torch.ones(T_new, T_ctx, dtype=torch.bool, device=device),
            torch.tril(torch.ones(T_new, T_new, dtype=torch.bool, device=device)),
        ], dim=1)
        q_freqs, k_freqs = self._rope_freqs(new_positions, all_positions) if self.pos_enc == "rotary" else (None, None)

        new_layer_caches: List[LayerKVCache] = []
        for layer, lc in zip(self.layers, cache.layers):
            x, new_lc = layer.forward_with_kv_cache(
                x, cache=lc, mask=mask, q_freqs=q_freqs, k_freqs=k_freqs,
            )
            new_layer_caches.append(new_lc)
        return KVCache(new_layer_caches, n=self.n_obs, m=self.n_act)

    def _run_single_pred_pass(self, cache: KVCache) -> torch.Tensor:
        """
        Run one copy of the prediction tokens against the current cache.
        Returns raw latents [B, n, D] — callers apply their own head.
        """
        n, m   = self.n_obs, self.n_act
        i      = cache.n_blocks_committed
        T_ctx  = cache.context_len
        device = cache.layers[0].k.device
        B      = cache.layers[0].k.shape[0]

        pred_x, pred_pos = self._pred_token_embeds(i, B, device)

        if self.pred_self_mask == "causal":
            self_block = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
        else:
            self_block = torch.ones(n, n, dtype=torch.bool, device=device)
        mask = torch.cat([
            torch.ones(n, T_ctx, dtype=torch.bool, device=device),
            self_block,
        ], dim=1)

        all_k_positions = torch.cat([torch.arange(T_ctx, device=device), pred_pos])
        q_freqs, k_freqs = self._rope_freqs(pred_pos, all_k_positions) if self.pos_enc == "rotary" else (None, None)

        x = pred_x
        for layer, lc in zip(self.layers, cache.layers):
            x, _ = layer.forward_with_kv_cache(
                x, cache=lc, mask=mask, q_freqs=q_freqs, k_freqs=k_freqs,
            )
        return x  # [B, n, D]

    def _append_x_and_predict_next_core(
        self,
        cache: KVCache,
        new_block_x: torch.Tensor,         # [B, n+m, D]  obs|act, pos enc already applied
        pred_x: torch.Tensor,              # [B, n, D]    pred tokens, pos enc already applied
        q_positions: Optional[torch.Tensor],  # [n+m+n] for RoPE Q (None for sinusoidal)
        k_positions: Optional[torch.Tensor],  # [T_ctx+n+m+n] for RoPE K (None for sinusoidal)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-pass: encode [obs|act] into the cache AND run prediction tokens.

        Runs all layers once over [new_block | pred] tokens attending to the
        existing cache.  Only obs+act K/V are committed; pred K/V are trimmed.

        Returns (updated_cache, pred_latents [B, n, D]).
        Equivalent to _append_x_to_cache followed by _run_single_pred_pass,
        but with one forward pass instead of two.
        """
        n, m    = self.n_obs, self.n_act
        T_ctx   = cache.context_len
        T_new   = n + m + n           # obs + act + pred
        T_total = T_ctx + T_new
        device  = new_block_x.device

        x = torch.cat([new_block_x, pred_x], dim=1)  # [B, n+m+n, D]

        # Attention mask [T_new, T_total]: True = attend
        mask = torch.zeros(T_new, T_total, dtype=torch.bool, device=device)
        mask[:, :T_ctx] = True                        # all new tokens see full cache
        # obs+act: causal self-attention
        mask[:n+m, T_ctx:T_ctx+n+m] = torch.tril(
            torch.ones(n+m, n+m, dtype=torch.bool, device=device)
        )
        # pred: attend to all obs+act new tokens
        mask[n+m:, T_ctx:T_ctx+n+m] = True
        # pred: self-attention within pred copy
        if self.pred_self_mask == "causal":
            self_block = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
        else:
            self_block = torch.ones(n, n, dtype=torch.bool, device=device)
        mask[n+m:, T_ctx+n+m:] = self_block

        q_freqs, k_freqs = self._rope_freqs(q_positions, k_positions) if q_positions is not None else (None, None)

        new_layer_caches: List[LayerKVCache] = []
        for layer, lc in zip(self.layers, cache.layers):
            x, new_lc = layer.forward_with_kv_cache(
                x, cache=lc, mask=mask, q_freqs=q_freqs, k_freqs=k_freqs,
            )
            # Commit only obs+act K/V; discard pred K/V from the cache
            new_layer_caches.append(LayerKVCache(
                k=new_lc.k[:, :, :T_ctx + n + m, :],
                v=new_lc.v[:, :, :T_ctx + n + m, :],
            ))

        new_cache     = KVCache(new_layer_caches, n=self.n_obs, m=self.n_act)
        pred_latents  = x[:, n + m:, :]   # [B, n, D]
        return new_cache, pred_latents

    # ══════════════════════════════════════════════════════════════════
    # TRAINING PATH  (token-based public API)
    # ══════════════════════════════════════════════════════════════════

    def _encode_to_kv_cache(self, tokens: torch.Tensor) -> KVCache:
        B, T   = tokens.shape
        device = tokens.device
        positions = torch.arange(T, device=device)
        x = self._embed(tokens, positions)
        _, kvc = self._encode_x_to_kv_cache(x, positions)
        return kvc

    def _predict_observations(
        self, k: int, context_cache: KVCache, device: torch.device,
    ) -> torch.Tensor:
        """Returns logits [B, k, n_obs, vocab_size]."""
        n = self.n_obs
        B = context_cache.layers[0].k.shape[0]
        x = self._run_prediction_pass(k, context_cache, device)
        return self.obs_head(x.view(B, k, n, self.d_model))

    def forward(
        self,
        tokens: torch.Tensor,        # [B, k*(n+m)]
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Returns
        -------
        logits  : [B, k, n_obs, vocab_size]
        targets : [B, k, n_obs]
        """
        n, m = self.n_obs, self.n_act
        B, T = tokens.shape
        if k is None:
            assert T % (n + m) == 0
            k = T // (n + m)

        context_cache = self._encode_to_kv_cache(tokens)
        logits = self._predict_observations(k, context_cache, tokens.device)
        targets = torch.stack([
            tokens[:, i * (n + m) : i * (n + m) + n] for i in range(k)
        ], dim=1)
        return logits, targets

    def loss(self, tokens: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        logits, targets = self.forward(tokens, k)
        B, _k, n, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * _k * n, V),
            targets.reshape(B * _k * n),
        )

    # ══════════════════════════════════════════════════════════════════
    # INFERENCE PATH  (token-based public API)
    # ══════════════════════════════════════════════════════════════════

    def init_cache(
        self,
        device: torch.device,
        batch_size: int = 1,
        context_tokens: Optional[torch.Tensor] = None,
    ) -> KVCache:
        """
        Create a KVCache, optionally pre-filled with a context sequence.

        context_tokens : [B, T]  where T must be a multiple of (n+m).
        """
        n, m = self.n_obs, self.n_act

        if context_tokens is None:
            layer_caches = [
                self._empty_layer_cache(batch_size, device)
                for _ in self.layers
            ]
            return KVCache(layer_caches, n=n, m=m)

        B, T = context_tokens.shape
        assert T % (n + m) == 0, "context_tokens length must be a multiple of (n+m)"
        return self._encode_to_kv_cache(context_tokens)

    def append_block(
        self,
        cache: KVCache,
        obs_tokens: torch.Tensor,    # [B, n]
        act_tokens: torch.Tensor,    # [B, m]
    ) -> KVCache:
        """
        Encode one new [obs | act] block and append its K/V to the cache.
        Returns a new KVCache; the original is not mutated.
        """
        device    = obs_tokens.device
        T_ctx     = cache.context_len
        T_new     = self.n_obs + self.n_act
        new_pos   = torch.arange(T_ctx, T_ctx + T_new, device=device)
        all_pos   = torch.arange(T_ctx + T_new, device=device)
        x = self._embed(torch.cat([obs_tokens, act_tokens], dim=1), new_pos)
        return self._append_x_to_cache(cache, x, new_pos, all_pos)

    def predict_next(self, cache: KVCache) -> torch.Tensor:
        """
        Run one copy of the prediction tokens against the current cache.
        Their K/V projections are NOT committed — call append_block after sampling.
        Returns logits [B, n_obs, vocab_size].
        """
        return self.obs_head(self._run_single_pred_pass(cache))

    def append_block_and_predict(
        self,
        cache: KVCache,
        obs_tokens: torch.Tensor,  # [B, n]
        act_tokens: torch.Tensor,  # [B, m]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combined: append [obs|act] to the cache and run the next prediction pass.
        Equivalent to append_block then predict_next but in a single forward pass.
        Returns (updated_cache, logits [B, n_obs, vocab_size]).
        """
        T_ctx  = cache.context_len
        n, m   = self.n_obs, self.n_act
        device = obs_tokens.device
        B      = obs_tokens.shape[0]

        block_pos   = torch.arange(T_ctx, T_ctx + n + m, device=device)
        new_block_x = self._embed(torch.cat([obs_tokens, act_tokens], dim=1), block_pos)

        pred_x, pred_pos = self._pred_token_embeds(cache.n_blocks_committed + 1, B, device)

        if self.pos_enc == "rotary":
            q_pos = torch.cat([block_pos, pred_pos])
            k_pos = torch.cat([torch.arange(T_ctx, device=device), block_pos, pred_pos])
        else:
            q_pos = k_pos = None

        new_cache, pred_latents = self._append_x_and_predict_next_core(
            cache, new_block_x, pred_x, q_pos, k_pos,
        )
        return new_cache, self.obs_head(pred_latents)

    # ══════════════════════════════════════════════════════════════════
    # EMBEDDING-FREE API  (used by POPWorldModel with multi-modal embs)
    # ══════════════════════════════════════════════════════════════════

    def encode_embs_to_kv_cache(self, x: torch.Tensor) -> Tuple[torch.Tensor, KVCache]:
        """
        Like _encode_to_kv_cache but accepts pre-computed [B, T, D] embeddings.
        Returns (hidden_states [B, T, D], KVCache) — hidden states let callers
        read action-token outputs without a second forward pass.
        """
        device    = x.device
        positions = torch.arange(x.shape[1], device=device)
        if self.pos_enc == "sinusoidal":
            x = x + self.sinusoidal(positions).unsqueeze(0)
        return self._encode_x_to_kv_cache(x, positions)

    def predict_latents(
        self, k: int, context_cache: KVCache, device: torch.device
    ) -> torch.Tensor:
        """
        Like _predict_observations but returns raw latents [B, k, n, D]
        instead of applying obs_head.
        """
        B = context_cache.layers[0].k.shape[0]
        x = self._run_prediction_pass(k, context_cache, device)
        return x.view(B, k, self.n_obs, self.d_model)

    def append_block_embs(self, cache: KVCache, block_emb: torch.Tensor) -> KVCache:
        """
        Like append_block but accepts pre-computed [B, T_new, D] embeddings.
        T_new is typically n+m but may be larger for multi-block context init.
        Returns a new KVCache; the original is not mutated.
        """
        device  = block_emb.device
        T_ctx   = cache.context_len
        T_new   = block_emb.shape[1]
        new_pos = torch.arange(T_ctx, T_ctx + T_new, device=device)
        all_pos = torch.arange(T_ctx + T_new, device=device)
        x = block_emb.clone()
        if self.pos_enc == "sinusoidal":
            x = x + self.sinusoidal(new_pos).unsqueeze(0)
        return self._append_x_to_cache(cache, x, new_pos, all_pos)

    def predict_next_latents(self, cache: KVCache) -> torch.Tensor:
        """
        Like predict_next but returns [B, n, D] latents instead of applying obs_head.
        """
        return self._run_single_pred_pass(cache)

    def append_block_embs_and_predict_latents(
        self,
        cache: KVCache,
        obs_emb: torch.Tensor,  # [B, n, D]
        act_emb: torch.Tensor,  # [B, m, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Like append_block_and_predict but accepts pre-computed embeddings.
        Combined: append [obs_emb|act_emb] to cache and run the next prediction pass.
        Returns (updated_cache, pred_latents [B, n, D]).
        """
        T_ctx  = cache.context_len
        n, m   = self.n_obs, self.n_act
        device = obs_emb.device
        B      = obs_emb.shape[0]

        block_pos   = torch.arange(T_ctx, T_ctx + n + m, device=device)
        new_block_x = torch.cat([obs_emb, act_emb], dim=1).clone()
        if self.pos_enc == "sinusoidal":
            new_block_x = new_block_x + self.sinusoidal(block_pos).unsqueeze(0)

        pred_x, pred_pos = self._pred_token_embeds(cache.n_blocks_committed + 1, B, device)

        if self.pos_enc == "rotary":
            q_pos = torch.cat([block_pos, pred_pos])
            k_pos = torch.cat([torch.arange(T_ctx, device=device), block_pos, pred_pos])
        else:
            q_pos = k_pos = None

        return self._append_x_and_predict_next_core(
            cache, new_block_x, pred_x, q_pos, k_pos,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Smoke tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    torch.manual_seed(0)

    B, k, n_obs, n_act = 2, 4, 3, 2
    d_model, n_heads, d_ff, vocab = 64, 4, 128, 32
    seq_len = k * (n_obs + n_act)
    tokens  = torch.randint(0, vocab, (B, seq_len))

    print("=" * 60)
    print("TRAINING PASS")
    print("=" * 60)
    for mode in ("sinusoidal", "rotary"):
        for psm in ("causal", "full"):
            model = POPTransformer(
                n_obs=n_obs, n_act=n_act, n_layers=2,
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                vocab_size=vocab, pos_enc=mode, pred_self_mask=psm,
            )
            logits, targets = model(tokens, k=k)
            loss = model.loss(tokens, k=k)
            print(f"  [{mode:>10} / {psm}]  logits {tuple(logits.shape)}  loss {loss.item():.4f}")

    print()
    print("=" * 60)
    print("INFERENCE PASS")
    print("=" * 60)
    for mode in ("sinusoidal", "rotary"):
        model = POPTransformer(
            n_obs=n_obs, n_act=n_act, n_layers=2,
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            vocab_size=vocab, pos_enc=mode,
        )
        model.eval()
        with torch.no_grad():
            # Step-by-step from scratch
            cache = model.init_cache(torch.device("cpu"), batch_size=B)
            for step in range(k):
                logits = model.predict_next(cache)
                obs  = logits.argmax(-1)
                act  = torch.randint(0, vocab, (B, n_act))
                cache = model.append_block(cache, obs, act)
            print(f"  [{mode:>10}]  step-by-step ok  "
                  f"context_len={cache.context_len}  blocks={cache.n_blocks_committed}")

            # Pre-fill then continue
            context = tokens[:, : 2 * (n_obs + n_act)]
            cache2  = model.init_cache(torch.device("cpu"), batch_size=B,
                                       context_tokens=context)
            logits2 = model.predict_next(cache2)
            print(f"  [{mode:>10}]  pre-fill ok  "
                  f"context_len={cache2.context_len}  "
                  f"logits {tuple(logits2.shape)}")

    print()
    print("=" * 60)
    print("MASK SANITY CHECK")
    print("=" * 60)
    orig_len = k * (n_obs + n_act)
    for psm in ("causal", "full"):
        mask = build_pop_mask(k, n_obs, n_act, torch.device("cpu"), psm)
        print(f"  [{psm}]  shape {tuple(mask.shape)}")
        for i in range(k):
            row    = i * n_obs
            kv_max = (mask[row, :orig_len].nonzero(as_tuple=True)[0].max().item()
                      if mask[row, :orig_len].any() else -1)
            self_c = mask[row, orig_len:].nonzero(as_tuple=True)[0].tolist()
            print(f"    copy {i}  KV 0..{kv_max:>2} (exp 0..{i*(n_obs+n_act)-1:>2})  "
                  f"self-cols {self_c}")
