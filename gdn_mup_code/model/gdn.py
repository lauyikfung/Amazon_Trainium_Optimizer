from __future__ import annotations
# Trainium-friendly rewrite of the original gdn.py.
# Original copyright (Songlin Yang, Yu Zhang) from https://github.com/fla-org/flash-linear-attention
#
# Changes vs. the original file:
#   1. The two Triton kernels `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule`
#      are replaced by pure-PyTorch implementations (naive recurrent + chunked).
#   2. `fla.layers.attn.Attention`  -> TorchSDPAAttention  (torch SDPA + RoPE, no flash-attn).
#   3. `fla.modules.GatedMLP`        -> TorchSwiGLUMLP      (pure-PyTorch SwiGLU).
#   4. Local fused `RMSNorm`         -> torch.nn.RMSNorm.
#   5. Varlen / cu_seqlens / get_unpad_data path removed (dynamic shapes break neuronx-cc cache).
#   6. The host-device sync `attention_mask.all().item()` is removed.
#   7. `config.fuse_norm` is forced False; FLA's fused-residual norm signature is gone.
#
# Everything else (model topology, parameter counts, init scheme, HF interface) is preserved.

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

# These three are pure-Python in fla / your repo, so they are Trainium-safe.
from .gpt_base import PastKeyValue
from .pydantic_config import validate_pretrained_config_kwargs

# fla.models.utils.Cache is a thin DynamicCache subclass with no Triton inside.
try:
    from fla.models.utils import Cache
except Exception:  # pragma: no cover
    from transformers.cache_utils import DynamicCache as Cache  # type: ignore

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

# Prefer HF's GradientCheckpointingLayer, fall back to a tiny local shim.
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    class GradientCheckpointingLayer(nn.Module):
        """Minimal stand-in: forwards as a normal nn.Module unless a parent toggles checkpointing."""
        gradient_checkpointing: bool = False


logger = logging.get_logger(__name__)


# =============================================================================
# Pure-PyTorch gated delta rule kernels (Trainium-safe)
# =============================================================================
#
# Reference recurrence (per head, batch element):
#
#     S_t = exp(g_t) * S_{t-1} + beta_t * (v_t - exp(g_t) * S_{t-1} @ k_t) @ k_t^T
#         = exp(g_t) * S_{t-1} * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
#     o_t = q_t^T @ S_t
#
# Two equivalent implementations are provided:
#   * `naive_recurrent_gated_delta_rule`: O(T) static-graph Python loop. Simple, exact,
#     and the right thing to use for short prefills / generation (q_len <= ~64).
#   * `chunk_gated_delta_rule_torch`: chunked, matmul-heavy form. Outer loop over chunks
#     (Python-unrolled), inner chunk solved with a Neumann series via BT-step matvec.
#     Use this for training and longer prefills.
#
# Both accept the same arg set as the FLA Triton kernels so the call sites in
# `GatedDeltaNet.forward` change only by replacing the function pointer.


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    # Promote to fp32 for the norm to keep stability on bf16 / fp16.
    n = x.float().pow(2).sum(dim=dim, keepdim=True).clamp_min(eps).rsqrt()
    return (x.float() * n).to(x.dtype)


def naive_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,  # accepted for API parity, must be None
    **_unused: Any,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Static-graph recurrent gated delta rule.

    Shapes (head-last / seq-first, matching the post-2024-11 FLA convention):
        q, k:  [B, T, H, K]
        v:     [B, T, H, V]
        g:     [B, T, H]      log-decay (typically <= 0; exp(g) is the per-step gate)
        beta:  [B, T, H]      learning-rate
        initial_state: [B, H, K, V] or None
    Returns:
        o:           [B, T, H, V]
        final_state: [B, H, K, V] or None
    """
    if cu_seqlens is not None:
        raise NotImplementedError("Variable-length packing is not supported on Trainium; pad the batch instead.")

    B, T, H, K = q.shape
    V = v.shape[-1]

    if use_qk_l2norm_in_kernel:
        q = _l2_normalize(q, dim=-1)
        k = _l2_normalize(k, dim=-1)

    out_dtype = q.dtype
    if initial_state is None:
        state = q.new_zeros(B, H, K, V, dtype=torch.float32)
    else:
        state = initial_state.to(torch.float32)

    # We accumulate outputs in fp32 for stability then cast back at the end.
    o_steps = []
    for t in range(T):
        q_t = q[:, t].float()                   # [B, H, K]
        k_t = k[:, t].float()                   # [B, H, K]
        v_t = v[:, t].float()                   # [B, H, V]
        g_t = g[:, t].float().exp()             # [B, H]
        b_t = beta[:, t].float()                # [B, H]

        # Decay the carried state.
        state = state * g_t.unsqueeze(-1).unsqueeze(-1)

        # delta-rule correction: u_t = beta_t * (v_t - state^T @ k_t)
        # state has shape [B, H, K, V]; predict v from current state at key k_t.
        pred = torch.einsum("bhk,bhkv->bhv", k_t, state)        # [B, H, V]
        u_t = b_t.unsqueeze(-1) * (v_t - pred)                  # [B, H, V]
        state = state + k_t.unsqueeze(-1) * u_t.unsqueeze(-2)   # outer product update

        # Read out: o_t = state @ q_t  (q_t indexes the K dim)
        o_t = torch.einsum("bhk,bhkv->bhv", q_t, state)         # [B, H, V]
        o_steps.append(o_t)

    o = torch.stack(o_steps, dim=1).to(out_dtype)  # [B, T, H, V]
    return o, state if output_final_state else None


def chunk_gated_delta_rule_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    **_unused: Any,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunked gated delta rule using the WY decomposition.

    Within a chunk of length BT we compute the corrected values
    ``u_j = beta_j (v_j - exp(g_cs[j]) S_start k_j) - sum_{l<j} A[j,l] u_l``
    where ``A[j,l] = beta_j * exp(g_cs[j] - g_cs[l]) * (k_j . k_l)`` for l<j.

    Solving the strictly lower-triangular system ``(I + A) u = w`` is done by
    BT-1 in-place subtractions (BT is a Python ``int``, so neuronx-cc fully
    unrolls this into a static graph). Every other operation in the chunk is
    a matmul, which is the Tensor Engine's fast path.

    Args mirror the FLA Triton API; ``cu_seqlens`` must be ``None`` because
    Neuron prefers static shapes.
    """
    if cu_seqlens is not None:
        raise NotImplementedError("Variable-length packing is not supported on Trainium; pad the batch instead.")

    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = int(chunk_size)
    if BT <= 0:
        raise ValueError(f"chunk_size must be positive, got {BT}")
    out_dtype = v.dtype  # preserve caller's dtype on the output

    if use_qk_l2norm_in_kernel:
        q = _l2_normalize(q, dim=-1)
        k = _l2_normalize(k, dim=-1)

    # Pad T to a multiple of BT. The pad amount is a Python int, so this is a
    # static shape change at trace time.
    pad = (-T) % BT
    if pad:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
    Tp = q.size(1)
    NC = Tp // BT

    # Reshape into [B, H, NC, BT, *] -- head dim moved next to batch so the
    # heavy matmuls below batch over (B, H) on the PE.
    q = q.view(B, NC, BT, H, K).permute(0, 3, 1, 2, 4).contiguous().float()
    k = k.view(B, NC, BT, H, K).permute(0, 3, 1, 2, 4).contiguous().float()
    v = v.view(B, NC, BT, H, V).permute(0, 3, 1, 2, 4).contiguous().float()
    g = g.view(B, NC, BT, H).permute(0, 3, 1, 2).contiguous().float()        # [B, H, NC, BT]
    beta = beta.view(B, NC, BT, H).permute(0, 3, 1, 2).contiguous().float()  # [B, H, NC, BT]

    g_cs = g.cumsum(dim=-1)              # [B, H, NC, BT]   log-cumsum within chunk
    g_chunk = g_cs[..., -1]              # [B, H, NC]       total log-decay per chunk

    # Causal mask + intra-chunk pairwise decay.
    causal = torch.tril(torch.ones(BT, BT, device=q.device, dtype=torch.bool))  # [BT, BT]
    # diff[..., i, j] = g_cs[..., i] - g_cs[..., j]
    diff = g_cs.unsqueeze(-1) - g_cs.unsqueeze(-2)              # [B, H, NC, BT, BT]
    decay_pairwise = diff.exp() * causal                        # 0 above the diagonal

    decay_from_start = g_cs.exp()        # [B, H, NC, BT]
    decay_to_end = (g_chunk.unsqueeze(-1) - g_cs).exp()         # [B, H, NC, BT]

    if initial_state is None:
        state = q.new_zeros(B, H, K, V)
    else:
        state = initial_state.to(q.dtype)

    out_chunks = []
    strict_lt = torch.tril(torch.ones(BT, BT, device=q.device, dtype=q.dtype), diagonal=-1)

    for c in range(NC):
        q_c = q[:, :, c]          # [B, H, BT, K]
        k_c = k[:, :, c]
        v_c = v[:, :, c]
        beta_c = beta[:, :, c]    # [B, H, BT]
        decay_c = decay_pairwise[:, :, c]               # [B, H, BT, BT]
        from_start_c = decay_from_start[:, :, c]        # [B, H, BT]
        to_end_c = decay_to_end[:, :, c]                # [B, H, BT]
        gchunk_c = g_chunk[:, :, c]                     # [B, H]

        # "Without state" target  w_j = beta_j (v_j - exp(g_cs[j]) * (k_j @ state))
        state_proj = torch.matmul(k_c, state)                        # [B, H, BT, V]
        w = beta_c.unsqueeze(-1) * (v_c - from_start_c.unsqueeze(-1) * state_proj)

        # Strict-lower-triangular A: A[j, l] = beta_j * decay_c[j, l] * (k_j . k_l)
        K_dot = torch.matmul(k_c, k_c.transpose(-1, -2))             # [B, H, BT, BT]
        A = beta_c.unsqueeze(-1) * decay_c * K_dot
        A = A * strict_lt                                            # zero diag + upper

        # Solve u = w - A u, sequentially within the chunk.
        # Each j depends only on l < j, so BT-1 unrolled subtractions suffice.
        u = w
        for j in range(1, BT):
            # contrib = A[..., j, :j] @ u[..., :j, :]
            contrib = torch.matmul(A[..., j:j + 1, :j], u[..., :j, :])  # [B, H, 1, V]
            u = torch.cat([u[..., :j, :], u[..., j:j + 1, :] - contrib, u[..., j + 1:, :]], dim=-2)

        # Outputs for the chunk:
        #   o_i = exp(g_cs[i]) * (q_i @ S_start) + sum_{l <= i} decay_c[i, l] * (q_i . k_l) * u_l
        o_state = torch.matmul(q_c, state) * from_start_c.unsqueeze(-1)   # [B, H, BT, V]
        QK = torch.matmul(q_c, k_c.transpose(-1, -2)) * decay_c            # [B, H, BT, BT]
        o_intra = torch.matmul(QK, u)                                      # [B, H, BT, V]
        out_chunks.append(o_state + o_intra)

        # State update for the next chunk:
        #   S_new = exp(g_chunk) * S_start + sum_l exp(g_cs[BT-1] - g_cs[l]) * k_l^T * u_l
        u_weighted = u * to_end_c.unsqueeze(-1)                            # [B, H, BT, V]
        contrib_state = torch.matmul(k_c.transpose(-1, -2), u_weighted)   # [B, H, K, V]
        state = state * gchunk_c.exp().unsqueeze(-1).unsqueeze(-1) + contrib_state

    o = torch.stack(out_chunks, dim=2)                # [B, H, NC, BT, V]
    o = o.view(B, H, NC * BT, V).transpose(1, 2)     # [B, T_padded, H, V]
    if pad:
        o = o[:, :T]
    o = o.to(out_dtype)

    return o.contiguous(), state if output_final_state else None


# =============================================================================
# Trainium-safe building blocks
# =============================================================================

# ELU and sum-norm utilities (unchanged) -------------------------------------
def elu_p1(x: torch.Tensor) -> torch.Tensor:
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    return (x / x.sum(-1, keepdim=True)).to(x)


class TorchShortConvolution(nn.Module):
    """Trainium-safe replacement for FLA's causal-conv1d ShortConvolution."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.activation = activation
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.kernel_size,
            padding=0,
            groups=hidden_size,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cu_seqlens is not None:
            raise NotImplementedError("TorchShortConvolution does not support packed cu_seqlens inputs.")
        if x.ndim != 3:
            raise ValueError(f"expected x of shape (B, T, C), got {tuple(x.shape)}")

        conv_input = x
        if cache is not None:
            conv_input = torch.cat((cache.to(device=x.device, dtype=x.dtype), x), dim=1)

        y = conv_input.transpose(1, 2).contiguous()
        if cache is None:
            y = F.pad(y, (self.kernel_size - 1, 0)).contiguous()
        y = self.conv(y).transpose(1, 2).contiguous()
        if cache is not None:
            y = y[:, -x.size(1):, :]

        if self.activation == "silu":
            y = F.silu(y)
        elif self.activation not in (None, "identity"):
            raise ValueError(f"Unsupported TorchShortConvolution activation={self.activation!r}")

        final_state = None
        if output_final_state:
            keep = max(self.kernel_size - 1, 0)
            final_state = conv_input[:, -keep:, :].detach() if keep > 0 else conv_input[:, :0, :].detach()
        return y, final_state


class TorchRMSNormGated(nn.Module):
    """Pure-PyTorch gated RMSNorm: y = (RMSNorm(x) * weight) * SiLU(gate)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = y * self.weight.float()
        y = y * F.silu(gate.float())
        return y.to(orig_dtype)


class TorchSwiGLUMLP(nn.Module):
    """SwiGLU MLP. Replaces fla.modules.GatedMLP, no Triton."""

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        fuse_swiglu: bool = False,  # accepted for config compat, but always pure torch
    ) -> None:
        super().__init__()
        del fuse_swiglu
        if intermediate_size is None:
            inter = int(hidden_size * (hidden_ratio or 4) * 2 / 3)
            # Round up to a multiple of 128 for nicer tile alignment on Trainium.
            inter = ((inter + 127) // 128) * 128
            intermediate_size = inter
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        if hidden_act in ("swish", "silu"):
            self.act = F.silu
        elif hidden_act == "gelu":
            self.act = F.gelu
        else:
            self.act = getattr(F, hidden_act)

    def forward(self, x: torch.Tensor, **_unused: Any) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class TorchSDPAAttention(nn.Module):
    """torch SDPA + rotary embedding. Replaces fla.layers.attn.Attention.

    Supports GQA (num_kv_heads <= num_heads) and optional sliding window.
    KV-cache implementation is intentionally simple: returns updated tuple in
    past_key_values for HF compatibility on the inference path.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        window_size: int | None = None,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
        layer_idx: int | None = None,
        **_unused: Any,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads {self.num_heads} not divisible by num_kv_heads {self.num_kv_heads}")
        self.head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # x:  [B, T, H, D]  with D even
        # position_ids: [B, T]
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.to(x.device)  # [B, T, D/2]
        cos = freqs.cos().unsqueeze(-2).to(x.dtype)   # [B, T, 1, D/2]
        sin = freqs.sin().unsqueeze(-2).to(x.dtype)
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, Cache | None]:
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)

        # Position ids -- prefer caller-provided, otherwise generate from cache offset.
        cache_offset = 0
        if past_key_values is not None and self.layer_idx is not None:
            cached = past_key_values[self.layer_idx] if len(past_key_values) > self.layer_idx else None
            if cached is not None and isinstance(cached, dict) and "k" in cached:
                cache_offset = cached["k"].shape[1]
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(cache_offset, cache_offset + T, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(B, -1)

        q = self._apply_rope(q, position_ids)
        k = self._apply_rope(k, position_ids)

        # Update KV cache (simple concat; for production inference, swap to a paged cache).
        if past_key_values is not None and use_cache and self.layer_idx is not None:
            cached = past_key_values[self.layer_idx] if len(past_key_values) > self.layer_idx else None
            if cached is not None and isinstance(cached, dict) and "k" in cached:
                k = torch.cat([cached["k"], k], dim=1)
                v = torch.cat([cached["v"], v], dim=1)
            past_key_values.update(k=k, v=v, layer_idx=self.layer_idx)

        # GQA: replicate kv heads to match q heads.
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=2)
            v = v.repeat_interleave(n_rep, dim=2)

        # SDPA expects [B, H, T, D].
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Build attention bias: causal mandatory; optionally a sliding window.
        attn_bias = None
        is_causal = True
        if self.window_size is not None:
            # Materialize a bias only when a window is actually requested. SDPA path
            # without bias is hot on Neuron.
            S_q = q.size(-2)
            S_k = k.size(-2)
            i = torch.arange(S_q, device=q.device).unsqueeze(-1) + (S_k - S_q)
            j = torch.arange(S_k, device=q.device).unsqueeze(0)
            mask = (j <= i) & (j > i - self.window_size)
            attn_bias = torch.zeros(S_q, S_k, device=q.device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(~mask, float("-inf"))
            is_causal = False  # the bias already encodes causality

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            is_causal=is_causal,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(attn), None, past_key_values


# =============================================================================
# GatedDeltaNet layer
# =============================================================================

class GatedDeltaNet(nn.Module):
    """Trainium-safe Gated DeltaNet linear-attention layer.

    Same parameter layout as the original; the Triton kernels are swapped for
    `chunk_gated_delta_rule_torch` / `naive_recurrent_gated_delta_rule`.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int | None = None,
        mode: str = "chunk",
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        mup: bool = False,
        hidden_size_base: int | None = None,
        chunk_size: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.output_multiplier = (
            float(hidden_size_base if hidden_size_base is not None else hidden_size) / float(hidden_size)
            if mup
            else 1.0
        )
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.chunk_size = int(chunk_size)

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.")
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}.",
            )
        if mode not in ("chunk", "fused_recurrent"):
            raise ValueError(f"Not supported mode `{mode}`.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus inverse
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = TorchShortConvolution(self.key_dim, conv_size, bias=conv_bias, activation="silu")
            self.k_conv1d = TorchShortConvolution(self.key_dim, conv_size, bias=conv_bias, activation="silu")
            self.v_conv1d = TorchShortConvolution(self.value_dim, conv_size, bias=conv_bias, activation="silu")
        else:
            warnings.warn(
                "ShortConvolution is crucial to performance; do not turn it off unless you know what you are doing.",
            )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = TorchRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, Cache | None]:
        # Trainium: we don't unpad/repack on the fly because dynamic shapes blow up
        # the compile cache. Callers should pre-pad to a static (B, T) and either
        # zero out invalid tokens upstream or rely on the gate at padding positions
        # being whatever the model produces. If you need explicit masking, multiply
        # `beta` and `g` by an attention_mask after projection.
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "Expected attention_mask as a 0/1 matrix [B, T]; arbitrary 2D masks aren't supported.",
            )
        if kwargs.get("cu_seqlens") is not None:
            raise NotImplementedError("cu_seqlens / varlen packing isn't supported on the Trainium path.")

        B, T, _ = hidden_states.shape
        # Pick recurrent for short prefills; otherwise use the chunked form.
        mode = "fused_recurrent" if (T <= 64 and not self.training) else self.mode
        if self.training and mode != "chunk":
            raise RuntimeError("Only chunk mode is supported in training.")

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state.get("conv_state", (None, None, None))
            q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q, output_final_state=bool(use_cache))
            k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k, output_final_state=bool(use_cache))
            v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v, output_final_state=bool(use_cache))
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))
            conv_state_q = conv_state_k = conv_state_v = None

        # [B, T, key_dim] -> [B, T, H, head_k_dim]
        q = q.view(B, T, self.num_heads, self.head_k_dim)
        k = k.view(B, T, self.num_heads, self.head_k_dim)
        v = v.view(B, T, self.num_v_heads, self.head_v_dim)

        # GVA: replicate q,k along the head dim if v has more heads.
        if self.num_v_heads > self.num_heads:
            n_rep = self.num_v_heads // self.num_heads
            q = q.repeat_interleave(n_rep, dim=2)
            k = k.repeat_interleave(n_rep, dim=2)

        beta = self.b_proj(hidden_states).sigmoid()                       # [B, T, num_v_heads]
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Gate: g = -exp(A_log) * softplus(a_proj(x) + dt_bias), per-head log-decay
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias.float())

        # If the caller provided a [B, T] padding mask, zero gating effect at padding.
        if attention_mask is not None:
            mask = attention_mask.to(dtype=g.dtype).unsqueeze(-1)  # [B, T, 1]
            # At padding positions: don't update state (beta=0) and don't decay (g=0).
            beta = beta * mask
            g = g * mask

        recurrent_state = last_state.get("recurrent_state") if last_state is not None else None

        if mode == "chunk":
            o, recurrent_state = chunk_gated_delta_rule_torch(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=bool(use_cache),
                use_qk_l2norm_in_kernel=True,
                chunk_size=self.chunk_size,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = naive_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=bool(use_cache),
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None and self.layer_idx is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=T,
            )

        if self.use_gate:
            g_out = self.g_proj(hidden_states).view(B, T, self.num_v_heads, self.head_v_dim)
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)
        o = o.reshape(B, T, self.num_v_heads * self.head_v_dim)
        o = self.o_proj(o)
        if self.output_multiplier != 1.0:
            o = o * self.output_multiplier
        return o, None, past_key_values


# =============================================================================
# Transformer block / GPT shell
# =============================================================================

class Block(GradientCheckpointingLayer):

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Always use stock nn.RMSNorm on Trainium (no FLA fused-residual variant).
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        if config.attn is not None and layer_idx in config.attn.get("layers", []):
            self.attn = TorchSDPAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.attn["num_attention_heads"],
                num_kv_heads=config.attn["num_kv_heads"],
                qkv_bias=config.attn["qkv_bias"],
                window_size=config.attn["window_size"],
                rope_theta=config.attn["rope_theta"],
                max_position_embeddings=config.block_size,
                layer_idx=layer_idx,
            )
        else:
            self.attn = GatedDeltaNet(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_attention_heads,
                num_v_heads=config.num_v_heads,
                use_gate=config.use_gate,
                use_short_conv=config.use_short_conv,
                allow_neg_eigval=config.allow_neg_eigval,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
                mup=config.mup,
                hidden_size_base=config.hidden_size_base,
                chunk_size=getattr(config, "gdn_chunk_size", 64),
            )

        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = TorchSwiGLUMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, Cache | None]:
        # Attn sub-block (pre-norm + residual)
        h = self.attn_norm(hidden_states)
        h, _, past_key_values = self.attn(
            hidden_states=h,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = hidden_states + h

        # MLP sub-block (pre-norm + residual)
        h = self.mlp_norm(hidden_states)
        h = self.mlp(h, **kwargs)
        hidden_states = hidden_states + h

        return hidden_states, None, past_key_values


@dataclass
class GPTConfig(PretrainedConfig):
    attn_mode: str = "chunk"
    expand_v: float = 2.0
    use_gate: bool = True
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    conv_size: int = 4
    num_v_heads: int | None = None
    hidden_ratio: int | None = 4
    intermediate_size: int | None = None
    hidden_act: str = "swish"
    norm_eps: float = 1e-6
    attn: dict | None = None
    linear_attn_layers: list[int] = None
    gdn_chunk_size: int = 64        # NEW: knob for the torch chunked GDR kernel

    use_cache: bool = True
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    fuse_norm: bool = False          # forced False on Trainium
    fuse_swiglu: bool = False        # forced False on Trainium
    mup: bool = False
    mymup: bool = False
    hidden_size_base: int = 768
    embedding_init_std: float = 0.02
    hidden_init_std_factor: float = 0.5
    embedding_lr_multiplier: float = 1.0
    model_type = "nanogpt"
    vocab_size: int = 50304
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    hidden_size: int = 768
    head_dim: int = 96
    block_size: int = 1024
    bias: bool = False

    def __init__(self, **kwargs: Any) -> None:
        raw = dict(kwargs)
        super().__init__(**validate_pretrained_config_kwargs(type(self), raw))
        if self.num_attention_heads == -1:
            self.num_attention_heads = int(self.hidden_size * 3 // (self.head_dim * 4))
        if self.head_dim == -1:
            self.head_dim = int(self.hidden_size * 0.75 // self.num_attention_heads)
        # Force the FLA-fused paths off on Trainium.
        self.fuse_norm = False
        self.fuse_swiglu = False
        self.attn = {
            "num_kv_heads": self.num_attention_heads,
            "qkv_bias": self.bias,
            "window_size": None,
            "rope_theta": 10000,
            "num_attention_heads": self.num_attention_heads,
            "layers": self.linear_attn_layers if self.linear_attn_layers is not None else [],
        }


class GPT(PreTrainedModel):

    config_class = GPTConfig
    base_model_prefix = "nanogpt"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _supports_cache_class = True

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_size),
                h=nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]),
            )
        )
        self.ln_f = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.mup = config.mup
        self.config = config
        self.post_init()
        if config.tie_word_embeddings:
            self.tie_weights()

    def post_init(self) -> None:
        super().post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.transformer.wte = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def tie_weights(self, *args: Any, **kwargs: Any) -> None:
        self.transformer.wte.weight = self.lm_head.weight

    def forward(
        self,
        idx: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        return_logits: bool = True,
        output_all_seq: bool = False,
        *,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[PastKeyValue, ...] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor | None, torch.Tensor | None]:
        del position_ids, cache_position, kwargs

        if (idx is None) == (input_ids is None):
            raise ValueError("Exactly one of `idx` or `input_ids` must be provided.")
        if idx is None:
            idx = input_ids
        if labels is not None and targets is not None:
            raise ValueError("Only one of `labels` or `targets` can be provided.")
        if targets is None:
            targets = labels

        use_cache_flag = bool(use_cache) if use_cache is not None else False
        return_dict_flag = bool(return_dict) if return_dict is not None else False
        output_hidden_states_flag = bool(output_hidden_states) if output_hidden_states is not None else False

        # NB: we deliberately do NOT collapse the attention_mask via .item() here,
        # because .item() forces a host/device sync on XLA. If you have an all-ones
        # mask, pass None instead.

        x = self.transformer.wte(idx)
        hidden_states: tuple[torch.Tensor, ...] | None = (x,) if output_hidden_states_flag else None

        if use_cache_flag and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values) if past_key_values is not None else Cache()

        for layer_idx, block in enumerate(self.transformer.h):
            x, _, past_key_values = block(
                x,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache_flag,
            )
            if output_hidden_states_flag:
                assert hidden_states is not None
                hidden_states = (*hidden_states, x)

        x = self.ln_f(x)

        logits_scale = 1.0
        if getattr(self.config, "mup", False):
            logits_scale = float(self.config.hidden_size_base) / float(self.config.hidden_size)

        if targets is not None:
            logits = self.lm_head(x).float() * logits_scale
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            loss = None
            if output_all_seq or return_dict_flag:
                logits = self.lm_head(x) * logits_scale
            else:
                logits = self.lm_head(x[:, [-1], :]).float() * logits_scale

        if not return_logits:
            logits = None
        if not return_dict_flag:
            return logits, loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=None,
        )

    def _init_weights(self, module: nn.Module) -> None:
        mup = getattr(self.config, "mup", False)
        if mup:
            hidden_std = float(getattr(self.config, "hidden_init_std_factor", 0.5)) / math.sqrt(float(self.config.hidden_size))
            emb_std = float(getattr(self.config, "embedding_init_std", 0.02))
        else:
            hidden_std = float(self.config.initializer_range)
            emb_std = hidden_std

        if isinstance(module, GatedDeltaNet) and next(module.parameters()).device.type != "meta":
            with torch.no_grad():
                module.A_log.copy_(nn.init.uniform_(module.A_log, a=0, b=16).log())
                module.A_log._no_weight_decay = True
                dt = torch.exp(
                    nn.init.uniform_(module.dt_bias) * (math.log(0.1) - math.log(0.001)) + math.log(0.001),
                ).clamp(min=1e-4)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                module.dt_bias.copy_(inv_dt)
                module.dt_bias._no_weight_decay = True
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            if "o_proj" in module._get_name().lower():
                module_std = hidden_std
                nn.init.normal_(module.weight, mean=0.0, std=module_std)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=hidden_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=emb_std)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def crop_block_size(self, block_size: int) -> None:
        block_size_int = int(block_size)
        if block_size_int <= 0:
            raise ValueError(f"block_size must be a positive integer, got {block_size_int}.")
        current = getattr(self.config, "block_size", None)
        if isinstance(current, int) and block_size_int > int(current):
            raise ValueError(f"block_size must be <= {int(current)} to crop, got {block_size_int}.")
        setattr(self.config, "block_size", block_size_int)

    def estimate_mfu(self, fwdbwd_per_iter: float, dt: float) -> float:
        """MFU estimate (A100 BF16 peak as the denominator, kept for parity with the original)."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 BF16 peak
        return flops_achieved / flops_promised

    def get_num_params(self, non_embedding: bool = True) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, save_directory: str) -> None:  # type: ignore[override]
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any) -> Any:
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        if isinstance(model, GPT) and config.tie_word_embeddings:
            model.tie_weights()
        return model
