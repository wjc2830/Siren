from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import SirenConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight


def _round_up(value: int, multiple: int) -> int:
    return multiple * ((value + multiple - 1) // multiple)


class SwiGLU(nn.Module):
    def __init__(self, config: SirenConfig) -> None:
        super().__init__()
        hidden = int(8 * config.dim / 3)
        if config.ffn_multiplier is not None:
            hidden = int(hidden * config.ffn_multiplier)
        hidden = _round_up(hidden, config.ffn_multiple_of)
        self.gate_value = nn.Linear(config.dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.gate_value(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * value))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: SirenConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        qkv_dim = (config.n_heads + 2 * config.n_kv_heads) * self.head_dim
        self.qkv = nn.Linear(config.dim, qkv_dim, bias=False)
        self.output = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = config.dropout

    def forward(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        batch, length, _ = x.shape
        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = self.qkv(x).split((self.n_heads * self.head_dim, kv_dim, kv_dim), dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        repeat = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        attention_mask = None
        is_causal = True
        if valid_mask is not None:
            if valid_mask.shape != (batch, length):
                raise ValueError("valid_mask must have shape [batch, length]")
            causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
            attention_mask = causal.view(1, 1, length, length) & valid_mask[:, None, None, :].bool()
            is_causal = False
        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        result = result.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.output(result)


class CrossAttention(nn.Module):
    def __init__(self, config: SirenConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.query = nn.Linear(config.dim, config.dim, bias=True)
        self.key_value = nn.Linear(config.prompt_dim, 2 * config.dim, bias=True)
        self.output = nn.Linear(config.dim, config.dim, bias=True)
        self.dropout = config.dropout

    def forward(self, x: Tensor, prompt: Tensor, prompt_mask: Tensor | None = None) -> Tensor:
        batch, query_len, dim = x.shape
        if prompt.ndim != 3 or prompt.shape[0] != batch:
            raise ValueError("prompt must have shape [batch, prompt_length, prompt_dim]")
        prompt_len = prompt.shape[1]
        q = self.query(x).view(batch, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.key_value(prompt).chunk(2, dim=-1)
        k = k.view(batch, prompt_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, prompt_len, self.n_heads, self.head_dim).transpose(1, 2)
        mask = None
        if prompt_mask is not None:
            if prompt_mask.shape != (batch, prompt_len):
                raise ValueError("prompt_mask must have shape [batch, prompt_length]")
            mask = prompt_mask[:, None, None, :].bool()
        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(result.transpose(1, 2).contiguous().view(batch, query_len, dim))


class TransformerBlock(nn.Module):
    def __init__(self, config: SirenConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.attention_norm(x), valid_mask)
        return x + self.feed_forward(self.ffn_norm(x))


class CrossTransformerBlock(nn.Module):
    def __init__(self, config: SirenConfig) -> None:
        super().__init__()
        self.cross_norm = RMSNorm(config.dim, config.norm_eps)
        self.cross_attention = CrossAttention(config)
        self.self_norm = RMSNorm(config.dim, config.norm_eps)
        self.self_attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(
        self,
        x: Tensor,
        prompt: Tensor,
        valid_mask: Tensor | None = None,
        prompt_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.cross_attention(self.cross_norm(x), prompt, prompt_mask)
        x = x + self.self_attention(self.self_norm(x), valid_mask)
        return x + self.feed_forward(self.ffn_norm(x))


def sinusoidal_positions(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    result = torch.zeros(length, dim, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * frequency)
    result[:, 1::2] = torch.cos(position * frequency[: result[:, 1::2].shape[1]])
    return result.to(dtype=dtype)
