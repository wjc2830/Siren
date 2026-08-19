from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import torch
from torch import Tensor, nn

from .config import SirenConfig
from .layers import CrossTransformerBlock, RMSNorm, TransformerBlock, sinusoidal_positions


@dataclass(frozen=True)
class DacEmbeddingState:
    """Frozen parameters for one official DAC decode-embedding mapping."""

    codebook: Tensor
    out_weight: Tensor
    out_bias: Tensor

    def __post_init__(self) -> None:
        expected = {
            "codebook": (1024, 8),
            "out_weight": (1024, 8),
            "out_bias": (1024,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(f"DAC {name} must have shape {shape}")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"DAC {name} must be a finite floating-point tensor")


class DacCodebookLayer(nn.Module):
    """One frozen DAC decode embedding followed by a trainable projector."""

    def __init__(self, config: SirenConfig) -> None:
        super().__init__()
        self.codebook = nn.Embedding(config.codebook_size, config.dac_code_dim)
        self.dac_out = nn.Linear(config.dac_code_dim, config.dac_feature_dim, bias=True)
        self.projector = nn.Linear(config.dac_feature_dim, config.dim, bias=True)
        self.codebook.weight.requires_grad_(False)
        self.dac_out.weight.requires_grad_(False)
        self.dac_out.bias.requires_grad_(False)

    def inject_embedding(self, state: DacEmbeddingState) -> None:
        parameters = (
            ("codebook", self.codebook.weight, state.codebook),
            ("out_weight", self.dac_out.weight, state.out_weight),
            ("out_bias", self.dac_out.bias, state.out_bias),
        )
        with torch.no_grad():
            for name, target, source in parameters:
                if target is None or tuple(source.shape) != tuple(target.shape):
                    expected = None if target is None else tuple(target.shape)
                    raise ValueError(f"DAC {name} must match layer shape {expected}")
                target.copy_(source.to(device=target.device, dtype=target.dtype))
                target.requires_grad_(False)

    def forward(self, indices: Tensor) -> Tensor:
        return self.projector(self.dac_out(self.codebook(indices)))


class DacCodebookBank(nn.Module):
    def __init__(
        self,
        config: SirenConfig,
        *,
        embedding_states: Sequence[DacEmbeddingState] | None = None,
        allow_random_codebooks: bool = False,
    ) -> None:
        super().__init__()
        if embedding_states is None and not allow_random_codebooks:
            raise ValueError(
                "DAC embedding states are required; set allow_random_codebooks=True only for tests"
            )
        self.layers = nn.ModuleList([DacCodebookLayer(config) for _ in range(12)])
        if embedding_states is not None:
            self.inject_embeddings(embedding_states)

    def inject_embeddings(self, embedding_states: Sequence[DacEmbeddingState]) -> None:
        states = list(embedding_states)
        if len(states) != len(self.layers):
            raise ValueError("DAC embedding states must contain exactly 12 entries")
        for index, (module, state) in enumerate(zip(self.layers, states)):
            if not isinstance(state, DacEmbeddingState):
                raise TypeError(f"DAC embedding state {index} has an invalid type")
            layer = cast(DacCodebookLayer, module)
            layer.inject_embedding(state)

    def forward(self, codes: Tensor) -> list[Tensor]:
        if codes.ndim != 3 or codes.shape[1] != 12:
            raise ValueError("codes must have shape [batch, 12, time]")
        if codes.dtype not in (torch.int32, torch.int64):
            raise TypeError("codes must use an integer dtype")
        return [layer(codes[:, index]) for index, layer in enumerate(self.layers)]


class SirenExpert(nn.Module):
    """Predicts one adjacent DAC pair (start_codebook, start_codebook + 1)."""

    def __init__(
        self,
        config: SirenConfig,
        start_codebook: int,
        *,
        embedding_states: Sequence[DacEmbeddingState] | None = None,
        allow_random_codebooks: bool = False,
    ) -> None:
        super().__init__()
        if start_codebook not in range(0, 12, 2):
            raise ValueError("start_codebook must be one of 0, 2, 4, 6, 8, 10")
        self.config = config
        self.start_codebook = start_codebook
        self.codebooks = DacCodebookBank(
            config,
            embedding_states=embedding_states,
            allow_random_codebooks=allow_random_codebooks,
        )
        self.sos = nn.Parameter(torch.empty(1, 1, config.dim))
        self.prompt_summary = nn.Linear(config.prompt_dim, config.dim, bias=True)
        self.spatial_blocks = nn.ModuleList(
            [CrossTransformerBlock(config) for _ in range(config.spatial_layers)]
        )
        self.spatial_norm = RMSNorm(config.dim, config.norm_eps)
        self.factorized_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.factorized_layers)]
        )
        self.factorized_norm = RMSNorm(config.dim, config.norm_eps)
        self.heads = nn.ModuleList(
            [nn.Linear(config.dim, config.codebook_size, bias=False) for _ in range(2)]
        )
        self.apply(self._initialize)
        nn.init.normal_(self.sos, mean=0.0, std=config.initializer_range)
        if embedding_states is not None:
            self.codebooks.inject_embeddings(embedding_states)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _prompt_mean(self, prompt: Tensor, prompt_mask: Tensor | None) -> Tensor:
        if prompt_mask is None:
            return prompt.mean(dim=1, keepdim=True)
        weights = prompt_mask.to(dtype=prompt.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (prompt * weights).sum(dim=1, keepdim=True) / denominator

    def forward(
        self,
        codes: Tensor,
        prompt: Tensor,
        sequence_mask: Tensor | None = None,
        prompt_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, _, time = codes.shape
        if time > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if prompt.shape[-1] != self.config.prompt_dim:
            raise ValueError("prompt embedding dimension does not match config")
        if prompt.shape[1] > self.config.max_prompt_len:
            raise ValueError("prompt exceeds max_prompt_len")
        embeddings = self.codebooks(codes)
        previous = torch.stack(embeddings, dim=0).sum(dim=0)[:, :-1]
        temporal = torch.cat((self.sos.expand(batch, -1, -1), previous), dim=1)
        temporal_positions = sinusoidal_positions(
            time, self.config.dim, codes.device, temporal.dtype
        ).unsqueeze(0)
        for block in self.spatial_blocks:
            temporal = block(temporal + temporal_positions, prompt, sequence_mask, prompt_mask)
        temporal = self.spatial_norm(temporal)

        prompt_state = self.prompt_summary(self._prompt_mean(prompt, prompt_mask))
        prompt_state = prompt_state.expand(-1, time, -1)
        before_parts = embeddings[: self.start_codebook]
        before = (
            prompt_state
            if not before_parts
            else prompt_state + torch.stack(before_parts, dim=0).sum(dim=0)
        )
        after_first = before + embeddings[self.start_codebook]
        inter = temporal.reshape(batch * time, 1, self.config.dim)
        before = before.reshape(batch * time, 1, self.config.dim)
        after_first = after_first.reshape(batch * time, 1, self.config.dim)
        intra = torch.cat((inter, before, after_first), dim=1)
        intra_positions = sinusoidal_positions(
            3, self.config.dim, codes.device, intra.dtype
        ).unsqueeze(0)
        for block in self.factorized_blocks:
            intra = block(intra + intra_positions)
        intra = self.factorized_norm(intra).view(batch, time, 3, self.config.dim)
        return self.heads[0](intra[:, :, 1]), self.heads[1](intra[:, :, 2])
