from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class SirenConfig:
    dim: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 16
    spatial_layers: int = 14
    factorized_layers: int = 2
    prompt_dim: int = 768
    num_codebooks: int = 12
    num_experts: int = 6
    codebook_size: int = 1024
    dac_code_dim: int = 8
    dac_feature_dim: int = 1024
    max_seq_len: int = 2048
    max_prompt_len: int = 128
    ffn_multiple_of: int = 256
    ffn_multiplier: float | None = None
    norm_eps: float = 1e-5
    dropout: float = 0.1
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        positive = (
            "dim",
            "n_heads",
            "n_kv_heads",
            "spatial_layers",
            "factorized_layers",
            "prompt_dim",
            "num_codebooks",
            "num_experts",
            "codebook_size",
            "dac_code_dim",
            "dac_feature_dim",
            "max_seq_len",
            "max_prompt_len",
            "ffn_multiple_of",
            "initializer_range",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.num_codebooks != 12 or self.num_experts != 6:
            raise ValueError("Stage 1 requires exactly 12 codebooks and 6 experts")
        if self.codebook_size != 1024:
            raise ValueError("Stage 1 DAC heads require 1024 classes")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

    @classmethod
    def tiny(cls) -> "SirenConfig":
        return cls(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            spatial_layers=2,
            factorized_layers=1,
            prompt_dim=32,
            dac_feature_dim=64,
            max_seq_len=32,
            max_prompt_len=16,
            ffn_multiple_of=32,
            dropout=0.0,
        )

    @classmethod
    def paper_1_6b(cls) -> "SirenConfig":
        return cls(spatial_layers=14, factorized_layers=2)

    @classmethod
    def paper_3_1b(cls) -> "SirenConfig":
        return cls(spatial_layers=24, factorized_layers=8)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SirenConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**value)

    @classmethod
    def from_json(cls, path: str | Path) -> "SirenConfig":
        resolved = Path(path).expanduser().resolve(strict=True)
        with resolved.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("config must be a JSON object")
        value = {key: item for key, item in value.items() if not key.startswith("_")}
        return cls.from_dict(value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
