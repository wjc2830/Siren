from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .model import DacEmbeddingState


class DacCodec:
    """Frozen Descript DAC 16 kHz adapter loaded from an explicit local checkpoint."""

    sample_rate = 16_000

    def __init__(self, model_path: str | Path, device: str | torch.device = "cpu") -> None:
        self.model_path = Path(model_path).expanduser().resolve(strict=True)
        self.device = torch.device(device)
        try:
            import dac
        except ImportError as error:
            raise RuntimeError(
                "descript-audio-codec is required for DAC operations; install it separately"
            ) from error
        self.model: Any = dac.DAC.load(str(self.model_path))
        configured_rate = getattr(
            self.model, "sample_rate", getattr(self.model, "sampling_rate", self.sample_rate)
        )
        if int(configured_rate) != self.sample_rate:
            raise ValueError("the supplied DAC checkpoint must be the official 16 kHz model")
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, audio: Tensor, *, sample_rate: int = sample_rate) -> Tensor:
        if sample_rate != self.sample_rate:
            raise ValueError("audio must be sampled at 16000 Hz")
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("audio must have shape [batch, time] or [batch, 1, time]")
        audio = audio.to(self.device, dtype=torch.float32)
        preprocess = getattr(self.model, "preprocess", None)
        if preprocess is not None:
            audio = preprocess(audio, self.sample_rate)
        encoded = self.model.encode(audio)
        if not isinstance(encoded, (tuple, list)) or len(encoded) < 2:
            raise RuntimeError("unexpected Descript DAC encode result")
        codes = encoded[1]
        if codes.ndim != 3 or codes.shape[1] != 12:
            raise ValueError("Descript DAC 16 kHz codes must have shape [batch, 12, time]")
        return codes.to(dtype=torch.long)

    @torch.no_grad()
    def decode(self, codes: Tensor) -> Tensor:
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        if codes.ndim != 3 or codes.shape[1] != 12:
            raise ValueError("codes must have shape [batch, 12, time]")
        codes = codes.to(self.device, dtype=torch.long)
        quantizer = getattr(self.model, "quantizer", None)
        if quantizer is None or not hasattr(quantizer, "from_codes"):
            raise RuntimeError("the supplied Descript DAC model cannot decode code tensors")
        quantized = quantizer.from_codes(codes)
        latent = quantized[0] if isinstance(quantized, (tuple, list)) else quantized
        audio = self.model.decode(latent)
        if audio.ndim not in (2, 3):
            raise RuntimeError("unexpected Descript DAC decode result")
        return audio

    def embedding_states(self) -> tuple[DacEmbeddingState, ...]:
        quantizer = getattr(self.model, "quantizer", None)
        quantizers = getattr(quantizer, "quantizers", None)
        if quantizers is None or len(quantizers) != 12:
            raise ValueError("the supplied DAC checkpoint must expose 12 residual quantizers")
        states: list[DacEmbeddingState] = []
        for index, residual_quantizer in enumerate(quantizers):
            codebook = getattr(residual_quantizer, "codebook", None)
            candidate_weight = getattr(codebook, "weight", None)
            if isinstance(codebook, nn.Embedding):
                codebook_weight = codebook.weight
            elif torch.is_tensor(codebook):
                codebook_weight = codebook
            elif torch.is_tensor(candidate_weight):
                codebook_weight = candidate_weight
            else:
                raise ValueError(f"DAC residual quantizer {index} has no tensor codebook")
            if tuple(codebook_weight.shape) == (8, 1024):
                codebook_weight = codebook_weight.transpose(0, 1)

            out_proj = getattr(residual_quantizer, "out_proj", None)
            out_weight = getattr(out_proj, "weight", None)
            out_bias = getattr(out_proj, "bias", None)
            if not torch.is_tensor(out_weight) or not torch.is_tensor(out_bias):
                raise ValueError(
                    f"DAC residual quantizer {index} must expose out_proj weight and bias"
                )
            if tuple(out_weight.shape) == (1024, 8, 1):
                out_weight = out_weight.squeeze(-1)

            try:
                state = DacEmbeddingState(
                    codebook=codebook_weight.detach().to(device="cpu", dtype=torch.float32).clone(),
                    out_weight=out_weight.detach().to(device="cpu", dtype=torch.float32).clone(),
                    out_bias=out_bias.detach().to(device="cpu", dtype=torch.float32).clone(),
                )
            except ValueError as error:
                raise ValueError(f"invalid DAC residual quantizer {index}: {error}") from error
            states.append(state)
        return tuple(states)
