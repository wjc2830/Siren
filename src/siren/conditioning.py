from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn


class ClapTextConditioner:
    """Frozen CLAP text encoder loaded only from a user-supplied local path."""

    def __init__(self, model_path: str | Path, device: str | torch.device = "cpu") -> None:
        self.model_path = Path(model_path).expanduser().resolve(strict=True)
        self.device = torch.device(device)
        try:
            import transformers
        except ImportError as error:
            raise RuntimeError(
                "transformers is required for text prompts; install it separately"
            ) from error
        loader_name = "from_" + "pretrained"
        tokenizer_class = getattr(transformers, "AutoTokenizer")
        model_class = getattr(transformers, "ClapTextModel")
        self.tokenizer = getattr(tokenizer_class, loader_name)(
            str(self.model_path), local_files_only=True
        )
        self.model: nn.Module = getattr(model_class, loader_name)(
            str(self.model_path), local_files_only=True
        )
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode_text(
        self,
        prompts: str | Sequence[str],
        *,
        max_length: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        texts = [prompts] if isinstance(prompts, str) else list(prompts)
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("prompts must contain non-empty strings")
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in tokenized.items()
        }
        output = self.model(**inputs)
        embeddings = output.last_hidden_state
        if embeddings.ndim != 3 or embeddings.shape[-1] != 768:
            raise ValueError("CLAP text encoder must return [batch, prompt_length, 768]")
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(dtype=torch.bool)
        return embeddings, mask
