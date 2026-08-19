from __future__ import annotations

import math
import os
import random
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import SirenExpert

PRECISIONS = ("fp32", "bf16", "fp16")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch so a run can be reproduced."""
    if not 0 <= seed < 2**32:
        raise ValueError("seed must fit in an unsigned 32-bit integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_dataloader_worker(worker_id: int) -> None:
    """Give every DataLoader worker a distinct but derived seed."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def learning_rate_at(
    step: int,
    *,
    base_learning_rate: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_ratio: float = 0.1,
) -> float:
    """Linear warmup followed by cosine decay to ``base_learning_rate * min_ratio``.

    ``step`` is 0-based, so the first optimizer step uses ``step=0``.
    """
    if base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be positive")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps]")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be in [0, 1]")
    if step < warmup_steps:
        return base_learning_rate * (step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return base_learning_rate * min_ratio
    # step is 0-based, so the final step (total_steps - 1) must land exactly on the floor.
    progress = min((step - warmup_steps) / (decay_steps - 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_learning_rate * (min_ratio + (1.0 - min_ratio) * cosine)


def autocast_dtype(precision: str) -> torch.dtype | None:
    """Map a precision name to an autocast dtype, or ``None`` for full fp32."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


class Stage1Module(nn.Module):
    def __init__(self, model: SirenExpert, start_codebook: int | None = None) -> None:
        super().__init__()
        selected_start = model.start_codebook if start_codebook is None else start_codebook
        if selected_start != model.start_codebook:
            raise ValueError("start_codebook must match the supplied expert")
        self.model = model
        self.start_codebook = selected_start

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        return self.model(
            batch["codes"],
            batch["prompt_embeddings"],
            batch.get("sequence_mask"),
            batch.get("prompt_mask"),
        )

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs = self(batch)
        mask = batch.get("sequence_mask")
        total = torch.zeros((), device=batch["codes"].device)
        metrics: dict[str, Tensor] = {}
        predicted = 0
        for offset, logits in enumerate(outputs):
            codebook = self.start_codebook + offset
            target = batch["codes"][:, codebook]
            flat_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
            ).view_as(target)
            loss = (
                flat_loss.mean()
                if mask is None
                else (flat_loss * mask).sum() / mask.sum().clamp_min(1)
            )
            metrics[f"ce_codebook_{codebook}"] = loss.detach()
            total = total + loss
            predicted += 1
        metrics["loss"] = total / max(predicted, 1)
        return metrics

    def configure_optimizer(
        self,
        learning_rate: float = 3e-4,
        weight_decay: float = 5e-2,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.AdamW:
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        decay = [parameter for parameter in parameters if parameter.ndim >= 2]
        no_decay = [parameter for parameter in parameters if parameter.ndim < 2]
        return torch.optim.AdamW(
            (
                ({"params": decay, "weight_decay": weight_decay}),
                ({"params": no_decay, "weight_decay": 0.0}),
            ),
            lr=learning_rate,
            betas=betas,
        )

    @torch.no_grad()
    def validate(self, batches: Iterable[dict[str, Tensor]]) -> dict[str, float]:
        """Token-weighted mean of the training loss over a held-out loader."""
        was_training = self.training
        self.eval()
        totals: dict[str, float] = {}
        weight = 0.0
        try:
            for batch in batches:
                mask = batch.get("sequence_mask")
                tokens = float(batch["codes"].shape[0] * batch["codes"].shape[2])
                if mask is not None:
                    tokens = float(mask.sum().item())
                if tokens <= 0:
                    continue
                for key, value in self.compute_loss(batch).items():
                    totals[key] = totals.get(key, 0.0) + float(value.item()) * tokens
                weight += tokens
        finally:
            if was_training:
                self.train()
        if weight <= 0:
            raise ValueError("validation loader produced no tokens")
        return {f"val_{key}": value / weight for key, value in totals.items()}
