from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


class SirenJsonlDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        max_npz_bytes: int = 512 * 1024 * 1024,
        max_seq_len: int | None = None,
        max_prompt_len: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
        self.root = self.manifest_path.parent.resolve(strict=True)
        self.max_npz_bytes = max_npz_bytes
        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len
        self.records = self._read_manifest()

    def _read_manifest(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at line {line_number}") from error
                if not isinstance(record, dict) or not isinstance(record.get("npz"), str):
                    raise ValueError(
                        f"line {line_number} requires an object with string field 'npz'"
                    )
                records.append(record)
        if not records:
            raise ValueError("dataset manifest is empty")
        return records

    def _safe_path(self, value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute() or candidate.suffix.lower() != ".npz":
            raise ValueError("npz paths must be relative and end in .npz")
        resolved = (self.root / candidate).resolve(strict=True)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError("npz path escapes the manifest directory") from error
        if resolved.stat().st_size > self.max_npz_bytes:
            raise ValueError("npz file exceeds max_npz_bytes")
        try:
            with zipfile.ZipFile(resolved) as archive:
                members = archive.infolist()
                total_uncompressed = sum(member.file_size for member in members)
                if total_uncompressed > self.max_npz_bytes:
                    raise ValueError("npz uncompressed content exceeds max_npz_bytes")
                if any(
                    member.file_size > 0
                    and member.compress_size > 0
                    and member.file_size / member.compress_size > 1000
                    for member in members
                ):
                    raise ValueError("npz compression ratio exceeds safety limit")
        except zipfile.BadZipFile as error:
            raise ValueError("invalid npz archive") from error
        return resolved

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self._safe_path(record["npz"])
        codes_key = record.get("codes_key", "codes")
        prompt_key = record.get("prompt_key", "prompt_embeddings")
        if not isinstance(codes_key, str) or not isinstance(prompt_key, str):
            raise ValueError("array keys must be strings")
        with np.load(path, allow_pickle=False) as archive:
            if codes_key not in archive or prompt_key not in archive:
                raise KeyError(f"{path.name} must contain {codes_key!r} and {prompt_key!r}")
            codes = np.asarray(archive[codes_key])
            prompt = np.asarray(archive[prompt_key])
        if codes.ndim != 2 or codes.shape[0] != 12 or codes.dtype.kind not in "iu":
            raise ValueError("codes must be an integer array with shape [12, time]")
        if prompt.ndim != 2 or prompt.dtype.kind != "f":
            raise ValueError("prompt_embeddings must be a float array [prompt_length, prompt_dim]")
        if codes.size == 0 or prompt.size == 0:
            raise ValueError("arrays must be non-empty")
        if codes.nbytes + prompt.nbytes > self.max_npz_bytes:
            raise ValueError("uncompressed arrays exceed max_npz_bytes")
        if int(codes.min()) < 0 or int(codes.max()) >= 1024:
            raise ValueError("codes must be in [0, 1023]")
        if not np.isfinite(prompt).all():
            raise ValueError("prompt embeddings must be finite")
        if self.max_seq_len is not None and codes.shape[1] > self.max_seq_len:
            raise ValueError("code sequence exceeds max_seq_len")
        if self.max_prompt_len is not None and prompt.shape[0] > self.max_prompt_len:
            raise ValueError("prompt sequence exceeds max_prompt_len")
        return {
            "id": str(record.get("id", index)),
            "codes": torch.from_numpy(codes.astype(np.int64, copy=False)),
            "prompt_embeddings": torch.from_numpy(prompt.astype(np.float32, copy=False)),
        }


def siren_collate(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(samples)
    if not items:
        raise ValueError("cannot collate an empty batch")
    prompt_dim = items[0]["prompt_embeddings"].shape[1]
    if any(item["prompt_embeddings"].shape[1] != prompt_dim for item in items):
        raise ValueError("all prompt embedding dimensions must match")
    batch = len(items)
    max_time = max(item["codes"].shape[1] for item in items)
    max_prompt = max(item["prompt_embeddings"].shape[0] for item in items)
    codes = torch.zeros(batch, 12, max_time, dtype=torch.long)
    prompts = torch.zeros(batch, max_prompt, prompt_dim, dtype=torch.float32)
    sequence_mask = torch.zeros(batch, max_time, dtype=torch.bool)
    prompt_mask = torch.zeros(batch, max_prompt, dtype=torch.bool)
    for row, item in enumerate(items):
        time = item["codes"].shape[1]
        prompt_length = item["prompt_embeddings"].shape[0]
        codes[row, :, :time] = item["codes"]
        prompts[row, :prompt_length] = item["prompt_embeddings"]
        sequence_mask[row, :time] = True
        prompt_mask[row, :prompt_length] = True
    return {
        "ids": [item["id"] for item in items],
        "codes": codes,
        "prompt_embeddings": prompts,
        "sequence_mask": sequence_mask,
        "prompt_mask": prompt_mask,
    }
