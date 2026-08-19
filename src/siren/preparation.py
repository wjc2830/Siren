from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np
import torch
from torch import Tensor

AUDIO_SUFFIXES = {".flac", ".mp3", ".ogg", ".opus", ".wav"}


class SupportsEncode(Protocol):
    """Minimal DAC surface used by preparation, so tests can substitute a fake."""

    sample_rate: int

    def encode(self, audio: Tensor, *, sample_rate: int) -> Tensor: ...


class SupportsEncodeText(Protocol):
    """Minimal CLAP surface used by preparation, so tests can substitute a fake."""

    def encode_text(
        self, prompts: str, *, max_length: int | None = None
    ) -> tuple[Tensor, Tensor]: ...


class PreparationError(ValueError):
    pass


def read_source_manifest(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    """Read a JSONL manifest of ``{"audio": ..., "caption": ...}`` records."""
    manifest_path = Path(path).expanduser().resolve(strict=True)
    root = manifest_path.parent
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise PreparationError(f"invalid JSON at line {line_number}") from error
            if not isinstance(record, dict):
                raise PreparationError(f"line {line_number} must be a JSON object")
            audio = record.get("audio")
            caption = record.get("caption")
            if not isinstance(audio, str) or not audio:
                raise PreparationError(f"line {line_number} requires a string field 'audio'")
            if not isinstance(caption, str) or not caption.strip():
                raise PreparationError(f"line {line_number} requires a non-empty 'caption'")
            records.append(record)
    if not records:
        raise PreparationError("source manifest is empty")
    return root, records


def resolve_audio_path(root: Path, value: str) -> Path:
    """Resolve a relative audio path without letting it escape the manifest directory."""
    candidate = Path(value)
    if candidate.is_absolute():
        raise PreparationError("audio paths must be relative to the source manifest")
    if candidate.suffix.lower() not in AUDIO_SUFFIXES:
        raise PreparationError(f"unsupported audio suffix: {candidate.suffix!r}")
    resolved = (root / candidate).resolve(strict=True)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise PreparationError("audio path escapes the manifest directory") from error
    return resolved


def load_audio(path: Path, sample_rate: int) -> Tensor:
    """Load mono audio at ``sample_rate`` using soundfile; resampling is refused."""
    try:
        import soundfile
    except ImportError as error:
        raise RuntimeError(
            "soundfile is required to read audio; install the 'audio' extra"
        ) from error
    data, file_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if int(file_rate) != int(sample_rate):
        raise PreparationError(
            f"{path.name} is sampled at {file_rate} Hz but {sample_rate} Hz is required; "
            "resample it before preparation so codes stay comparable"
        )
    mono = np.asarray(data, dtype=np.float32).mean(axis=1)
    if mono.size == 0:
        raise PreparationError(f"{path.name} contains no samples")
    if not np.isfinite(mono).all():
        raise PreparationError(f"{path.name} contains non-finite samples")
    return torch.from_numpy(mono).unsqueeze(0)


def write_record_npz(
    path: Path,
    codes: np.ndarray,
    prompt_embeddings: np.ndarray,
) -> None:
    """Write one training record, validating the contract enforced by the dataset."""
    if codes.ndim != 2 or codes.shape[0] != 12:
        raise PreparationError("codes must have shape [12, time]")
    if prompt_embeddings.ndim != 2:
        raise PreparationError("prompt_embeddings must have shape [prompt_length, prompt_dim]")
    if codes.min() < 0 or codes.max() >= 1024:
        raise PreparationError("codes must be in [0, 1023]")
    if not np.isfinite(prompt_embeddings).all():
        raise PreparationError("prompt embeddings must be finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        codes=codes.astype(np.int64, copy=False),
        prompt_embeddings=prompt_embeddings.astype(np.float32, copy=False),
    )


def prepare_dataset(
    source_manifest: str | Path,
    output_dir: str | Path,
    *,
    codec: SupportsEncode,
    conditioner: SupportsEncodeText,
    max_prompt_len: int | None = None,
    max_seq_len: int | None = None,
    overwrite: bool = False,
) -> Iterator[dict[str, Any]]:
    """Tokenize audio and captions into NPZ records, yielding manifest entries.

    The caller writes the manifest, so a failure part-way through does not leave a
    manifest pointing at records that were never written.
    """
    root, records = read_source_manifest(source_manifest)
    destination = Path(output_dir).expanduser().resolve()
    data_dir = destination / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records):
        identifier = str(record.get("id", f"sample-{index:06d}"))
        if "/" in identifier or identifier in {"", ".", ".."}:
            raise PreparationError(f"record id {identifier!r} is not a safe file name")
        npz_path = data_dir / f"{identifier}.npz"
        if npz_path.exists() and not overwrite:
            raise PreparationError(f"{npz_path} already exists; pass overwrite=True to replace it")

        audio_path = resolve_audio_path(root, record["audio"])
        audio = load_audio(audio_path, codec.sample_rate)
        codes = codec.encode(audio.unsqueeze(0), sample_rate=codec.sample_rate)
        if codes.ndim == 3:
            codes = codes[0]
        if max_seq_len is not None and codes.shape[1] > max_seq_len:
            codes = codes[:, :max_seq_len]

        prompt, _mask = conditioner.encode_text(record["caption"], max_length=max_prompt_len)
        if prompt.ndim == 3:
            prompt = prompt[0]

        write_record_npz(
            npz_path,
            codes.detach().to(device="cpu").numpy(),
            prompt.detach().to(device="cpu", dtype=torch.float32).numpy(),
        )
        yield {
            "id": identifier,
            "npz": f"data/{npz_path.name}",
            "codes_key": "codes",
            "prompt_key": "prompt_embeddings",
        }


def write_manifest(path: str | Path, entries: list[dict[str, Any]]) -> Path:
    """Write a training manifest consumable by ``SirenJsonlDataset``."""
    if not entries:
        raise PreparationError("refusing to write an empty manifest")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return target


def synthesize_dataset(
    output_dir: str | Path,
    *,
    records: int = 4,
    time_steps: int = 16,
    prompt_len: int = 8,
    prompt_dim: int = 32,
    seed: int = 0,
    manifest_name: str = "train.jsonl",
) -> Path:
    """Build a random dataset that satisfies the data contract, for CPU tests only.

    The codes are noise, so the resulting loss is meaningless. This exists purely so the
    documented quickstart runs without shipping audio or weights in the repository.
    """
    if records <= 0 or time_steps <= 0 or prompt_len <= 0 or prompt_dim <= 0:
        raise PreparationError("synthetic dataset dimensions must be positive")
    destination = Path(output_dir).expanduser().resolve()
    data_dir = destination / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    entries: list[dict[str, Any]] = []
    for index in range(records):
        identifier = f"synthetic-{index:04d}"
        npz_path = data_dir / f"{identifier}.npz"
        write_record_npz(
            npz_path,
            rng.integers(0, 1024, size=(12, time_steps), dtype=np.int64),
            rng.standard_normal((prompt_len, prompt_dim)).astype(np.float32),
        )
        entries.append(
            {
                "id": identifier,
                "npz": f"data/{npz_path.name}",
                "codes_key": "codes",
                "prompt_key": "prompt_embeddings",
            }
        )
    return write_manifest(destination / manifest_name, entries)
