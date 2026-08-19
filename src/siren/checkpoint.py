from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import SirenConfig
from .model import DacEmbeddingState, SirenExpert


class CheckpointManifestError(ValueError):
    pass


def _relative_local_file(root: Path, value: Any, suffix: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CheckpointManifestError("manifest paths must be non-empty strings")
    path = Path(value)
    if path.is_absolute() or "://" in value or path.suffix.lower() != suffix:
        raise CheckpointManifestError(f"path must be a relative local {suffix} file")
    resolved = (root / path).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise CheckpointManifestError("manifest path escapes its directory") from error
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    if manifest_path.suffix.lower() != ".json":
        raise CheckpointManifestError("checkpoint manifest must be JSON")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise CheckpointManifestError("unsupported checkpoint manifest")
    return manifest_path, manifest


def _expert_entries(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    experts = manifest.get("experts")
    if not isinstance(experts, list):
        raise CheckpointManifestError("experts must be a list")
    by_start: dict[int, dict[str, Any]] = {}
    for entry in experts:
        if not isinstance(entry, dict) or not isinstance(entry.get("start_codebook"), int):
            raise CheckpointManifestError("each expert requires integer start_codebook")
        start = entry["start_codebook"]
        if start not in range(0, 12, 2):
            raise CheckpointManifestError("invalid expert start_codebook")
        if start in by_start:
            raise CheckpointManifestError("duplicate expert entry")
        by_start[start] = entry
    return by_start


def _load_expert_state(
    model: SirenExpert,
    root: Path,
    entry: dict[str, Any],
    *,
    allow_random_init: bool,
) -> bool:
    try:
        weights = _relative_local_file(root, entry.get("weights"), ".safetensors")
    except (FileNotFoundError, CheckpointManifestError):
        if allow_random_init:
            return False
        raise
    expected_hash = entry.get("sha256")
    if not allow_random_init and (not isinstance(expected_hash, str) or len(expected_hash) != 64):
        raise CheckpointManifestError("production expert entries require sha256")
    if expected_hash is not None and _sha256(weights) != expected_hash.lower():
        raise CheckpointManifestError(f"sha256 mismatch for expert {model.start_codebook}")
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("safetensors is required to load checkpoints") from error
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    return True


def load_expert_checkpoint_manifest(
    path: str | Path,
    start_codebook: int,
    *,
    allow_random_init: bool = False,
    embedding_states: Sequence[DacEmbeddingState] | None = None,
    allow_random_codebooks: bool = False,
) -> SirenExpert:
    if start_codebook not in range(0, 12, 2):
        raise ValueError("start_codebook must be one of 0, 2, 4, 6, 8, 10")
    manifest_path, manifest = read_checkpoint_manifest(path)
    root = manifest_path.parent.resolve(strict=True)
    config_path = _relative_local_file(root, manifest.get("config"), ".json")
    entry = _expert_entries(manifest).get(start_codebook)
    if entry is None and not allow_random_init:
        raise CheckpointManifestError(f"manifest has no expert {start_codebook}")
    model = SirenExpert(
        SirenConfig.from_json(config_path),
        start_codebook,
        embedding_states=embedding_states,
        allow_random_codebooks=allow_random_codebooks or embedding_states is None,
    )
    loaded = entry is not None and _load_expert_state(
        model, root, entry, allow_random_init=allow_random_init
    )
    if not loaded and embedding_states is None and not allow_random_codebooks:
        raise ValueError(
            "DAC embedding states are required when the expert checkpoint is not loaded"
        )
    return model


def build_expert(
    start_codebook: int,
    *,
    config_path: str | Path | None = None,
    checkpoint_manifest: str | Path | None = None,
    allow_random_init: bool = False,
    embedding_states: Sequence[DacEmbeddingState] | None = None,
    allow_random_codebooks: bool = False,
) -> SirenExpert:
    if checkpoint_manifest is not None:
        return load_expert_checkpoint_manifest(
            checkpoint_manifest,
            start_codebook,
            allow_random_init=allow_random_init,
            embedding_states=embedding_states,
            allow_random_codebooks=allow_random_codebooks,
        )
    if not allow_random_init:
        raise FileNotFoundError(
            "checkpoint manifest is required; use allow_random_init=True only for explicit training"
        )
    if config_path is None:
        raise ValueError("config_path is required for random initialization")
    return SirenExpert(
        SirenConfig.from_json(config_path),
        start_codebook,
        embedding_states=embedding_states,
        allow_random_codebooks=allow_random_codebooks,
    )


TRAINING_STATE_VERSION = 1


def _flatten_optimizer_state(
    optimizer: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split AdamW state into flat tensors plus a JSON-safe scalar sidecar.

    ``torch.save`` is deliberately avoided so resumption never unpickles data.
    """
    state = optimizer.state_dict()
    tensors: dict[str, Any] = {}
    scalars: dict[str, Any] = {"param_groups": [], "state": {}}
    for group in state["param_groups"]:
        scalars["param_groups"].append({key: value for key, value in group.items()})
    for index, entries in state["state"].items():
        recorded: dict[str, Any] = {}
        for key, value in entries.items():
            if torch.is_tensor(value):
                if value.numel() == 1 and value.ndim == 0:
                    recorded[key] = {"scalar": value.item(), "tensor": True}
                else:
                    name = f"opt.{index}.{key}"
                    tensors[name] = value.detach().to(device="cpu").contiguous()
                    recorded[key] = {"ref": name}
            else:
                recorded[key] = {"scalar": value, "tensor": False}
        scalars["state"][str(index)] = recorded
    return tensors, scalars


def _restore_optimizer_state(
    optimizer: Any,
    tensors: dict[str, Any],
    scalars: dict[str, Any],
) -> None:
    rebuilt: dict[int, dict[str, Any]] = {}
    for index, entries in scalars.get("state", {}).items():
        restored: dict[str, Any] = {}
        for key, record in entries.items():
            if "ref" in record:
                name = record["ref"]
                if name not in tensors:
                    raise CheckpointManifestError(f"training state is missing tensor {name!r}")
                restored[key] = tensors[name]
            elif record.get("tensor"):
                restored[key] = torch.tensor(record["scalar"])
            else:
                restored[key] = record["scalar"]
        rebuilt[int(index)] = restored
    optimizer.load_state_dict({"param_groups": scalars.get("param_groups", []), "state": rebuilt})


def save_training_state(
    path: str | Path,
    *,
    model: SirenExpert,
    optimizer: Any,
    step: int,
    epoch: int,
    start_codebook: int,
) -> Path:
    """Write a resumable training state as safetensors plus a JSON sidecar."""
    target = Path(path).expanduser()
    if target.suffix.lower() != ".safetensors":
        raise ValueError("training state must be written to a .safetensors path")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("safetensors is required to save checkpoints") from error

    optimizer_tensors, optimizer_scalars = _flatten_optimizer_state(optimizer)
    payload = {
        f"model.{name}": tensor.detach().to(device="cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    payload.update(optimizer_tensors)
    save_file(payload, str(target))
    sidecar = target.with_suffix(".state.json")
    sidecar.write_text(
        json.dumps(
            {
                "format_version": TRAINING_STATE_VERSION,
                "step": int(step),
                "epoch": int(epoch),
                "start_codebook": int(start_codebook),
                "optimizer": optimizer_scalars,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def load_training_state(
    path: str | Path,
    *,
    model: SirenExpert,
    optimizer: Any,
) -> tuple[int, int]:
    """Restore model and optimizer in place; returns ``(step, epoch)``."""
    target = Path(path).expanduser().resolve(strict=True)
    if target.suffix.lower() != ".safetensors":
        raise ValueError("training state must be a .safetensors path")
    sidecar = target.with_suffix(".state.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing training state sidecar: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("format_version") != TRAINING_STATE_VERSION:
        raise CheckpointManifestError("unsupported training state version")
    if metadata.get("start_codebook") != model.start_codebook:
        raise CheckpointManifestError(
            f"training state belongs to expert {metadata.get('start_codebook')}, "
            f"not {model.start_codebook}"
        )
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("safetensors is required to load checkpoints") from error

    payload = load_file(str(target), device="cpu")
    model_state = {
        name[len("model.") :]: tensor
        for name, tensor in payload.items()
        if name.startswith("model.")
    }
    model.load_state_dict(model_state, strict=True)
    optimizer_tensors = {
        name: tensor for name, tensor in payload.items() if name.startswith("opt.")
    }
    _restore_optimizer_state(optimizer, optimizer_tensors, metadata.get("optimizer", {}))
    return int(metadata["step"]), int(metadata["epoch"])
