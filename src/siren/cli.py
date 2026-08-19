from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import (
    build_expert,
    load_training_state,
    save_training_state,
)
from .codec import DacCodec
from .conditioning import ClapTextConditioner
from .data import SirenJsonlDataset, siren_collate
from .preparation import prepare_dataset, synthesize_dataset, write_manifest
from .training import (
    PRECISIONS,
    Stage1Module,
    autocast_dtype,
    learning_rate_at,
    seed_dataloader_worker,
    seed_everything,
)

EXPERT_STARTS = (0, 2, 4, 6, 8, 10)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def _seed(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed < 2**32:
        raise argparse.ArgumentTypeError("seed must fit in an unsigned 32-bit integer")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def _safetensors_path(value: str) -> str:
    if Path(value).suffix.lower() != ".safetensors":
        raise argparse.ArgumentTypeError("output must end in .safetensors")
    return value


def _device_list(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in value.split(",") if part.strip())
    if not devices:
        raise argparse.ArgumentTypeError("devices must be a non-empty comma-separated list")
    return devices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dac_codec(arguments: argparse.Namespace) -> DacCodec | None:
    path = getattr(arguments, "dac_model", None)
    return None if path is None else DacCodec(path, arguments.device)


def _expert(arguments: argparse.Namespace, codec: DacCodec | None = None):
    return build_expert(
        arguments.expert_start,
        config_path=arguments.config,
        checkpoint_manifest=arguments.checkpoint_manifest,
        allow_random_init=arguments.allow_random_init,
        embedding_states=None if codec is None else codec.embedding_states(),
        allow_random_codebooks=arguments.allow_random_codebooks,
    )


def _dataset(path: str, model) -> SirenJsonlDataset:
    return SirenJsonlDataset(
        path,
        max_seq_len=model.config.max_seq_len,
        max_prompt_len=model.config.max_prompt_len,
    )


def _loader(path: str, model, arguments: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    generator = None
    if shuffle and arguments.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(arguments.seed)
    return DataLoader(
        _dataset(path, model),
        batch_size=arguments.batch_size,
        shuffle=shuffle,
        num_workers=arguments.num_workers,
        collate_fn=siren_collate,
        generator=generator,
        worker_init_fn=seed_dataloader_worker if arguments.seed is not None else None,
    )


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def _save_expert_weights(model, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("safetensors is required to save checkpoints") from error
    state = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(output))


def command_train(arguments: argparse.Namespace) -> None:
    if arguments.seed is not None:
        seed_everything(arguments.seed)
    device = torch.device(arguments.device)
    codec = _dac_codec(arguments)
    expert = _expert(arguments, codec)
    stage = Stage1Module(expert, arguments.expert_start).to(device)
    loader = _loader(arguments.dataset, stage.model, arguments, shuffle=True)
    validation = (
        None
        if arguments.val_dataset is None
        else _loader(arguments.val_dataset, stage.model, arguments, shuffle=False)
    )
    optimizer = stage.configure_optimizer(arguments.learning_rate, arguments.weight_decay)

    steps = 0
    epoch = 0
    if arguments.resume is not None:
        steps, epoch = load_training_state(arguments.resume, model=stage.model, optimizer=optimizer)
        print(json.dumps({"event": "resumed", "step": steps, "epoch": epoch}))

    batches_per_epoch = len(loader)
    if batches_per_epoch == 0:
        raise RuntimeError("training dataset produced no batches")
    total_steps = (
        arguments.max_steps
        if arguments.max_steps is not None
        else batches_per_epoch * arguments.epochs
    )
    warmup_steps = min(arguments.warmup_steps, total_steps)
    amp_dtype = autocast_dtype(arguments.precision)
    scaler = torch.amp.GradScaler(
        device.type, enabled=amp_dtype is torch.float16 and device.type == "cuda"
    )
    output = Path(arguments.output).expanduser().resolve()

    stage.train()
    stop = False
    validated_at = -1
    while not stop and (arguments.max_steps is not None or epoch < arguments.epochs):
        for batch_index, batch in enumerate(loader):
            tensors = _to_device(batch, device)
            learning_rate = learning_rate_at(
                steps,
                base_learning_rate=arguments.learning_rate,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_ratio=arguments.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                metrics = stage.compute_loss(tensors)
            loss = metrics["loss"]
            scaler.scale(loss).backward()
            grad_norm = None
            if arguments.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(stage.parameters(), arguments.grad_clip).item()
                )
            scaler.step(optimizer)
            scaler.update()
            steps += 1

            record = {
                "epoch": epoch,
                "batch": batch_index,
                "step": steps,
                "expert_start": arguments.expert_start,
                "loss": float(loss.item()),
                "learning_rate": learning_rate,
            }
            if grad_norm is not None:
                record["grad_norm"] = grad_norm
            print(json.dumps(record))

            if validation is not None and arguments.val_every and steps % arguments.val_every == 0:
                validation_metrics = stage.validate(_to_device(item, device) for item in validation)
                validated_at = steps
                print(json.dumps({"step": steps, **validation_metrics}, sort_keys=True))

            if arguments.save_every and steps % arguments.save_every == 0:
                save_training_state(
                    output.with_name(f"{output.stem}.step{steps}.safetensors"),
                    model=stage.model,
                    optimizer=optimizer,
                    step=steps,
                    epoch=epoch,
                    start_codebook=arguments.expert_start,
                )

            if arguments.max_steps is not None and steps >= arguments.max_steps:
                stop = True
                break
        epoch += 1

    if validation is not None and validated_at != steps:
        final_validation = stage.validate(_to_device(item, device) for item in validation)
        print(json.dumps({"step": steps, **final_validation}, sort_keys=True))

    _save_expert_weights(stage.model, output)
    if arguments.save_every:
        save_training_state(
            output.with_name(f"{output.stem}.final.safetensors"),
            model=stage.model,
            optimizer=optimizer,
            step=steps,
            epoch=epoch,
            start_codebook=arguments.expert_start,
        )


def _train_all_commands(arguments: argparse.Namespace) -> list[list[str]]:
    commands: list[list[str]] = []
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    for start in EXPERT_STARTS:
        command = [
            sys.executable,
            "-m",
            "siren",
            "train",
            "--expert-start",
            str(start),
            "--dataset",
            arguments.dataset,
            "--output",
            str(output_dir / f"expert_{start}.safetensors"),
            "--device",
            arguments.device,
            "--batch-size",
            str(arguments.batch_size),
            "--num-workers",
            str(arguments.num_workers),
            "--epochs",
            str(arguments.epochs),
            "--learning-rate",
            str(arguments.learning_rate),
            "--weight-decay",
            str(arguments.weight_decay),
            "--warmup-steps",
            str(arguments.warmup_steps),
            "--min-lr-ratio",
            str(arguments.min_lr_ratio),
            "--grad-clip",
            str(arguments.grad_clip),
            "--precision",
            arguments.precision,
            "--save-every",
            str(arguments.save_every),
            "--val-every",
            str(arguments.val_every),
        ]
        for option, value in (
            ("--config", arguments.config),
            ("--checkpoint-manifest", arguments.checkpoint_manifest),
            ("--dac-model", arguments.dac_model),
            ("--val-dataset", arguments.val_dataset),
        ):
            if value is not None:
                command.extend((option, value))
        if arguments.max_steps is not None:
            command.extend(("--max-steps", str(arguments.max_steps)))
        if arguments.seed is not None:
            command.extend(("--seed", str(arguments.seed + start)))
        if arguments.allow_random_init:
            command.append("--allow-random-init")
        if arguments.allow_random_codebooks:
            command.append("--allow-random-codebooks")
        commands.append(command)
    return commands


def _write_training_manifest(arguments: argparse.Namespace, output_dir: Path) -> None:
    if arguments.config is None:
        raise ValueError("--config is required with --launch to copy the public config")
    config_source = Path(arguments.config).expanduser().resolve(strict=True)
    if config_source.suffix.lower() != ".json":
        raise ValueError("training config must be a JSON file")
    config_output = output_dir / "config.json"
    if config_source != config_output.resolve():
        shutil.copyfile(config_source, config_output)

    experts = []
    for start in EXPERT_STARTS:
        weights = output_dir / f"expert_{start}.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"missing expert output: {weights}")
        experts.append(
            {
                "start_codebook": start,
                "weights": weights.name,
                "sha256": _sha256(weights),
            }
        )
    manifest = {
        "format_version": 1,
        "config": config_output.name,
        "experts": experts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def command_train_all(arguments: argparse.Namespace) -> None:
    commands = _train_all_commands(arguments)
    for command in commands:
        print(shlex.join(command))
    if arguments.devices is None:
        print(
            "warning: --devices was not provided; launching all experts would contend for "
            f"the same --device {arguments.device!r}",
            file=sys.stderr,
        )
    if not arguments.launch:
        return

    output_dir = Path(arguments.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for index, command in enumerate(commands):
        if arguments.devices is None:
            processes.append(subprocess.Popen(command))  # noqa: S603 - argv is constructed above
        else:
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = arguments.devices[index % len(arguments.devices)]
            processes.append(
                subprocess.Popen(command, env=environment)  # noqa: S603 - argv is constructed above
            )
    return_codes = [process.wait() for process in processes]
    failures = [code for code in return_codes if code != 0]
    if failures:
        raise RuntimeError(f"{len(failures)} expert training process(es) failed")
    _write_training_manifest(arguments, output_dir)


def command_prepare_data(arguments: argparse.Namespace) -> None:
    if arguments.dac_model is None:
        raise ValueError("--dac-model is required to tokenize audio")
    if arguments.clap_model is None:
        raise ValueError("--clap-model is required to encode captions")
    codec = DacCodec(arguments.dac_model, arguments.device)
    conditioner = ClapTextConditioner(arguments.clap_model, arguments.device)
    entries = []
    for entry in prepare_dataset(
        arguments.source,
        arguments.output_dir,
        codec=codec,
        conditioner=conditioner,
        max_prompt_len=arguments.max_prompt_len,
        max_seq_len=arguments.max_seq_len,
        overwrite=arguments.overwrite,
    ):
        entries.append(entry)
        print(json.dumps(entry, sort_keys=True))
    manifest = write_manifest(Path(arguments.output_dir) / arguments.manifest_name, entries)
    print(json.dumps({"manifest": str(manifest), "records": len(entries)}))


def command_make_tiny_dataset(arguments: argparse.Namespace) -> None:
    manifest = synthesize_dataset(
        arguments.output_dir,
        records=arguments.records,
        time_steps=arguments.time_steps,
        prompt_len=arguments.prompt_len,
        prompt_dim=arguments.prompt_dim,
        seed=arguments.seed if arguments.seed is not None else 0,
        manifest_name=arguments.manifest_name,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "records": arguments.records,
                "synthetic": True,
                "warning": "random codes; loss values are meaningless",
            },
            sort_keys=True,
        )
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config")
    parser.add_argument("--checkpoint-manifest")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--dac-model")
    parser.add_argument("--allow-random-codebooks", action="store_true")
    parser.add_argument("--device", default="cpu")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    _add_model_arguments(parser)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--val-dataset")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=_positive_int, default=1)
    parser.add_argument("--max-steps", type=_positive_int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--seed", type=_seed, help="seed Python, NumPy, Torch and DataLoader")
    parser.add_argument(
        "--warmup-steps",
        type=_non_negative_int,
        default=0,
        help="linear warmup before cosine decay",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=_ratio,
        default=0.1,
        help="cosine decay floor as a fraction of --learning-rate",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="global grad-norm clip; 0 disables clipping"
    )
    parser.add_argument("--precision", choices=PRECISIONS, default="fp32")
    parser.add_argument(
        "--save-every",
        type=_non_negative_int,
        default=0,
        help="write a resumable training state every N steps; 0 disables",
    )
    parser.add_argument("--val-every", type=_non_negative_int, default=0)
    parser.add_argument("--resume", help="resumable training state written by --save-every")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="siren")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    _add_training_arguments(train)
    train.add_argument("--expert-start", type=int, choices=EXPERT_STARTS, required=True)
    train.add_argument("--output", type=_safetensors_path, required=True)
    train.set_defaults(function=command_train)

    train_all = commands.add_parser("train-all")
    _add_training_arguments(train_all)
    train_all.add_argument("--output-dir", required=True)
    train_all.add_argument("--devices", type=_device_list)
    train_all.add_argument(
        "--launch",
        action="store_true",
        help="start the six independent commands after printing them",
    )
    train_all.set_defaults(function=command_train_all)

    prepare = commands.add_parser(
        "prepare-data",
        help="tokenize audio and captions into the NPZ + JSONL training format",
    )
    prepare.add_argument("--source", required=True, help="JSONL of {audio, caption} records")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--dac-model", required=True)
    prepare.add_argument("--clap-model", required=True)
    prepare.add_argument("--device", default="cpu")
    prepare.add_argument("--max-seq-len", type=_positive_int)
    prepare.add_argument("--max-prompt-len", type=_positive_int)
    prepare.add_argument("--manifest-name", default="train.jsonl")
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(function=command_prepare_data)

    tiny = commands.add_parser(
        "make-tiny-dataset",
        help="write a random CPU-test dataset; codes are noise and losses are meaningless",
    )
    tiny.add_argument("--output-dir", required=True)
    tiny.add_argument("--records", type=_positive_int, default=4)
    tiny.add_argument("--time-steps", type=_positive_int, default=16)
    tiny.add_argument("--prompt-len", type=_positive_int, default=8)
    tiny.add_argument("--prompt-dim", type=_positive_int, default=32)
    tiny.add_argument("--seed", type=_seed, default=0)
    tiny.add_argument("--manifest-name", default="train.jsonl")
    tiny.set_defaults(function=command_make_tiny_dataset)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
