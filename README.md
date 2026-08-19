# Siren

Stage 1 training code for **"Language Model Based Text-to-Audio Generation: Anti-Causally Aligned Collaborative Residual Transformers"** (EMNLP 2025 Main).

Juncheng Wang, Chao Xu, Cheng Yu, Zhe Hu, Haoyu Xie, Guoqi Yu, Lei Shang, Shujun Wang

[Paper](https://aclanthology.org/2025.emnlp-main.1322/)

## Scope

This is the supervised Stage 1 training code only: six isolated experts over the 12 DAC residual codebooks, conditioned on CLAP text embeddings. Expert `i` predicts codebooks `(2i, 2i+1)` and receives the accumulated embeddings of all earlier codebooks.

The paper's second mechanism — anti-causal alignment of the coarse codes via reinforcement learning — is **not** included, so the paper's reported numbers cannot be reproduced from this code alone. There is also no inference, sampling, or evaluation code here, and no checkpoints or weights.

## Install

```bash
pip install -e .
```

Add the `audio` extra only if you need to build data from raw audio:

```bash
pip install -e ".[audio]"
```

## Data

A JSONL manifest pointing at relative NPZ files, each containing integer `codes` of shape `[12, time]` in `[0, 1023]` and float `prompt_embeddings` of shape `[prompt_length, prompt_dim]`:

```json
{"id": "sample-0001", "npz": "data/sample-0001.npz"}
```

Build it from 16 kHz audio and captions using your own local Descript DAC 16 kHz and CLAP text checkpoints. The source manifest is JSONL of `{"id", "audio", "caption"}`:

```bash
siren prepare-data \
  --source data/source.jsonl --output-dir data/prepared \
  --dac-model /path/to/dac-16khz \
  --clap-model /path/to/clap-text-encoder \
  --device cuda --max-seq-len 300
```

Audio must already be 16 kHz; a mismatched rate is rejected rather than resampled, because codes from a different rate are not comparable. The paper crops 6-second caption-aligned segments, which is 300 frames at the DAC frame rate. `--max-seq-len` here truncates from the start, so do the cropping in your own source manifest if you need to match the paper.

## Train

One expert per process:

```bash
siren train --config configs/siren_1_6b.json \
  --allow-random-init --expert-start 0 --dac-model /path/to/dac-16khz \
  --dataset data/train.jsonl --val-dataset data/val.jsonl \
  --output outputs/expert_0.safetensors --device cuda \
  --seed 0 --precision bf16 --warmup-steps 2000 --grad-clip 1.0 \
  --save-every 1000 --val-every 1000
```

`--dac-model` injects and freezes the 12 DAC decode embeddings that produced your codes; only the projector and transformer train. The paper uses AdamW at `3e-4` with batch 24 per GPU.

Print the six per-expert commands, or launch them across GPUs:

```bash
siren train-all --config configs/siren_1_6b.json \
  --allow-random-init --dac-model /path/to/dac-16khz \
  --dataset data/train.jsonl --output-dir outputs/run-1 --seed 0 \
  --device cuda --devices 0,1,2,3,4,5 --launch
```

Without `--launch` this is a dry run that prints the commands. On success it writes `manifest.json` listing the six expert files with SHA-256 hashes.

### Key flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--seed` | unset | Seeds Python, NumPy, Torch, and data shuffling. Set it for reported runs. |
| `--warmup-steps` | `0` | Linear warmup before cosine decay. |
| `--min-lr-ratio` | `0.1` | Cosine floor as a fraction of the base LR, reached on the final step. |
| `--grad-clip` | `1.0` | Global grad-norm clip; `0` disables it. |
| `--precision` | `fp32` | `bf16` / `fp16` enable autocast. |
| `--save-every` | `0` | Write a resumable state every N steps. |
| `--resume` | unset | Resume from such a state. |
| `--max-steps` | unset | Overrides `--epochs` as the stopping criterion. |

Total steps is `--max-steps` if given, else `batches_per_epoch * --epochs`; the LR schedule derives from that total. Each step prints one JSON line with `loss`, `learning_rate`, and `grad_norm`.

Checkpoints use `safetensors` plus a JSON sidecar, so nothing is ever unpickled. `--output` holds weights only; `--save-every` additionally writes `<stem>.stepN.safetensors` carrying optimizer moments, step, and epoch for `--resume`.

## Quick check

Runs on CPU with no audio and no weights. The codes are random, so the loss is meaningless — this only confirms the pipeline executes:

```bash
siren make-tiny-dataset --output-dir outputs/tiny --records 4 --time-steps 16
siren train --config configs/tiny.json \
  --allow-random-init --allow-random-codebooks --expert-start 0 \
  --dataset outputs/tiny/train.jsonl \
  --output outputs/expert_0.safetensors --device cpu --seed 0
```

## Configs

| Config | Layers (main + decoder) | Per expert | Total |
| --- | --- | --- | --- |
| `configs/siren_1_6b.json` | 14 + 2 | 273M | 1.64B |
| `configs/siren_3_1b.json` | 24 + 8 | 515M | 3.09B |
| `configs/tiny.json` | 2 + 1 | — | CPU tests only |

Both paper configs use a 1024-dim hidden space. `--allow-random-codebooks` substitutes random DAC embeddings and is for tests only; real training must pass `--dac-model`.

## Layout

```
src/siren/
  model.py         SirenExpert, DAC codebook bank
  layers.py        RMSNorm, SwiGLU, causal + cross attention
  config.py        SirenConfig
  data.py          JSONL/NPZ dataset and collator
  training.py      Stage1Module, LR schedule, seeding
  checkpoint.py    safetensors save / load / resume
  preparation.py   audio + caption -> NPZ
  codec.py         local DAC adapter
  conditioning.py  local CLAP text adapter
  cli.py           train, train-all, prepare-data, make-tiny-dataset
```

## Citation

```bibtex
@inproceedings{wang-etal-2025-language-model,
  title     = {Language Model Based Text-to-Audio Generation: Anti-Causally Aligned Collaborative Residual Transformers},
  author    = {Wang, Juncheng and Xu, Chao and Yu, Cheng and Hu, Zhe and Xie, Haoyu and Yu, Guoqi and Shang, Lei and Wang, Shujun},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages     = {26025--26043},
  year      = {2025},
  doi       = {10.18653/v1/2025.emnlp-main.1322}
}
```
