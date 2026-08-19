# Siren: Language Model Based Text-to-Audio Generation via Anti-Causally Aligned Collaborative Residual Transformers (EMNLP 2025)

[![venue](https://img.shields.io/badge/EMNLP-2025-blue.svg?style=flat-square)](https://aclanthology.org/2025.emnlp-main.1322/)
[![paper](https://img.shields.io/badge/ACL_Anthology-2025.emnlp--main.1322-b31b1b.svg?style=flat-square)](https://aclanthology.org/2025.emnlp-main.1322/)
[![python](https://img.shields.io/badge/Python-3.10+-brightgreen.svg?style=flat-square)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)

Official implementation of [**Language Model Based Text-to-Audio Generation: Anti-Causally Aligned Collaborative Residual Transformers**](https://aclanthology.org/2025.emnlp-main.1322/), EMNLP 2025 Main Conference.

Adding RVQ layers improves audio reconstruction fidelity but exceeds the generation capacity of a conventional LM. Siren splits the 12 residual codebooks of a DAC tokenizer across six isolated transformers, each learning a narrower objective, while accumulated conditions preserve the codec's coarse-to-fine dependency structure.

<hr>

## What this repository contains

This release covers the **Stage-1 supervised training pipeline**: data preparation, model definition, and per-expert training.

- **Included** — Stage-1 training, resumable checkpointing, validation, data preparation from audio and captions.
- **Not included** — the anti-causal alignment stage (reinforcement learning). The metrics reported in the paper depend on it, so they are not reproducible from this code alone.
- **Checkpoints** — not released. Training runs from scratch with the configs in `configs/`.

- [ ] Anti-causal alignment (RL) stage
- [ ] Pretrained checkpoints

<hr>

## Method

The paper identifies two properties of RVQ that break a single monolithic LM: features are near-orthogonal across layers, which impedes training, and semantic richness descends with codebook depth, which worsens exposure bias during autoregressive decoding.

Siren therefore assigns each expert one adjacent codebook pair. Expert `k` additionally receives the accumulated embeddings of every codebook below `2k`, which is what keeps the collaboration causal.

| Expert | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| Codebooks | 0, 1 | 2, 3 | 4, 5 | 6, 7 | 8, 9 | 10, 11 |

Each expert runs a temporal path over the summed previous-frame embeddings of all 12 codebooks with prompt cross-attention, then a factorized path that predicts the first codebook of its pair and conditions the second prediction on that first code. Blocks use pre-norm RMSNorm, grouped-query-capable causal attention, and SwiGLU.

<hr>

## Installation

```bash
git clone https://github.com/wjc2830/Siren.git
cd Siren
pip install -e .
```

The `audio` extra is required only to build data from raw audio:

```bash
pip install -e ".[audio]"
```

<hr>

## Quick start

Runs on CPU with no audio and no pretrained weights. The synthetic codes are random, so the loss carries no meaning — this only verifies the pipeline executes.

```bash
siren make-tiny-dataset --output-dir outputs/tiny --records 4 --time-steps 16

siren train --config configs/tiny.json \
  --allow-random-init --allow-random-codebooks --expert-start 0 \
  --dataset outputs/tiny/train.jsonl \
  --output outputs/expert_0.safetensors --device cpu --seed 0
```

<hr>

## Data preparation

Training consumes a JSONL manifest pointing at relative NPZ files, each holding integer `codes` of shape `[12, time]` in `[0, 1023]` and float `prompt_embeddings` of shape `[prompt_length, prompt_dim]`:

```json
{"id": "sample-0001", "npz": "data/sample-0001.npz"}
```

Build it from 16 kHz audio and captions with your own local Descript DAC 16 kHz and CLAP text checkpoints. The source manifest is JSONL of `{"id", "audio", "caption"}`:

```bash
siren prepare-data \
  --source data/source.jsonl --output-dir data/prepared \
  --dac-model /path/to/dac-16khz \
  --clap-model /path/to/clap-text-encoder \
  --device cuda --max-seq-len 300
```

Audio must already be 16 kHz. A mismatched rate is rejected rather than resampled, because codes produced at a different rate are not comparable. The paper crops 6-second caption-aligned segments, which is 300 frames at the DAC frame rate; `--max-seq-len` truncates from the start, so perform the cropping in your own source manifest to match the paper.

Datasets are not redistributed here.

<hr>

## Training

Stage 1 trains the six experts independently, one per process, with pairwise masked cross-entropy. The DAC tokenizer and CLAP text encoder stay frozen.

```bash
siren train --config configs/siren_1_6b.json \
  --allow-random-init --expert-start 0 --dac-model /path/to/dac-16khz \
  --dataset data/train.jsonl --val-dataset data/val.jsonl \
  --output outputs/expert_0.safetensors --device cuda \
  --seed 0 --precision bf16 --warmup-steps 2000 --grad-clip 1.0 \
  --save-every 1000 --val-every 1000
```

`--dac-model` injects and freezes the 12 DAC decode embeddings that produced your codes; only the projector and transformer train. The paper uses AdamW at `3e-4` with batch 24 per GPU.

To print the six per-expert commands, or launch them across GPUs:

```bash
siren train-all --config configs/siren_1_6b.json \
  --allow-random-init --dac-model /path/to/dac-16khz \
  --dataset data/train.jsonl --output-dir outputs/run-1 --seed 0 \
  --device cuda --devices 0,1,2,3,4,5 --launch
```

Without `--launch` this is a dry run that only prints the commands. On success it writes `manifest.json` listing the six expert files with SHA-256 hashes.

### Options

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

Total steps is `--max-steps` when given, otherwise `batches_per_epoch * --epochs`; the LR schedule derives from that total. Every step prints one JSON line with `loss`, `learning_rate`, and `grad_norm`.

Checkpoints use `safetensors` plus a JSON sidecar, so nothing is ever unpickled. `--output` holds weights only, while `--save-every` additionally writes `<stem>.stepN.safetensors` carrying optimizer moments, step, and epoch for `--resume`.

<hr>

## Configurations

| Config | Main layers | Decoder layers | Per expert | Total |
| --- | --- | --- | --- | --- |
| `configs/siren_1_6b.json` | 14 | 2 | 273M | 1.64B |
| `configs/siren_3_1b.json` | 24 | 8 | 515M | 3.09B |
| `configs/tiny.json` | 2 | 1 | — | CPU tests only |

Both paper configurations use a 1024-dimensional hidden space. `--allow-random-codebooks` substitutes random DAC embeddings for tests only; real training must pass `--dac-model`.

<hr>

## Results

Reported on the AudioCaps test set, from Table 1 of the paper. These require the anti-causal alignment stage, which this repository does not implement.

| Model | FAD ↓ | FD ↓ | ISC ↑ | KL ↓ | CLAP ↑ |
| --- | --- | --- | --- | --- | --- |
| Siren 1.6B | 1.35 | 10.65 | 12.85 | 1.33 | 24.18 |
| Siren 3.1B | **1.28** | **10.35** | **13.93** | 1.36 | **25.64** |

<hr>

## Repository structure

```
src/siren/
├── model.py          SirenExpert, DAC codebook bank
├── layers.py         RMSNorm, SwiGLU, causal and cross attention
├── config.py         SirenConfig
├── data.py           JSONL/NPZ dataset and collator
├── training.py       Stage1Module, LR schedule, seeding
├── checkpoint.py     safetensors save / load / resume
├── preparation.py    audio and captions to NPZ
├── codec.py          local DAC adapter
├── conditioning.py   local CLAP text adapter
└── cli.py            train, train-all, prepare-data, make-tiny-dataset
```

<hr>

## Citation

```bibtex
@inproceedings{wang-etal-2025-language-model,
    title     = "Language Model Based Text-to-Audio Generation: Anti-Causally Aligned Collaborative Residual Transformers",
    author    = "Wang, Juncheng and Xu, Chao and Yu, Cheng and Hu, Zhe and Xie, Haoyu and Yu, Guoqi and Shang, Lei and Wang, Shujun",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    year      = "2025",
    publisher = "Association for Computational Linguistics",
    pages     = "26025--26043",
    doi       = "10.18653/v1/2025.emnlp-main.1322",
    url       = "https://aclanthology.org/2025.emnlp-main.1322/"
}
```

<hr>

## Acknowledgements

Siren builds on the [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) and [CLAP](https://github.com/LAION-AI/CLAP), neither of which is bundled or downloaded by this repository. The transformer blocks follow the standard Llama-style RMSNorm and SwiGLU formulations.

This work was partially supported by the RGC Collaborative Research Fund (C5055-24G), The Hong Kong Polytechnic University Start-up Fund (P0045999), the Research Institute for Smart Ageing Seed Fund (P0050946), the Tsinghua–PolyU Joint Research Initiative Fund (P0056509), and PolyU UGC funding (P0053716).
