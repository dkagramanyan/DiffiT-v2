# DiffiT: Diffusion Vision Transformers for Image Generation

Official PyTorch implementation of [**DiffiT: Diffusion Vision Transformers for Image Generation**](https://arxiv.org/abs/2312.02139).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

[![Star on GitHub](https://img.shields.io/github/stars/NVlabs/DiffiT.svg?style=social)](https://github.com/NVlabs/DiffiT/stargazers)

**DiffiT** (Diffusion Vision Transformers) is a generative model that combines the expressive power of diffusion models with Vision Transformers (ViTs), introducing **Time-dependent Multihead Self Attention (TMSA)** for fine-grained control over the denoising at each timestep. DiffiT achieves SOTA performance on class-conditional ImageNet generation at multiple resolutions, notably an **FID score of 1.73** on ImageNet-256.

![teaser](./assets/imagenet.png)

![teaser](./assets/latent_diffit.png)

## News
- **[03.08.2026]** DiffiT code and pretrained model are released!
- **[07.01.2024]** DiffiT has been accepted to [ECCV 2024](https://eccv.ecva.net/)!
- **[04.02.2024]** Updated [manuscript](https://arxiv.org/abs/2312.02139) now available on arXiv!
- **[12.04.2023]** Paper is published on arXiv!

## Models

### ImageNet-256

| Model | Dataset | Resolution | FID-50K | Inception Score | Download |
|-------|---------|-----------|---------|-----------------|----------|
| **DiffiT** | ImageNet | 256x256 | **1.73** | **276.49** | [model](https://huggingface.co/nvidia/DiffiT/resolve/main/diffit_256.safetensors) |

### ImageNet-512

| Model | Dataset | Resolution | FID-50K | Inception Score | Download |
|-------|---------|-----------|---------|-----------------|----------|
| **DiffiT** | ImageNet | 512x512 | **2.67** | **252.12** | [model](https://huggingface.co/nvidia/DiffiT/resolve/main/diffit_512.safetensors) |

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Preparation

To prepare an ImageNet dataset for training, use the dataset tool:

```bash
python dataset_tool_for_imagenet.py \
    --source /path/to/ILSVRC \
    --dest ./datasets/imagenet_256x256.zip \
    --resolution 256x256 \
    --transform center-crop
```

This creates a ZIP archive with resized images and a `dataset.json` containing class labels.

## Training

Train DiffiT on ImageNet with multi-GPU DDP:

```bash
python train.py \
    --outdir ./training-runs \
    --data ./datasets/imagenet_256x256.zip \
    --image-size 256 \
    --gpus 4 \
    --batch 256 \
    --batch-gpu 64 \
    --kimg 400000 \
    --snap 50
```

Or use the provided SLURM script:

```bash
sbatch slurm_train_4_gpu.sh
```

Training uses PyTorch DDP for multi-GPU parallelization, mixed precision via `MixedPrecisionTrainer`, and EMA weight averaging.

## Sampling

### Bulk Sampling for FID Evaluation

Generate 50K samples as `.npz` for FID evaluation:

**ImageNet-256:**
```bash
torchrun --nproc_per_node=4 sample.py \
    --model-path ./ckpts/diffit_256.safetensors \
    --outdir ./samples/256 \
    --image-size 256 \
    --cfg-scale 4.4 \
    --num-samples 50000 \
    --batch-size 16 \
    --num-sampling-steps 250 \
    --cfg-cond
```

**ImageNet-512:**
```bash
torchrun --nproc_per_node=4 sample.py \
    --model-path ./ckpts/diffit_512.safetensors \
    --outdir ./samples/512 \
    --image-size 512 \
    --cfg-scale 1.49 \
    --num-samples 50000 \
    --batch-size 8 \
    --num-sampling-steps 250 \
    --cfg-cond
```

### Individual Image Generation

Generate individual PNG images for visual inspection:

```bash
python gen_images.py \
    --model-path ./ckpts/diffit_256.safetensors \
    --seeds 0-49 \
    --outdir ./out \
    --image-size 256 \
    --cfg-scale 4.4 \
    --num-sampling-steps 250
```

Options:
- `--seeds`: Comma-separated list or ranges (e.g., `0,1,4-6`)
- `--class-idx`: Specific class label (random if not specified)
- `--batch-sz`: Batch size per seed
- `--use-ddim`: Use DDIM sampling instead of DDPM

SLURM scripts are also provided:
```bash
sbatch slurm_sample_256.sh
sbatch slurm_sample_512.sh
```

## Evaluation (FID-50K)

Compute FID-50K and other metrics using the PyTorch-based evaluator:

```bash
python evaluator.py \
    --ref-batch ./VIRTUAL_imagenet256_labeled.npz \
    --sample-batch ./samples/256/samples_50000x256x256x3.npz
```

Or use the convenience script:
```bash
bash eval_run.sh 256 ./samples/256
```

Metrics computed: **Inception Score**, **FID**, **sFID**, **Precision**, **Recall**.

### Expected Results

**ImageNet-256:**

| Inception Score | FID | sFID | Precision | Recall |
|:-:|:-:|:-:|:-:|:-:|
| 276.49 | 1.73 | 4.54 | 0.8024 | 0.6205 |

**ImageNet-512:**

| Inception Score | FID | sFID | Precision | Recall |
|:-:|:-:|:-:|:-:|:-:|
| 252.13 | 2.67 | 4.99 | 0.8277 | 0.5500 |

> **Note:** Small variations in the reported numbers are expected depending on the device used for sampling and due to numerical precision differences.

## Project Structure

```
DiffiT-v2/
├── diffit/                          # Core model architecture
│   ├── __init__.py                 # Diffusion creation & defaults
│   ├── diffit.py                   # DiffiT model (ViT + TMSA)
│   ├── gaussian_diffusion.py       # Diffusion process (DDPM/DDIM)
│   ├── respace.py                  # Timestep respacing
│   ├── dist_util.py                # Distributed training (PyTorch DDP)
│   ├── image_datasets.py           # Dataset loading (dir/zip + DistributedSampler)
│   ├── logger.py                   # Logging (stdout, JSON, CSV)
│   ├── fp16_util.py                # Mixed precision training
│   ├── nn.py                       # Neural network utilities
│   ├── timestep_sampler.py         # Timestep sampling strategies
│   ├── diffusion_utils.py          # KL divergence & likelihood
│   └── pos_emb.py                  # Positional embeddings (CoordConv, Swin)
├── train.py                         # Training script (DDP, click CLI)
├── sample.py                        # Bulk sampling for FID (.npz output)
├── gen_images.py                    # Individual PNG generation (click CLI)
├── evaluator.py                     # FID/IS evaluation (PyTorch)
├── dataset_tool_for_imagenet.py     # ImageNet → ZIP dataset converter
├── eval_run.sh                      # Evaluation convenience script
├── slurm_train_4_gpu.sh             # SLURM: 4-GPU training
├── slurm_sample_256.sh              # SLURM: 256x256 sampling + eval
├── slurm_sample_512.sh              # SLURM: 512x512 sampling + eval
├── requirements.txt                 # Python dependencies
└── README.md
```

## Citation

```
@inproceedings{hatamizadeh2025diffit,
  title={Diffit: Diffusion vision transformers for image generation},
  author={Hatamizadeh, Ali and Song, Jiaming and Liu, Guilin and Kautz, Jan and Vahdat, Arash},
  booktitle={European Conference on Computer Vision},
  pages={37--55},
  year={2025},
  organization={Springer}
}
```

## Licenses

Copyright 2026, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

The pre-trained models are shared under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

## Acknowledgement
We gratefully acknowledge the authors of [Guided-Diffusion](https://github.com/openai/guided-diffusion/tree/main/), [DiT](https://github.com/facebookresearch/DiT/tree/main) and [MDT](https://github.com/sail-sg/MDT/tree/mdtv1) for making their excellent codebases publicly available.
