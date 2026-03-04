# MedHetLoRA

**First Benchmark of Heterogeneous-Rank Federated LoRA for Multi-Center Skin Lesion Classification**

Research by Cristian Mendoza, UCI 2026

---

## Overview

MedHetLoRA is a federated learning framework where 4 hospitals fine-tune a shared ViT-base/16 on skin lesion data (ISIC 2019) using LoRA adapters of different ranks (2, 4, 8, 16), simulating real-world compute heterogeneity. We benchmark three aggregation strategies and introduce **DQAW** — Data-Quality-Adaptive Weighting.

### The Problem

When hospitals train together without sharing data (federated learning), they often use different compute budgets. A large hospital might use LoRA rank 16; a smaller one might use rank 2. Naive aggregation (zero-padding) causes **rank collapse** — the shared model degrades significantly.

### Our Contribution: DQAW

Instead of weighting hospitals by data count (standard FedAvg), DQAW weights by **per-sample LoRA update magnitude** — rewarding clients who learned coherent, strong representations per data point. This mitigates rank-heterogeneity bias.

### Results

| Method | BACC | Notes |
|--------|------|-------|
| FedAvg | 74.0% | Homogeneous rank 8 (baseline) |
| DQAW (Ours) | 71.0% | Quality-adaptive weighting |
| FlexLoRA | 70.9% | ΔW + SVD |
| HetLoRA | 64.2% | Zero-padding (rank collapse) |

---

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Local Development

```bash
# Verify imports
python -c "from models.lora_utils import LoRALinear; print('OK')"
```

### Run Experiments (Google Colab with GPU recommended)

```bash
# Methods: fedavg, hetlora, flexlora, dqaw
python experiments/run_experiment.py --method dqaw --use_kaggle --kaggle_dir /content/isic2019 --device cuda
```

### Gradio Demo

```bash
python demo/gradio_app.py
```

Upload a dermoscopy image to compare predictions across FedAvg, HetLoRA, FlexLoRA, and DQAW.

---

## Project Structure

```
medhetlora/
├── configs/           # YAML config
├── data/              # FedISIC loader, Kaggle ISIC loader
├── models/            # ViT + LoRA (timm, manual injection)
├── federated/         # Server, client, aggregators
│   └── aggregators/   # fedavg, hetlora, flexlora, dqaw
├── evaluation/       # Metrics, per-class analysis
├── experiments/      # Run scripts
├── colab_notebooks/  # Colab workflows
├── demo/             # Gradio app
└── results/          # Model checkpoints (not in repo)
```

---

## References

- LoRA: Hu et al., ICLR 2022
- FedAvg: McMahan et al., AISTATS 2017
- HetLoRA: Cho et al., EMNLP 2024
- FlexLoRA: Bai et al., 2024
- FedISIC: Terrail et al., NeurIPS 2022 (Flamby)

---

## Model Weights

**FedAvg, HetLoRA, and FlexLoRA** checkpoints are available for the Gradio demo. **DQAW** weights are not available at this time — the model was trained and achieved 71% BACC, but the checkpoint was lost when the Colab runtime disconnected. You can still run the demo with the three available methods and train DQAW yourself using the experiment script above.
