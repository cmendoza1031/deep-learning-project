# MedHetLoRA

**First Benchmark of Heterogeneous-Rank Federated LoRA for Multi-Center Skin Lesion Classification**

Research by Cristian Mendoza, UCI 2026

---

## Overview

MedHetLoRA is a federated learning framework where 4 hospitals fine-tune a shared ViT-base/16 on skin lesion data (ISIC 2019) using LoRA adapters of different ranks (2, 4, 8, 16), simulating real-world compute heterogeneity. We benchmark three aggregation strategies and introduce **DQAW** — Data-Quality-Adaptive Weighting.

### The Problem: Rank Collapse in Heterogeneous FedLoRA

When hospitals train together without sharing data (federated learning), they often use different compute budgets — a large hospital might use LoRA rank 16, a smaller one rank 2. Under standard aggregation, this causes **rank collapse**: the global update's energy concentrates on the minimum shared rank, while higher-rank directions decay geometrically across rounds.

The root cause is a **rank-wise averaging mismatch**: uniform aggregation applies the same weight to every rank direction, even though higher-rank components are only supported by the subset of clients whose rank is large enough to represent them. As a result, those components are systematically down-weighted each round until they vanish. This phenomenon affects all naive FedLoRA aggregation schemes with heterogeneous ranks — not just zero-padding approaches.

### Our Contribution: DQAW

Instead of weighting hospitals by data count (standard FedAvg), DQAW weights each client by its **per-sample LoRA update magnitude** — the Frobenius norm of the full reconstructed weight delta divided by the client's sample count. This rewards clients that learned coherent, high-signal representations per data point. Since higher-rank clients typically produce larger update magnitudes (more expressive adapters), DQAW naturally up-weights their contributions, offering a quality-adaptive alternative to both count-based and rank-partitioned aggregation schemes.

### Results


| Method      | BACC  | Notes                         |
| ----------- | ----- | ----------------------------- |
| FedAvg      | 74.0% | Homogeneous rank 8 (baseline) |
| DQAW (Ours) | 71.0% | Quality-adaptive weighting    |
| FlexLoRA    | 70.9% | ΔW + SVD                      |
| HetLoRA     | 64.2% | Zero-padding (rank collapse)  |


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
- raFLoRA (rank collapse theory): Wu et al., arXiv 2602.13486, 2026
- FedISIC: Terrail et al., NeurIPS 2022 (Flamby)

---

## Model Weights

All four model checkpoints (FedAvg, HetLoRA, FlexLoRA, DQAW) are available for the Gradio demo. Run `python demo/gradio_app.py` to compare predictions across all methods.