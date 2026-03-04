"""
Main experiment script. Run in Google Colab A100.

Usage:
    python experiments/run_experiment.py --method dqaw --device cuda

Methods:
    fedavg  : FedAvg with homogeneous rank 8 (baseline)
    hetlora : Zero-padding heterogeneous ranks [16,8,4,2]
    flexlora: FlexLoRA heterogeneous ranks [16,8,4,2]
    dqaw    : DQAW (ours) heterogeneous ranks [16,8,4,2]
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
import timm

from data.fed_isic_loader import get_client_dataloaders
from data.kaggle_isic_loader import get_kaggle_client_dataloaders
from models.vit_lora import apply_lora_to_vit
from federated.client import FederatedClient
from federated.server import FederatedServer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Rank assignment
    if args.method == 'fedavg':
        ranks = [8, 8, 8, 8]
    else:
        ranks = [16, 8, 4, 2]

    # Build clients
    if args.use_kaggle:
        data_loader_fn = lambda cid: get_kaggle_client_dataloaders(
            cid, base_dir=args.kaggle_dir, batch_size=args.batch_size
        )
        print(f"Using Kaggle ISIC data from {args.kaggle_dir}")
    else:
        data_loader_fn = lambda cid: get_client_dataloaders(cid, batch_size=args.batch_size)
        print("Using FedISIC data from Flamby")

    clients = []
    for client_id in range(4):
        print(f"Setting up Client {client_id} (rank={ranks[client_id]})...")

        train_loader, test_loader = data_loader_fn(client_id)

        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=7
        )
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

        model = apply_lora_to_vit(model, rank=ranks[client_id], alpha=16)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_trainable:,}")

        client = FederatedClient(
            client_id=client_id,
            model=model,
            rank=ranks[client_id],
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            lr=args.lr,
        )

        client.n_samples = len(train_loader.dataset)
        clients.append(client)

    # Run federated training
    save_dir = os.path.join(args.save_dir, args.method)
    server = FederatedServer(
        clients=clients,
        aggregation_method=args.method,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        save_dir=save_dir,
    )

    history = server.run()
    print(f"\nFinal average BACC: {sum(history['bacc_per_client'][-1]) / 4:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        choices=['fedavg', 'hetlora', 'flexlora', 'dqaw'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--local_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--use_kaggle', action='store_true',
                        help='Use Kaggle ISIC 2019 instead of Flamby FedISIC')
    parser.add_argument('--kaggle_dir', type=str, default='/content/isic2019',
                        help='Path to Kaggle ISIC 2019 data (when --use_kaggle)')
    args = parser.parse_args()
    main(args)
