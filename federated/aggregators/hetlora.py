"""
HetLoRA: heterogeneous rank aggregation via zero-padding.
From: Cho et al., EMNLP 2024.

Method:
- Pad smaller rank matrices with zeros to match the largest rank
- Average padded matrices with FedAvg weights
- Truncate back to each client's rank for distribution

This is the naive baseline. raFLoRA proves it causes rank collapse via
geometric energy decay in higher-rank directions.
"""

import torch
from typing import Dict, List


def pad_lora_to_rank(A: torch.Tensor, B: torch.Tensor, target_rank: int):
    """
    Zero-pads LoRA matrices to target_rank.
    A: (rank, d_in) → (target_rank, d_in)
    B: (d_out, rank) → (d_out, target_rank)
    """
    current_rank = A.shape[0]
    if current_rank == target_rank:
        return A, B

    pad_size = target_rank - current_rank
    A_padded = torch.cat([A, torch.zeros(pad_size, A.shape[1], device=A.device)], dim=0)
    B_padded = torch.cat([B, torch.zeros(B.shape[0], pad_size, device=B.device)], dim=1)
    return A_padded, B_padded


def truncate_lora_to_rank(A: torch.Tensor, B: torch.Tensor, target_rank: int):
    """Truncates padded LoRA matrices back to target_rank."""
    return A[:target_rank, :], B[:, :target_rank]


def hetlora_aggregate(
    client_lora_matrices: List[Dict],
    client_head_states: List[Dict],
    client_ranks: List[int],
    client_sizes: List[int],
) -> Dict:
    """
    HetLoRA aggregation:
    1. Pad all client LoRA matrices to max rank
    2. Weighted average (FedAvg weights by n_k)
    3. Return padded global matrices (clients truncate to their rank on receipt)

    Returns dict with:
        'lora': {layer_name: {'A': tensor, 'B': tensor}} at max_rank
        'head': aggregated head state dict
    """
    max_rank = max(client_ranks)
    total = sum(client_sizes)
    weights = [n / total for n in client_sizes]

    # Get all layer names from first client
    layer_names = list(client_lora_matrices[0].keys())

    global_lora = {}
    for layer in layer_names:
        # Pad all clients to max_rank
        padded_As = []
        padded_Bs = []

        for client_idx in range(len(client_lora_matrices)):
            A = client_lora_matrices[client_idx][layer]['A']
            B = client_lora_matrices[client_idx][layer]['B']
            A_pad, B_pad = pad_lora_to_rank(A, B, max_rank)
            padded_As.append(A_pad)
            padded_Bs.append(B_pad)

        # Weighted average
        global_A = sum(w * A for w, A in zip(weights, padded_As))
        global_B = sum(w * B for w, B in zip(weights, padded_Bs))

        global_lora[layer] = {
            'A': global_A,
            'B': global_B,
            'rank': max_rank,
        }

    # Aggregate classification head via standard FedAvg
    head_keys = list(client_head_states[0].keys())
    global_head = {}
    for key in head_keys:
        global_head[key] = sum(
            w * sd[key].float()
            for w, sd in zip(weights, client_head_states)
        )

    return {'lora': global_lora, 'head': global_head}
