"""
Standard FedAvg aggregation.
Used for baseline: homogeneous LoRA rank 8 for all clients.
Weight by n_k / sum(n_k).
"""

import torch
from typing import Dict, List


def fedavg_aggregate(
    client_state_dicts: List[Dict[str, torch.Tensor]],
    client_sizes: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Standard FedAvg: weighted average of client state dicts.

    Args:
        client_state_dicts: list of state dicts (LoRA + head params) from each client
        client_sizes: number of training samples per client

    Returns:
        Aggregated global state dict
    """
    total = sum(client_sizes)
    weights = [n / total for n in client_sizes]

    # Validate all clients have same keys (required for homogeneous setting)
    keys = set(client_state_dicts[0].keys())
    for sd in client_state_dicts:
        assert set(sd.keys()) == keys, "State dict keys mismatch"

    aggregated = {}
    for key in keys:
        # Weighted average
        aggregated[key] = sum(
            w * sd[key].float()
            for w, sd in zip(weights, client_state_dicts)
        )

    return aggregated
