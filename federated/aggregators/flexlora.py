"""
FlexLoRA: heterogeneous rank aggregation via ΔW reconstruction + SVD.
From: Bai et al., 2024.

Method:
1. Each client computes ΔW_k = B_k @ A_k * scaling (full weight update matrix)
2. Weighted average: ΔW_global = Σ (n_k/N) * ΔW_k
3. SVD decomposition of ΔW_global: U, S, Vt = svd(ΔW_global)
4. Redistribute to each client at their rank r_k:
   B_k_new = U[:, :r_k] * sqrt(S[:r_k])
   A_k_new = sqrt(S[:r_k, None]) * Vt[:r_k, :]

This is unbiased (no zero-padding noise) and supports any rank.
"""

import torch
from typing import Dict, List


def flexlora_aggregate(
    client_lora_matrices: List[Dict],
    client_head_states: List[Dict],
    client_ranks: List[int],
    client_sizes: List[int],
) -> Dict:
    """
    FlexLoRA aggregation.

    Returns: {
        'lora_per_client': [{layer: {'A', 'B'}} for each client]
        'head': aggregated head state dict
    }
    """
    total = sum(client_sizes)
    weights = [n / total for n in client_sizes]
    layer_names = list(client_lora_matrices[0].keys())

    # Per-client results
    client_new_loras = [{} for _ in range(len(client_ranks))]

    for layer in layer_names:
        # Step 1: Compute ΔW_k = B_k @ A_k * scaling for each client
        delta_Ws = []
        for client_idx in range(len(client_lora_matrices)):
            info = client_lora_matrices[client_idx][layer]
            A, B = info['A'], info['B']
            scaling = info['scaling']
            delta_W = (B @ A) * scaling  # shape: (d_out, d_in)
            delta_Ws.append(delta_W)

        # Step 2: Weighted average of ΔW matrices
        delta_W_global = sum(w * dW for w, dW in zip(weights, delta_Ws))

        # Step 3: SVD of global ΔW
        U, S, Vt = torch.linalg.svd(delta_W_global, full_matrices=False)

        # Step 4: Redistribute to each client at their rank
        for client_idx, rank in enumerate(client_ranks):
            info = client_lora_matrices[client_idx][layer]
            scaling = info['scaling']

            # Take top-rank singular components
            r = min(rank, S.shape[0])
            sqrt_S = torch.sqrt(S[:r])

            # New B: (d_out, rank), New A: (rank, d_in)
            new_B = U[:, :r] * sqrt_S.unsqueeze(0)
            new_A = sqrt_S.unsqueeze(1) * Vt[:r, :]

            # Scale back (undo the scaling applied earlier)
            new_B = new_B / (scaling ** 0.5)
            new_A = new_A / (scaling ** 0.5)

            client_new_loras[client_idx][layer] = {
                'A': new_A,
                'B': new_B,
                'rank': rank,
                'alpha': info['alpha'],
                'scaling': info['scaling'],
                'in_features': info['in_features'],
                'out_features': info['out_features'],
            }

    # Aggregate head
    head_keys = list(client_head_states[0].keys())
    global_head = {}
    for key in head_keys:
        global_head[key] = sum(
            w * sd[key].float()
            for w, sd in zip(weights, client_head_states)
        )

    return {
        'lora_per_client': client_new_loras,
        'head': global_head
    }
