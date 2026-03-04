"""
DQAW: Data-Quality-Adaptive Weighting for Heterogeneous Federated LoRA.
Novel contribution by Cristian Mendoza, UCI, 2026.

MOTIVATION:
Standard FedAvg weights client contributions by n_k / N (data count).
Under rank heterogeneity, this is problematic:
- Client with rank 2 and 4521 samples may contribute a very weak LoRA update
  because rank 2 severely constrains the expressive power of the adapter
- Weighting this client by n_k/N gives it significant influence despite its
  update being low-quality due to rank underfitting
- Conversely, a client with rank 16 and fewer samples may learn a much
  stronger, more coherent update

DQAW IDEA:
Weight each client's contribution by the per-sample update magnitude:

    w_k = (||ΔW_k||_F / n_k) / Σ_j (||ΔW_j||_F / n_j)

where ΔW_k = B_k @ A_k * scaling

This measures: "how much did this client learn per data point?"

RELATIONSHIP TO EXISTING WORK:
- Different from FedAvg: uses signal quality, not data quantity
- Combines naturally with FlexLoRA's ΔW reconstruction (no zero-padding bias)
"""

import torch
from typing import Dict, List


def compute_dqaw_weights(
    client_lora_matrices: List[Dict],
    client_sizes: List[int],
    epsilon: float = 1e-8
) -> List[float]:
    """
    Computes DQAW weights for each client.

    w_k = (||ΔW_k||_F / n_k) / Σ_j (||ΔW_j||_F / n_j)

    Args:
        client_lora_matrices: list of {layer: {'A', 'B', 'scaling', ...}} per client
        client_sizes: [n_0, n_1, n_2, n_3]
        epsilon: numerical stability

    Returns:
        List of weights summing to 1.0
    """
    per_sample_magnitudes = []

    for client_idx, lora_dict in enumerate(client_lora_matrices):
        total_frob_sq = 0.0

        for layer_name, info in lora_dict.items():
            A = info['A']
            B = info['B']
            scaling = info['scaling']

            delta_W = (B @ A) * scaling
            total_frob_sq += (delta_W ** 2).sum().item()

        total_frob = (total_frob_sq ** 0.5) + epsilon
        n_k = client_sizes[client_idx]

        per_sample_magnitudes.append(total_frob / n_k)

    total = sum(per_sample_magnitudes)
    weights = [m / total for m in per_sample_magnitudes]

    return weights


def dqaw_aggregate(
    client_lora_matrices: List[Dict],
    client_head_states: List[Dict],
    client_ranks: List[int],
    client_sizes: List[int],
) -> Dict:
    """
    DQAW aggregation:
    Same as FlexLoRA (ΔW reconstruction + SVD redistribution) but uses
    DQAW weights instead of FedAvg weights.

    Returns: {
        'lora_per_client': [{layer: {'A', 'B'}} for each client]
        'head': aggregated head state dict
        'weights': the DQAW weights (for logging/analysis)
    }
    """
    dqaw_weights = compute_dqaw_weights(client_lora_matrices, client_sizes)

    layer_names = list(client_lora_matrices[0].keys())
    client_new_loras = [{} for _ in range(len(client_ranks))]

    for layer in layer_names:
        delta_Ws = []
        for client_idx in range(len(client_lora_matrices)):
            info = client_lora_matrices[client_idx][layer]
            A, B, scaling = info['A'], info['B'], info['scaling']
            delta_W = (B @ A) * scaling
            delta_Ws.append(delta_W)

        # DQAW weighted average (KEY DIFFERENCE from FlexLoRA)
        delta_W_global = sum(w * dW for w, dW in zip(dqaw_weights, delta_Ws))

        U, S, Vt = torch.linalg.svd(delta_W_global, full_matrices=False)

        for client_idx, rank in enumerate(client_ranks):
            info = client_lora_matrices[client_idx][layer]
            scaling = info['scaling']
            r = min(rank, S.shape[0])
            sqrt_S = torch.sqrt(S[:r])

            new_B = U[:, :r] * sqrt_S.unsqueeze(0)
            new_A = sqrt_S.unsqueeze(1) * Vt[:r, :]

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

    # Aggregate head — use standard FedAvg for head
    total = sum(client_sizes)
    fedavg_weights = [n / total for n in client_sizes]
    head_keys = list(client_head_states[0].keys())
    global_head = {}
    for key in head_keys:
        global_head[key] = sum(
            w * sd[key].float()
            for w, sd in zip(fedavg_weights, client_head_states)
        )

    return {
        'lora_per_client': client_new_loras,
        'head': global_head,
        'weights': dqaw_weights
    }
