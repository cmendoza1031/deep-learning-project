"""
Federated server: orchestrates training rounds and aggregation.
"""

import torch
import torch.nn as nn
from typing import Dict, List
import json
import os

from federated.client import FederatedClient
from federated.aggregators.hetlora import hetlora_aggregate, truncate_lora_to_rank
from federated.aggregators.flexlora import flexlora_aggregate
from federated.aggregators.dqaw import dqaw_aggregate
from federated.aggregators.fedavg import fedavg_aggregate


CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC']


class FederatedServer:
    def __init__(
        self,
        clients: List[FederatedClient],
        aggregation_method: str,  # 'fedavg', 'hetlora', 'flexlora', 'dqaw'
        num_rounds: int = 20,
        local_epochs: int = 3,
        save_dir: str = './results',
    ):
        assert aggregation_method in ['fedavg', 'hetlora', 'flexlora', 'dqaw']

        self.clients = clients
        self.aggregation_method = aggregation_method
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.history = {
            'round': [],
            'train_loss_per_client': [],
            'bacc_per_client': [],
            'per_class_recall': [],
            'dqaw_weights': [],
        }

    def run(self):
        """Main federated training loop."""
        print(f"\n{'='*60}")
        print(f"Starting Federated Training: {self.aggregation_method.upper()}")
        print(f"Rounds: {self.num_rounds}, Local epochs: {self.local_epochs}")
        print(f"Client ranks: {[c.rank for c in self.clients]}")
        print(f"Client sizes: {[c.n_samples for c in self.clients]}")
        print(f"{'='*60}\n")

        best_bacc = 0.0

        for round_num in range(1, self.num_rounds + 1):
            print(f"\n--- Round {round_num}/{self.num_rounds} ---")

            # === LOCAL TRAINING ===
            train_losses = []
            for client in self.clients:
                loss = client.local_train(self.local_epochs)
                train_losses.append(loss)
                print(f"  Client {client.client_id} (rank={client.rank}): loss={loss:.4f}")

            # === AGGREGATION ===
            self._aggregate()

            # === EVALUATION ===
            round_baccs = []
            round_per_class = []

            for client in self.clients:
                bacc, per_class = client.evaluate()
                round_baccs.append(bacc)
                round_per_class.append(per_class)
                print(f"  Client {client.client_id}: BACC={bacc:.4f}")

                mel_recall = per_class[0]['recall']
                nv_recall = per_class[1]['recall']
                print(f"    MEL recall={mel_recall:.4f}, NV recall={nv_recall:.4f}")

            avg_bacc = sum(round_baccs) / len(round_baccs)
            print(f"  Average BACC: {avg_bacc:.4f}")

            # Save best checkpoint
            if avg_bacc > best_bacc:
                best_bacc = avg_bacc
                torch.save(
                    self.clients[0].model.state_dict(),
                    os.path.join(self.save_dir, 'best_model.pt')
                )
                print(f"  ✓ New best BACC: {best_bacc:.4f}, checkpoint saved")

            # Log
            self.history['round'].append(round_num)
            self.history['train_loss_per_client'].append(train_losses)
            self.history['bacc_per_client'].append(round_baccs)
            self.history['per_class_recall'].append([
                {cls: pc[cls]['recall'] for cls in pc}
                for pc in round_per_class
            ])

        # Save results
        self._save_results()
        print(f"\nTraining complete. Results saved to {self.save_dir}")

        return self.history

    def _aggregate(self):
        """Dispatches to the appropriate aggregation method."""
        client_loras = [c.get_lora_matrices() for c in self.clients]
        client_heads = [c.get_head_state_dict() for c in self.clients]
        client_ranks = [c.rank for c in self.clients]
        client_sizes = [c.n_samples for c in self.clients]

        if self.aggregation_method == 'fedavg':
            combined_states = []
            for lora, head in zip(client_loras, client_heads):
                state = {}
                state.update(head)
                for layer, info in lora.items():
                    state[f'{layer}.lora_A'] = info['A']
                    state[f'{layer}.lora_B'] = info['B']
                combined_states.append(state)

            global_state = fedavg_aggregate(combined_states, client_sizes)

            for client in self.clients:
                lora_dict = {}
                for layer in client_loras[0].keys():
                    info = client_loras[0][layer].copy()
                    info['A'] = global_state[f'{layer}.lora_A']
                    info['B'] = global_state[f'{layer}.lora_B']
                    lora_dict[layer] = info
                client.set_lora_matrices(lora_dict)

                head_dict = {k: v for k, v in global_state.items() if 'head' in k}
                client.set_head_state_dict(head_dict)

        elif self.aggregation_method == 'hetlora':
            result = hetlora_aggregate(client_loras, client_heads, client_ranks, client_sizes)
            global_lora = result['lora']
            global_head = result['head']

            for client in self.clients:
                client_lora = {}
                for layer, info in global_lora.items():
                    A_trunc, B_trunc = truncate_lora_to_rank(
                        info['A'], info['B'], client.rank
                    )
                    client_lora[layer] = {
                        'A': A_trunc,
                        'B': B_trunc,
                        'rank': client.rank,
                        'alpha': client_loras[0][layer]['alpha'],
                        'scaling': client_loras[0][layer]['alpha'] / client.rank,
                        'in_features': client_loras[0][layer]['in_features'],
                        'out_features': client_loras[0][layer]['out_features'],
                    }
                client.set_lora_matrices(client_lora)
                client.set_head_state_dict(global_head)

        elif self.aggregation_method == 'flexlora':
            result = flexlora_aggregate(client_loras, client_heads, client_ranks, client_sizes)
            for i, client in enumerate(self.clients):
                client.set_lora_matrices(result['lora_per_client'][i])
                client.set_head_state_dict(result['head'])

        elif self.aggregation_method == 'dqaw':
            result = dqaw_aggregate(client_loras, client_heads, client_ranks, client_sizes)
            for i, client in enumerate(self.clients):
                client.set_lora_matrices(result['lora_per_client'][i])
                client.set_head_state_dict(result['head'])
            self.history['dqaw_weights'].append(result['weights'])
            print(f"  DQAW weights: {[f'{w:.3f}' for w in result['weights']]}")

    def _save_results(self):
        """Saves training history to JSON."""
        save_path = os.path.join(self.save_dir, f'{self.aggregation_method}_results.json')
        serializable = {
            'method': self.aggregation_method,
            'rounds': self.history['round'],
            'bacc_per_client': self.history['bacc_per_client'],
            'train_loss_per_client': self.history['train_loss_per_client'],
            'per_class_recall': self.history['per_class_recall'],
        }
        if self.history['dqaw_weights']:
            serializable['dqaw_weights'] = self.history['dqaw_weights']
        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)
