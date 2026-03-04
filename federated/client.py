"""
Federated client: local training logic.
Each client holds a local model, trains for local_epochs,
then extracts LoRA matrices for aggregation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from models.lora_utils import LoRALinear, get_model_lora_matrices


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        rank: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.rank = rank
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.n_samples = len(train_loader.dataset)

        # Only optimize LoRA parameters and head
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay
        )

        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, num_epochs: int = 3) -> float:
        """
        Runs num_epochs of local training.
        Returns average training loss.
        """
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0
                )

                self.optimizer.step()
                epoch_loss += loss.item()
                total_batches += 1

            total_loss += epoch_loss

        return total_loss / total_batches

    def get_lora_matrices(self) -> Dict:
        """Returns all LoRA A and B matrices for aggregation."""
        return get_model_lora_matrices(self.model)

    def get_head_state_dict(self) -> Dict:
        """Returns classification head parameters."""
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if 'head' in name and param.requires_grad
        }

    def set_lora_matrices(self, lora_dict: Dict):
        """Updates local model with aggregated LoRA matrices from server."""
        for module_name, module in self.model.named_modules():
            if isinstance(module, LoRALinear) and module_name in lora_dict:
                info = lora_dict[module_name]
                module.lora_A.data.copy_(info['A'])
                module.lora_B.data.copy_(info['B'])

    def set_head_state_dict(self, head_dict: Dict):
        """Updates classification head from server."""
        for name, param in self.model.named_parameters():
            if 'head' in name and name in head_dict:
                param.data.copy_(head_dict[name])

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, Dict[int, Dict]]:
        """
        Evaluates model on local test set.
        Returns:
            overall_bacc: balanced accuracy across all classes
            per_class_stats: {class_id: {'correct': int, 'total': int, 'recall': float}}
        """
        self.model.eval()

        per_class_correct = {i: 0 for i in range(7)}
        per_class_total = {i: 0 for i in range(7)}

        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            preds = outputs.argmax(dim=1)

            for class_id in range(7):
                mask = labels == class_id
                per_class_correct[class_id] += (preds[mask] == labels[mask]).sum().item()
                per_class_total[class_id] += mask.sum().item()

        # BACC = average recall per class
        recalls = []
        for class_id in range(7):
            if per_class_total[class_id] > 0:
                recall = per_class_correct[class_id] / per_class_total[class_id]
                recalls.append(recall)

        bacc = sum(recalls) / len(recalls) if recalls else 0.0

        per_class_stats = {
            i: {
                'correct': per_class_correct[i],
                'total': per_class_total[i],
                'recall': per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0.0
            }
            for i in range(7)
        }

        return bacc, per_class_stats
