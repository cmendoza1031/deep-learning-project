"""
Manual LoRA injection for timm ViT.

timm's ViT combines Q, K, V into a single linear layer (768 → 2304).
We need to split this and apply LoRA to Q and V only.

LoRA math:
    output = W_pretrained @ x + (B @ A) @ x * (alpha / rank)
    where A: rank × d_in (initialized random), B: d_out × rank (initialized zero)

    The product BA has shape d_out × d_in, same as the pretrained weight.
    At init, BA = 0, so no change from pretrained model.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class LoRALinear(nn.Module):
    """
    A linear layer with LoRA adapters.

    Replaces: y = x @ W^T + b
    With:     y = x @ W^T + x @ (A^T @ B^T) * (alpha / rank) + b

    Only A and B are trained. W is frozen.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int = 16,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen pretrained weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=False
        )
        self.bias_param = nn.Parameter(
            torch.zeros(out_features),
            requires_grad=False
        ) if bias else None

        # LoRA matrices — these ARE trained
        # A: initialized with Kaiming uniform (standard LoRA init)
        # B: initialized to zero (so BA = 0 at start, preserving pretrained behavior)
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features),
            requires_grad=True
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank),
            requires_grad=True
        )

        # Init A with Kaiming
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pretrained forward
        base_out = nn.functional.linear(x, self.weight, self.bias_param)
        # LoRA forward: x @ A^T gives (batch, seq, rank), then @ B^T gives (batch, seq, out)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def get_delta_W(self) -> torch.Tensor:
        """Returns the LoRA weight update ΔW = B @ A * scaling. Shape: (out, in)."""
        return (self.lora_B @ self.lora_A) * self.scaling


class SplitQKVWithLoRA(nn.Module):
    """
    Replaces a combined QKV linear layer with separate Q, K, V projections,
    where Q and V have LoRA adapters and K is frozen.

    Input: combined QKV weight of shape (3*d_model, d_model)
    Output: splits into Q (d_model, d_model), K (d_model, d_model), V (d_model, d_model)
    """

    def __init__(
        self,
        qkv_weight: torch.Tensor,
        qkv_bias: Optional[torch.Tensor],
        rank: int,
        alpha: int = 16,
        d_model: int = 768
    ):
        super().__init__()
        self.d_model = d_model

        # Q projection with LoRA
        self.q_proj = LoRALinear(d_model, d_model, rank=rank, alpha=alpha)
        self.q_proj.weight.data = qkv_weight[:d_model, :].clone()
        if qkv_bias is not None:
            self.q_proj.bias_param.data = qkv_bias[:d_model].clone()

        # K projection — frozen, no LoRA
        self.k_proj = nn.Linear(d_model, d_model, bias=(qkv_bias is not None))
        self.k_proj.weight.data = qkv_weight[d_model:2*d_model, :].clone()
        if qkv_bias is not None:
            self.k_proj.bias.data = qkv_bias[d_model:2*d_model].clone()
        for p in self.k_proj.parameters():
            p.requires_grad = False

        # V projection with LoRA
        self.v_proj = LoRALinear(d_model, d_model, rank=rank, alpha=alpha)
        self.v_proj.weight.data = qkv_weight[2*d_model:, :].clone()
        if qkv_bias is not None:
            self.v_proj.bias_param.data = qkv_bias[2*d_model:].clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns concatenated QKV — same output shape as original qkv layer."""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return torch.cat([q, k, v], dim=-1)


def replace_qkv_with_lora(
    qkv_layer: nn.Linear,
    rank: int,
    alpha: int = 16,
    d_model: int = 768
) -> SplitQKVWithLoRA:
    """
    Takes an existing combined QKV linear layer from timm ViT,
    returns a SplitQKVWithLoRA that has the same forward behavior
    but with LoRA adapters on Q and V.
    """
    weight = qkv_layer.weight.data
    bias = qkv_layer.bias.data if qkv_layer.bias is not None else None

    return SplitQKVWithLoRA(
        qkv_weight=weight,
        qkv_bias=bias,
        rank=rank,
        alpha=alpha,
        d_model=d_model
    )


def get_model_lora_matrices(model: nn.Module) -> dict:
    """
    Extracts all LoRA A and B matrices from a ViT model with LoRA injection.

    Returns dict: {layer_name: {'A': tensor, 'B': tensor}}

    Used by aggregators to compute ΔW = B @ A and by DQAW to compute update magnitudes.
    """
    lora_matrices = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_matrices[name] = {
                'A': module.lora_A.data.clone(),
                'B': module.lora_B.data.clone(),
                'rank': module.rank,
                'alpha': module.alpha,
                'scaling': module.scaling,
                'in_features': module.in_features,
                'out_features': module.out_features
            }

    return lora_matrices
