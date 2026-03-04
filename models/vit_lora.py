"""
ViT-base/16 with LoRA adapters injected via manual PEFT-style injection.

Key design decisions:
- We use timm's ViT-base/16 (ImageNet-21k pretrained) for strong transfer
- LoRA injected on Q and V matrices of all 12 attention blocks
- Classification head is a standard Linear layer (7 classes), NOT LoRA-adapted
- Classification head is aggregated via FedAvg (full parameter)
- Only LoRA matrices (A and B) + classification head are updated during training
"""

import torch
import torch.nn as nn
import timm

from models.lora_utils import LoRALinear, replace_qkv_with_lora


def create_vit_lora(rank: int, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """
    Creates ViT-base/16 with LoRA adapters of the given rank.

    Args:
        rank: LoRA rank for this client (2, 4, 8, or 16)
        num_classes: 7 for FedISIC
        pretrained: use ImageNet-21k pretrained weights

    Returns:
        nn.Module with LoRA adapters applied, ready for fine-tuning
    """
    # Load ViT-base/16 pretrained
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes
    )

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA via manual injection
    model = apply_lora_to_vit(model, rank=rank, alpha=16)

    # Unfreeze classification head
    for param in model.head.parameters():
        param.requires_grad = True

    return model


def apply_lora_to_vit(model: nn.Module, rank: int, alpha: int = 16) -> nn.Module:
    """
    Manually injects LoRA adapters into Q and V projection matrices
    of all 12 ViT attention blocks.

    Each attention block has: qkv = nn.Linear(768, 768*3)
    We split this into Q (768→768), K (768→768), V (768→768) and apply LoRA to Q and V.

    Args:
        model: timm ViT model with frozen weights
        rank: LoRA rank
        alpha: LoRA scaling factor (typically 2x rank or fixed at 16)

    Returns:
        model with LoRA adapters added
    """
    for i, block in enumerate(model.blocks):
        # Replace the combined QKV linear layer with LoRA-enabled version
        block.attn.qkv = replace_qkv_with_lora(
            block.attn.qkv,
            rank=rank,
            alpha=alpha,
            d_model=768  # ViT-base hidden dimension
        )

    return model


def get_lora_parameters(model: nn.Module):
    """Returns only the LoRA adapter parameters (A and B matrices) and head."""
    lora_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'head' in name:
                head_params.append(param)
            elif 'lora' in name.lower():
                lora_params.append(param)

    return lora_params, head_params


def get_lora_state_dict(model: nn.Module) -> dict:
    """
    Returns state dict containing ONLY LoRA matrices and classification head.
    This is what gets communicated to the server each round.
    """
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.data.clone()
    return state


def set_lora_state_dict(model: nn.Module, state_dict: dict):
    """Loads LoRA + head parameters into model."""
    for name, param in model.named_parameters():
        if param.requires_grad and name in state_dict:
            param.data.copy_(state_dict[name])
