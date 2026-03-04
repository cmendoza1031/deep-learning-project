"""
FedISIC data loading via Flamby library.
Flamby partitions FedISIC by acquisition center — real non-IID splits.
No artificial Dirichlet partitioning needed.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def get_fedisic_transforms(split='train', image_size=224):
    """Standard ViT preprocessing."""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_client_dataloaders(client_id, batch_size=32, image_size=224):
    """
    Returns (train_loader, test_loader) for a given FedISIC client.

    Client IDs: 0=VIDIR-Graz, 1=Ham10000-A, 2=Ham10000-B, 3=BCN

    NOTE: Requires flamby to be installed and FedISIC downloaded.
    Run colab_notebooks/01_setup_and_data.ipynb first.
    """
    try:
        from flamby.datasets.fed_isic2019 import FedIsic2019
    except ImportError:
        raise ImportError(
            "Flamby not installed. Run: pip install 'flamby[fed_isic2019]'\n"
            "Then download data: python -c \"from flamby.datasets.fed_isic2019 import FedIsic2019; FedIsic2019(center=0, train=True)\""
        )

    train_dataset = FedIsic2019(
        center=client_id,
        train=True,
        transform=get_fedisic_transforms('train', image_size)
    )
    test_dataset = FedIsic2019(
        center=client_id,
        train=False,
        transform=get_fedisic_transforms('test', image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader


def get_all_client_dataloaders(batch_size=32, image_size=224):
    """Returns dict of {client_id: (train_loader, test_loader)} for all 4 clients."""
    return {
        i: get_client_dataloaders(i, batch_size, image_size)
        for i in range(4)
    }
