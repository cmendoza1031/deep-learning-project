"""
Alternative data loader for raw ISIC 2019 from Kaggle.

Use this when you have downloaded ISIC 2019 from Kaggle but not FedISIC from Flamby.
This creates a SYNTHETIC 4-client partition (by splitting data) — NOT the real
FedISIC partition by acquisition center. For proper FedISIC (real non-IID by
hospital), use fed_isic_loader.py with Flamby.

Expected Kaggle structure:
  /content/isic2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/*.jpg
  /content/isic2019/ISIC_2019_Training_GroundTruth.csv
  /content/isic2019/ISIC_2019_Training_Metadata.csv (optional, for center info)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image


def get_isic_transforms(split='train', image_size=224):
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


class KaggleISICDataset(Dataset):
    """ISIC 2019 from Kaggle with 7-class labels (MEL, NV, BCC, AK, BKL, DF, VASC)."""

    CLASS_COLS = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC']

    def __init__(self, df, img_dir, transform=None, client_ids=None):
        """
        Args:
            df: DataFrame with 'image' and class columns
            img_dir: path to image folder (e.g. .../ISIC_2019_Training_Input/ISIC_2019_Training_Input/)
            transform: torchvision transform
            client_ids: optional list of indices to use (for client partition)
        """
        self.df = df.copy()
        if client_ids is not None:
            self.df = self.df.iloc[client_ids].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self._get_labels()

    def _get_labels(self):
        labels = []
        for _, row in self.df.iterrows():
            vec = [row[c] for c in self.CLASS_COLS]
            label = vec.index(1.0) if 1.0 in vec else 0
            labels.append(label)
        return labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_kaggle_client_dataloaders(
    client_id: int,
    base_dir: str = '/content/isic2019',
    batch_size: int = 32,
    image_size: int = 224,
):
    """
    Returns (train_loader, test_loader) for a synthetic client partition.

    Partitions data into 4 clients by index. Uses 80/20 train/val split per client.
    """
    gt_path = os.path.join(base_dir, 'ISIC_2019_Training_GroundTruth.csv')
    img_dir = os.path.join(base_dir, 'ISIC_2019_Training_Input', 'ISIC_2019_Training_Input')

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth not found at {gt_path}. Download from Kaggle first.")

    df = pd.read_csv(gt_path)
    # Filter to 7 classes (exclude SCC, UNK) - only rows with exactly one of our classes
    df = df[['image'] + KaggleISICDataset.CLASS_COLS]
    df = df[df[KaggleISICDataset.CLASS_COLS].sum(axis=1) == 1]
    n = len(df)

    # Synthetic partition: split into 4 clients
    partition_size = n // 4
    start = client_id * partition_size
    end = start + partition_size if client_id < 3 else n
    client_indices = list(range(start, end))

    # 80/20 train/val
    n_client = len(client_indices)
    n_train = int(0.8 * n_client)
    train_ids = client_indices[:n_train]
    val_ids = client_indices[n_train:]

    train_ds = KaggleISICDataset(
        df, img_dir,
        transform=get_isic_transforms('train', image_size),
        client_ids=train_ids
    )
    val_ds = KaggleISICDataset(
        df, img_dir,
        transform=get_isic_transforms('test', image_size),
        client_ids=val_ids
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
