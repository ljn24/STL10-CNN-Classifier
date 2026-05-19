from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_BASE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

_AUG_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(96, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class TransformedSubset(Dataset):
    """Wraps a Subset with a custom transform applied to PIL images."""

    def __init__(self, subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

    def __len__(self):
        return len(self.subset)


def build_train_val_loaders(
    root: str,
    val_ratio: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    augment: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders from an ImageFolder layout."""
    base_dataset = ImageFolder(Path(root) / "train")

    val_size = int(len(base_dataset) * val_ratio)
    train_size = len(base_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(
        base_dataset, [train_size, val_size], generator=generator,
    )

    train_transform = _AUG_TRANSFORM if augment else _BASE_TRANSFORM
    train_set = TransformedSubset(train_sub, train_transform)
    val_set = TransformedSubset(val_sub, _BASE_TRANSFORM)

    common = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, **common),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, **common),
    )


def build_test_loader(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, list[str]]:
    """Build a test-only dataloader from an ImageFolder layout."""
    test_set = ImageFolder(Path(root) / "test", transform=_BASE_TRANSFORM)
    loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return loader, test_set.classes
