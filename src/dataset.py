from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_train_val_loaders(
    root: str,
    val_ratio: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build train / val dataloaders from an ImageFolder layout."""
    root = Path(root)
    full_train = ImageFolder(root / "train", transform=get_transforms())

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader


def build_test_loader(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, list[str]]:
    """Build a test-only dataloader from an ImageFolder layout."""
    root = Path(root)
    test_set = ImageFolder(root / "test", transform=get_transforms())
    loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return loader, test_set.classes
