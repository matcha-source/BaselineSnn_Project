from __future__ import annotations
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

from configs.config import BATCH_SIZE, NUM_WORKERS, RANDOM_SEED

def create_datasets():
    transform = transforms.ToTensor()

    full_trainset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_size = 50_000
    validation_size = 10_000

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, validation_dataset = random_split(
        full_trainset,
        [train_size, validation_size],
        generator=generator,
    )
    return train_dataset, validation_dataset, test_dataset

def create_dataloaders():
    train_dataset, validation_dataset, test_dataset = create_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        #num_workers=NUM_WORKERS,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader

