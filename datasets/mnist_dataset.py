"""
MNIST dataset loader.

This module provides functions for downloading and loading the MNIST dataset.
"""

from torchvision import datasets
from torchvision import transforms

from configs.config import DATASET_ROOT


def load_mnist(train: bool = True):
    """
    Load the MNIST dataset.

    Parameters
    ----------
    train : bool
        True for the training set, False for the tests set.

    Returns
    -------
    torchvision.datasets.MNIST
        A PyTorch MNIST dataset object.
    """

    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root=DATASET_ROOT,
        train=train,
        download=True,
        transform=transform,
    )

    return dataset