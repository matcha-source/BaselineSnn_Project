from torch.utils.data import DataLoader

from configs.config import BATCH_SIZE, NUM_WORKERS
from datasets.mnist_dataset import load_mnist


def create_dataloader(train: bool = True) -> DataLoader:
    """
    Create a DataLoader for the MNIST dataset.

    Parameters
    ----------
    train : bool
        True to load the training set, False for the tests set.

    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader.
    """

    dataset = load_mnist(train=train)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return dataloader