""" Train the SNN model on MNIST """

from __future__ import annotations
import torch
from torch import nn

from configs.config import (
    EPOCHS,
    LEARNING_RATE,
    TIME_STEPS
)
from datasets.data_loader import create_dataloader
from models.snn_model import BaselineSNN
from training.trainer import train_one_epoch
from configs.config import RANDOM_SEED
from utils.reproducibility import set_seed

def main() -> None:
    set_seed(RANDOM_SEED)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    train_loader = create_dataloader(train=True)
    model = BaselineSNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        loss, accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            time_steps=TIME_STEPS,
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {loss:.4f} "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

if __name__ == "__main__":
    main()