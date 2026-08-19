""" Train the SNN model on MNIST """

from __future__ import annotations
import torch
from torch import nn
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from configs.config import (
    EPOCHS,
    LEARNING_RATE,
    TIME_STEPS,
    RANDOM_SEED
)
from datasets.data_loader import create_dataloaders
from models.snn_model import BaselineSNN
from training.trainer import train_one_epoch
from utils.reproducibility import set_seed
from evaluation.evaluator import evaluate
from evaluation.metrics import calculate_class_accuracy, collect_predictions

def main() -> None:
    set_seed(RANDOM_SEED)
    labels = torch.Tensor
    predictions = torch.Tensor
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    train_loader, validation_loader, test_loader = create_dataloaders()
    model = BaselineSNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_validation_accuracy = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            time_steps=TIME_STEPS,
        )
        validation_loss, validation_accuracy = evaluate(
            model=model,
            dataloader=validation_loader,
            device=device,
            time_steps=TIME_STEPS,
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Accuracy: {train_accuracy * 100:.2f}% "
            f"Validation Loss: {validation_loss:.4f} "
            f"Validation Accuracy: {validation_accuracy * 100:.2f}% "
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            # torch.save(
            #     model.state_dict(),
            #     "./checkpoints/best_model.pth"
            # )

            checkpoint_dir = Path("checkpoints")

            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                model.state_dict(),
                checkpoint_dir / "best_model.pth",
            )
    model.load_state_dict(
        torch.load(
            "checkpoints/best_model.pth",
            map_location=device,
        )
    )
    test_loss, test_accuracy = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        time_steps=TIME_STEPS,
    )

    print(
        f"Final Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    labels, predictions = collect_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        time_steps=TIME_STEPS
    )
    class_accuracy = calculate_class_accuracy(
        labels=labels,
        predictions=predictions,
    )
    for class_id, accuracy in class_accuracy.items():
        print(
            f"Class {class_id}: "
            f"{accuracy * 100:.2f}%"
        )

    report = classification_report(
        labels.cpu().numpy(),
        predictions.cpu().numpy(),
        digits=4,
    )
    print(f"classification_report: {report}")

    matrix = confusion_matrix(
        labels.cpu().numpy(),
        predictions.cpu().numpy(),
    )
    print(f"Confusion matrix: {matrix}")

if __name__ == "__main__":
    main()