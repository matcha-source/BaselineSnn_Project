""" Train the SNN model on MNIST """

from __future__ import annotations

import time

import torch
from matplotlib import pyplot as plt
from torch import nn
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from configs.config import (
    EPOCHS,
    LEARNING_RATE,
    TIME_STEPS,
    RANDOM_SEED, NUM_CLASSES
)
from datasets.data_loader import create_dataloaders
from models.snn_model import BaselineSNN
from training.trainer import train_one_epoch
from utils.result import ResultsManager
from utils.reproducibility import set_seed
from evaluation.evaluator import evaluate, parameter_count, inference_time
from evaluation.metrics import calculate_class_accuracy, collect_predictions
from evaluation.snn_metrics import calculate_spike_rate
from evaluation.plot_results import plot_confusion_matrix

def baseline_main() -> None:
    set_seed(RANDOM_SEED)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    results_manager = ResultsManager(
        experiment_name="baseline_snn",
    )
    results_manager.add_experiment_config({
        "model": "BaselineSNN",
        "dataset": "MNIST",
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "time_steps": TIME_STEPS,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss"
    })

    total_start_time = time.perf_counter()

    train_loader, validation_loader, test_loader = create_dataloaders()
    model = BaselineSNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_validation_accuracy = 0.0

    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()
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
        epoch_end_time = time.perf_counter()
        epoch_time = epoch_end_time - epoch_start_time

        results_manager.add_training_epoch(
            epochs=epoch + 1,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            epoch_time_seconds=epoch_time
        )
        results_manager.save()

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Accuracy: {validation_accuracy * 100:.2f}% "
            f"Time: {epoch_time:.2f} s"
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

    total_end_time = time.perf_counter()
    total_training_time = total_end_time - total_start_time
    # Find the best validation accuracy and best epoch
    results_manager.add_training_summary(
        total_training_time=total_training_time
    )
    results_manager.save()
    #save_history(history, "../results/history/baseline_snn_history.json")

    print(
        f"\nTotal training time: "
        f"{total_training_time:.2f} seconds"
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

    # classification_matrix = calculate_classification_matrix(labels=labels, predictions=predictions)
    # print(f"classification_matrix: {classification_matrix}")

    report = classification_report(
        labels.cpu().numpy(),
        predictions.cpu().numpy(),
        digits=4,
    )
    print(f"classification_report: {report}")

    param_count = parameter_count(model)
    print(f"param_count: {param_count}")

    total_time, average_time, throughput = inference_time(
        model,
        test_loader,
        device
    )

    results_manager.add_evaluation_results(
        classification_metrics=report,
        performance={
            "inference_time": total_time,
            "average_time": average_time,
            "throughput": throughput,
        },
        model={
            "trainable_parameters": param_count,
        },
        snn_metrics={
            "time_steps": TIME_STEPS,
            "total_spikes": 0.0,
            "average_spike_count": 0.0,
            "average_spike_rate": 0.0,
        }
    )
    results_manager.save()

    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Average time: {average_time * 1000:.4f} ms/sample")
    print(f"Throughput: {throughput:.2f} samples/sec")

    matrix = confusion_matrix(
        labels.cpu().numpy(),
        predictions.cpu().numpy(),
    )

    matrix_display = ConfusionMatrixDisplay(
        matrix,
        display_labels=list(range(NUM_CLASSES)),
    )
    matrix_display.plot()
    plt.show()

if __name__ == "__main__":
    baseline_main()