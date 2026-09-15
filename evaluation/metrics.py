""" Evaluation Metrics """

from __future__ import annotations
import torch

from utils.decoding import decode_spike_count
from utils.encoding import poisson_encoder
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix
# )

def collect_predictions(
        model,
        dataloader,
        device,
        time_steps,
):
    """ Collect predictions and labels """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            spikes = poisson_encoder(
                images=images,
                time_steps=time_steps,
            ).to(device)
            output_spikes = model(spikes)
            predictions = decode_spike_count(output_spikes)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    labels = torch.cat(all_labels)
    predictions = torch.cat(all_predictions)

    return labels, predictions

# def calculate_classification_matrix(
#         predictions: torch.Tensor,
#         labels: torch.Tensor,
# ) :
#     return {
#         "accuracy": accuracy_score(labels, predictions),
#         "precision": precision_score(labels, predictions, average="macro", zero_division=0),
#         "recall": recall_score(labels, predictions, average="macro", zero_division=0),
#         "f1": f1_score(labels, predictions, average="macro", zero_division=0),
#         "confusion_matrix": confusion_matrix(labels, predictions),
#     }

    # Accuracy for each class
def calculate_class_accuracy(
        labels: torch.Tensor,
        predictions: torch.Tensor,
) -> dict[int, float]:
    """ Calculate class accuracy """

    class_accuracy = {}
    classes = torch.unique(labels)

    for class_id in classes:
        mask = labels == class_id
        correct = (
            predictions[mask] == labels[mask]
        ).sum().item()
        total = mask.sum().item()
        class_accuracy[int(class_id)] = correct / total

    return class_accuracy
