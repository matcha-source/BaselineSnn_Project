""" Evaluating the baseline model """

from __future__ import annotations
import torch
from torch import nn

from utils.encoding import poisson_encoder
from  utils.decoding import decode_spike_count

def evaluate(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        time_steps: int,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            spikes = poisson_encoder(
                images=images,
                time_steps=time_steps,
            ).to(device)
            output_spikes = model(spikes)
            spike_counts = output_spikes.sum(dim=0)
            loss = criterion(spike_counts, labels)
            predictions = decode_spike_count(output_spikes)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    avrg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avrg_loss, accuracy
