""" Train the model on one epoch and return the loss and accuracy """

from __future__ import annotations
import torch
from torch import nn

from utils.decoding import decode_spike_count
from utils.encoding import poisson_encoder

def train_one_epoch(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        time_steps: int,
) -> tuple[float, float]:

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        spikes = poisson_encoder(
            images=images,
            time_steps=time_steps,
        )
        spikes = spikes.to(device)
        output_spikes = model(spikes)
        spike_counts = output_spikes.sum(dim=0)

        loss = criterion(spike_counts, labels)
        loss.backward()
        optimizer.step()

        predictions = decode_spike_count(output_spikes)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy
