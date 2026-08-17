""" Decoding SNN spikes: Predict classes using output spike counts """

from __future__ import annotations
import torch


def decode_spike_count(
        output_spikes: torch.Tensor,
) -> torch.Tensor:
    if output_spikes.ndim != 3:
        raise ValueError("Expected output shape is [time, batch, classes]")

    spike_count = output_spikes.sum(dim=0)
    predictions = spike_count.argmax(dim=1)

    return predictions