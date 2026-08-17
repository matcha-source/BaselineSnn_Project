"""
This module contains a function for converting conventional image tensors into spike trains
(suitable for SNN processing)
"""

import torch

def poisson_encoder(
        images: torch.Tensor,
        time_steps: int,
) -> torch.Tensor:
    if images.min() < 0 or images.max() > 1:
        raise ValueError(
            "Poisson encoding requires image values between 0 and 1"
        )

    random_values = torch.rand(
        (time_steps, *images.shape),
        device=images.device,
    )

    spikes = random_values < images.unsqueeze(0)
    return spikes.float()

def calculate_spike_rate(spikes: torch.Tensor) -> float:
    return spikes.float().mean().item()