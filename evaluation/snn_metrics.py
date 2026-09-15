import torch

def calculate_spike_count(spikes: torch.Tensor)  -> float:
    return spikes.sum().item()

def calculate_spike_rate(
        spikes: torch.Tensor,
) -> float:
    total_spikes = spikes.sum().item()
    total_possible = spikes.numel()

    return total_spikes / total_possible