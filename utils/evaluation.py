import torch

def calculate_spike_rate(spikes: torch.Tensor) -> float:
    return spikes.float().mean().item()

def total_spikes(spikes: torch.Tensor) -> float:
    return torch.sum(spikes).item()

def average_spikes(spikes: torch.Tensor) -> float:
    return torch.mean(spikes).item()