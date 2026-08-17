from __future__ import annotations
import torch
from torch import nn
from models.surrogate import fast_sigmoid

class LIFNeuron(nn.Module):
    def __init__(
            self,
            beta: float = 0.9,
            threshold: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < beta < 1.0:
            raise (ValueError("beta must be between 0 and 1"))
        if threshold <= 0.0:
            raise (ValueError("threshold must be greater than 0.0"))
        self.beta = beta
        self.threshold = threshold

    def forward(
        self,
        input_current: torch.Tensor,
        membrane:torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        #LIF Operation
        membrane = (self.beta * membrane + input_current)
        # Generating the spike
        #spikes = (membrane > self.threshold).float()

        #Making the LIF neuron trainable
        membrane_difference = (membrane - self.threshold)
        spikes = fast_sigmoid(membrane_difference)

        # Resetting the membrane potential using the reset-to-zero approach
        membrane = torch.where(
            spikes.bool(),
            torch.zeros_like(membrane),
            membrane,
        )
        return spikes, membrane