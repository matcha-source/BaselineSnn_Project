""" Baseline spiking neural network for MNIST classification"""

from __future__ import annotations
import torch
from torch import nn
from configs.config import (BETA, HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE, SURROGATE_SLOPE, THRESHOLD)
from models.surrogate import fast_sigmoid


class LIFLayer(nn.Module):
    """
        Leaky Integrate-and-Fire layer.

        The layer receives an input current and a membrane potential,
        updates the membrane potential, generates spikes, and resets
        the membrane after a spike.
        """
    def __init__(self, beta: float = BETA, threshold: float = THRESHOLD) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input_current: torch.Tensor, membrane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Performs one simulation step"""
        if input_current.shape != membrane.shape:
            raise ValueError("Input and membrane shapes most match"
                             f"{input_current.shape} and {membrane.shape}")

        membrane = self.beta * membrane + input_current
        membrane_difference = (membrane - self.threshold)
        spikes = fast_sigmoid(
            membrane_difference,
            slope=SURROGATE_SLOPE
        )
        membrane = torch.where(
            spikes.bool(),
            torch.zeros_like(membrane),
            membrane,
        )
        return spikes, membrane

class BaselineSNN(nn.Module):
    """ Fully connected spiking neural network for MNIST classification """
    def __init__(self) -> None:
        super().__init__()

        self.first_linear_layer = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.lif1 = LIFLayer()

        self.second_linear_layer = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.lif2 = LIFLayer()

    def forward(
            self,
            spike_input: torch.Tensor,
    ) -> torch.Tensor:
        """ Performs a forward pass through the network
        The input spike is of the form [time, batch, channel, height, width]
        """
        time_steps = spike_input.shape[0]
        batch_size = spike_input.shape[1]

        device = spike_input.device
        membrane_hidden = torch.zeros(
            batch_size,
            HIDDEN_SIZE,
            device=device,
        )
        membrane_output = torch.zeros(
            batch_size,
            OUTPUT_SIZE,
            device=device,
        )
        output_spikes = []

        for time_step in range(time_steps):
            current_input = spike_input[time_step]
            current_input = current_input.flatten(start_dim=1)
            
            hidden_current = self.first_linear_layer(current_input)
            hidden_spikes, membrane_hidden = self.lif1(hidden_current, membrane_hidden)
            
            output_current = self.second_linear_layer(hidden_spikes)
            output_spike, membrane_output = self.lif2(output_current, membrane_output)
            
            output_spikes.append(output_spike)
        
        return torch.stack(output_spikes, dim=0)
