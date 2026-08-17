import torch
from models.lif_neuron import LIFNeuron

def test_lif_neuron_generates_spike() -> None:
    """Verify that sufficient input generates spike"""

    neuron = LIFNeuron(
        beta=0.9,
        threshold=1.0,
    )
    membrane = torch.Tensor([0.0])
    input_current = torch.Tensor([1.2])
    spikes, membrane = neuron(
        input_current,
        membrane,
    )

    assert spikes.item() == 1.0
    assert membrane.item() == 0.0

def test_lif_neuron_does_not_spike_below_threshold() -> None:
    """Verify that insufficient input does not generate a spike."""

    neuron = LIFNeuron(
        beta=0.9,
        threshold=1.0,
    )

    membrane = torch.tensor([0.0])
    input_current = torch.tensor([0.2])

    spikes, membrane = neuron(
        input_current,
        membrane,
    )

    assert spikes.item() == 0.0
    assert membrane.item() == 0.20000000298023224

def test_lif_neuron_leaks_membrane() -> None:
    """Verify membrane potential decays without input."""

    neuron = LIFNeuron(
        beta=0.9,
        threshold=1.0,
    )

    membrane = torch.tensor([0.5])
    input_current = torch.tensor([0.0])

    spikes, updated_membrane = neuron(
        input_current,
        membrane,
    )

    assert spikes.item() == 0.0
    assert torch.isclose(
        updated_membrane,
        torch.tensor([0.45]),
    )