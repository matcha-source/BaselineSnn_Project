import torch

from models.snn_model import BaselineSNN


def test_snn_output_shape() -> None:
    """Verify the SNN output dimensions."""

    time_steps = 25
    batch_size = 8

    spike_input = torch.rand(
        time_steps,
        batch_size,
        1,
        28,
        28,
    )

    spike_input = (
        spike_input > 0.5
    ).float()

    model = BaselineSNN()

    output = model(spike_input)

    expected_shape = (
        time_steps,
        batch_size,
        10,
    )

    assert output.shape == expected_shape