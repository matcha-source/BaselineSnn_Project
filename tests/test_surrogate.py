import torch

from models.surrogate import fast_sigmoid


def test_surrogate_generates_binary_spikes() -> None:
    """
    Verify that the forward pass produces binary spikes.
    """

    membrane_difference = torch.tensor(
        [-1.0, 0.0, 1.0],
        requires_grad=True,
    )

    spikes = fast_sigmoid(membrane_difference)

    expected = torch.tensor([0.0, 1.0, 1.0])

    assert torch.equal(spikes, expected)


def test_surrogate_produces_gradient() -> None:
    """
    Verify that the surrogate function produces gradients.
    """

    membrane_difference = torch.tensor(
        [0.0],
        requires_grad=True,
    )

    spikes = fast_sigmoid(membrane_difference)

    loss = spikes.sum()

    loss.backward()

    assert membrane_difference.grad is not None
    assert membrane_difference.grad.item() > 0