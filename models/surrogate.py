""" Surrogate-gradient functions for spiking neural networks"""

from __future__ import annotations
import torch

class FastSigmoid(torch.autograd.Function):
    """
    Fast-sigmoid surrogate gradient function

    Forward pass produces a binary spike.
    Backword pass uses a smooth approximation of the sigmoid function.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        membrane_difference: torch.Tensor,
        slope: float,
    ) -> torch.Tensor:
        """ Generates a binary spike """
        ctx.save_for_backward(membrane_difference)
        ctx.slope = slope
        return (membrane_difference >= 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Compute the surrogate gradient during backward pass or backpropagation"""
        (membrane_difference,) = ctx.saved_tensors
        slope = ctx.slope

        gradient = 1.0 / (1.0 + slope * membrane_difference.abs()) ** 2
        grad_input = grad_output * gradient
        return grad_input, None

def fast_sigmoid(
    membrane_difference: torch.Tensor,
    slope: float = 25.0,
) -> torch.Tensor:
    """
            Apply the fast-sigmoid surrogate spike function.

            Parameters
            ----------
            membrane_difference:
                Difference between membrane potential and threshold.

            slope:
                Controls the sharpness of the surrogate gradient.

            Returns
            -------
            torch.Tensor
                Binary spike tensor.
    """

    return FastSigmoid.apply(
        membrane_difference,
        slope,
    )