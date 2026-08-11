"""
models/attention.py

Squeeze-and-Excitation (SE) channel attention block. A lightweight,
well-established attention mechanism that lets the model learn to
dynamically reweight feature channels per-image, rather than treating
all learned features as equally important.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SqueezeExciteBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Given a feature vector (already globally pooled, e.g. the output
    of a CNN backbone before its classification head), learns a
    per-channel importance weight and applies it via elementwise
    multiplication -- amplifying useful channels, suppressing less
    relevant ones, on a per-image basis.

    Attributes:
        excitation: A two-layer bottleneck network producing channel
            importance weights in [0, 1] via a final sigmoid.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 16) -> None:
        """Initialize the SE block.

        Args:
            num_channels: Number of input feature channels (must match
                the backbone's output feature dimension).
            reduction_ratio: How much to compress the bottleneck by.
                E.g. 16 means the hidden layer has num_channels/16
                units -- a standard default balancing expressiveness
                against added parameter count.

        Raises:
            ValueError: If num_channels is too small for the given
                reduction_ratio to produce at least 1 hidden unit.
        """
        super().__init__()

        hidden_dim = num_channels // reduction_ratio
        if hidden_dim < 1:
            raise ValueError(
                f"num_channels={num_channels} with reduction_ratio="
                f"{reduction_ratio} produces {hidden_dim} hidden units; "
                f"must be at least 1. Use a smaller reduction_ratio."
            )

        self.excitation = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_channels),
            nn.Sigmoid(),
        )

        logger.info(
            "SqueezeExciteBlock initialized | num_channels=%d | "
            "reduction_ratio=%d | hidden_dim=%d",
            num_channels, reduction_ratio, hidden_dim,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to already-pooled feature vectors.

        Args:
            features: FloatTensor, shape (batch_size, num_channels) --
                the output of global average pooling over a CNN
                backbone's final feature maps.

        Returns:
            FloatTensor, same shape as input, with each channel
            rescaled by its learned importance weight.
        """
        channel_weights = self.excitation(features)
        reweighted = features * channel_weights
        return reweighted