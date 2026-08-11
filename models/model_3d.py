"""
models/model_3d.py

True 3D CNN classifier for knee MRI volumes, using MONAI's 3D
DenseNet121. Unlike the 2D backbones in models/baseline_cnn.py
(which process one slice at a time), this model consumes an entire
fixed-depth volume at once, allowing it to learn patterns that span
multiple adjacent slices.
"""

import logging

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

logger = logging.getLogger(__name__)

NUM_FINDINGS = 12


class KneeMRI3DClassifier(nn.Module):
    """3D DenseNet121-based multi-label classifier for knee MRI volumes.

    Consumes a full 3D volume (fixed depth, from
    preprocessing.volume_transforms.build_3d_volume_from_slices) and
    produces 12 independent logits, one per finding. Unlike the 2D
    backbones, this model has no separate backbone/head split exposed
    -- MONAI's DenseNet121 handles the full architecture internally,
    configured here to output num_findings logits directly.

    Attributes:
        network: The full 3D DenseNet121 network, configured for our
            single-channel (grayscale MRI) input and 12-way output.
    """

    def __init__(self, num_findings: int = NUM_FINDINGS) -> None:
        """Initialize the 3D classifier.

        Args:
            num_findings: Number of independent binary findings to
                predict (12 for this competition).
        """
        super().__init__()

        self.network = DenseNet121(
            spatial_dims=3,
            in_channels=1,  # grayscale MRI, unlike the 3-channel trick used for 2D backbones
            out_channels=num_findings,
        )

        logger.info(
            "KneeMRI3DClassifier initialized | architecture=DenseNet121-3D | "
            "num_findings=%d",
            num_findings,
        )

    def forward(self, volumes: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            volumes: FloatTensor, shape (batch_size, 1, D, H, W) --
                note the extra channel dimension (1, for grayscale)
                and depth dimension (D), compared to the 2D models'
                (batch_size, 3, H, W) input.

        Returns:
            FloatTensor of raw logits, shape (batch_size, num_findings).
        """
        logits = self.network(volumes)
        return logits