"""
models/baseline_cnn.py

Configurable multi-label classifier for knee MRI slices: a pretrained
CNN backbone (via timm) with a custom classification head producing
12 independent logits, one per finding. Supports multiple backbone
architectures (ResNet18, EfficientNet-B0, ...) through a single
config-driven class, so the rest of the pipeline (training loop,
dataloaders, metrics) works identically regardless of which backbone
is active.
"""

import logging

import timm
import torch
import torch.nn as nn

from models.attention import SqueezeExciteBlock

logger = logging.getLogger(__name__)

NUM_FINDINGS = 12

SUPPORTED_BACKBONES = (
    "resnet18",
    "efficientnet_b0",
    "efficientnet_b1",
    "convnext_tiny",
    "swin_tiny_patch4_window7_224",
)


class KneeMRIClassifier(nn.Module):
    """Configurable CNN-based multi-label classifier for knee MRI findings.

    Uses a timm-provided backbone pretrained on ImageNet, with its
    original classification head removed and replaced with a dropout
    layer followed by a linear layer producing 12 independent logits
    (one per finding). No sigmoid is applied here -- raw logits are
    returned, since the loss function (BCEWithLogitsLoss / FocalLoss)
    applies sigmoid internally for numerical stability.

    Attributes:
        backbone: The pretrained feature-extraction network.
        dropout: Dropout layer applied to pooled features before the
            classification head, active only during training.
        head: Final linear layer mapping features to 12 logits.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_findings: int = NUM_FINDINGS,
        pretrained: bool = True,
        use_attention: bool = False,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initialize the model.

        Args:
            backbone_name: Which timm backbone to use. Must be one of
                SUPPORTED_BACKBONES.
            num_findings: Number of independent binary findings to
                predict (12 for this competition).
            pretrained: If True, load ImageNet-pretrained weights for
                the backbone. Should be True for real training; False
                is mainly useful for fast architecture testing without
                a network download.
            use_attention: If True, apply a Squeeze-and-Excitation
                channel attention block to the backbone's pooled
                features before the classification head.
            dropout_rate: Probability of zeroing each feature before
                the classification head, applied only during training
                (has no effect in model.eval() mode). Helps prevent
                overfitting on small datasets.

        Raises:
            ValueError: If backbone_name is not in SUPPORTED_BACKBONES.
        """
        super().__init__()

        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Supported backbones: {SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone_name

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # removes the original ImageNet classification head
        )

        backbone_out_features = self.backbone.num_features

        self.use_attention = use_attention
        if use_attention:
            self.attention = SqueezeExciteBlock(backbone_out_features)
        else:
            self.attention = None

        self.dropout = nn.Dropout(p=dropout_rate)
        self.head = nn.Linear(backbone_out_features, num_findings)

        logger.info(
            "KneeMRIClassifier initialized | backbone=%s | pretrained=%s | "
            "backbone_features=%d | num_findings=%d | dropout_rate=%.2f",
            backbone_name, pretrained, backbone_out_features, num_findings, dropout_rate,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            images: FloatTensor, shape (batch_size, 3, H, W).

        Returns:
            FloatTensor of raw logits, shape (batch_size, num_findings).
            Apply torch.sigmoid() to convert to probabilities, or let
            BCEWithLogitsLoss/FocalLoss handle that internally during
            training.
        """
        features = self.backbone(images)

        if self.attention is not None:
            features = self.attention(features)

        features = self.dropout(features)
        logits = self.head(features)
        return logits


class BaselineKneeCNN(KneeMRIClassifier):
    """Backward-compatible alias: ResNet18-only version of KneeMRIClassifier.

    Kept so existing code (e.g. notebooks/test_train_loop.py) that
    imports BaselineKneeCNN continues to work unchanged. New code
    should prefer KneeMRIClassifier(backbone_name=...) directly.
    """

    def __init__(
        self,
        num_findings: int = NUM_FINDINGS,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initialize a ResNet18-backboned classifier.

        Args:
            num_findings: Number of independent binary findings to predict.
            pretrained: If True, load ImageNet-pretrained weights.
            dropout_rate: Probability of zeroing each feature before
                the classification head, applied only during training.
        """
        super().__init__(
            backbone_name="resnet18",
            num_findings=num_findings,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )