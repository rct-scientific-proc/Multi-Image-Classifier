"""
Model — configurable CNN / torchvision backbone wrapper.

Two options:
  1. SimpleCNN  — lightweight custom CNN, good baseline for small images (e.g. 28x28)
  2. BackboneModel — thin wrapper around a torchvision backbone (e.g. ResNet, EfficientNet)
                     with the classifier head replaced to match num_classes.

Both accept in_channels=1 (grayscale) or in_channels=3 (RGB).

Usage:
    from src.model import build_model

    # Small custom CNN
    model = build_model("simple_cnn", in_channels=1, num_classes=11)

    # Torchvision backbone
    model = build_model("resnet18", in_channels=1, num_classes=11, pretrained=True)
"""

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Simple CNN — fast baseline for small images
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Lightweight CNN suitable for small images (28x28 upwards).

    Architecture: 3 conv blocks (conv → BN → ReLU → MaxPool) followed by
    an adaptive average pool and a two-layer classifier head.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 11):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Torchvision backbone wrapper
# ---------------------------------------------------------------------------

# Backbones supported and how to access their classifier head
_BACKBONE_CONFIGS: dict[str, dict] = {
    "resnet18":        {"builder": models.resnet18,        "head_attr": "fc",         "in_features": lambda m: m.fc.in_features},
    "resnet34":        {"builder": models.resnet34,        "head_attr": "fc",         "in_features": lambda m: m.fc.in_features},
    "resnet50":        {"builder": models.resnet50,        "head_attr": "fc",         "in_features": lambda m: m.fc.in_features},
    "efficientnet_b0": {"builder": models.efficientnet_b0, "head_attr": "classifier", "in_features": lambda m: m.classifier[1].in_features},
    "efficientnet_b1": {"builder": models.efficientnet_b1, "head_attr": "classifier", "in_features": lambda m: m.classifier[1].in_features},
    "mobilenet_v3_small": {"builder": models.mobilenet_v3_small, "head_attr": "classifier", "in_features": lambda m: m.classifier[0].in_features},
}

AVAILABLE_BACKBONES = ["simple_cnn"] + list(_BACKBONE_CONFIGS.keys())


class BackboneModel(nn.Module):
    """Torchvision backbone with a replaced classification head.

    When in_channels != 3 a 1x1 conv adapter is prepended so the backbone
    always receives a 3-channel input without altering pretrained weights.
    """

    def __init__(
        self,
        backbone_name: str,
        in_channels: int = 3,
        num_classes: int = 11,
        pretrained: bool = False,
    ):
        super().__init__()

        cfg     = _BACKBONE_CONFIGS[backbone_name]
        weights = "DEFAULT" if pretrained else None
        base    = cfg["builder"](weights=weights)

        # Channel adapter — keeps pretrained conv1 weights intact
        self.channel_adapter = (
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            if in_channels != 3
            else nn.Identity()
        )

        # Replace the classification head
        in_features = cfg["in_features"](base)
        new_head    = nn.Linear(in_features, num_classes)

        head_attr = cfg["head_attr"]
        if head_attr == "classifier" and isinstance(getattr(base, head_attr), nn.Sequential):
            # Replace only the final Linear inside the Sequential
            seq   = getattr(base, head_attr)
            last  = len(seq) - 1
            seq[last] = new_head
        else:
            setattr(base, head_attr, new_head)

        self.backbone = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_adapter(x)
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    backbone_name: str,
    in_channels: int = 1,
    num_classes: int = 11,
    pretrained: bool = False,
) -> nn.Module:
    """Build and return a model by name.

    Parameters
    ----------
    backbone_name:
        ``"simple_cnn"`` or any key in AVAILABLE_BACKBONES.
    in_channels:
        1 for grayscale, 3 for RGB.
    num_classes:
        Number of output classes (including hard_negative if used).
    pretrained:
        Load ImageNet weights (ignored for simple_cnn).
    """
    if backbone_name == "simple_cnn":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)

    if backbone_name not in _BACKBONE_CONFIGS:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Choose from: {AVAILABLE_BACKBONES}"
        )

    return BackboneModel(
        backbone_name=backbone_name,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
    )
