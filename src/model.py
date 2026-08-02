"""
Model — configurable CNN / torchvision backbone wrapper.

Two options:
  1. SimpleCNN  — lightweight custom CNN, good baseline for small images (e.g. 28x28)
  2. BackboneModel — thin wrapper around a torchvision backbone (e.g. ResNet, EfficientNet)
                     with the classifier head replaced to match num_classes.

Both accept in_channels=1 (grayscale) or in_channels=3 (RGB).

Offline machines: `pretrained=True` normally makes torchvision download the
ImageNet weights on first use, which fails without internet access. Instead,
run `download_pretrained_weights()` on a connected machine (Models menu in the
GUI), copy the folder across, and pass it as `weights_dir` — the weights are
then read from disk and the network is never touched.

Usage:
    from src.model import build_model

    # Small custom CNN
    model = build_model("simple_cnn", in_channels=1, num_classes=11)

    # Torchvision backbone (downloads ImageNet weights on first use)
    model = build_model("resnet18", in_channels=1, num_classes=11, pretrained=True)

    # Same, but weights come from a folder copied onto an offline machine
    model = build_model("resnet18", in_channels=1, num_classes=11,
                        pretrained=True, weights_dir="pretrained_weights")
"""

import re
from pathlib import Path
from urllib.parse import urlparse

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
    # [-1], not [0]: this classifier is Linear(576,1024) → Hardswish → Dropout
    # → Linear(1024, n), and the replacement below swaps the LAST linear — so
    # the new head must accept that layer's 1024 inputs, not the first's 576.
    "mobilenet_v3_small": {"builder": models.mobilenet_v3_small, "head_attr": "classifier", "in_features": lambda m: m.classifier[-1].in_features},
}

# Backbones with downloadable ImageNet weights — everything but simple_cnn.
PRETRAINED_BACKBONES = list(_BACKBONE_CONFIGS.keys())

AVAILABLE_BACKBONES = ["simple_cnn"] + PRETRAINED_BACKBONES


# ---------------------------------------------------------------------------
# Pretrained weight files (offline support)
# ---------------------------------------------------------------------------

def pretrained_weights_url(backbone_name: str) -> str:
    """The download URL of *backbone_name*'s DEFAULT ImageNet weights."""
    if backbone_name not in _BACKBONE_CONFIGS:
        raise ValueError(
            f"'{backbone_name}' has no pretrained weights. "
            f"Choose from: {PRETRAINED_BACKBONES}"
        )
    return models.get_model_weights(backbone_name).DEFAULT.url


def pretrained_weights_filename(backbone_name: str) -> str:
    """torchvision's own filename for the weights, e.g. resnet18-f37072fd.pth."""
    return Path(urlparse(pretrained_weights_url(backbone_name)).path).name


def find_local_weights(backbone_name: str, weights_dir: str | None) -> Path | None:
    """The weight file for *backbone_name* inside *weights_dir*, if present."""
    if not weights_dir or backbone_name not in _BACKBONE_CONFIGS:
        return None
    path = Path(weights_dir) / pretrained_weights_filename(backbone_name)
    return path if path.is_file() else None


def download_pretrained_weights(backbone_name: str, weights_dir: str,
                                progress: bool = False) -> Path:
    """Fetch *backbone_name*'s ImageNet weights into *weights_dir*.

    Copy the folder to an offline machine and pass it to build_model as
    `weights_dir`. Returns the file path; skips the download when the file is
    already there. torch.hub verifies the sha256 prefix embedded in the
    filename and downloads via a temp file, so an interrupted or corrupted
    transfer never leaves a plausible-looking `.pth` behind.
    """
    url = pretrained_weights_url(backbone_name)
    target = Path(weights_dir) / pretrained_weights_filename(backbone_name)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    match = re.search(r"-([0-9a-f]{8,})\.", target.name)
    torch.hub.download_url_to_file(
        url, str(target),
        hash_prefix=match.group(1) if match else None,
        progress=progress,
    )
    return target


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
        weights_path: str | Path | None = None,
    ):
        super().__init__()

        cfg = _BACKBONE_CONFIGS[backbone_name]
        if weights_path is not None:
            # Offline path: build empty, then load the copied weight file.
            base = cfg["builder"](weights=None)
            # weights_only=True — a state dict is all this file should contain.
            state = torch.load(str(weights_path), map_location="cpu",
                               weights_only=True)
            base.load_state_dict(state)
        else:
            base = cfg["builder"](weights="DEFAULT" if pretrained else None)

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
    weights_dir: str | None = None,
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
    weights_dir:
        Folder of pre-downloaded weight files (see
        ``download_pretrained_weights``). When it holds this backbone's file,
        the weights are read from disk and nothing is downloaded — the offline
        case. Otherwise torchvision downloads as usual. Ignored unless
        *pretrained*.
    """
    if backbone_name == "simple_cnn":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)

    if backbone_name not in _BACKBONE_CONFIGS:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Choose from: {AVAILABLE_BACKBONES}"
        )

    weights_path = find_local_weights(backbone_name, weights_dir) if pretrained else None

    return BackboneModel(
        backbone_name=backbone_name,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        weights_path=weights_path,
    )
