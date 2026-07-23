"""
GPU-side batch augmentation and normalisation.

Two objects with deliberately different lifecycles:

    Normalizer   preprocessing. Must run on train, validation AND inference —
                 if the three disagree, every metric is quietly wrong. Its
                 stats therefore travel in the checkpoint.
    GpuAugment   train only. Never on validation or inference.

Everything operates on a whole batch already resident on the GPU, in float
[0, 1], shaped (B, C, H, W) — i.e. straight after the uint8 -> float step in
Trainer.train_one_epoch, so no data leaves the device and nothing is done
per-image on the CPU.

Every random parameter is drawn PER SAMPLE. This is the reason these ops are
hand-written rather than delegated to torchvision.transforms.v2: v2 draws once
per *call*, so handing it a batch of 256 applies one identical flip/crop/jitter
to all 256 images. Measured cost of the hand-rolled path is ~35x cheaper than
looping v2 per image (see test/bench_augment.py, which re-checks both claims).

Op order is fixed: geometric -> photometric -> erasing -> normalise. Erasing
comes after geometry so the erased box is not resampled or rotated away, and
normalisation is last so augmentation magnitudes stay interpretable in [0, 1].

Usage:
    from src.augment import GpuAugment, Normalizer, AUGMENT_DEFAULTS

    aug  = GpuAugment(AUGMENT_DEFAULTS, in_channels=3).to(device)
    norm = Normalizer.from_config({"normalize": "imagenet"}, in_channels=3).to(device)

    images = images.float().mul_(1 / 255)   # (B, C, H, W) in [0, 1]
    images = norm(aug(images))              # train
    images = norm(images)                   # validate / infer
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Flat scalars on purpose: this dict is merged into the settings dict that
# becomes `hyperparams`, and ExperimentLogger.log_hparams drops anything that
# is not int/float/str/bool. A nested dict would vanish from TensorBoard.
AUGMENT_DEFAULTS: dict = {
    "aug_enabled":      False,
    # geometric
    "aug_hflip_p":      0.5,
    "aug_vflip_p":      0.0,
    "aug_crop_padding": 4,      # pixels; 0 disables
    "aug_rotate_deg":   0.0,
    "aug_translate":    0.0,    # fraction of image size
    "aug_scale_jitter": 0.0,    # +/- fraction
    "aug_shear_deg":    0.0,
    # photometric (saturation/hue need 3 channels; skipped when C == 1)
    "aug_brightness":   0.0,
    "aug_contrast":     0.0,
    "aug_saturation":   0.0,
    "aug_hue":          0.0,    # fraction of the colour wheel, 0..0.5
    # erasing
    "aug_erase_p":      0.0,
    "aug_erase_min":    0.02,   # min fraction of image area
    "aug_erase_max":    0.20,
    "aug_erase_value":  0.0,    # fill, in [0, 1] space
}

# Named starting points for the GUI. "standard" is the usual CIFAR recipe:
# pad-and-crop plus a horizontal flip, which is what most published numbers use.
AUGMENT_PRESETS: dict[str, dict] = {
    "none": {**AUGMENT_DEFAULTS, "aug_enabled": False},
    "light": {
        **AUGMENT_DEFAULTS, "aug_enabled": True,
        "aug_hflip_p": 0.5, "aug_crop_padding": 4,
    },
    "standard": {
        **AUGMENT_DEFAULTS, "aug_enabled": True,
        "aug_hflip_p": 0.5, "aug_crop_padding": 4,
        "aug_brightness": 0.2, "aug_contrast": 0.2,
        "aug_erase_p": 0.25,
    },
    "heavy": {
        **AUGMENT_DEFAULTS, "aug_enabled": True,
        "aug_hflip_p": 0.5, "aug_crop_padding": 4,
        "aug_rotate_deg": 15.0, "aug_translate": 0.1, "aug_scale_jitter": 0.1,
        "aug_brightness": 0.4, "aug_contrast": 0.4,
        "aug_saturation": 0.4, "aug_hue": 0.05,
        "aug_erase_p": 0.5, "aug_erase_max": 0.33,
    },
}

NORMALIZE_MODES = ("none", "imagenet", "dataset", "custom")
DEFAULT_NORMALIZE = "none"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Per-sample random helpers
# ---------------------------------------------------------------------------

def _rand(b: int, device, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
    """(B,) uniform in [low, high) on *device*."""
    return torch.rand(b, device=device) * (high - low) + low


def _symmetric(b: int, device, mag: float) -> torch.Tensor:
    """(B,) uniform in [-mag, +mag)."""
    return (torch.rand(b, device=device) * 2 - 1) * mag


def _bernoulli(b: int, device, p: float) -> torch.Tensor:
    """(B,) bool mask, True with probability p."""
    return torch.rand(b, device=device) < p


# ---------------------------------------------------------------------------
# Geometric ops
# ---------------------------------------------------------------------------

def random_hflip(x: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return x
    m = _bernoulli(x.shape[0], x.device, p)
    return torch.where(m[:, None, None, None], x.flip(-1), x)


def random_vflip(x: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return x
    m = _bernoulli(x.shape[0], x.device, p)
    return torch.where(m[:, None, None, None], x.flip(-2), x)


def random_crop(x: torch.Tensor, padding: int) -> torch.Tensor:
    """Zero-pad by *padding* then take a random HxW crop, offset per sample."""
    if padding <= 0:
        return x
    b, _, h, w = x.shape
    xp = F.pad(x, (padding,) * 4)
    hp, wp = xp.shape[-2:]
    top = torch.randint(0, hp - h + 1, (b,), device=x.device)
    left = torch.randint(0, wp - w + 1, (b,), device=x.device)
    rows = top[:, None] + torch.arange(h, device=x.device)[None, :]      # (B,H)
    cols = left[:, None] + torch.arange(w, device=x.device)[None, :]     # (B,W)
    idx = torch.arange(b, device=x.device)[:, None, None]
    # Advanced-index B,H,W with a slice on C. Mixing a slice between advanced
    # indices moves the advanced dims first, giving (B,H,W,C) — hence permute.
    return xp[idx, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)


def random_affine(
    x: torch.Tensor,
    rotate_deg: float,
    translate: float,
    scale_jitter: float,
    shear_deg: float,
) -> torch.Tensor:
    """Per-sample rotate / translate / scale / shear in one resample.

    Composed as a single 2x3 theta per image so the batch costs one
    grid_sample rather than one per op. Note affine_grid maps *output* coords
    back to input, so theta is the inverse transform — immaterial here because
    every parameter is drawn from a symmetric range.
    """
    if rotate_deg <= 0 and translate <= 0 and scale_jitter <= 0 and shear_deg <= 0:
        return x

    b = x.shape[0]
    dev = x.device
    ang = _symmetric(b, dev, rotate_deg) * (math.pi / 180.0)
    shear = _symmetric(b, dev, shear_deg) * (math.pi / 180.0)
    scale = 1.0 + _symmetric(b, dev, scale_jitter)
    scale = scale.clamp(min=1e-3)
    # Grid coords span [-1, 1], so a translation of `translate` image-fractions
    # is 2*translate in grid units.
    tx = _symmetric(b, dev, translate) * 2.0
    ty = _symmetric(b, dev, translate) * 2.0

    cos, sin = torch.cos(ang), torch.sin(ang)
    tan_sh = torch.tan(shear)

    theta = torch.zeros(b, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0] = cos / scale
    theta[:, 0, 1] = (tan_sh - sin) / scale
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin / scale
    theta[:, 1, 1] = cos / scale
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)


# ---------------------------------------------------------------------------
# Photometric ops
# ---------------------------------------------------------------------------

# Rec.601 luma weights, used for both contrast and saturation.
_LUMA = (0.299, 0.587, 0.114)


def _grayscale(x: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) luma. Assumes 3 channels."""
    w = torch.tensor(_LUMA, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def random_brightness(x: torch.Tensor, mag: float) -> torch.Tensor:
    if mag <= 0:
        return x
    f = 1.0 + _symmetric(x.shape[0], x.device, mag)
    return (x * f[:, None, None, None]).clamp_(0.0, 1.0)


def random_contrast(x: torch.Tensor, mag: float) -> torch.Tensor:
    """Scale deviation from each image's own mean luma, so mean is preserved."""
    if mag <= 0:
        return x
    f = 1.0 + _symmetric(x.shape[0], x.device, mag)
    ref = (_grayscale(x) if x.shape[1] == 3 else x).mean(dim=(1, 2, 3), keepdim=True)
    return ((x - ref) * f[:, None, None, None] + ref).clamp_(0.0, 1.0)


def random_saturation(x: torch.Tensor, mag: float) -> torch.Tensor:
    """Blend towards greyscale. No-op on single-channel input."""
    if mag <= 0 or x.shape[1] != 3:
        return x
    f = 1.0 + _symmetric(x.shape[0], x.device, mag)
    gray = _grayscale(x)
    return (gray + (x - gray) * f[:, None, None, None]).clamp_(0.0, 1.0)


def random_hue(x: torch.Tensor, mag: float) -> torch.Tensor:
    """Rotate hue via YIQ, which is a plain 3x3 linear map — far cheaper than a
    round trip through HSV and differentiable-friendly. No-op unless C == 3.

    *mag* is a fraction of the colour wheel, so 0.5 is a full half-turn.
    """
    if mag <= 0 or x.shape[1] != 3:
        return x
    b = x.shape[0]
    theta = _symmetric(b, x.device, mag) * 2.0 * math.pi
    cos, sin = torch.cos(theta), torch.sin(theta)

    to_yiq = torch.tensor([[0.299, 0.587, 0.114],
                           [0.596, -0.274, -0.322],
                           [0.211, -0.523, 0.312]],
                          device=x.device, dtype=x.dtype)
    to_rgb = torch.tensor([[1.0, 0.956, 0.621],
                           [1.0, -0.272, -0.647],
                           [1.0, -1.106, 1.703]],
                          device=x.device, dtype=x.dtype)

    rot = torch.zeros(b, 3, 3, device=x.device, dtype=x.dtype)
    rot[:, 0, 0] = 1.0
    rot[:, 1, 1] = cos
    rot[:, 1, 2] = -sin
    rot[:, 2, 1] = sin
    rot[:, 2, 2] = cos

    m = to_rgb @ rot @ to_yiq                      # (B,3,3)
    out = torch.einsum("bij,bjhw->bihw", m, x)
    return out.clamp_(0.0, 1.0)


# ---------------------------------------------------------------------------
# Erasing
# ---------------------------------------------------------------------------

def random_erase(
    x: torch.Tensor,
    p: float,
    area_min: float,
    area_max: float,
    value: float,
    ratio: tuple[float, float] = (0.3, 3.3),
) -> torch.Tensor:
    """Cutout with a per-sample box position, size and aspect ratio.

    Variable box sizes still vectorise: the mask is built from broadcast
    comparisons against per-sample bounds rather than by slicing.
    """
    if p <= 0 or area_max <= 0:
        return x

    b, _, h, w = x.shape
    dev = x.device
    area = _rand(b, dev, area_min, area_max) * (h * w)
    log_r = _rand(b, dev, math.log(ratio[0]), math.log(ratio[1]))
    r = torch.exp(log_r)

    eh = (area * r).sqrt().round().long().clamp_(1, h)
    ew = (area / r).sqrt().round().long().clamp_(1, w)
    top = (torch.rand(b, device=dev) * (h - eh + 1).to(x.dtype)).long()
    left = (torch.rand(b, device=dev) * (w - ew + 1).to(x.dtype)).long()

    rows = torch.arange(h, device=dev)[None, :]
    cols = torch.arange(w, device=dev)[None, :]
    rmask = (rows >= top[:, None]) & (rows < (top + eh)[:, None])       # (B,H)
    cmask = (cols >= left[:, None]) & (cols < (left + ew)[:, None])     # (B,W)
    box = rmask[:, :, None] & cmask[:, None, :]                          # (B,H,W)
    box = box & _bernoulli(b, dev, p)[:, None, None]
    return x.masked_fill(box[:, None, :, :], value)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class GpuAugment(nn.Module):
    """Batched, per-sample augmentation for a GPU-resident float batch.

    Expects (B, C, H, W) float in [0, 1] and returns the same shape and range.
    Train-only: forward is a no-op in eval mode, and Trainer must not call it
    during validation regardless.
    """

    def __init__(self, config: dict | None = None, in_channels: int = 3):
        super().__init__()
        cfg = {**AUGMENT_DEFAULTS, **(config or {})}
        self.in_channels = int(in_channels)
        self.config = {k: cfg[k] for k in AUGMENT_DEFAULTS}

        c = self.config
        self.enabled = bool(c["aug_enabled"])
        self._hflip_p = float(c["aug_hflip_p"])
        self._vflip_p = float(c["aug_vflip_p"])
        self._crop_padding = int(c["aug_crop_padding"])
        self._rotate_deg = float(c["aug_rotate_deg"])
        self._translate = float(c["aug_translate"])
        self._scale_jitter = float(c["aug_scale_jitter"])
        self._shear_deg = float(c["aug_shear_deg"])
        self._brightness = float(c["aug_brightness"])
        self._contrast = float(c["aug_contrast"])
        self._saturation = float(c["aug_saturation"])
        self._hue = float(c["aug_hue"])
        self._erase_p = float(c["aug_erase_p"])
        self._erase_min = float(c["aug_erase_min"])
        self._erase_max = float(c["aug_erase_max"])
        self._erase_value = float(c["aug_erase_value"])

    @property
    def active_ops(self) -> list[str]:
        """Names of the ops that will actually do something — for logging."""
        if not self.enabled:
            return []
        colour = self.in_channels == 3
        checks = [
            ("hflip", self._hflip_p > 0),
            ("vflip", self._vflip_p > 0),
            ("crop", self._crop_padding > 0),
            ("affine", any((self._rotate_deg > 0, self._translate > 0,
                            self._scale_jitter > 0, self._shear_deg > 0))),
            ("brightness", self._brightness > 0),
            ("contrast", self._contrast > 0),
            ("saturation", self._saturation > 0 and colour),
            ("hue", self._hue > 0 and colour),
            ("erase", self._erase_p > 0),
        ]
        return [name for name, on in checks if on]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.training:
            return x

        # geometric
        x = random_hflip(x, self._hflip_p)
        x = random_vflip(x, self._vflip_p)
        x = random_crop(x, self._crop_padding)
        x = random_affine(x, self._rotate_deg, self._translate,
                          self._scale_jitter, self._shear_deg)
        # photometric
        x = random_brightness(x, self._brightness)
        x = random_contrast(x, self._contrast)
        x = random_saturation(x, self._saturation)
        x = random_hue(x, self._hue)
        # erasing last, so geometry cannot resample the box away
        x = random_erase(x, self._erase_p, self._erase_min,
                         self._erase_max, self._erase_value)
        return x

    def extra_repr(self) -> str:
        return f"enabled={self.enabled}, ops={self.active_ops}"


class Normalizer(nn.Module):
    """Per-channel (x - mean) / std, as buffers so .to(device) moves them.

    Applies to train, validation and inference alike. The stats are part of the
    model contract, not a training detail — a checkpoint evaluated with
    different stats than it was trained on gives silently wrong numbers, so
    `state()` is stored in the checkpoint and replayed at inference.
    """

    def __init__(self, mean, std, in_channels: int = 3):
        super().__init__()
        mean_t = self._as_tensor(mean, in_channels, "mean")
        std_t = self._as_tensor(std, in_channels, "std")
        if bool((std_t <= 0).any()):
            raise ValueError(f"normalisation std must be positive, got {std}")
        self.register_buffer("mean", mean_t.view(1, in_channels, 1, 1))
        self.register_buffer("std", std_t.view(1, in_channels, 1, 1))
        self.in_channels = int(in_channels)

    @staticmethod
    def _as_tensor(v, c: int, label: str) -> torch.Tensor:
        t = torch.as_tensor(v, dtype=torch.float32).flatten()
        if t.numel() == 1:
            t = t.repeat(c)
        if t.numel() != c:
            raise ValueError(f"{label} has {t.numel()} values, expected 1 or {c}")
        return t

    @property
    def is_identity(self) -> bool:
        return bool((self.mean == 0).all() and (self.std == 1).all())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_identity:
            return x
        return (x - self.mean) / self.std

    def state(self) -> dict:
        """Plain-Python stats for the checkpoint payload."""
        return {
            "normalize_mean": [round(float(v), 6) for v in self.mean.flatten()],
            "normalize_std": [round(float(v), 6) for v in self.std.flatten()],
        }

    def extra_repr(self) -> str:
        return (f"mean={[round(float(v), 4) for v in self.mean.flatten()]}, "
                f"std={[round(float(v), 4) for v in self.std.flatten()]}")

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict, in_channels: int = 3) -> "Normalizer":
        """Build from a settings dict.

        Recognised keys: ``normalize`` (one of NORMALIZE_MODES) and, for
        ``custom``/``dataset``, ``normalize_mean`` / ``normalize_std``.
        An unknown mode falls back to identity rather than raising, so a
        hand-edited settings file cannot stop training from starting.
        """
        mode = str(config.get("normalize", DEFAULT_NORMALIZE))
        if mode == "imagenet":
            mean, std = _IMAGENET_MEAN, _IMAGENET_STD
            if in_channels == 1:
                # Collapse to luma-weighted scalars so 1-channel input still
                # gets sensible centring instead of an error.
                mean = (sum(m * w for m, w in zip(_IMAGENET_MEAN, _LUMA)),)
                std = (sum(s * w for s, w in zip(_IMAGENET_STD, _LUMA)),)
            return cls(mean, std, in_channels)

        if mode in ("dataset", "custom"):
            mean = config.get("normalize_mean")
            std = config.get("normalize_std")
            if mean is None or std is None:
                # Nothing computed yet — identity beats guessing.
                return cls(0.0, 1.0, in_channels)
            return cls(mean, std, in_channels)

        return cls(0.0, 1.0, in_channels)

    @classmethod
    def from_checkpoint(cls, hyperparams: dict, in_channels: int = 3) -> "Normalizer":
        """Rebuild the exact normalisation a checkpoint was trained with.

        Prefers the explicit stats the Trainer wrote over re-deriving them from
        the mode name: the stored numbers are what the weights actually saw, and
        for mode="dataset" they cannot be recomputed without the training data.

        A checkpoint predating this feature has neither key and yields identity,
        which is exactly the plain ``/255`` those weights were trained under.
        """
        hp = hyperparams or {}
        mean, std = hp.get("normalize_mean"), hp.get("normalize_std")
        if mean is not None and std is not None:
            return cls(mean, std, in_channels)
        return cls.from_config(hp, in_channels)


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_dataset_stats(
    loader,
    device: str = "cpu",
    max_batches: int | None = None,
    progress=None,
) -> tuple[list[float], list[float]]:
    """Stream *loader* and return per-channel (mean, std) in [0, 1].

    Uses sum / sum-of-squares rather than accumulating batch means, so an
    undersized last batch does not skew the result.

    Parameters
    ----------
    loader:
        Yields (images, labels, gt) — images uint8 or float, (B, C, H, W).
    max_batches:
        Stop early. Useful for a quick estimate on a large dataset.
    progress:
        Optional callable(batches_done, total_or_None) for GUI feedback.
    """
    total = n = 0
    csum = csq = None

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        if images.dtype == torch.uint8:
            images = images.float().mul_(1.0 / 255.0)
        else:
            images = images.float()

        b, c = images.shape[0], images.shape[1]
        if csum is None:
            csum = torch.zeros(c, device=images.device, dtype=torch.float64)
            csq = torch.zeros(c, device=images.device, dtype=torch.float64)

        flat = images.permute(1, 0, 2, 3).reshape(c, -1).double()
        csum += flat.sum(dim=1)
        csq += (flat * flat).sum(dim=1)
        n += flat.shape[1]
        total += b
        if progress is not None:
            progress(i + 1, None)

    if csum is None or n == 0:
        raise ValueError("compute_dataset_stats: loader yielded no batches")

    mean = csum / n
    var = (csq / n) - mean * mean
    std = var.clamp_min(1e-12).sqrt()
    return ([round(float(v), 6) for v in mean],
            [round(float(v), 6) for v in std])
