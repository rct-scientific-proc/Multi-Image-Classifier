"""
Sliding-window CPU inference for a Multi-Image-Classifier checkpoint.

Sweeps a trained chip classifier over one large image (a geotiff, as a numpy
array): the image is tiled into fixed-size snippets with a set stride, each
snippet is classified independently, and every result is tagged with the
snippet's pixel centre in the original image.

CPU only, by design — modern torch on a modern CPU classifies these snippets
fast enough, it keeps this dependency-light and deployable anywhere, and nothing
here touches CUDA.

Input
-----
One image as a numpy array, channels-LAST (the layout the H5 `images` dataset
uses):

    (H, W)      a single-channel scene
    (H, W, C)   a C-channel scene

uint8/integer pixels are scaled by 1/255; float pixels are assumed already in
[0, 1] — the same convention the trainer uses. The checkpoint's own
normalisation is then replayed, so each snippet is fed exactly as the model saw
its training chips. A snippet whose channel count differs from the model's is
converted (grayscale<->RGB) when ``auto_channels`` is on.

Stride
------
``stride`` is the pixel step between window origins. Precedence: an explicit
``stride`` wins; otherwise ``overlap`` (the fraction of the window neighbours
share) sets ``stride = round(window·(1-overlap))``; otherwise the default steps
20% of the window (``stride = round(0.2·window)`` = 51 for a 256px window,
matching the "next snippet at column ~50" convention).

Return
------
A plain dict of JSON-serialisable primitives (no numpy types), with a
``predictions`` list of one record per snippet:

    {
      "checkpoint": "...", "class_names": [...], "num_classes": 61,
      "window": 256, "stride": 51,
      "image_height": H, "image_width": W, "grid_rows": R, "grid_cols": C,
      "num_samples": N,
      "predictions": [
        {"index": 0, "row": 0, "col": 0, "x0": 0, "y0": 0,
         "center_x": 128, "center_y": 128,
         "pred_index": 12, "pred_class": "...", "pred_prob": 0.87,
         "probs": [ ... per class ... ]},
        ...
      ],
    }

The ``predictions`` list is a plain list, so sweeps of several scenes concatenate
trivially; ``center_x`` / ``center_y`` are the pixel coordinates you map through
the raster transform to geo-coordinates.

Usage
-----
    from inference.inference import load_model, infer_sliding

    clf = load_model("checkpoints/best.pt")             # load once
    result = infer_sliding(scene_hwc, clf,              # sweep a scene
                           window=256, stride=51)       # uses all CPU cores

    # one-off (loads the checkpoint for this call):
    result = infer_sliding(scene_hwc, "checkpoints/best.pt")

CLI:
    python inference/inference.py --checkpoint best.pt --input scene.npy \
        --window 256 --stride 51 --output windows.json [--top-k 5] [--no-probs]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch

# Make ``src`` importable regardless of working directory — this file lives in a
# sibling package of src/, not under it.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.augment import Normalizer          # noqa: E402
from src.model import build_model           # noqa: E402

# Rec.601 luma weights, for the grayscale <-> RGB channel reconciliations.
_LUMA = (0.299, 0.587, 0.114)
_DECIMALS = 6


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _infer_num_classes(state_dict: dict) -> int:
    """Fallback for checkpoints predating stored class names.

    The classification head is the last 2-D weight in the state dict; its first
    dimension is the class count.
    """
    n = None
    for val in state_dict.values():
        if hasattr(val, "ndim") and val.ndim == 2:
            n = int(val.shape[0])
    if n is None:
        raise ValueError("could not infer num_classes from the checkpoint")
    return n


class Classifier:
    """A checkpoint loaded once, ready to classify snippet batches on CPU."""

    def __init__(self, checkpoint_path: Union[str, Path]):
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        # weights_only=True: refuse arbitrary pickle payloads. The checkpoint
        # holds only tensors, lists and dicts, so this loads cleanly.
        ckpt = torch.load(self.checkpoint_path, map_location="cpu",
                          weights_only=True)
        hp = ckpt.get("hyperparams", {}) or {}

        self.backbone    = str(hp.get("backbone", "simple_cnn"))
        self.in_channels = int(hp.get("in_channels", 1))
        self.class_names = list(ckpt.get("classes") or [])

        state = ckpt["model_state_dict"]
        num_classes = len(self.class_names) or _infer_num_classes(state)
        if not self.class_names:
            self.class_names = [f"class_{i}" for i in range(num_classes)]
        self.num_classes = num_classes

        # pretrained=False on purpose: the state dict overwrites every weight
        # anyway, and pretrained=True would trigger an ImageNet download that a
        # deployed, possibly-offline inference host should never need.
        model = build_model(self.backbone, in_channels=self.in_channels,
                            num_classes=num_classes, pretrained=False)
        model.load_state_dict(state)
        model.eval()
        self.model = model

        # Replay training-time normalisation; a checkpoint without stored stats
        # yields identity (plain [0, 1]), exactly what those weights trained on.
        self.normalizer = Normalizer.from_checkpoint(hp, self.in_channels)
        self.normalizer.eval()

    def _forward(self, batch: np.ndarray, auto_channels: bool = True) -> np.ndarray:
        """Classify one batch of snippets -> softmax probs (N, num_classes).

        *batch* is (N, H, W) or (N, H, W, C), channels last, as sliced from the
        scene. uint8 is scaled by 1/255; float is taken as-is. Normalisation and
        softmax mirror the training / evaluation path exactly.
        """
        nhwc = _to_nhwc(batch, self.in_channels, auto_channels)
        if np.issubdtype(nhwc.dtype, np.integer):
            x = nhwc.astype(np.float32) / 255.0
        else:
            x = nhwc.astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        with torch.inference_mode():
            tensor = self.normalizer(tensor)
            probs = torch.softmax(self.model(tensor), dim=1)
        return probs.numpy()

    def __repr__(self) -> str:
        return (f"Classifier(backbone={self.backbone!r}, "
                f"in_channels={self.in_channels}, "
                f"num_classes={self.num_classes})")


def load_model(checkpoint_path: Union[str, Path]) -> Classifier:
    """Load a checkpoint into a reusable :class:`Classifier` (CPU)."""
    return Classifier(checkpoint_path)


# ---------------------------------------------------------------------------
# Input shaping
# ---------------------------------------------------------------------------

def _to_nhwc(array: np.ndarray, in_channels: int, auto_channels: bool) -> np.ndarray:
    """Coerce a snippet batch to (N, H, W, in_channels)."""
    a = np.asarray(array)
    if a.ndim == 3:                          # (N, H, W) single-channel batch
        a = a[..., None]
    elif a.ndim != 4:
        raise ValueError(
            f"expected a snippet batch of shape (N,H,W) or (N,H,W,C), got {a.shape}")

    c = a.shape[-1]
    if c != in_channels:
        if not auto_channels:
            raise ValueError(
                f"snippet has {c} channel(s) but the model expects {in_channels}; "
                f"pass auto_channels=True to convert automatically")
        a = _reconcile_channels(a, c, in_channels)
    return a


def _reconcile_channels(a: np.ndarray, have: int, want: int) -> np.ndarray:
    if have == 1 and want == 3:
        return np.repeat(a, 3, axis=-1)                     # gray -> RGB
    if have == 3 and want == 1:                             # RGB -> luma
        w = np.asarray(_LUMA, dtype=np.float64)
        gray = (a.astype(np.float64) * w).sum(axis=-1, keepdims=True)
        return gray.astype(a.dtype)
    raise ValueError(f"cannot convert {have}-channel input to {want} channels")


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def _window_origins(length: int, window: int, stride: int, edge: str) -> list[int]:
    """Start offsets tiling *length* with a *window*/*stride*.

    edge="cover" (default) appends a final window flush against the far edge so
    the right/bottom strip is not dropped when the image does not divide evenly;
    that last window is full-size and simply overlaps its neighbour more.
    edge="drop" keeps only whole-stride windows.
    """
    if window > length:
        raise ValueError(
            f"window ({window}) is larger than the image dimension ({length}); "
            f"reduce the window or pad the image")
    pos = list(range(0, length - window + 1, max(1, stride)))
    if edge == "cover" and (not pos or pos[-1] != length - window):
        pos.append(length - window)
    return pos


def _prediction_fields(prob: np.ndarray, class_names: list[str],
                       top_k: int | None, include_probs: bool) -> dict:
    """The classification part of a record (no geometry)."""
    j = int(prob.argmax())
    rec: dict = {
        "pred_index": j,
        "pred_class": class_names[j],
        "pred_prob":  round(float(prob[j]), _DECIMALS),
    }
    if include_probs:
        rec["probs"] = [round(float(v), _DECIMALS) for v in prob]
    if top_k:
        order = np.argsort(prob)[::-1][:top_k]
        rec["top_k"] = [
            {"index": int(k), "class": class_names[int(k)],
             "prob": round(float(prob[int(k)]), _DECIMALS)}
            for k in order
        ]
    return rec


def infer_sliding(
    image: np.ndarray,
    model: Union[str, Path, Classifier],
    *,
    window: int = 256,
    stride: int | None = None,
    overlap: float | None = None,
    edge: str = "cover",
    batch_size: int = 64,
    top_k: int | None = None,
    include_probs: bool = True,
    auto_channels: bool = True,
    num_threads: int | None = None,
) -> dict:
    """Sweep the classifier over one image as a sliding window.

    *image* is a single (H, W) or (H, W, C) array (a whole scene), not a batch.
    *model* is a loaded :class:`Classifier` (reused as-is) or a checkpoint path
    (loaded for this call — pass a loaded Classifier when sweeping many scenes).

    *num_threads* sets torch's CPU intra-op thread count for the sweep; None
    (default) uses every core, which is ~4x faster than single-threaded on a
    many-core box. The previous global setting is restored on return, so calling
    this does not quietly reconfigure torch for the rest of the process.

    See the module docstring for the input contract, stride precedence and the
    returned record shape. Windows are streamed through the model *batch_size* at
    a time, so memory is bounded by one batch of snippets rather than the whole
    tiling — safe on large rasters.
    """
    clf = model if isinstance(model, Classifier) else load_model(model)

    a = np.asarray(image)
    if a.ndim not in (2, 3):
        raise ValueError(
            f"infer_sliding expects one (H,W) or (H,W,C) image, got shape {a.shape}")
    h, w = a.shape[0], a.shape[1]

    if stride is None:
        stride = (max(1, round(window * (1.0 - overlap))) if overlap is not None
                  else max(1, round(window * 0.2)))

    ys = _window_origins(h, window, stride, edge)
    xs = _window_origins(w, window, stride, edge)
    grid = [(ri, ci, x0, y0)
            for ri, y0 in enumerate(ys)
            for ci, x0 in enumerate(xs)]          # row-major: x advances fastest

    threads = (os.cpu_count() or 1) if num_threads is None else max(1, int(num_threads))
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    half = window // 2
    records: list[dict] = []
    try:
        for start in range(0, len(grid), max(1, batch_size)):
            cells = grid[start:start + batch_size]
            batch = np.stack([a[y0:y0 + window, x0:x0 + window]
                              for _, _, x0, y0 in cells])
            probs = clf._forward(batch, auto_channels=auto_channels)
            for (ri, ci, x0, y0), prob in zip(cells, probs):
                records.append({
                    "index":    len(records),
                    "row":      int(ri),
                    "col":      int(ci),
                    "x0":       int(x0),
                    "y0":       int(y0),
                    "center_x": int(x0 + half),
                    "center_y": int(y0 + half),
                    **_prediction_fields(prob, clf.class_names, top_k, include_probs),
                })
    finally:
        # Leave torch's global thread count as we found it.
        torch.set_num_threads(prev_threads)

    return {
        "checkpoint":   clf.checkpoint_path,
        "class_names":  list(clf.class_names),
        "num_classes":  clf.num_classes,
        "in_channels":  clf.in_channels,
        "backbone":     clf.backbone,
        "window":       int(window),
        "stride":       int(stride),
        "num_threads":  int(threads),
        "image_height": int(h),
        "image_width":  int(w),
        "grid_rows":    len(ys),
        "grid_cols":    len(xs),
        "num_samples":  len(records),
        "predictions":  records,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sliding-window CPU inference over one image (numpy array).")
    ap.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    ap.add_argument("--input", required=True,
                    help="Path to a .npy of one (H,W) or (H,W,C) image")
    ap.add_argument("--output", help="Write JSON here (default: stdout)")
    ap.add_argument("--window", type=int, default=256, help="Window size (px)")
    ap.add_argument("--stride", type=int, default=None,
                    help="Pixels between windows (default: 20%% of --window)")
    ap.add_argument("--overlap", type=float, default=None,
                    help="Fraction shared by neighbours; sets stride if --stride omitted")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-threads", type=int, default=None,
                    help="CPU threads for the sweep (default: all cores)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="Also include the top-K classes per snippet")
    ap.add_argument("--no-probs", action="store_true",
                    help="Omit the full per-class probability vector")
    args = ap.parse_args(argv)

    try:
        array = np.load(args.input)
    except (OSError, ValueError) as exc:
        print(f"ERROR: could not load {args.input}: {exc}", file=sys.stderr)
        return 1

    result = infer_sliding(array, args.checkpoint, window=args.window,
                           stride=args.stride, overlap=args.overlap,
                           batch_size=args.batch_size, top_k=args.top_k,
                           include_probs=not args.no_probs,
                           num_threads=args.num_threads)

    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {result['num_samples']} predictions -> {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
