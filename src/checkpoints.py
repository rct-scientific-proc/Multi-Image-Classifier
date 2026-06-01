"""
Checkpoints — save and load .pt checkpoint dicts.

Naming convention: epoch_{epoch:03d}_val{val_accuracy:.4f}.pt
A 'best.pt' symlink/copy is always kept for the highest val accuracy seen.

Usage:
    save_checkpoint(path, epoch, model, optimizer, scheduler, metrics, hyperparams)
    ckpt = load_checkpoint(path, model, optimizer, scheduler)
"""

import shutil
from pathlib import Path

import torch
import torch.nn as nn


def checkpoint_name(epoch: int, val_accuracy: float) -> str:
    return f"epoch_{epoch:03d}_val{val_accuracy:.4f}.pt"


def save_checkpoint(
    checkpoint_dir: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,                          # LR scheduler or None
    metrics: dict,
    hyperparams: dict,
    keep_last: int = 3,
) -> Path:
    """Save a checkpoint and maintain a rolling window of the last N files.

    Always writes/overwrites 'best.pt' when val_accuracy is the highest seen.
    Returns the path of the file that was written.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    val_accuracy = metrics.get("accuracy", 0.0)
    name         = checkpoint_name(epoch, val_accuracy)
    path         = checkpoint_dir / name

    payload = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        # Strip numpy arrays (e.g. confusion_matrix) — not safe with weights_only=True
        # and not needed for resuming training.
        "metrics":              {k: v for k, v in metrics.items()
                                 if not hasattr(v, "__array__")},
        "hyperparams":          hyperparams,
    }
    torch.save(payload, path)

    # Overwrite best.pt if this is the best val accuracy so far
    best_path = checkpoint_dir / "best.pt"
    if not best_path.exists():
        shutil.copy2(path, best_path)
    else:
        prev = torch.load(best_path, weights_only=True)
        if val_accuracy >= prev["metrics"].get("accuracy", 0.0):
            shutil.copy2(path, best_path)

    # Rolling window — delete oldest checkpoints beyond keep_last
    # (excludes best.pt)
    existing = sorted(
        [p for p in checkpoint_dir.glob("epoch_*.pt")],
        key=lambda p: p.stat().st_mtime,
    )
    for old in existing[:-keep_last]:
        old.unlink(missing_ok=True)

    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> dict:
    """Load a checkpoint into model (and optionally optimizer/scheduler).

    Returns the full checkpoint dict so callers can read epoch, metrics, etc.
    weights_only=True prevents arbitrary code execution from untrusted files.
    """
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt

