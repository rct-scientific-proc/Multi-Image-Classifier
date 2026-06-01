"""
Checkpoints — save and load .pt checkpoint dicts.

Naming convention: epoch_{epoch:03d}_acc{val_accuracy:.4f}_{metric}{val:.4f}.pt
A 'best.pt' copy is always kept for the best target-metric value seen.

Usage:
    save_checkpoint(path, epoch, model, optimizer, scheduler, metrics, hyperparams)
    ckpt = load_checkpoint(path, model, optimizer, scheduler)
"""

import shutil
from pathlib import Path

import torch
import torch.nn as nn

# Metrics where a lower value is better (everything else is maximised).
_MINIMISE = {"avg_loss"}


def _is_better(new_val: float, old_val: float, metric: str) -> bool:
    if metric in _MINIMISE:
        return new_val <= old_val
    return new_val >= old_val


def checkpoint_name(
    epoch: int,
    val_accuracy: float,
    target_metric: str = "accuracy",
    target_val: float | None = None,
) -> str:
    # Sanitise metric name: keep alphanumerics + underscore only
    safe = "".join(c if c.isalnum() or c == "_" else "" for c in target_metric)
    if target_val is not None and safe != "accuracy":
        return f"epoch_{epoch:03d}_acc{val_accuracy:.4f}_{safe}{target_val:.4f}.pt"
    return f"epoch_{epoch:03d}_acc{val_accuracy:.4f}.pt"


def save_checkpoint(
    checkpoint_dir: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,                          # LR scheduler or None
    metrics: dict,
    hyperparams: dict,
    keep_last: int = 3,
    target_metric: str = "accuracy",
) -> Path:
    """Save a checkpoint and maintain a rolling window of the last N files.

    Always writes/overwrites 'best.pt' when the target metric is the best seen.
    Returns the path of the file that was written.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    val_accuracy = metrics.get("accuracy", 0.0)
    target_val   = metrics.get(target_metric, val_accuracy)
    name         = checkpoint_name(epoch, val_accuracy, target_metric, target_val)
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

    # Overwrite best.pt if this checkpoint is better on the target metric
    best_path = checkpoint_dir / "best.pt"
    if not best_path.exists():
        shutil.copy2(path, best_path)
    else:
        prev      = torch.load(best_path, weights_only=True)
        prev_val  = prev["metrics"].get(target_metric,
                        prev["metrics"].get("accuracy", 0.0))
        if _is_better(target_val, prev_val, target_metric):
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

