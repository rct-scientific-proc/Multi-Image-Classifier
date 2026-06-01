"""
Trainer — train_one_epoch(), validate(), cancel token, metric callbacks.

Design:
  - All heavy work runs inside Trainer methods; the GUI calls them from a QThread.
  - Metrics are emitted via an on_epoch_end callback so both CLI and GUI can consume them.
  - A threading.Event cancel token lets the GUI stop training cleanly between epochs.

Usage:
    import threading
    from src.trainer import Trainer

    cancel = threading.Event()
    trainer = Trainer(
        model, optimizer, scheduler,
        train_loader, val_loader,
        device="cuda",
        on_epoch_end=lambda info: print(info),
        cancel_event=cancel,
    )
    trainer.fit(epochs=20, checkpoint_dir="checkpoints", hyperparams={})
"""

import threading
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.metrics import MetricTracker, DEFAULT_TARGET_METRIC
from src.checkpoints import save_checkpoint
from src.logger import ExperimentLogger


class FocalLoss(nn.Module):
    """Multi-class focal loss  FL = -α (1-p)^γ log(p).

    Parameters
    ----------
    gamma : float
        Focusing parameter. 0 = standard cross-entropy. Typical value 2.
    alpha : float | None
        Optional uniform class weight scalar. Pass a 1-D tensor for per-class
        weights (same semantics as `nn.CrossEntropyLoss(weight=...)).
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p  = F.log_softmax(logits, dim=1)
        ce     = F.nll_loss(log_p, targets, weight=self.alpha, reduction="none")
        p      = torch.exp(-ce)                      # p_t
        focal  = (1.0 - p) ** self.gamma * ce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,                          # LR scheduler or None
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        on_epoch_end: Callable[[dict], None] | None = None,
        on_batch_end: Callable[[dict], None] | None = None,
        cancel_event: threading.Event | None = None,
        target_metric: str = DEFAULT_TARGET_METRIC,
        logger: "ExperimentLogger | None" = None,
        keep_last: int = 3,
        criterion: nn.Module | None = None,
    ):
        self.model        = model.to(device)
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.on_epoch_end = on_epoch_end
        self.on_batch_end = on_batch_end
        self.cancel_event  = cancel_event or threading.Event()
        self.criterion     = criterion if criterion is not None else nn.CrossEntropyLoss()
        self._num_classes  = len(train_loader.dataset.classes)
        self.target_metric = target_metric
        self.logger        = logger
        self.keep_last     = keep_last
        self._class_names  = list(train_loader.dataset.classes)

    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        tracker = MetricTracker(self._num_classes)

        for batch_idx, (images, labels, _gt) in enumerate(self.train_loader):
            if self.cancel_event.is_set():
                break

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            tracker.update(logits.detach(), labels.detach(), loss.item())

            if self.on_batch_end is not None:
                self.on_batch_end({
                    "phase":     "train",
                    "epoch":     epoch,
                    "batch":     batch_idx,
                    "num_batches": len(self.train_loader),
                    "loss":      loss.item(),
                })

        return tracker.compute()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        tracker = MetricTracker(self._num_classes)

        for images, labels, _gt in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            tracker.update(logits, labels, loss.item())

        return tracker.compute()

    # ------------------------------------------------------------------
    def fit(
        self,
        epochs: int,
        checkpoint_dir: str | Path,
        hyperparams: dict,
        start_epoch: int = 0,
    ):
        """Run the full training loop.

        Calls on_epoch_end(info) at the end of each epoch with:
            epoch, train_loss, train_accuracy, val_loss, val_accuracy, lr
        Saves a checkpoint after every epoch.
        """
        for epoch in range(start_epoch, start_epoch + epochs):
            if self.cancel_event.is_set():
                break

            train_metrics = self.train_one_epoch(epoch)
            val_metrics   = self.validate(epoch)

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["avg_loss"])
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            target_val = val_metrics.get(self.target_metric, 0.0)

            info = {
                "epoch":          epoch,
                "train_loss":     train_metrics["avg_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss":       val_metrics["avg_loss"],
                "val_accuracy":   val_metrics["accuracy"],
                "target_metric":  self.target_metric,
                "target_val":     target_val,
                "lr":             lr,
                "val_metrics":    val_metrics,
                "train_metrics":  train_metrics,
            }

            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=val_metrics,
                hyperparams=hyperparams,
                keep_last=self.keep_last,
                target_metric=self.target_metric,
            )

            if self.logger is not None:
                self.logger.log_epoch(info, self._class_names)

            if self.on_epoch_end is not None:
                self.on_epoch_end(info)

        if self.logger is not None:
            best_ckpt_path = Path(checkpoint_dir) / "best.pt"
            if best_ckpt_path.exists():
                best = torch.load(best_ckpt_path, weights_only=True)
                self.logger.log_hparams(hyperparams, best.get("metrics", {}))

