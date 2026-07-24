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

import math
import threading
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.augment import GpuAugment, Normalizer
from src.metrics import MetricTracker, DEFAULT_TARGET_METRIC
from src.checkpoints import save_checkpoint, is_improvement
from src.logger import ExperimentLogger


class FocalLoss(nn.Module):
    """Multi-class focal loss  FL = -α (1-p)^γ log(p).

    Parameters
    ----------
    gamma : float
        Focusing parameter. 0 = standard cross-entropy. Typical value 2.
    alpha : 1-D tensor | sequence | None
        Optional per-class weight, same semantics as
        ``nn.CrossEntropyLoss(weight=...)``. Registered as a buffer so
        ``FocalLoss(...).to(device)`` moves it alongside the model — without
        that, a CPU alpha against CUDA logits raises a device-mismatch error.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        alpha_t = (torch.as_tensor(alpha, dtype=torch.float32)
                   if alpha is not None else None)
        # register_buffer accepts None; .to(device) then moves it when present.
        self.register_buffer("alpha", alpha_t)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p  = F.log_softmax(logits, dim=1)
        ce     = F.nll_loss(log_p, targets, weight=self.alpha, reduction="none")
        p      = torch.exp(-ce)                      # p_t
        focal  = (1.0 - p) ** self.gamma * ce
        if self.reduction == "mean":
            # nll_loss(reduction="mean") divides by the summed weights, not the
            # sample count, when a weight is given; match that so the loss scale
            # is comparable with and without alpha.
            if self.alpha is not None:
                w = self.alpha[targets]
                return focal.sum() / w.sum().clamp_min(1e-8)
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def class_weights(counts, scheme: str = "effective_number",
                  beta: float = 0.999) -> torch.Tensor:
    """Per-class loss weights from ground-truth counts.

    Re-weighting, not re-sampling: every sample is still trained on, but the
    majority class contributes less per example. This is how the full detector
    output (with its ~50:1 background ratio) can be trained on without dropping
    any hard negatives while the genuine classes still get gradient.

    scheme:
        "inverse_freq"     — w_c ∝ 1 / n_c. Simple, but over-weights ultra-rare
                             classes at extreme ratios.
        "effective_number" — w_c ∝ (1 - β) / (1 - β^{n_c})  (Cui et al. 2019).
                             The "effective number of samples" saturates as n_c
                             grows, so it is gentler than raw inverse frequency
                             at 50:1 — the recommended default.

    Weights are normalised to average 1 over the classes actually present, so
    the overall loss scale is unchanged and only the *balance* shifts. Absent
    classes get weight 0; their index is never selected by nll_loss anyway.
    """
    import numpy as np
    counts = np.asarray(counts, dtype=np.float64)
    safe   = np.maximum(counts, 1.0)                 # guard n_c = 0 in the maths
    if scheme == "inverse_freq":
        raw = 1.0 / safe
    elif scheme == "effective_number":
        beta = min(max(float(beta), 0.0), 1.0 - 1e-9)
        raw  = (1.0 - beta) / (1.0 - np.power(beta, safe))
    else:
        raise ValueError(f"unknown class-weight scheme: {scheme!r}")

    present = counts > 0
    if present.any():
        raw = raw / raw[present].mean()
    raw[~present] = 0.0
    return torch.as_tensor(raw, dtype=torch.float32)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
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
        recall_targets: list[float] | None = None,
        specificity_targets: list[float] | None = None,
        use_amp: bool = False,
        augment: "GpuAugment | None" = None,
        normalizer: "Normalizer | None" = None,
        seed: int | None = None,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best: bool = True,
        smart_training: bool = False,
        max_restarts: int = 3,
        restart_lr_factor: float = 1.0,
        restart_from_best: bool = True,
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
        # Move the criterion to the device too: a weighted loss (FocalLoss alpha
        # or CrossEntropyLoss weight) carries a per-class tensor that must sit
        # beside the logits, or the first backward pass raises a device mismatch.
        criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.criterion     = criterion.to(device)
        self._num_classes  = len(train_loader.dataset.classes)
        self.target_metric = target_metric
        self.logger        = logger
        self.keep_last     = keep_last
        self._class_names  = list(train_loader.dataset.classes)
        self.recall_targets = list(recall_targets) if recall_targets else []
        self.specificity_targets = (list(specificity_targets)
                                    if specificity_targets else [])
        # AMP is only meaningful on CUDA; silently disable elsewhere
        self.use_amp       = bool(use_amp) and str(device).startswith("cuda")
        self._scaler       = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.seed          = seed
        # Augmentation is train-only; normalisation applies to both loops and
        # must match at inference, so its stats go into the checkpoint (see fit).
        self.augment    = augment.to(device) if augment is not None else None
        self.normalizer = normalizer.to(device) if normalizer is not None else None
        # Early stopping — patience on the *target* metric, same metric best.pt
        # tracks, so "best for early stopping" and best.pt always agree.
        self.early_stopping = bool(early_stopping)
        self.patience       = int(patience)
        self.min_delta      = float(min_delta)
        self.restore_best   = bool(restore_best)
        # Smart training — a plateau (patience epochs without improvement) spikes
        # the LR to escape the basin instead of stopping, optionally rewinding to
        # the best weights first. After max_restarts *consecutive unproductive*
        # restarts it gives up and stops. When on, it owns the LR (the scheduler
        # is bypassed) and its give-up supersedes plain early stopping. It shares
        # `patience` / `min_delta` with early stopping — the same plateau window.
        self.smart_training    = bool(smart_training)
        self.max_restarts      = int(max_restarts)
        self.restart_lr_factor = float(restart_lr_factor)
        self.restart_from_best = bool(restart_from_best)

    # ------------------------------------------------------------------
    def _prepare_batch(self, images: torch.Tensor, training: bool) -> torch.Tensor:
        """uint8 -> float [0,1] -> augment (train only) -> normalise.

        Called from both loops with ``training`` set accordingly, so the one
        asymmetry that matters — augmentation on train but never on validation,
        normalisation on both — lives in a single place rather than being
        duplicated and drifting.

        Deliberately runs before the autocast block: geometric resampling and
        colour maths in fp16 lose precision, and at these tensor sizes there is
        no speed to gain from doing them in half.
        """
        if images.dtype == torch.uint8:
            images = images.float().mul_(1.0 / 255.0)

        if training and self.augment is not None:
            images = self.augment(images)
        if self.normalizer is not None:
            images = self.normalizer(images)
        return images

    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        if self.augment is not None:
            self.augment.train()          # matches the module's own eval guard
        tracker = MetricTracker(self._num_classes,
                                 recall_targets=self.recall_targets,
                                 specificity_targets=self.specificity_targets)

        for batch_idx, (images, labels, _gt) in enumerate(self.train_loader):
            if self.cancel_event.is_set():
                break

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            images = self._prepare_batch(images, training=True)

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)
            self._scaler.scale(loss).backward()
            self._scaler.step(self.optimizer)
            self._scaler.update()

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
        if self.augment is not None:
            self.augment.eval()           # belt and braces; not called below
        tracker = MetricTracker(self._num_classes,
                                 recall_targets=self.recall_targets,
                                 specificity_targets=self.specificity_targets)

        for images, labels, _gt in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            # training=False: normalise, never augment
            images = self._prepare_batch(images, training=False)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
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
        """Run the training loop from *start_epoch* up to *epochs*.

        *epochs* is the TOTAL number of epochs for the run, not a count of
        additional ones. Resuming a checkpoint saved at epoch 4 with epochs=10
        therefore trains epochs 5-9, and the LR schedule (built with
        T_max=epochs) stays aligned with the run it was created for.
        Returns immediately when start_epoch >= epochs.

        Calls on_epoch_end(info) at the end of each epoch with:
            epoch, train_loss, train_accuracy, val_loss, val_accuracy, lr
        Saves a checkpoint after every epoch.
        """
        if self.seed is not None:
            # Makes the augmentation draw sequence repeatable. Note this seeds
            # from here, so it does not cover model init or the DataLoader
            # shuffle that happened before the Trainer was constructed.
            torch.manual_seed(self.seed)

        # Record what the weights were actually trained under. Copied rather
        # than mutated: the caller's settings dict is reused elsewhere, and the
        # normalisation stats may be derived (mode="dataset") rather than typed
        # in, so re-deriving them at inference is not possible.
        hyperparams = dict(hyperparams)
        if self.normalizer is not None:
            hyperparams.update(self.normalizer.state())
        if self.augment is not None:
            hyperparams.update(self.augment.config)

        # Early-stopping state. best_state is kept in memory (on CPU) so a
        # restore does not depend on best.pt existing on disk, and the counter
        # resets on any genuine improvement — which, under cyclic LR, happens at
        # each cycle's minimum, so a plateau across cycles is what stops the run.
        best_value: float | None = None
        best_epoch = start_epoch
        best_state = None
        epochs_no_improve = 0
        stop_reason = ""

        # Smart-training state. base_lrs is captured once so a restart can spike
        # back to the run's original LR; epochs_in_cycle drives the within-cycle
        # cosine decay and resets on each restart.
        base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        epochs_in_cycle = 0
        restarts_used = 0

        for epoch in range(start_epoch, epochs):
            if self.cancel_event.is_set():
                break

            # Smart training owns the LR: a cosine decay from the (spiked) peak
            # over `patience` epochs, floored at 5% so a long cycle cannot stall
            # at zero. epoch 0 of a cycle sits at the peak.
            if self.smart_training:
                horizon = max(self.patience, 1)
                t = min(epochs_in_cycle, horizon)
                scale = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * t / horizon))
                for g, base in zip(self.optimizer.param_groups, base_lrs):
                    g["lr"] = base * self.restart_lr_factor * scale
                epochs_in_cycle += 1

            sampler = self.train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            train_metrics = self.train_one_epoch(epoch)
            val_metrics   = self.validate(epoch)

            # The scheduler is bypassed under smart training, which sets the LR
            # itself above — otherwise the two would fight over param_groups.
            if self.scheduler is not None and not self.smart_training:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["avg_loss"])
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            target_val = val_metrics.get(self.target_metric, 0.0)

            # ---- plateau bookkeeping (before info, so the flags ride along on
            #      the same full dict the GUI already knows how to read) ----
            improved = best_value is None or is_improvement(
                target_val, best_value, self.target_metric, self.min_delta)
            if improved:
                best_value = target_val
                best_epoch = epoch
                epochs_no_improve = 0
                # A productive epoch refills the restart budget: restarts only
                # "give up" after max_restarts in a row that beat nothing.
                restarts_used = 0
                if (self.early_stopping or self.smart_training) and self.restore_best:
                    # Snapshot to CPU so a second model's worth of GPU memory is
                    # not pinned for the whole run.
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
            else:
                epochs_no_improve += 1

            plateau = self.patience > 0 and epochs_no_improve >= self.patience
            should_stop = False
            restarted = False
            restart_msg = ""

            if self.smart_training and plateau:
                restarts_used += 1
                if restarts_used > self.max_restarts:
                    should_stop = True
                    stop_reason = (
                        f"Smart training gave up at epoch {epoch}: "
                        f"{self.max_restarts} restarts without beating the best "
                        f"{self.target_metric} ({best_value:.4f} at epoch "
                        f"{best_epoch})."
                    )
                else:
                    # Rewind to the best weights when the current model has
                    # drifted below them — explore outward from the good point,
                    # not from a worse one.
                    rewound = False
                    if (self.restart_from_best and best_state is not None
                            and is_improvement(best_value, target_val,
                                               self.target_metric)):
                        self.model.load_state_dict(
                            {k: v.to(self.device) for k, v in best_state.items()})
                        rewound = True
                    # Perturb: reset the optimizer's momentum/variance history so
                    # it does not simply retrace its path, and restart the LR
                    # cosine (the spike takes effect on the next epoch's top).
                    self.optimizer.state.clear()
                    epochs_in_cycle = 0
                    epochs_no_improve = 0
                    restarted = True
                    restart_msg = (
                        f"Restart {restarts_used}/{self.max_restarts} at epoch "
                        f"{epoch}: LR spiked"
                        + (" and weights rewound to best" if rewound else "")
                        + "."
                    )
            elif self.early_stopping and plateau:
                should_stop = True
                stop_reason = (
                    f"Early stop at epoch {epoch}: no improvement in "
                    f"{self.target_metric} for {epochs_no_improve} epochs "
                    f"(best {best_value:.4f} at epoch {best_epoch})."
                )

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
                "epochs_no_improve": epochs_no_improve,
                "best_epoch":     best_epoch,
                "best_target_val": best_value,
                "restarts_used":  restarts_used,
            }
            if restarted:
                info["restarted"]       = True
                info["restart_message"] = restart_msg
            if should_stop:
                info["early_stopped"] = True
                info["stop_reason"]   = stop_reason

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
                model_name=str(hyperparams.get("backbone", "")) or type(self.model).__name__,
                classes=self._class_names,
            )

            if self.logger is not None:
                self.logger.log_epoch(info, self._class_names)

            if self.on_epoch_end is not None:
                self.on_epoch_end(info)

            if should_stop:
                break

        # Restore the best-seen weights into the live model. best.pt already
        # holds them on disk, so this matters for anything that keeps using the
        # model object after fit() returns; it is a no-op if no epoch ran.
        if ((self.early_stopping or self.smart_training) and self.restore_best
                and best_state is not None):
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})

        if self.logger is not None:
            best_ckpt_path = Path(checkpoint_dir) / "best.pt"
            if best_ckpt_path.exists():
                best = torch.load(best_ckpt_path, weights_only=True)
                self.logger.log_hparams(hyperparams, best.get("metrics", {}))

