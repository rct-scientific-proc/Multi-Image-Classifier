"""
Control panel — Start / Pause / Stop buttons + QThread training worker.

Test-split evaluation lives in inference_panel.py; this module is training only.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback

import torch
import torch.optim as optim
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.augment import GpuAugment, Normalizer
from src.dataset import (
    H5Dataset, count_labels, make_dataloader, peek_h5_meta,
    SPLIT_TRAIN, SPLIT_VALIDATE,
)
from src.checkpoints import load_checkpoint
from src.logger import ExperimentLogger
from src.metrics import parse_target_list
from src.model import build_model
from src.trainer import FocalLoss, Trainer, class_weights


# ── Worker ────────────────────────────────────────────────────────────────────

class TrainingWorker(QThread):
    """Runs Trainer.fit() in a background thread and emits Qt signals."""

    sig_log        = pyqtSignal(str)
    sig_epoch_end  = pyqtSignal(dict)
    sig_batch_end  = pyqtSignal(dict)
    sig_finished   = pyqtSignal()
    sig_error      = pyqtSignal(str)
    sig_checkpoint = pyqtSignal(str)   # emits checkpoint_dir path

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self._s            = settings
        self._cancel_event = threading.Event()
        self._pause_event  = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()
        self._pause_event.clear()   # unblock a paused loop immediately

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def run(self) -> None:
        s = self._s
        logger: ExperimentLogger | None = None
        try:
            self.sig_log.emit(f"Loading dataset: {s['h5_path']}")
            train_ds = H5Dataset(s["h5_path"], split=SPLIT_TRAIN)
            val_ds   = H5Dataset(s["h5_path"], split=SPLIT_VALIDATE)
            num_classes = len(train_ds.classes)
            self.sig_log.emit(
                f"  Train: {len(train_ds)} samples  "
                f"Val: {len(val_ds)} samples  "
                f"Classes: {num_classes}"
            )

            pin = bool(s.get("pin_memory", False))
            nw  = int(s.get("num_workers", 0))
            if nw > 0 and sys.platform == "win32" and str(s.get("device", "")).startswith("cuda"):
                self.sig_log.emit(
                    f"[WARN] num_workers={nw} on Windows+CUDA spawns worker processes "
                    f"that each re-import torch and load CUDA DLLs (~1 GB committed memory "
                    f"per worker). If you hit 'WinError 1455 (paging file too small)', "
                    f"set num_workers=0 or increase the Windows page file."
                )
            train_loader = make_dataloader(
                train_ds, batch_size=s["batch_size"],
                shuffle=True, shuffle_every=int(s.get("shuffle_every_n_epochs", 1)),
                num_workers=nw, pin_memory=pin,
            )
            val_loader = make_dataloader(
                # 2× batch in val is safe: no gradients, no backward pass memory
                val_ds, batch_size=s["batch_size"] * 2,
                shuffle=False, num_workers=nw, pin_memory=pin,
            )

            self.sig_log.emit(f"Building model: {s['backbone']}")
            model = build_model(
                backbone_name=s["backbone"],
                in_channels=s["in_channels"],
                num_classes=num_classes,
                pretrained=s["pretrained"],
            )

            optimizer = _build_optimizer(model, s)
            scheduler = _build_scheduler(optimizer, s)

            # Both read from the settings dict and fall back to no-ops, so this
            # is inert until the Augmentation controls land. Normalisation
            # applies to train and validation alike; augmentation to train only.
            augment = GpuAugment(s, in_channels=s["in_channels"])
            normalizer = Normalizer.from_config(s, in_channels=s["in_channels"])
            if augment.enabled:
                self.sig_log.emit(f"Augmentation: {', '.join(augment.active_ops)}")
            if not normalizer.is_identity:
                st = normalizer.state()
                self.sig_log.emit(
                    f"Normalisation: mean={st['normalize_mean']} "
                    f"std={st['normalize_std']}"
                )

            # Class weights re-balance the loss without dropping any samples —
            # counted from the train split so they match the data actually used.
            weight = None
            cw_mode = s.get("class_weight_mode", "none")
            if cw_mode != "none":
                counts = count_labels(s["h5_path"], SPLIT_TRAIN, num_classes)
                weight = class_weights(counts, scheme=cw_mode,
                                       beta=float(s.get("class_weight_beta", 0.999)))
                hi = int(counts.argmax())
                self.sig_log.emit(
                    f"Class weights ({cw_mode}): range "
                    f"{float(weight.min()):.3f}–{float(weight.max()):.3f}; "
                    f"largest class '{train_ds.classes[hi]}' (n={int(counts[hi])}) "
                    f"-> {float(weight[hi]):.3f}"
                )

            os.makedirs(s["checkpoint_dir"], exist_ok=True)
            os.makedirs(s["log_dir"], exist_ok=True)

            logger     = ExperimentLogger(s["log_dir"], s["experiment_name"])
            pause_ev   = self._pause_event
            cancel_ev  = self._cancel_event
            ck_dir     = s["checkpoint_dir"]

            def on_batch_end(info: dict) -> None:
                self.sig_batch_end.emit(info)
                while pause_ev.is_set() and not cancel_ev.is_set():
                    self.msleep(100)

            def on_epoch_end(info: dict) -> None:
                self.sig_epoch_end.emit(info)
                self.sig_checkpoint.emit(ck_dir)
                self.sig_log.emit(
                    f"Epoch {info['epoch']:3d}  "
                    f"train_loss={info['train_loss']:.4f}  "
                    f"val_loss={info['val_loss']:.4f}  "
                    f"{info['target_metric']}={info['target_val']:.4f}  "
                    f"lr={info['lr']:.2e}"
                )
                if info.get("early_stopped"):
                    self.sig_log.emit(f"[INFO] {info['stop_reason']} "
                                      f"Best weights restored.")

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                device=s["device"],
                on_epoch_end=on_epoch_end,
                on_batch_end=on_batch_end,
                cancel_event=cancel_ev,
                target_metric=s["target_metric"],
                logger=logger,
                keep_last=int(s.get("keep_last", 3)),
                criterion=_build_criterion(s, weight=weight),
                recall_targets=parse_target_list(s.get("recall_targets", "")),
                specificity_targets=parse_target_list(
                    s.get("specificity_targets", "")),
                use_amp=bool(s.get("use_amp", False)),
                augment=augment,
                normalizer=normalizer,
                seed=s.get("seed"),
                early_stopping=bool(s.get("early_stopping", False)),
                patience=int(s.get("patience", 10)),
                min_delta=float(s.get("min_delta", 0.0)),
            )

            self.sig_log.emit(
                f"Starting training — {s['epochs']} epochs on {s['device']}"
            )

            start_epoch = 0
            resume_path = s.get("resume_checkpoint", "").strip()
            if resume_path:
                self.sig_log.emit(f"Resuming from: {resume_path}")
                ckpt = load_checkpoint(resume_path, model, optimizer, scheduler)
                start_epoch = ckpt.get("epoch", 0) + 1
                self.sig_log.emit(
                    f"  Resuming at epoch {start_epoch} — training through "
                    f"epoch {s['epochs'] - 1}"
                )
                if start_epoch >= s["epochs"]:
                    self.sig_log.emit(
                        f"[WARN] Nothing to do: the checkpoint is already at epoch "
                        f"{start_epoch - 1} and 'Epochs' is set to {s['epochs']} "
                        f"(total, not additional). Raise 'Epochs' above "
                        f"{start_epoch} to continue this run."
                    )

            trainer.fit(
                epochs=s["epochs"],
                checkpoint_dir=ck_dir,
                hyperparams=s,
                start_epoch=start_epoch,
            )

        except Exception:
            self.sig_error.emit(traceback.format_exc())
        finally:
            if logger is not None:
                try:
                    logger.close()
                except Exception:
                    pass
            self.sig_finished.emit()


def _build_optimizer(model: torch.nn.Module, s: dict) -> torch.optim.Optimizer:
    name   = s.get("optimizer", "Adam")
    lr     = float(s.get("lr", 1e-3))
    params = model.parameters()
    if name == "Adam":
        return optim.Adam(params, lr=lr)
    if name == "AdamW":
        return optim.AdamW(params, lr=lr)
    if name == "SGD":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    if name == "RMSprop":
        return optim.RMSprop(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def _build_scheduler(optimizer: torch.optim.Optimizer, s: dict):
    name   = s.get("scheduler", "CosineAnnealing")
    epochs = max(int(s.get("epochs", 10)), 1)
    if name == "CosineAnnealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "CosineWarmRestarts":
        # Cyclic (SGDR): the LR decays over T_0 epochs, jumps back to the top,
        # and repeats — with each cycle T_mult× longer than the last. Unlike
        # CosineAnnealing this restarts within a run. Steps once per epoch, so
        # it falls through the trainer's plain scheduler.step() branch.
        t0 = max(int(s.get("restart_period", 5)), 1)
        t_mult = max(int(s.get("restart_mult", 1)), 1)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=t_mult
        )
    if name == "StepLR":
        step = max(epochs // 3, 1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.1)
    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    return None   # "None" — no scheduler


def _build_criterion(s: dict, weight=None) -> torch.nn.Module:
    name = s.get("loss_fn", "CrossEntropy")
    if name == "FocalLoss":
        gamma = float(s.get("focal_gamma", 2.0))
        return FocalLoss(gamma=gamma, alpha=weight)
    return torch.nn.CrossEntropyLoss(weight=weight)




# ── Panel ─────────────────────────────────────────────────────────────────────

class ControlPanel(QWidget):
    sig_log_message       = pyqtSignal(str)
    sig_epoch_complete    = pyqtSignal(dict)
    sig_batch_complete    = pyqtSignal(dict)
    sig_training_started  = pyqtSignal(dict)   # emits settings dict
    sig_training_finished = pyqtSignal()
    sig_checkpoint_saved  = pyqtSignal(str)

    def __init__(self, settings_panel, parent=None):
        super().__init__(parent)
        self._settings              = settings_panel
        self._worker: TrainingWorker | None = None

        # No enclosing QGroupBox — the "Train" tab label already names this.
        layout = QVBoxLayout(self)

        # ── buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("▶  Start")
        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_stop  = QPushButton("⏹  Stop")
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_pause)
        btn_row.addWidget(self._btn_stop)
        layout.addLayout(btn_row)

        self._btn_start.clicked.connect(self._on_start)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_stop.clicked.connect(self._on_stop)

        # ── progress ──────────────────────────────────────────────────────
        self._epoch_label  = QLabel("Epoch: —")
        self._batch_bar    = QProgressBar()
        self._batch_bar.setTextVisible(True)
        self._batch_bar.setFormat("Batch %v / %m")
        self._metric_label = QLabel("")
        self._metric_label.setWordWrap(True)

        layout.addWidget(self._epoch_label)
        layout.addWidget(self._batch_bar)
        layout.addWidget(self._metric_label)
        layout.addStretch()

    # ── button handlers ───────────────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        settings = self._settings.get_settings()
        error = self._validate_settings(settings)
        if error:
            QMessageBox.warning(self, "Cannot start training", error)
            return

        self._worker = TrainingWorker(settings)
        self._worker.sig_log.connect(self.sig_log_message)
        self._worker.sig_epoch_end.connect(self._on_epoch_end)
        self._worker.sig_batch_end.connect(self._on_batch_end)
        self._worker.sig_finished.connect(self._on_finished)
        self._worker.sig_error.connect(self._on_error)
        self._worker.sig_checkpoint.connect(self.sig_checkpoint_saved)

        self._btn_start.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._batch_bar.setValue(0)
        self.sig_training_started.emit(settings)
        self._worker.start()

    @staticmethod
    def _validate_settings(s: dict) -> str:
        """Return an error string if settings are invalid, else empty string."""
        h5 = s.get("h5_path", "").strip()
        if not h5:
            return "No H5 dataset path set in Settings."
        if not os.path.isfile(h5):
            return f"H5 dataset not found:\n{h5}"
        if not s.get("experiment_name", "").strip():
            return "Experiment name cannot be empty."
        resume = s.get("resume_checkpoint", "").strip()
        if resume and not os.path.isfile(resume):
            return f"Resume checkpoint not found:\n{resume}"

        # Check the file's channel count up front — a mismatch would otherwise
        # only surface as an opaque shape error at the first conv layer.
        try:
            meta = peek_h5_meta(h5)
        except (OSError, KeyError) as exc:
            return f"Could not read H5 dataset:\n{h5}\n\n{exc}"
        if meta["channels"] != int(s.get("in_channels", 1)):
            return (
                f"Input channels mismatch.\n\n"
                f"Settings say in_channels={s.get('in_channels')}, but "
                f"{os.path.basename(h5)} stores images with "
                f"{meta['channels']} channel(s).\n\n"
                f"Set 'Input channels' to {meta['channels']}."
            )
        return ""

    def _on_pause(self) -> None:
        if self._worker is None:
            return
        if self._worker.is_paused:
            self._worker.resume()
            self._btn_pause.setText("⏸  Pause")
            self.sig_log_message.emit("[INFO] Training resumed.")
        else:
            self._worker.pause()
            self._btn_pause.setText("▶  Resume")
            self.sig_log_message.emit("[INFO] Training paused.")

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.sig_log_message.emit("[INFO] Stop requested — finishing current batch…")

    # ── worker signal handlers ─────────────────────────────────────────────────

    def _on_epoch_end(self, info: dict) -> None:
        total = self._settings.get_settings().get("epochs", "?")
        self._epoch_label.setText(f"Epoch: {info['epoch'] + 1} / {total}")
        self._metric_label.setText(
            f"{info['target_metric']} = {info['target_val']:.4f}"
            f"  │  val_loss = {info['val_loss']:.4f}"
        )
        self.sig_epoch_complete.emit(info)

    def _on_batch_end(self, info: dict) -> None:
        if info.get("phase") == "train":
            self._batch_bar.setMaximum(info["num_batches"])
            self._batch_bar.setValue(info["batch"] + 1)
        self.sig_batch_complete.emit(info)

    def _on_finished(self) -> None:
        self._btn_start.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_pause.setText("⏸  Pause")
        self._btn_stop.setEnabled(False)
        self._epoch_label.setText("Epoch: —")
        self._batch_bar.setValue(0)
        self.sig_training_finished.emit()

    def _on_error(self, tb: str) -> None:
        self.sig_log_message.emit(f"[ERROR]\n{tb}")
        # Detect the most common Windows pitfall and give actionable advice
        if "WinError 1455" in tb or "paging file is too small" in tb:
            QMessageBox.critical(
                self,
                "Out of virtual memory (WinError 1455)",
                "Windows ran out of page-file space while a DataLoader worker tried "
                "to load the CUDA DLLs.\n\n"
                "Fix one of:\n"
                "  • Set 'DataLoader workers' to 0 in Settings (recommended).\n"
                "  • Increase the Windows page file size "
                "(System Properties → Advanced → Performance → Virtual memory).\n\n"
                "Each worker process re-imports torch and commits ~1 GB; with several "
                "workers + a small page file this exceeds Windows' commit limit."
            )
            return
        # Generic fallback: show last line of the traceback
        first_line = tb.strip().splitlines()[-1] if tb.strip() else "Unknown error"
        QMessageBox.critical(self, "Training error", first_line)
