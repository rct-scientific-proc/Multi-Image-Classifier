"""
Control panel — Start / Pause / Stop buttons + QThread training worker.
"""

from __future__ import annotations

import os
import threading
import traceback

import torch
import torch.optim as optim
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.dataset import H5Dataset, make_dataloader, SPLIT_TRAIN, SPLIT_VALIDATE
from src.logger import ExperimentLogger
from src.model import build_model
from src.trainer import Trainer


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

            pin = s["device"] != "cpu"
            train_loader = make_dataloader(
                train_ds, batch_size=s["batch_size"],
                shuffle=True, num_workers=s["num_workers"], pin_memory=pin,
            )
            val_loader = make_dataloader(
                val_ds, batch_size=s["batch_size"] * 2,
                shuffle=False, num_workers=s["num_workers"], pin_memory=pin,
            )

            self.sig_log.emit(f"Building model: {s['backbone']}")
            model = build_model(
                backbone_name=s["backbone"],
                in_channels=s["in_channels"],
                num_classes=num_classes,
                pretrained=s["pretrained"],
            )

            optimizer = _build_optimizer(model, s)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(s["epochs"], 1)
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
            )

            self.sig_log.emit(
                f"Starting training — {s['epochs']} epochs on {s['device']}"
            )
            trainer.fit(
                epochs=s["epochs"],
                checkpoint_dir=ck_dir,
                hyperparams=s,
            )
            logger.close()

        except Exception:
            self.sig_error.emit(traceback.format_exc())
        finally:
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


# ── Panel ─────────────────────────────────────────────────────────────────────

class ControlPanel(QWidget):
    sig_log_message       = pyqtSignal(str)
    sig_epoch_complete    = pyqtSignal(dict)
    sig_batch_complete    = pyqtSignal(dict)
    sig_training_finished = pyqtSignal()
    sig_checkpoint_saved  = pyqtSignal(str)

    def __init__(self, settings_panel, parent=None):
        super().__init__(parent)
        self._settings              = settings_panel
        self._worker: TrainingWorker | None = None

        # ── buttons ───────────────────────────────────────────────────────
        btn_box = QGroupBox("Training")
        btn_lay = QHBoxLayout(btn_box)
        self._btn_start = QPushButton("▶  Start")
        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_stop  = QPushButton("⏹  Stop")
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        btn_lay.addWidget(self._btn_start)
        btn_lay.addWidget(self._btn_pause)
        btn_lay.addWidget(self._btn_stop)

        self._btn_start.clicked.connect(self._on_start)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_stop.clicked.connect(self._on_stop)

        # ── progress ──────────────────────────────────────────────────────
        self._epoch_label  = QLabel("Epoch: —")
        self._batch_bar    = QProgressBar()
        self._batch_bar.setTextVisible(True)
        self._batch_bar.setFormat("Batch %v / %m")
        self._metric_label = QLabel("")

        layout = QVBoxLayout(self)
        layout.addWidget(btn_box)
        layout.addWidget(self._epoch_label)
        layout.addWidget(self._batch_bar)
        layout.addWidget(self._metric_label)
        layout.addStretch()

    # ── button handlers ───────────────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        settings = self._settings.get_settings()
        if not settings.get("h5_path"):
            self.sig_log_message.emit("[ERROR] No H5 dataset path set in Settings.")
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
        self._worker.start()

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
