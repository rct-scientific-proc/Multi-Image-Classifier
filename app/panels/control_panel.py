"""
Control panel — Start / Pause / Stop buttons + QThread training worker.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.dataset import H5Dataset, make_dataloader, SPLIT_TRAIN, SPLIT_VALIDATE, SPLIT_TEST
from src.checkpoints import load_checkpoint
from src.logger import ExperimentLogger
from src.metrics import MetricTracker
from src.model import build_model
from src.trainer import FocalLoss, Trainer


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
                keep_last=int(s.get("keep_last", 3)),
                criterion=_build_criterion(s),
                recall_targets=_parse_recall_targets(s.get("recall_targets", "")),
                use_amp=bool(s.get("use_amp", False)),
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
                self.sig_log.emit(f"  Resuming at epoch {start_epoch}")

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


# ── Inference Worker ──────────────────────────────────────────────────────────

class InferenceWorker(QThread):
    """Runs inference on the test split using a saved checkpoint."""

    sig_log      = pyqtSignal(str)
    sig_progress = pyqtSignal(int, int)        # (current_batch, total_batches)
    sig_done     = pyqtSignal(dict)            # final metrics dict (json-safe)
    sig_error    = pyqtSignal(str)
    sig_finished = pyqtSignal()

    def __init__(
        self,
        checkpoint_path: str,
        h5_path: str,
        device: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        use_amp: bool,
        save_json_path: str | None,
        parent=None,
    ):
        super().__init__(parent)
        self._ckpt_path     = checkpoint_path
        self._h5_path       = h5_path
        self._device        = device
        self._batch_size    = batch_size
        self._num_workers   = num_workers
        self._pin_memory    = pin_memory
        self._use_amp       = use_amp and str(device).startswith("cuda")
        self._save_json     = save_json_path
        self._cancel_event  = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            self.sig_log.emit(f"Loading checkpoint: {self._ckpt_path}")
            ckpt = torch.load(self._ckpt_path, weights_only=True)
            hp   = ckpt.get("hyperparams", {}) or {}

            backbone    = hp.get("backbone", "simple_cnn")
            in_channels = int(hp.get("in_channels", 1))
            pretrained  = bool(hp.get("pretrained", False))

            self.sig_log.emit(f"Loading test split from: {self._h5_path}")
            test_ds = H5Dataset(self._h5_path, split=SPLIT_TEST)
            num_classes = len(test_ds.classes)
            self.sig_log.emit(
                f"  Test: {len(test_ds)} samples  Classes: {num_classes}"
            )
            if len(test_ds) == 0:
                raise RuntimeError("Test split contains 0 samples.")

            test_loader = make_dataloader(
                test_ds,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
            )

            self.sig_log.emit(
                f"Building model: {backbone} (in_channels={in_channels}, "
                f"num_classes={num_classes})"
            )
            model = build_model(
                backbone_name=backbone,
                in_channels=in_channels,
                num_classes=num_classes,
                pretrained=pretrained,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self._device).eval()

            criterion = torch.nn.CrossEntropyLoss()
            tracker   = MetricTracker(num_classes)
            total     = len(test_loader)
            self.sig_log.emit(
                f"Running inference on {self._device} "
                f"(AMP={'on' if self._use_amp else 'off'})…"
            )

            with torch.no_grad():
                for batch_idx, (images, labels, _gt) in enumerate(test_loader):
                    if self._cancel_event.is_set():
                        self.sig_log.emit("[INFO] Inference cancelled.")
                        return

                    images = images.to(self._device, non_blocking=True)
                    labels = labels.to(self._device, non_blocking=True)
                    if images.dtype == torch.uint8:
                        images = images.float().mul_(1.0 / 255.0)

                    with torch.amp.autocast("cuda", enabled=self._use_amp):
                        logits = model(images)
                        loss   = criterion(logits, labels)
                    tracker.update(logits, labels, loss.item())
                    self.sig_progress.emit(batch_idx + 1, total)

            metrics = tracker.compute()
            payload = _metrics_to_jsonable(metrics, class_names=list(test_ds.classes))
            payload["checkpoint"] = os.path.abspath(self._ckpt_path)
            payload["h5_path"]    = os.path.abspath(self._h5_path)
            payload["device"]     = self._device
            payload["num_samples"] = len(test_ds)
            payload["timestamp"]  = datetime.now().isoformat(timespec="seconds")

            if self._save_json:
                os.makedirs(os.path.dirname(os.path.abspath(self._save_json)) or ".",
                            exist_ok=True)
                with open(self._save_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                self.sig_log.emit(f"Saved metrics → {self._save_json}")

            self.sig_done.emit(payload)

        except Exception:
            self.sig_error.emit(traceback.format_exc())
        finally:
            self.sig_finished.emit()


def _metrics_to_jsonable(metrics: dict, class_names: list[str]) -> dict:
    """Convert a MetricTracker result dict into JSON-serialisable form."""
    out: dict = {"class_names": list(class_names)}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, list):
            out[k] = [x.item() if isinstance(x, (np.floating, np.integer)) else x
                      for x in v]
        elif isinstance(v, dict):
            out[k] = _metrics_to_jsonable(v, class_names)
        else:
            out[k] = v
    return out


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
    if name == "StepLR":
        step = max(epochs // 3, 1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.1)
    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    return None   # "None" — no scheduler


def _build_criterion(s: dict) -> torch.nn.Module:
    name = s.get("loss_fn", "CrossEntropy")
    if name == "FocalLoss":
        gamma = float(s.get("focal_gamma", 2.0))
        return FocalLoss(gamma=gamma)
    return torch.nn.CrossEntropyLoss()


def _parse_recall_targets(text: str) -> list[float]:
    out: list[float] = []
    for tok in (text or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 < v <= 1.0:
            out.append(v)
    return sorted(set(out))


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
        self._inf_worker: InferenceWorker | None = None

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

        # ── Inference ─────────────────────────────────────────────────────
        inf_box = QGroupBox("Inference (test split)")
        inf_lay = QVBoxLayout(inf_box)

        # Checkpoint row
        ck_row = QHBoxLayout()
        ck_row.addWidget(QLabel("Checkpoint:"))
        self._inf_ckpt_edit = QLineEdit()
        self._inf_ckpt_edit.setPlaceholderText("Path to .pt checkpoint")
        ck_row.addWidget(self._inf_ckpt_edit)
        btn_browse_ckpt = QPushButton("Browse…")
        btn_browse_ckpt.clicked.connect(self._on_browse_inf_ckpt)
        ck_row.addWidget(btn_browse_ckpt)
        inf_lay.addLayout(ck_row)

        # Save-to-JSON row
        json_row = QHBoxLayout()
        self._inf_save_json = QCheckBox("Save metrics to JSON:")
        self._inf_save_json.setChecked(True)
        json_row.addWidget(self._inf_save_json)
        self._inf_json_edit = QLineEdit()
        self._inf_json_edit.setPlaceholderText("Output .json path")
        json_row.addWidget(self._inf_json_edit)
        btn_browse_json = QPushButton("Browse…")
        btn_browse_json.clicked.connect(self._on_browse_inf_json)
        json_row.addWidget(btn_browse_json)
        self._inf_save_json.toggled.connect(self._inf_json_edit.setEnabled)
        self._inf_save_json.toggled.connect(btn_browse_json.setEnabled)
        inf_lay.addLayout(json_row)

        # Run / Cancel + progress
        run_row = QHBoxLayout()
        self._btn_inf_run    = QPushButton("▶  Run Inference")
        self._btn_inf_cancel = QPushButton("⏹  Cancel")
        self._btn_inf_cancel.setEnabled(False)
        run_row.addWidget(self._btn_inf_run)
        run_row.addWidget(self._btn_inf_cancel)
        inf_lay.addLayout(run_row)

        self._inf_progress = QProgressBar()
        self._inf_progress.setTextVisible(True)
        self._inf_progress.setFormat("Batch %v / %m")
        inf_lay.addWidget(self._inf_progress)

        self._inf_result_label = QLabel("")
        self._inf_result_label.setWordWrap(True)
        inf_lay.addWidget(self._inf_result_label)

        self._btn_inf_run.clicked.connect(self._on_inf_run)
        self._btn_inf_cancel.clicked.connect(self._on_inf_cancel)

        layout.addWidget(inf_box)
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

    # ── inference handlers ────────────────────────────────────────────────────

    def _on_browse_inf_ckpt(self) -> None:
        s = self._settings.get_settings()
        start_dir = s.get("checkpoint_dir", "") or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint for inference", start_dir,
            "PyTorch checkpoint (*.pt);;All files (*)"
        )
        if path:
            self._inf_ckpt_edit.setText(path)
            if self._inf_save_json.isChecked() and not self._inf_json_edit.text().strip():
                base, _ = os.path.splitext(path)
                self._inf_json_edit.setText(base + "_test_metrics.json")

    def _on_browse_inf_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save metrics JSON", self._inf_json_edit.text() or "",
            "JSON files (*.json);;All files (*)"
        )
        if path:
            self._inf_json_edit.setText(path)

    def _on_inf_run(self) -> None:
        if self._inf_worker and self._inf_worker.isRunning():
            return
        s = self._settings.get_settings()
        ckpt = self._inf_ckpt_edit.text().strip()
        h5   = s.get("h5_path", "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            QMessageBox.warning(self, "Cannot run inference",
                                f"Checkpoint not found:\n{ckpt}")
            return
        if not h5 or not os.path.isfile(h5):
            QMessageBox.warning(self, "Cannot run inference",
                                f"H5 dataset not found:\n{h5}")
            return
        save_json = self._inf_json_edit.text().strip() if self._inf_save_json.isChecked() else None
        if self._inf_save_json.isChecked() and not save_json:
            QMessageBox.warning(self, "Cannot run inference",
                                "Output JSON path is empty.")
            return

        self._inf_worker = InferenceWorker(
            checkpoint_path=ckpt,
            h5_path=h5,
            device=s.get("device", "cpu"),
            batch_size=int(s.get("batch_size", 32)) * 2,
            num_workers=int(s.get("num_workers", 0)),
            pin_memory=bool(s.get("pin_memory", False)),
            use_amp=bool(s.get("use_amp", False)),
            save_json_path=save_json,
        )
        self._inf_worker.sig_log.connect(self.sig_log_message)
        self._inf_worker.sig_progress.connect(self._on_inf_progress)
        self._inf_worker.sig_done.connect(self._on_inf_done)
        self._inf_worker.sig_error.connect(self._on_inf_error)
        self._inf_worker.sig_finished.connect(self._on_inf_finished)

        self._btn_inf_run.setEnabled(False)
        self._btn_inf_cancel.setEnabled(True)
        self._inf_progress.setValue(0)
        self._inf_result_label.setText("")
        self._inf_worker.start()

    def _on_inf_cancel(self) -> None:
        if self._inf_worker:
            self._inf_worker.cancel()
            self.sig_log_message.emit("[INFO] Inference cancel requested…")

    def _on_inf_progress(self, current: int, total: int) -> None:
        self._inf_progress.setMaximum(total)
        self._inf_progress.setValue(current)

    def _on_inf_done(self, metrics: dict) -> None:
        acc = metrics.get("accuracy", float("nan"))
        f1  = metrics.get("f1_macro", float("nan"))
        mcc = metrics.get("mcc", float("nan"))
        loss = metrics.get("avg_loss", float("nan"))
        self._inf_result_label.setText(
            f"acc={acc:.4f}  f1_macro={f1:.4f}  mcc={mcc:.4f}  loss={loss:.4f}"
        )
        self.sig_log_message.emit(
            f"[INFER] accuracy={acc:.4f}  f1_macro={f1:.4f}  "
            f"mcc={mcc:.4f}  avg_loss={loss:.4f}"
        )

    def _on_inf_error(self, tb: str) -> None:
        self.sig_log_message.emit(f"[INFER ERROR]\n{tb}")
        first_line = tb.strip().splitlines()[-1] if tb.strip() else "Unknown error"
        QMessageBox.critical(self, "Inference error", first_line)

    def _on_inf_finished(self) -> None:
        self._btn_inf_run.setEnabled(True)
        self._btn_inf_cancel.setEnabled(False)
