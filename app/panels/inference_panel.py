"""
Inference panel — run a saved checkpoint over the test split.

Split out of control_panel.py so training and inference each own a tab rather
than competing for height in one stacked column.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from datetime import datetime

import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.augment import Normalizer
from src.dataset import H5Dataset, make_dataloader, SPLIT_TEST
from src.metrics import MetricTracker
from src.model import build_model


# ── Worker ────────────────────────────────────────────────────────────────────

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

            # Replay the exact normalisation these weights were trained under.
            # Getting this wrong does not raise — it silently shifts every
            # metric — so it is read from the checkpoint rather than from the
            # current settings, which may since have changed.
            normalizer = Normalizer.from_checkpoint(hp, in_channels).to(self._device)
            if normalizer.is_identity:
                self.sig_log.emit("  Normalisation: none (raw [0,1])")
            else:
                st = normalizer.state()
                self.sig_log.emit(
                    f"  Normalisation from checkpoint: "
                    f"mean={st['normalize_mean']} std={st['normalize_std']}"
                )

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
                    images = normalizer(images)   # never augment at inference

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


# ── Panel ─────────────────────────────────────────────────────────────────────

class InferencePanel(QWidget):
    """Checkpoint picker + run controls for test-split evaluation.

    Public signals
    --------------
    sig_log_message(str)
        Forwarded worker output, consumed by the console panel.
    """

    sig_log_message = pyqtSignal(str)

    def __init__(self, settings_panel, parent=None):
        super().__init__(parent)
        self._settings = settings_panel
        self._worker: InferenceWorker | None = None

        layout = QVBoxLayout(self)

        # ── checkpoint ────────────────────────────────────────────────────
        layout.addWidget(QLabel("Checkpoint:"))
        ck_row = QHBoxLayout()
        self._ckpt_edit = QLineEdit()
        self._ckpt_edit.setPlaceholderText("Path to .pt checkpoint")
        ck_row.addWidget(self._ckpt_edit)
        btn_browse_ckpt = QPushButton("Browse…")
        btn_browse_ckpt.clicked.connect(self._on_browse_ckpt)
        ck_row.addWidget(btn_browse_ckpt)
        layout.addLayout(ck_row)

        # ── save-to-JSON ──────────────────────────────────────────────────
        self._save_json = QCheckBox("Save metrics to JSON:")
        self._save_json.setChecked(True)
        layout.addWidget(self._save_json)

        json_row = QHBoxLayout()
        self._json_edit = QLineEdit()
        self._json_edit.setPlaceholderText("Output .json path")
        json_row.addWidget(self._json_edit)
        btn_browse_json = QPushButton("Browse…")
        btn_browse_json.clicked.connect(self._on_browse_json)
        json_row.addWidget(btn_browse_json)
        layout.addLayout(json_row)

        self._save_json.toggled.connect(self._json_edit.setEnabled)
        self._save_json.toggled.connect(btn_browse_json.setEnabled)

        # ── run / cancel ──────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self._btn_run    = QPushButton("▶  Run Inference")
        self._btn_cancel = QPushButton("⏹  Cancel")
        self._btn_cancel.setEnabled(False)
        run_row.addWidget(self._btn_run)
        run_row.addWidget(self._btn_cancel)
        layout.addLayout(run_row)

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setFormat("Batch %v / %m")
        layout.addWidget(self._progress)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        layout.addStretch()

        self._btn_run.clicked.connect(self._on_run)
        self._btn_cancel.clicked.connect(self._on_cancel)

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _autofill_json_path(self, ckpt_path: str) -> None:
        if self._save_json.isChecked() and not self._json_edit.text().strip():
            base, _ = os.path.splitext(ckpt_path)
            self._json_edit.setText(base + "_test_metrics.json")

    def _on_browse_ckpt(self) -> None:
        s = self._settings.get_settings()
        start_dir = s.get("checkpoint_dir", "") or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint for inference", start_dir,
            "PyTorch checkpoint (*.pt);;All files (*)"
        )
        if path:
            self._ckpt_edit.setText(path)
            self._autofill_json_path(path)

    def _on_browse_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save metrics JSON", self._json_edit.text() or "",
            "JSON files (*.json);;All files (*)"
        )
        if path:
            self._json_edit.setText(path)

    def _on_run(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        s = self._settings.get_settings()
        ckpt = self._ckpt_edit.text().strip()
        h5   = s.get("h5_path", "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            QMessageBox.warning(self, "Cannot run inference",
                                f"Checkpoint not found:\n{ckpt}")
            return
        if not h5 or not os.path.isfile(h5):
            QMessageBox.warning(self, "Cannot run inference",
                                f"H5 dataset not found:\n{h5}")
            return
        save_json = self._json_edit.text().strip() if self._save_json.isChecked() else None
        if self._save_json.isChecked() and not save_json:
            QMessageBox.warning(self, "Cannot run inference",
                                "Output JSON path is empty.")
            return

        self._worker = InferenceWorker(
            checkpoint_path=ckpt,
            h5_path=h5,
            device=s.get("device", "cpu"),
            batch_size=int(s.get("batch_size", 32)) * 2,
            num_workers=int(s.get("num_workers", 0)),
            pin_memory=bool(s.get("pin_memory", False)),
            use_amp=bool(s.get("use_amp", False)),
            save_json_path=save_json,
        )
        self._worker.sig_log.connect(self.sig_log_message)
        self._worker.sig_progress.connect(self._on_progress)
        self._worker.sig_done.connect(self._on_done)
        self._worker.sig_error.connect(self._on_error)
        self._worker.sig_finished.connect(self._on_finished)

        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setValue(0)
        self._result_label.setText("")
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.sig_log_message.emit("[INFO] Inference cancel requested…")

    def _on_progress(self, current: int, total: int) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(current)

    def _on_done(self, metrics: dict) -> None:
        acc  = metrics.get("accuracy", float("nan"))
        f1   = metrics.get("f1_macro", float("nan"))
        mcc  = metrics.get("mcc", float("nan"))
        loss = metrics.get("avg_loss", float("nan"))
        self._result_label.setText(
            f"acc={acc:.4f}  f1_macro={f1:.4f}  mcc={mcc:.4f}  loss={loss:.4f}"
        )
        self.sig_log_message.emit(
            f"[INFER] accuracy={acc:.4f}  f1_macro={f1:.4f}  "
            f"mcc={mcc:.4f}  avg_loss={loss:.4f}"
        )

    def _on_error(self, tb: str) -> None:
        self.sig_log_message.emit(f"[INFER ERROR]\n{tb}")
        first_line = tb.strip().splitlines()[-1] if tb.strip() else "Unknown error"
        QMessageBox.critical(self, "Inference error", first_line)

    def _on_finished(self) -> None:
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
