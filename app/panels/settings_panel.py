"""
Settings panel — H5 path, backbone, hyperparameters, directories.
Settings are persisted to a JSON file next to the H5 path via QSettings fallback.
"""

from __future__ import annotations

import json
import os

import torch
from PyQt5.QtCore import QStandardPaths
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.metrics import DEFAULT_TARGET_METRIC, TARGET_METRICS
from src.model import AVAILABLE_BACKBONES

_SETTINGS_FILE = os.path.join(
    QStandardPaths.writableLocation(QStandardPaths.AppDataLocation),
    "image_classifier",
    "settings.json",
)

_DEFAULTS: dict = {
    "h5_path":          "",
    "resume_checkpoint": "",
    "backbone":         "simple_cnn",
    "in_channels":      1,
    "lr":               1e-3,
    "batch_size":       32,
    "epochs":           10,
    "optimizer":        "Adam",
    "scheduler":        "CosineAnnealing",
    "loss_fn":          "CrossEntropy",
    "focal_gamma":      2.0,
    "pretrained":       False,
    "target_metric":    DEFAULT_TARGET_METRIC,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir":   "checkpoints",
    "log_dir":          "runs",
    "experiment_name":  "experiment",
    "tensorboard_port": 6006,
    "num_workers":      0,
    "pin_memory":       torch.cuda.is_available(),
    "shuffle_every_n_epochs": 1,
    "keep_last":        3,
    "recall_targets":   "0.95, 0.99",
}


class SettingsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ── scrollable container ──────────────────────────────────────────
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        form = QFormLayout(inner)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ── Dataset ───────────────────────────────────────────────────────
        ds_box = QGroupBox("Dataset")
        ds_lay = QFormLayout(ds_box)

        self._h5_edit = QLineEdit()
        self._h5_edit.setPlaceholderText("Path to .h5 dataset file")
        btn_browse_h5 = QPushButton("Browse…")
        btn_browse_h5.clicked.connect(self._browse_h5)
        h5_row = QHBoxLayout()
        h5_row.addWidget(self._h5_edit)
        h5_row.addWidget(btn_browse_h5)
        ds_lay.addRow("H5 file:", h5_row)

        self._in_channels = QSpinBox()
        self._in_channels.setRange(1, 3)
        self._in_channels.setValue(1)
        ds_lay.addRow("Input channels:", self._in_channels)

        form.addRow(ds_box)

        # ── Model ─────────────────────────────────────────────────────────
        model_box = QGroupBox("Model")
        model_lay = QFormLayout(model_box)

        self._backbone = QComboBox()
        self._backbone.addItems(AVAILABLE_BACKBONES)
        self._backbone.currentTextChanged.connect(self._on_backbone_changed)
        model_lay.addRow("Backbone:", self._backbone)

        self._pretrained = QCheckBox("Use pretrained weights")
        model_lay.addRow("", self._pretrained)

        form.addRow(model_box)

        # ── Training ──────────────────────────────────────────────────────
        train_box = QGroupBox("Training")
        train_lay = QFormLayout(train_box)

        self._optimizer = QComboBox()
        self._optimizer.addItems(["Adam", "AdamW", "SGD", "RMSprop"])
        train_lay.addRow("Optimizer:", self._optimizer)

        self._scheduler = QComboBox()
        self._scheduler.addItems(["CosineAnnealing", "StepLR", "ReduceLROnPlateau", "None"])
        self._scheduler.setCurrentText("CosineAnnealing")
        train_lay.addRow("LR scheduler:", self._scheduler)

        self._loss_fn = QComboBox()
        self._loss_fn.addItems(["CrossEntropy", "FocalLoss"])
        self._loss_fn.setCurrentText("CrossEntropy")
        train_lay.addRow("Loss function:", self._loss_fn)

        self._focal_gamma = QDoubleSpinBox()
        self._focal_gamma.setDecimals(1)
        self._focal_gamma.setRange(0.0, 10.0)
        self._focal_gamma.setSingleStep(0.5)
        self._focal_gamma.setValue(2.0)
        self._focal_gamma.setToolTip(
            "Focal loss focusing parameter γ.  γ=0 reduces to standard cross-entropy.  "
            "Typical value is 2 for imbalanced classes."
        )
        train_lay.addRow("Focal γ:", self._focal_gamma)
        self._loss_fn.currentTextChanged.connect(
            lambda t: self._focal_gamma.setEnabled(t == "FocalLoss")
        )
        self._focal_gamma.setEnabled(False)

        self._lr = QDoubleSpinBox()
        self._lr.setDecimals(6)
        self._lr.setRange(1e-7, 1.0)
        self._lr.setSingleStep(1e-4)
        self._lr.setValue(1e-3)
        train_lay.addRow("Learning rate:", self._lr)

        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 2048)
        self._batch_size.setValue(32)
        train_lay.addRow("Batch size:", self._batch_size)

        self._epochs = QSpinBox()
        self._epochs.setRange(1, 9999)
        self._epochs.setValue(10)
        train_lay.addRow("Epochs:", self._epochs)

        self._num_workers = QSpinBox()
        self._num_workers.setRange(0, 32)
        self._num_workers.setValue(0)
        self._num_workers.setToolTip(
            "Number of DataLoader worker processes.\n"
            "On Windows + CUDA, set to 0: each worker re-imports torch and "
            "commits ~1 GB of virtual memory for the CUDA DLLs, which often "
            "exceeds the default Windows page-file size (WinError 1455)."
        )
        train_lay.addRow("DataLoader workers:", self._num_workers)

        self._pin_memory = QCheckBox("Pin memory (faster CPU→GPU transfer)")
        self._pin_memory.setChecked(torch.cuda.is_available())
        self._pin_memory.setToolTip(
            "Enables pinned (page-locked) memory in the DataLoader.\n"
            "Only beneficial when training on a CUDA GPU."
        )
        train_lay.addRow("", self._pin_memory)

        self._shuffle_every = QSpinBox()
        self._shuffle_every.setRange(0, 9999)
        self._shuffle_every.setValue(1)
        self._shuffle_every.setToolTip(
            "How often (in epochs) to re-shuffle the training data.\n"
            "1 = every epoch (default), N = every N epochs, 0 = never shuffle."
        )
        train_lay.addRow("Shuffle every N epochs:", self._shuffle_every)

        self._keep_last = QSpinBox()
        self._keep_last.setRange(1, 100)
        self._keep_last.setValue(3)
        self._keep_last.setToolTip("Number of recent epoch checkpoints to keep on disk (best.pt is always kept)")
        train_lay.addRow("Keep last N checkpoints:", self._keep_last)

        self._target_metric = QComboBox()
        self._target_metric.addItems(TARGET_METRICS)
        self._target_metric.setCurrentText(DEFAULT_TARGET_METRIC)
        train_lay.addRow("Target metric:", self._target_metric)

        self._recall_targets = QLineEdit("0.95, 0.99")
        self._recall_targets.setPlaceholderText("e.g. 0.90, 0.95, 0.99 — leave blank to disable")
        self._recall_targets.setToolTip(
            "Comma-separated target recall values in (0, 1]. For each value, the validation "
            "epoch logs a per-class probability threshold (one-vs-rest) achieving that recall, "
            "plus the resulting precision. Logged to TensorBoard under val/threshold@rX.XX/<class>."
        )
        train_lay.addRow("Recall targets:", self._recall_targets)

        self._device = QComboBox()
        self._device.addItem("CPU", "cpu")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                self._device.addItem(f"GPU {i}: {name}", f"cuda:{i}")
        device_default = "cuda:0" if torch.cuda.is_available() else "cpu"
        idx = self._device.findData(device_default)
        if idx >= 0:
            self._device.setCurrentIndex(idx)
        train_lay.addRow("Device:", self._device)
        # auto-toggle pin_memory when device changes
        self._device.currentIndexChanged.connect(self._on_device_changed)

        form.addRow(train_box)

        # ── Resume ───────────────────────────────────────────────────────
        resume_box = QGroupBox("Resume from checkpoint (optional)")
        resume_lay = QFormLayout(resume_box)

        self._resume_edit = QLineEdit()
        self._resume_edit.setPlaceholderText("Leave blank to start fresh")
        btn_browse_resume = QPushButton("Browse…")
        btn_browse_resume.clicked.connect(self._browse_resume)
        btn_clear_resume = QPushButton("✕")
        btn_clear_resume.setFixedWidth(28)
        btn_clear_resume.setToolTip("Clear — start from scratch")
        btn_clear_resume.clicked.connect(lambda: self._resume_edit.clear())
        resume_row = QHBoxLayout()
        resume_row.addWidget(self._resume_edit)
        resume_row.addWidget(btn_browse_resume)
        resume_row.addWidget(btn_clear_resume)
        resume_lay.addRow("Checkpoint file:", resume_row)

        form.addRow(resume_box)

        # ── Directories ───────────────────────────────────────────────────
        dir_box = QGroupBox("Directories & Logging")
        dir_lay = QFormLayout(dir_box)

        self._checkpoint_dir = QLineEdit("checkpoints")
        btn_ck = QPushButton("Browse…")
        btn_ck.clicked.connect(lambda: self._browse_dir(self._checkpoint_dir))
        ck_row = QHBoxLayout()
        ck_row.addWidget(self._checkpoint_dir)
        ck_row.addWidget(btn_ck)
        dir_lay.addRow("Checkpoint dir:", ck_row)

        self._log_dir = QLineEdit("runs")
        btn_log = QPushButton("Browse…")
        btn_log.clicked.connect(lambda: self._browse_dir(self._log_dir))
        log_row = QHBoxLayout()
        log_row.addWidget(self._log_dir)
        log_row.addWidget(btn_log)
        dir_lay.addRow("Log dir:", log_row)

        self._experiment_name = QLineEdit("experiment")
        dir_lay.addRow("Experiment name:", self._experiment_name)

        self._tb_port = QSpinBox()
        self._tb_port.setRange(1024, 65535)
        self._tb_port.setValue(6006)
        dir_lay.addRow("TensorBoard port:", self._tb_port)

        form.addRow(dir_box)

        # ── Persist buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save settings")
        btn_save.clicked.connect(self.save_settings)
        btn_load = QPushButton("Load settings")
        btn_load.clicked.connect(self._browse_and_load)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_load)
        form.addRow(btn_row)

        # ── Load persisted settings ───────────────────────────────────────
        self._load_from_file(_SETTINGS_FILE)

    # ── Private helpers ───────────────────────────────────────────────────

    def _on_backbone_changed(self, name: str) -> None:
        self._pretrained.setEnabled(name != "simple_cnn")

    def _on_device_changed(self) -> None:
        is_gpu = self._device.currentData() not in (None, "cpu")
        self._pin_memory.setEnabled(is_gpu)
        if not is_gpu:
            self._pin_memory.setChecked(False)

    def _browse_h5(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select H5 dataset", "", "HDF5 files (*.h5 *.hdf5);;All files (*)"
        )
        if path:
            self._h5_edit.setText(path)

    def _browse_resume(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint to resume from",
            self._checkpoint_dir.text(),
            "PyTorch checkpoint (*.pt);;All files (*)"
        )
        if path:
            self._resume_edit.setText(path)

    def _browse_dir(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select directory", line_edit.text())
        if path:
            line_edit.setText(path)

    def _browse_and_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load settings", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            self._load_from_file(path)

    def _load_from_file(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data: dict = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self._apply_settings({**_DEFAULTS, **data})

    def _apply_settings(self, s: dict) -> None:
        self._h5_edit.setText(s.get("h5_path", ""))
        idx = self._backbone.findText(s.get("backbone", "simple_cnn"))
        self._backbone.setCurrentIndex(max(0, idx))
        self._in_channels.setValue(int(s.get("in_channels", 1)))
        self._pretrained.setChecked(bool(s.get("pretrained", False)))
        idx = self._optimizer.findText(s.get("optimizer", "Adam"))
        self._optimizer.setCurrentIndex(max(0, idx))
        idx = self._scheduler.findText(s.get("scheduler", "CosineAnnealing"))
        self._scheduler.setCurrentIndex(max(0, idx))
        idx = self._loss_fn.findText(s.get("loss_fn", "CrossEntropy"))
        self._loss_fn.setCurrentIndex(max(0, idx))
        self._focal_gamma.setValue(float(s.get("focal_gamma", 2.0)))
        self._focal_gamma.setEnabled(s.get("loss_fn", "CrossEntropy") == "FocalLoss")
        self._lr.setValue(float(s.get("lr", 1e-3)))
        self._batch_size.setValue(int(s.get("batch_size", 32)))
        self._epochs.setValue(int(s.get("epochs", 10)))
        self._num_workers.setValue(int(s.get("num_workers", 0)))
        self._keep_last.setValue(int(s.get("keep_last", 3)))
        self._shuffle_every.setValue(int(s.get("shuffle_every_n_epochs", 1)))
        self._recall_targets.setText(str(s.get("recall_targets", "0.95, 0.99")))
        self._pin_memory.setChecked(bool(s.get("pin_memory", torch.cuda.is_available())))
        idx = self._target_metric.findText(s.get("target_metric", DEFAULT_TARGET_METRIC))
        self._target_metric.setCurrentIndex(max(0, idx))
        device_data = s.get("device", "cpu")
        idx = self._device.findData(device_data)
        if idx < 0:  # fall back: map bare "cuda" → first cuda entry
            idx = self._device.findData("cuda:0") if "cuda" in device_data else 0
        self._device.setCurrentIndex(max(0, idx))
        self._resume_edit.setText(s.get("resume_checkpoint", ""))
        self._checkpoint_dir.setText(s.get("checkpoint_dir", "checkpoints"))
        self._log_dir.setText(s.get("log_dir", "runs"))
        self._experiment_name.setText(s.get("experiment_name", "experiment"))
        self._tb_port.setValue(int(s.get("tensorboard_port", 6006)))

    # ── Public API ────────────────────────────────────────────────────────

    def get_settings(self) -> dict:
        """Return the current settings as a plain dict."""
        return {
            "h5_path":          self._h5_edit.text().strip(),
            "backbone":         self._backbone.currentText(),
            "in_channels":      self._in_channels.value(),
            "pretrained":       self._pretrained.isChecked(),
            "optimizer":        self._optimizer.currentText(),
            "scheduler":        self._scheduler.currentText(),
            "loss_fn":          self._loss_fn.currentText(),
            "focal_gamma":      self._focal_gamma.value(),
            "lr":               self._lr.value(),
            "batch_size":       self._batch_size.value(),
            "epochs":           self._epochs.value(),
            "num_workers":      self._num_workers.value(),
            "pin_memory":       self._pin_memory.isChecked(),
            "shuffle_every_n_epochs": self._shuffle_every.value(),
            "keep_last":        self._keep_last.value(),
            "recall_targets":   self._recall_targets.text().strip(),
            "target_metric":    self._target_metric.currentText(),
            "device":           self._device.currentData(),
            "resume_checkpoint": self._resume_edit.text().strip(),
            "checkpoint_dir":   self._checkpoint_dir.text().strip(),
            "log_dir":          self._log_dir.text().strip(),
            "experiment_name":  self._experiment_name.text().strip(),
            "tensorboard_port": self._tb_port.value(),
        }

    def save_settings(self, path: str | None = None) -> None:
        """Persist settings to *path* (defaults to the app-data JSON file)."""
        target = path or _SETTINGS_FILE
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.get_settings(), f, indent=2)
