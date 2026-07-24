"""
Settings panel — H5 path, backbone, hyperparameters, directories.
Settings are persisted to a JSON file next to the H5 path via QSettings fallback.
"""

from __future__ import annotations

import json
import os

import torch
from PyQt5.QtCore import QStandardPaths, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.panels.common import fit_tabs, scrollable
from app.theme import DEFAULT_THEME
from src.augment import (
    AUGMENT_DEFAULTS, AUGMENT_PRESETS, DEFAULT_NORMALIZE, NORMALIZE_MODES,
)
from src.metrics import DEFAULT_TARGET_METRIC, TARGET_METRICS
from src.model import AVAILABLE_BACKBONES

# Preset name shown once any individual control is touched.
_CUSTOM_PRESET = "custom"

class _StatsWorker(QThread):
    """Measures per-channel mean/std off the GUI thread.

    Streaming a large train split takes long enough to freeze the window if
    done inline, and this is a button a user presses expecting a result.
    """

    sig_done = pyqtSignal(list, list)
    sig_error = pyqtSignal(str)

    def __init__(self, h5_path: str, in_channels: int, parent=None):
        super().__init__(parent)
        self._h5 = h5_path
        self._c = in_channels

    def run(self) -> None:
        try:
            # Imported here, not at module scope: pulling torch/h5py in during
            # panel construction would slow app start for a button that is
            # rarely pressed.
            from src.augment import compute_dataset_stats
            from src.dataset import H5Dataset, make_dataloader, SPLIT_TRAIN

            ds = H5Dataset(self._h5, split=SPLIT_TRAIN)
            if len(ds) == 0:
                raise ValueError("Train split contains 0 samples.")
            loader = make_dataloader(ds, batch_size=256, shuffle=False,
                                     num_workers=0, pin_memory=False)
            mean, std = compute_dataset_stats(loader, device="cpu")
            self.sig_done.emit(list(mean), list(std))
        except Exception as exc:                       # surfaced in a dialog
            self.sig_error.emit(f"{type(exc).__name__}: {exc}")


class _WeightPreviewWorker(QThread):
    """Counts the train split and computes class weights off the GUI thread."""

    # (labels, counts, weights) as plain lists, plus the largest-class index.
    sig_done = pyqtSignal(list, list, list, int)
    sig_error = pyqtSignal(str)

    def __init__(self, h5_path: str, scheme: str, beta: float, parent=None):
        super().__init__(parent)
        self._h5 = h5_path
        self._scheme = scheme
        self._beta = beta

    def run(self) -> None:
        try:
            from src.dataset import count_labels, peek_h5_meta, SPLIT_TRAIN
            from src.trainer import class_weights

            meta = peek_h5_meta(self._h5)
            names = meta["classes"]
            counts = count_labels(self._h5, SPLIT_TRAIN, len(names))
            weights = class_weights(counts, scheme=self._scheme, beta=self._beta)
            self.sig_done.emit(list(names), [int(c) for c in counts],
                               [float(w) for w in weights], int(counts.argmax()))
        except Exception as exc:
            self.sig_error.emit(f"{type(exc).__name__}: {exc}")


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
    "restart_period":   5,
    "restart_mult":     1,
    "loss_fn":          "CrossEntropy",
    "focal_gamma":      2.0,
    "class_weight_mode": "none",
    "class_weight_beta": 0.999,
    "pretrained":       False,
    "target_metric":    DEFAULT_TARGET_METRIC,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir":   "checkpoints",
    "log_dir":          "runs",
    "experiment_name":  "experiment",
    "tensorboard_port": 6006,
    "num_workers":      0,
    "pin_memory":       torch.cuda.is_available(),
    "use_amp":          torch.cuda.is_available(),
    "shuffle_every_n_epochs": 1,
    "keep_last":        3,
    "recall_targets":   "0.95, 0.99",
    "specificity_targets": "0.95, 0.99",
    "theme":            DEFAULT_THEME,
    "seed":             0,
    # Preprocessing — applies to train, validation and inference alike.
    "normalize":        DEFAULT_NORMALIZE,
    "normalize_mean":   [],
    "normalize_std":    [],
    "aug_preset":       "none",
    # Train-only augmentation; 16 flat scalars, see src/augment.py
    **AUGMENT_DEFAULTS,
}


class SettingsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set before _load_from_file: _apply_settings is the only other writer
        # and it never runs when there is no settings file yet.
        self._theme = DEFAULT_THEME

        # ── tabbed container ──────────────────────────────────────────────
        # One 24-row form meant every new option lengthened a single 1025px
        # scroll. Five pages of ~7 rows each leaves room to grow per topic.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs)

        def _page(title: str) -> QFormLayout:
            """Add a scrollable form page and return its layout."""
            page = QWidget()
            lay  = QFormLayout(page)
            lay.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            self._tabs.addTab(scrollable(page), title)
            return lay

        ds_lay    = _page("Data")
        model_lay = _page("Model")
        train_lay = _page("Optimizer")
        aug_lay   = _page("Augment")
        hw_lay    = _page("Hardware")
        out_lay   = _page("Output")

        # Guards the preset <-> widget feedback loop: setting widgets from a
        # preset must not itself flip the preset back to "custom".
        self._loading = False
        self._normalize_mean: list[float] = []
        self._normalize_std: list[float] = []

        # ── Data ──────────────────────────────────────────────────────────
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

        self._shuffle_every = QSpinBox()
        self._shuffle_every.setRange(0, 9999)
        self._shuffle_every.setValue(1)
        self._shuffle_every.setToolTip(
            "How often (in epochs) to re-shuffle the training data.\n"
            "1 = every epoch (default), N = every N epochs, 0 = never shuffle."
        )
        ds_lay.addRow("Shuffle every N epochs:", self._shuffle_every)

        # ── Normalisation (Data, not Augment: this is preprocessing and runs
        #    on train, validation and inference alike) ──────────────────────
        self._normalize = QComboBox()
        self._normalize.addItems(NORMALIZE_MODES)
        self._normalize.setCurrentText(DEFAULT_NORMALIZE)
        self._normalize.setToolTip(
            "Per-channel (x - mean) / std, applied to every split.\n"
            "  none     — raw [0, 1]\n"
            "  imagenet — ImageNet statistics, for pretrained backbones\n"
            "  dataset  — measured from your train split (use the button below)\n"
            "  custom   — whatever mean/std are currently stored\n"
            "The values used are written into the checkpoint so inference "
            "reproduces them exactly."
        )
        self._normalize.currentTextChanged.connect(self._on_normalize_changed)
        ds_lay.addRow("Normalisation:", self._normalize)

        self._norm_label = QLabel("—")
        self._norm_label.setWordWrap(True)
        ds_lay.addRow("Current stats:", self._norm_label)

        self._btn_stats = QPushButton("Compute from dataset")
        self._btn_stats.setToolTip(
            "Stream the train split and measure per-channel mean/std. "
            "Sets the mode to 'dataset'."
        )
        self._btn_stats.clicked.connect(self._on_compute_stats)
        ds_lay.addRow("", self._btn_stats)

        # ── Augment (train only) ──────────────────────────────────────────
        self._aug_enabled = QCheckBox("Enable augmentation (training batches only)")
        self._aug_enabled.setToolTip(
            "Augmentation is applied on the GPU to whole batches, with "
            "independent random parameters per image.\n"
            "Never applied to validation or inference."
        )
        aug_lay.addRow("", self._aug_enabled)

        self._aug_preset = QComboBox()
        self._aug_preset.addItems(list(AUGMENT_PRESETS) + [_CUSTOM_PRESET])
        self._aug_preset.setToolTip(
            "Starting points. 'standard' is the usual CIFAR recipe "
            "(pad-and-crop + horizontal flip + mild jitter).\n"
            "Changing any control below switches this to 'custom'."
        )
        self._aug_preset.activated.connect(self._on_preset_chosen)
        aug_lay.addRow("Preset:", self._aug_preset)

        def _prob(tip: str, maximum: float = 1.0, step: float = 0.05,
                  decimals: int = 2) -> QDoubleSpinBox:
            sb = QDoubleSpinBox()
            sb.setDecimals(decimals)
            sb.setRange(0.0, maximum)
            sb.setSingleStep(step)
            sb.setToolTip(tip)
            sb.valueChanged.connect(self._on_aug_widget_changed)
            return sb

        self._aug_hflip_p = _prob("Probability of a horizontal flip, per image.")
        aug_lay.addRow("Horizontal flip p:", self._aug_hflip_p)

        self._aug_vflip_p = _prob(
            "Probability of a vertical flip. Usually wrong for natural images — "
            "an upside-down cat is not a cat.")
        aug_lay.addRow("Vertical flip p:", self._aug_vflip_p)

        self._aug_crop_padding = QSpinBox()
        self._aug_crop_padding.setRange(0, 64)
        self._aug_crop_padding.setToolTip(
            "Zero-pad by N pixels then take a random crop back to the original "
            "size. 4 is the standard choice for 32x32. 0 disables.")
        self._aug_crop_padding.valueChanged.connect(self._on_aug_widget_changed)
        aug_lay.addRow("Crop padding (px):", self._aug_crop_padding)

        self._aug_rotate_deg = _prob("Max rotation in degrees, +/-.", 180.0, 1.0, 1)
        aug_lay.addRow("Rotation (deg):", self._aug_rotate_deg)

        self._aug_translate = _prob("Max shift as a fraction of image size, +/-.", 0.5)
        aug_lay.addRow("Translate:", self._aug_translate)

        self._aug_scale_jitter = _prob("Max scale change, +/- this fraction.", 0.9)
        aug_lay.addRow("Scale jitter:", self._aug_scale_jitter)

        self._aug_shear_deg = _prob("Max shear in degrees, +/-.", 45.0, 1.0, 1)
        aug_lay.addRow("Shear (deg):", self._aug_shear_deg)

        self._aug_brightness = _prob("Max brightness change, +/- this fraction.")
        aug_lay.addRow("Brightness:", self._aug_brightness)

        self._aug_contrast = _prob("Max contrast change, +/- this fraction.")
        aug_lay.addRow("Contrast:", self._aug_contrast)

        self._aug_saturation = _prob(
            "Max saturation change, +/-. Ignored when input channels = 1.")
        aug_lay.addRow("Saturation:", self._aug_saturation)

        self._aug_hue = _prob(
            "Max hue rotation as a fraction of the colour wheel, +/-. "
            "Ignored when input channels = 1.", 0.5, 0.01)
        aug_lay.addRow("Hue:", self._aug_hue)

        self._aug_erase_p = _prob("Probability of blanking a random box, per image.")
        aug_lay.addRow("Random erase p:", self._aug_erase_p)

        self._aug_erase_min = _prob("Smallest erased box, as a fraction of area.",
                                    1.0, 0.01)
        aug_lay.addRow("Erase min area:", self._aug_erase_min)

        self._aug_erase_max = _prob("Largest erased box, as a fraction of area.",
                                    1.0, 0.01)
        aug_lay.addRow("Erase max area:", self._aug_erase_max)

        self._aug_erase_value = _prob("Fill value for the erased box, in [0, 1].")
        aug_lay.addRow("Erase fill:", self._aug_erase_value)

        self._aug_enabled.toggled.connect(self._on_aug_enabled_toggled)

        # ── Model ─────────────────────────────────────────────────────────
        self._backbone = QComboBox()
        self._backbone.addItems(AVAILABLE_BACKBONES)
        self._backbone.currentTextChanged.connect(self._on_backbone_changed)
        model_lay.addRow("Backbone:", self._backbone)

        self._pretrained = QCheckBox("Use pretrained weights")
        model_lay.addRow("", self._pretrained)

        # ── Optimizer ─────────────────────────────────────────────────────
        self._optimizer = QComboBox()
        self._optimizer.addItems(["Adam", "AdamW", "SGD", "RMSprop"])
        train_lay.addRow("Optimizer:", self._optimizer)

        self._scheduler = QComboBox()
        self._scheduler.addItems(["CosineAnnealing", "CosineWarmRestarts",
                                  "StepLR", "ReduceLROnPlateau", "None"])
        self._scheduler.setCurrentText("CosineAnnealing")
        self._scheduler.setToolTip(
            "CosineAnnealing       — smooth decay to ~0 over the whole run\n"
            "CosineWarmRestarts    — cyclic: decay, jump back up, repeat (SGDR)\n"
            "StepLR                — drop 10× at 1/3 and 2/3 of the run\n"
            "ReduceLROnPlateau     — halve when val loss stalls\n"
            "None                  — constant LR"
        )
        train_lay.addRow("LR scheduler:", self._scheduler)

        # Warm-restart parameters — only meaningful for CosineWarmRestarts.
        self._restart_period = QSpinBox()
        self._restart_period.setRange(1, 9999)
        self._restart_period.setValue(5)
        self._restart_period.setToolTip(
            "T_0: epochs in the first cycle before the LR restarts.\n"
            "If this is ≥ total epochs, there is no restart — it becomes a "
            "single truncated cosine decay."
        )
        train_lay.addRow("Restart period (T_0):", self._restart_period)

        self._restart_mult = QSpinBox()
        self._restart_mult.setRange(1, 16)
        self._restart_mult.setValue(1)
        self._restart_mult.setToolTip(
            "T_mult: each cycle is this many times longer than the previous "
            "one. 1 = every cycle is T_0 epochs; 2 = cycles of T_0, 2·T_0, "
            "4·T_0, …"
        )
        train_lay.addRow("Restart mult (T_mult):", self._restart_mult)

        self._scheduler.currentTextChanged.connect(self._on_scheduler_changed)
        self._on_scheduler_changed(self._scheduler.currentText())

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

        # ── Class weighting ────────────────────────────────────────────────
        # Re-weight the loss instead of re-sampling the data: every sample is
        # still trained on, the majority class just counts less per example.
        self._class_weight = QComboBox()
        self._class_weight.addItems(["none", "effective_number", "inverse_freq"])
        self._class_weight.setToolTip(
            "Down-weight frequent classes in the loss, keeping every sample.\n"
            "  none              — no weighting\n"
            "  effective_number  — Cui et al. 2019, gentle at extreme ratios "
            "(recommended for a ~50:1 background)\n"
            "  inverse_freq      — w ∝ 1/count, stronger and simpler\n"
            "Applies to both CrossEntropy and FocalLoss."
        )
        train_lay.addRow("Class weighting:", self._class_weight)

        self._cw_beta = QDoubleSpinBox()
        self._cw_beta.setDecimals(4)
        self._cw_beta.setRange(0.0, 0.9999)
        self._cw_beta.setSingleStep(0.001)
        self._cw_beta.setValue(0.999)
        self._cw_beta.setToolTip(
            "Effective-number β. Higher = stronger down-weighting of frequent "
            "classes. 0.999 or 0.9999 are usual for large imbalance."
        )
        train_lay.addRow("Weight β:", self._cw_beta)

        self._cw_label = QLabel("—")
        self._cw_label.setWordWrap(True)
        train_lay.addRow("Weights:", self._cw_label)

        self._btn_cw_preview = QPushButton("Preview weights")
        self._btn_cw_preview.setToolTip(
            "Count the train split and show the resulting weights without "
            "training. The largest class is the one to watch — that is your "
            "hard-negative bucket."
        )
        self._btn_cw_preview.clicked.connect(self._on_preview_weights)
        train_lay.addRow("", self._btn_cw_preview)

        self._class_weight.currentTextChanged.connect(self._on_class_weight_changed)
        self._on_class_weight_changed(self._class_weight.currentText())

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

        self._seed = QSpinBox()
        self._seed.setRange(0, 2_147_483_647)
        self._seed.setValue(0)
        self._seed.setToolTip(
            "Seeds the augmentation draw sequence at the start of fit().\n"
            "Does not cover model initialisation or the DataLoader shuffle, "
            "which happen before the Trainer is built."
        )
        train_lay.addRow("Random seed:", self._seed)

        # ── Hardware ──────────────────────────────────────────────────────
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
        hw_lay.addRow("Device:", self._device)
        # auto-toggle pin_memory when device changes
        self._device.currentIndexChanged.connect(self._on_device_changed)

        self._num_workers = QSpinBox()
        self._num_workers.setRange(0, 32)
        self._num_workers.setValue(0)
        self._num_workers.setToolTip(
            "Number of DataLoader worker processes.\n"
            "On Windows + CUDA, set to 0: each worker re-imports torch and "
            "commits ~1 GB of virtual memory for the CUDA DLLs, which often "
            "exceeds the default Windows page-file size (WinError 1455)."
        )
        hw_lay.addRow("DataLoader workers:", self._num_workers)

        self._pin_memory = QCheckBox("Pin memory (faster CPU→GPU transfer)")
        self._pin_memory.setChecked(torch.cuda.is_available())
        self._pin_memory.setToolTip(
            "Enables pinned (page-locked) memory in the DataLoader.\n"
            "Only beneficial when training on a CUDA GPU."
        )
        hw_lay.addRow("", self._pin_memory)

        self._use_amp = QCheckBox("Mixed precision (fp16 autocast)")
        self._use_amp.setChecked(torch.cuda.is_available())
        self._use_amp.setToolTip(
            "Use automatic mixed precision (torch.cuda.amp) during training and validation.\n"
            "Typically 1.5–3× faster and ~40% less GPU memory on Tensor Core GPUs\n"
            "(Volta/Turing/Ampere/Ada). Silently disabled on CPU."
        )
        hw_lay.addRow("", self._use_amp)

        # ── Output ────────────────────────────────────────────────────────
        self._keep_last = QSpinBox()
        self._keep_last.setRange(1, 100)
        self._keep_last.setValue(3)
        self._keep_last.setToolTip("Number of recent epoch checkpoints to keep on disk (best.pt is always kept)")
        out_lay.addRow("Keep last N checkpoints:", self._keep_last)

        self._target_metric = QComboBox()
        self._target_metric.addItems(TARGET_METRICS)
        self._target_metric.setCurrentText(DEFAULT_TARGET_METRIC)
        self._target_metric.setToolTip(
            "Metric used to decide which epoch becomes best.pt."
        )
        out_lay.addRow("Target metric:", self._target_metric)

        self._recall_targets = QLineEdit("0.95, 0.99")
        self._recall_targets.setPlaceholderText("e.g. 0.90, 0.95, 0.99 — leave blank to disable")
        self._recall_targets.setToolTip(
            "Comma-separated target recall values in (0, 1]. For each value, the validation "
            "epoch logs a per-class probability threshold (one-vs-rest) achieving that recall, "
            "plus the resulting precision. Logged to TensorBoard under val/threshold@rX.XX/<class>."
        )
        out_lay.addRow("Recall targets:", self._recall_targets)

        self._specificity_targets = QLineEdit("0.95, 0.99")
        self._specificity_targets.setPlaceholderText(
            "e.g. 0.95, 0.99 — leave blank to disable")
        self._specificity_targets.setToolTip(
            "Comma-separated target specificity values in (0, 1]. The mirror of "
            "recall targets: for each value the validation epoch logs the per-class "
            "probability threshold (one-vs-rest) holding false positives to that "
            "specificity, plus the recall and precision you get there.\n"
            "Use recall targets when missing a positive is the expensive error, "
            "and specificity targets when a false alarm is."
        )
        out_lay.addRow("Specificity targets:", self._specificity_targets)

        self._checkpoint_dir = QLineEdit("checkpoints")
        btn_ck = QPushButton("Browse…")
        btn_ck.clicked.connect(lambda: self._browse_dir(self._checkpoint_dir))
        ck_row = QHBoxLayout()
        ck_row.addWidget(self._checkpoint_dir)
        ck_row.addWidget(btn_ck)
        out_lay.addRow("Checkpoint dir:", ck_row)

        self._log_dir = QLineEdit("runs")
        btn_log = QPushButton("Browse…")
        btn_log.clicked.connect(lambda: self._browse_dir(self._log_dir))
        log_row = QHBoxLayout()
        log_row.addWidget(self._log_dir)
        log_row.addWidget(btn_log)
        out_lay.addRow("Log dir:", log_row)

        self._experiment_name = QLineEdit("experiment")
        out_lay.addRow("Experiment name:", self._experiment_name)

        self._tb_port = QSpinBox()
        self._tb_port.setRange(1024, 65535)
        self._tb_port.setValue(6006)
        out_lay.addRow("TensorBoard port:", self._tb_port)

        # ── Resume (on the Model page — it selects what to start from) ────
        self._resume_edit = QLineEdit()
        self._resume_edit.setPlaceholderText("Leave blank to start fresh")
        btn_browse_resume = QPushButton("Browse…")
        btn_browse_resume.clicked.connect(self._browse_resume)
        btn_clear_resume = QPushButton("✕")
        btn_clear_resume.setFixedWidth(28)
        # See checkpoint_panel — opts out of the stylesheet's button padding.
        btn_clear_resume.setObjectName("iconButton")
        btn_clear_resume.setToolTip("Clear — start from scratch")
        btn_clear_resume.clicked.connect(lambda: self._resume_edit.clear())
        resume_row = QHBoxLayout()
        resume_row.addWidget(self._resume_edit)
        resume_row.addWidget(btn_browse_resume)
        resume_row.addWidget(btn_clear_resume)
        model_lay.addRow("Resume from:", resume_row)

        # ── Persist buttons ───────────────────────────────────────────────
        # Outside the tabs so they stay reachable from any page.
        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save settings")
        btn_save.clicked.connect(self.save_settings)
        btn_load = QPushButton("Load settings")
        btn_load.clicked.connect(self._browse_and_load)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_load)
        outer.addLayout(btn_row)

        fit_tabs(self._tabs)

        # ── Load persisted settings ───────────────────────────────────────
        self._load_from_file(_SETTINGS_FILE)

    # ── Private helpers ───────────────────────────────────────────────────

    # ── Augmentation / normalisation handlers ─────────────────────────────

    _AUG_WIDGETS = {
        "aug_hflip_p": "_aug_hflip_p", "aug_vflip_p": "_aug_vflip_p",
        "aug_crop_padding": "_aug_crop_padding", "aug_rotate_deg": "_aug_rotate_deg",
        "aug_translate": "_aug_translate", "aug_scale_jitter": "_aug_scale_jitter",
        "aug_shear_deg": "_aug_shear_deg", "aug_brightness": "_aug_brightness",
        "aug_contrast": "_aug_contrast", "aug_saturation": "_aug_saturation",
        "aug_hue": "_aug_hue", "aug_erase_p": "_aug_erase_p",
        "aug_erase_min": "_aug_erase_min", "aug_erase_max": "_aug_erase_max",
        "aug_erase_value": "_aug_erase_value",
    }

    def _set_aug_widgets(self, cfg: dict) -> None:
        """Push a config into the widgets without tripping the custom guard."""
        prev, self._loading = self._loading, True
        try:
            for key, attr in self._AUG_WIDGETS.items():
                getattr(self, attr).setValue(
                    type(getattr(self, attr).value())(cfg.get(key, AUGMENT_DEFAULTS[key]))
                )
            self._aug_enabled.setChecked(bool(cfg.get("aug_enabled", False)))
            # Inside the guard: this re-syncs the enabled/disabled state, but it
            # also routes through _on_aug_widget_changed, which would otherwise
            # flag the freshly-applied preset as "custom".
            self._on_aug_enabled_toggled(self._aug_enabled.isChecked())
        finally:
            self._loading = prev

    def _on_preset_chosen(self, _index: int) -> None:
        name = self._aug_preset.currentText()
        if name == _CUSTOM_PRESET:
            return
        self._set_aug_widgets(AUGMENT_PRESETS[name])

    def _on_aug_widget_changed(self, *_args) -> None:
        """Any manual edit means the config no longer matches a named preset."""
        if self._loading:
            return
        if self._aug_preset.currentText() != _CUSTOM_PRESET:
            self._aug_preset.setCurrentText(_CUSTOM_PRESET)

    def _on_aug_enabled_toggled(self, on: bool) -> None:
        for attr in self._AUG_WIDGETS.values():
            getattr(self, attr).setEnabled(on)
        self._aug_preset.setEnabled(on)
        if not self._loading:
            self._on_aug_widget_changed()

    def _refresh_norm_label(self) -> None:
        mode = self._normalize.currentText()
        if mode == "none":
            self._norm_label.setText("none — raw [0, 1]")
        elif mode == "imagenet":
            self._norm_label.setText("ImageNet statistics")
        elif self._normalize_mean and self._normalize_std:
            self._norm_label.setText(
                f"mean {[round(v, 4) for v in self._normalize_mean]}\n"
                f"std  {[round(v, 4) for v in self._normalize_std]}"
            )
        else:
            self._norm_label.setText("not set — will fall back to raw [0, 1]")

    def _on_normalize_changed(self, _mode: str) -> None:
        self._refresh_norm_label()

    def _on_compute_stats(self) -> None:
        h5 = self._h5_edit.text().strip()
        if not h5 or not os.path.isfile(h5):
            QMessageBox.warning(self, "Compute statistics",
                                f"H5 dataset not found:\n{h5}")
            return
        self._btn_stats.setEnabled(False)
        self._btn_stats.setText("Computing…")
        self._stats_worker = _StatsWorker(h5, self._in_channels.value())
        self._stats_worker.sig_done.connect(self._on_stats_done)
        self._stats_worker.sig_error.connect(self._on_stats_error)
        self._stats_worker.start()

    def _on_stats_done(self, mean: list, std: list) -> None:
        self._normalize_mean = list(mean)
        self._normalize_std = list(std)
        self._normalize.setCurrentText("dataset")
        self._refresh_norm_label()
        self._btn_stats.setEnabled(True)
        self._btn_stats.setText("Compute from dataset")

    def _on_stats_error(self, msg: str) -> None:
        self._btn_stats.setEnabled(True)
        self._btn_stats.setText("Compute from dataset")
        QMessageBox.critical(self, "Compute statistics", msg)

    # ── Class-weight handlers ─────────────────────────────────────────────

    def _on_class_weight_changed(self, mode: str) -> None:
        on = mode != "none"
        self._cw_beta.setEnabled(on and mode == "effective_number")
        self._btn_cw_preview.setEnabled(on)
        if not on:
            self._cw_label.setText("—")

    def _on_preview_weights(self) -> None:
        h5 = self._h5_edit.text().strip()
        if not h5 or not os.path.isfile(h5):
            QMessageBox.warning(self, "Preview weights",
                                f"H5 dataset not found:\n{h5}")
            return
        self._btn_cw_preview.setEnabled(False)
        self._btn_cw_preview.setText("Counting…")
        self._cw_worker = _WeightPreviewWorker(
            h5, self._class_weight.currentText(), self._cw_beta.value())
        self._cw_worker.sig_done.connect(self._on_weights_done)
        self._cw_worker.sig_error.connect(self._on_weights_error)
        self._cw_worker.start()

    def _on_weights_done(self, names, counts, weights, hi) -> None:
        self._btn_cw_preview.setEnabled(True)
        self._btn_cw_preview.setText("Preview weights")
        lo = min(weights)
        self._cw_label.setText(
            f"{len(weights)} classes · weights {lo:.3f}–{max(weights):.3f}\n"
            f"largest '{names[hi]}' (n={counts[hi]}) → {weights[hi]:.3f}"
        )

    def _on_weights_error(self, msg: str) -> None:
        self._btn_cw_preview.setEnabled(True)
        self._btn_cw_preview.setText("Preview weights")
        QMessageBox.critical(self, "Preview weights", msg)

    # ── Other handlers ────────────────────────────────────────────────────

    def _on_scheduler_changed(self, name: str) -> None:
        on = name == "CosineWarmRestarts"
        self._restart_period.setEnabled(on)
        self._restart_mult.setEnabled(on)

    def _on_backbone_changed(self, name: str) -> None:
        self._pretrained.setEnabled(name != "simple_cnn")

    def _on_device_changed(self) -> None:
        is_gpu = self._device.currentData() not in (None, "cpu")
        self._pin_memory.setEnabled(is_gpu)
        if not is_gpu:
            self._pin_memory.setChecked(False)
        self._use_amp.setEnabled(is_gpu)
        if not is_gpu:
            self._use_amp.setChecked(False)

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
        # Theme is persisted alongside the training settings but deliberately
        # kept out of get_settings(): that dict is handed to the Trainer as
        # `hyperparams` and logged via add_hparams(), where a UI preference
        # would just be noise.
        self._theme = s.get("theme", DEFAULT_THEME)
        self._h5_edit.setText(s.get("h5_path", ""))
        idx = self._backbone.findText(s.get("backbone", "simple_cnn"))
        self._backbone.setCurrentIndex(max(0, idx))
        self._in_channels.setValue(int(s.get("in_channels", 1)))
        self._pretrained.setChecked(bool(s.get("pretrained", False)))
        idx = self._optimizer.findText(s.get("optimizer", "Adam"))
        self._optimizer.setCurrentIndex(max(0, idx))
        idx = self._scheduler.findText(s.get("scheduler", "CosineAnnealing"))
        self._scheduler.setCurrentIndex(max(0, idx))
        self._restart_period.setValue(int(s.get("restart_period", 5)))
        self._restart_mult.setValue(int(s.get("restart_mult", 1)))
        self._on_scheduler_changed(self._scheduler.currentText())
        idx = self._loss_fn.findText(s.get("loss_fn", "CrossEntropy"))
        self._loss_fn.setCurrentIndex(max(0, idx))
        self._focal_gamma.setValue(float(s.get("focal_gamma", 2.0)))
        idx = self._class_weight.findText(s.get("class_weight_mode", "none"))
        self._class_weight.setCurrentIndex(max(0, idx))
        self._cw_beta.setValue(float(s.get("class_weight_beta", 0.999)))
        self._on_class_weight_changed(self._class_weight.currentText())
        self._focal_gamma.setEnabled(s.get("loss_fn", "CrossEntropy") == "FocalLoss")
        self._lr.setValue(float(s.get("lr", 1e-3)))
        self._batch_size.setValue(int(s.get("batch_size", 32)))
        self._epochs.setValue(int(s.get("epochs", 10)))
        self._num_workers.setValue(int(s.get("num_workers", 0)))
        self._keep_last.setValue(int(s.get("keep_last", 3)))
        self._shuffle_every.setValue(int(s.get("shuffle_every_n_epochs", 1)))
        self._recall_targets.setText(str(s.get("recall_targets", "0.95, 0.99")))
        self._specificity_targets.setText(
            str(s.get("specificity_targets", "0.95, 0.99")))
        self._pin_memory.setChecked(bool(s.get("pin_memory", torch.cuda.is_available())))
        self._use_amp.setChecked(bool(s.get("use_amp", torch.cuda.is_available())))
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
        self._seed.setValue(int(s.get("seed", 0)))

        # Normalisation + augmentation. _set_aug_widgets holds the guard so a
        # restored config is not misread as a manual edit.
        self._normalize_mean = list(s.get("normalize_mean") or [])
        self._normalize_std = list(s.get("normalize_std") or [])
        idx = self._normalize.findText(s.get("normalize", DEFAULT_NORMALIZE))
        self._normalize.setCurrentIndex(max(0, idx))
        self._refresh_norm_label()

        self._set_aug_widgets(s)
        preset = s.get("aug_preset", "none")
        idx = self._aug_preset.findText(preset)
        prev, self._loading = self._loading, True
        try:
            self._aug_preset.setCurrentIndex(max(0, idx))
        finally:
            self._loading = prev

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
            "restart_period":   self._restart_period.value(),
            "restart_mult":     self._restart_mult.value(),
            "loss_fn":          self._loss_fn.currentText(),
            "focal_gamma":      self._focal_gamma.value(),
            "class_weight_mode": self._class_weight.currentText(),
            "class_weight_beta": self._cw_beta.value(),
            "lr":               self._lr.value(),
            "batch_size":       self._batch_size.value(),
            "epochs":           self._epochs.value(),
            "num_workers":      self._num_workers.value(),
            "pin_memory":       self._pin_memory.isChecked(),
            "use_amp":          self._use_amp.isChecked(),
            "shuffle_every_n_epochs": self._shuffle_every.value(),
            "keep_last":        self._keep_last.value(),
            "recall_targets":   self._recall_targets.text().strip(),
            "specificity_targets": self._specificity_targets.text().strip(),
            "target_metric":    self._target_metric.currentText(),
            "device":           self._device.currentData(),
            "resume_checkpoint": self._resume_edit.text().strip(),
            "checkpoint_dir":   self._checkpoint_dir.text().strip(),
            "log_dir":          self._log_dir.text().strip(),
            "experiment_name":  self._experiment_name.text().strip(),
            "tensorboard_port": self._tb_port.value(),
            "seed":             self._seed.value(),
            "normalize":        self._normalize.currentText(),
            "normalize_mean":   list(self._normalize_mean),
            "normalize_std":    list(self._normalize_std),
            "aug_preset":       self._aug_preset.currentText(),
            "aug_enabled":      self._aug_enabled.isChecked(),
            **{key: getattr(self, attr).value()
               for key, attr in self._AUG_WIDGETS.items()},
        }

    @property
    def theme(self) -> str:
        """Persisted UI theme name — see _apply_settings for why it lives here
        rather than in get_settings()."""
        return self._theme

    @theme.setter
    def theme(self, name: str) -> None:
        self._theme = name

    def save_settings(self, path: str | None = None) -> None:
        """Persist settings to *path* (defaults to the app-data JSON file)."""
        target = path or _SETTINGS_FILE
        os.makedirs(os.path.dirname(target), exist_ok=True)
        payload = self.get_settings()
        payload["theme"] = self._theme
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
