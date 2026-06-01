"""
Settings panel — H5 path, backbone, hyperparameters, directories.
(Full implementation in Phase 6)
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel


class SettingsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Settings — Phase 6"))
        layout.addStretch()

    def get_settings(self) -> dict:
        """Return current settings as a dict. Stub returns defaults."""
        return {
            "h5_path":          "",
            "backbone":         "simple_cnn",
            "in_channels":      1,
            "lr":               1e-3,
            "batch_size":       32,
            "epochs":           10,
            "optimizer":        "Adam",
            "pretrained":       False,
            "target_metric":    "f1_macro",
            "device":           "cpu",
            "checkpoint_dir":   "checkpoints",
            "log_dir":          "runs",
            "experiment_name":  "experiment",
            "tensorboard_port": 6006,
        }
