"""
Main window — QMainWindow with dockable panels.

Layout:
    Left dock   — Settings panel   (fixed width)
    Center      — Console panel    (expands)
    Right dock  — Control + Checkpoint panels (stacked vertically)
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QStatusBar,
)

from app.panels.settings_panel    import SettingsPanel
from app.panels.control_panel     import ControlPanel
from app.panels.console_panel     import ConsolePanel
from app.panels.checkpoint_panel  import CheckpointPanel
from app.panels.tensorboard_panel import TensorBoardPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier")
        self.resize(1200, 700)

        # ---- Central widget: console ----
        self.console_panel = ConsolePanel()
        self.setCentralWidget(self.console_panel)

        # ---- Left dock: settings ----
        self.settings_panel = SettingsPanel()
        left_dock = QDockWidget("Settings", self)
        left_dock.setWidget(self.settings_panel)
        left_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        left_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

        # ---- Right dock: controls + checkpoints stacked ----
        self.control_panel    = ControlPanel(self.settings_panel)
        self.checkpoint_panel = CheckpointPanel()

        right_container = QWidget()
        right_layout    = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.control_panel)
        right_layout.addWidget(self.checkpoint_panel)
        self.tensorboard_panel = TensorBoardPanel()
        right_layout.addWidget(self.tensorboard_panel)
        right_layout.addStretch()

        right_dock = QDockWidget("Controls", self)
        right_dock.setWidget(right_container)
        right_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        right_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

        # ---- Status bar ----
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

        # ---- Wire control panel signals to console ----
        self.control_panel.sig_log_message.connect(self.console_panel.append_message)
        self.control_panel.sig_epoch_complete.connect(self._on_epoch_complete)
        self.control_panel.sig_training_finished.connect(self._on_training_finished)
        self.control_panel.sig_checkpoint_saved.connect(
            self.checkpoint_panel.refresh
        )

        # ---- Wire checkpoint panel signals ----
        self.checkpoint_panel.sig_resume_requested.connect(self._on_resume_requested)

        # ---- TensorBoard panel: configure once per training run ----
        self.control_panel.sig_training_started.connect(self._on_training_started)

        # ---- Configure TB panel from persisted settings at startup ----
        _s = self.settings_panel.get_settings()
        self.tensorboard_panel.configure(
            log_dir=_s.get("log_dir", "runs"),
            port=_s.get("tensorboard_port", 6006),
        )

    # ------------------------------------------------------------------
    def _on_epoch_complete(self, info: dict):
        self._status.showMessage(
            f"Epoch {info['epoch']}  │  "
            f"val_loss={info['val_loss']:.4f}  "
            f"{info['target_metric']}={info['target_val']:.4f}"
        )

    def _on_training_finished(self):
        self._status.showMessage("Training complete")

    def _on_resume_requested(self, path: str):
        self.settings_panel._resume_edit.setText(path)
        self.console_panel.append_message(f"[INFO] Resume checkpoint set: {path}")

    def _on_training_started(self, settings: dict) -> None:
        self.tensorboard_panel.configure(
            log_dir=settings.get("log_dir", "runs"),
            port=settings.get("tensorboard_port", 6006),
        )
        self._status.showMessage("Training started…")

    def closeEvent(self, event):
        self.settings_panel.save_settings()
        self.tensorboard_panel.cleanup()
        super().closeEvent(event)
