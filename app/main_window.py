"""
Main window — QMainWindow with dockable panels.

Layout:
    Left dock   — Settings panel   (scrollable)
    Center      — Console panel    (expands)
    Right dock  — Tabbed: Train / Inference / Checkpoints / TensorBoard
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QDockWidget, QStatusBar, QTabWidget,
)

from app.panels.common            import fit_tabs, scrollable
from app.panels.settings_panel    import SettingsPanel
from app.panels.control_panel     import ControlPanel
from app.panels.inference_panel   import InferencePanel
from app.panels.console_panel     import ConsolePanel
from app.panels.checkpoint_panel  import CheckpointPanel
from app.panels.tensorboard_panel import TensorBoardPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier")
        # The tabbed right dock only needs ~220px now, so height is driven by the
        # settings form (~1025px natural — it scrolls at any realistic size) and
        # by how much log you want visible. 820 is a comfortable default that
        # still fits a 1080p screen once the taskbar is accounted for.
        self.resize(1200, 820)

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

        # ---- Right dock: one tab per task ----
        # Stacking these four vertically needed ~685px and overflowed the window.
        # Tabbed, the dock only ever needs the tallest single page (~220px), and
        # each page has room to grow.
        self.control_panel     = ControlPanel(self.settings_panel)
        self.inference_panel   = InferencePanel(self.settings_panel)
        self.checkpoint_panel  = CheckpointPanel()
        self.tensorboard_panel = TensorBoardPanel()

        right_tabs = QTabWidget()
        right_tabs.addTab(scrollable(self.control_panel),     "Train")
        right_tabs.addTab(scrollable(self.inference_panel),   "Inference")
        right_tabs.addTab(scrollable(self.checkpoint_panel),  "Checkpoints")
        right_tabs.addTab(scrollable(self.tensorboard_panel), "TensorBoard")
        self.right_tabs = right_tabs

        right_dock = QDockWidget("Controls", self)
        right_dock.setWidget(right_tabs)
        right_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        right_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        # Stop either dock being dragged narrow enough to elide its button text.
        # fit_tabs() has already pinned the tab widgets' own minimums, so these
        # floors are the binding constraint rather than the tab labels.
        fit_tabs(right_tabs)
        left_dock.setMinimumWidth(300)
        right_dock.setMinimumWidth(320)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

        # ---- Status bar ----
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

        # ---- Wire control panel signals to console ----
        self.control_panel.sig_log_message.connect(self.console_panel.append_message)
        self.inference_panel.sig_log_message.connect(self.console_panel.append_message)
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
        self.checkpoint_panel.refresh(settings.get("checkpoint_dir", "checkpoints"))
        self._status.showMessage("Training started…")

    def closeEvent(self, event):
        self.settings_panel.save_settings()
        self.tensorboard_panel.cleanup()
        super().closeEvent(event)
