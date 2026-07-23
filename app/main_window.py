"""
Main window — QMainWindow with dockable panels.

Layout:
    Left dock   — Settings panel   (scrollable)
    Center      — Console panel    (expands)
    Right dock  — Tabbed: Train / Inference / Checkpoints / TensorBoard
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction, QActionGroup, QApplication, QMainWindow, QDockWidget,
    QMessageBox, QStatusBar, QTabWidget,
)

from app.theme                    import THEMES, apply_theme
from app.panels.common            import fit_tabs, scrollable
from app.panels.settings_panel    import SettingsPanel
from app.panels.control_panel     import ControlPanel
from app.panels.inference_panel   import InferencePanel
from app.panels.console_panel     import ConsolePanel
from app.panels.metrics_panel     import MetricsPanel
from app.panels.preview_panel     import PreviewPanel
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

        # ---- Settings first: the centre and right panels are constructed with
        #      a reference to it, so it has to exist before either. ----
        self.settings_panel = SettingsPanel()

        # ---- Central widget: log + live charts + augmentation preview ----
        # The largest area was showing the least dense content. The log stays,
        # but the curves now live here too instead of only in TensorBoard.
        self.console_panel = ConsolePanel()
        self.metrics_panel = MetricsPanel()
        self.preview_panel = PreviewPanel(self.settings_panel)
        centre = QTabWidget()
        centre.addTab(self.console_panel, "Log")
        centre.addTab(self.metrics_panel, "Metrics")
        centre.addTab(self.preview_panel, "Augment preview")
        fit_tabs(centre, min_width=160)
        self.centre_tabs = centre
        self.setCentralWidget(centre)

        # ---- Left dock: settings ----
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

        self._left_dock  = left_dock
        self._right_dock = right_dock

        # ---- Menu bar ----
        self._build_menus()

        # ---- Status bar ----
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

        # ---- Wire control panel signals to console ----
        self.control_panel.sig_log_message.connect(self.console_panel.append_message)
        self.inference_panel.sig_log_message.connect(self.console_panel.append_message)
        self.preview_panel.sig_log_message.connect(self.console_panel.append_message)
        self.control_panel.sig_epoch_complete.connect(self._on_epoch_complete)
        self.control_panel.sig_epoch_complete.connect(self.metrics_panel.add_epoch)
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

        # ---- Apply the persisted theme (settings are loaded by now) ----
        self._set_theme(self.settings_panel.theme)

    # ------------------------------------------------------------------
    def _build_menus(self) -> None:
        """File / View / Help menus.

        The dock toggles matter beyond tidiness: the docks were previously not
        closable, so there was no way to reclaim their width for the log.
        """
        bar = self.menuBar()

        # ---- File ----
        file_menu = bar.addMenu("&File")
        act_save = QAction("&Save Settings", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(lambda: self.settings_panel.save_settings())
        file_menu.addAction(act_save)

        act_load = QAction("&Load Settings…", self)
        act_load.setShortcut("Ctrl+O")
        act_load.triggered.connect(self.settings_panel._browse_and_load)
        file_menu.addAction(act_load)

        file_menu.addSeparator()
        act_quit = QAction("E&xit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # ---- View ----
        view_menu = bar.addMenu("&View")
        # toggleViewAction() gives a ready-made checkable action bound to the
        # dock's visibility, so it stays in sync if the dock is closed directly.
        for dock in (self._left_dock, self._right_dock):
            dock.setFeatures(dock.features() | QDockWidget.DockWidgetClosable)
            view_menu.addAction(dock.toggleViewAction())

        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("&Theme")
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        self._theme_actions: dict[str, QAction] = {}
        for name in THEMES:
            act = QAction(name.capitalize(), self, checkable=True)
            act.triggered.connect(lambda _checked, n=name: self._set_theme(n))
            self._theme_group.addAction(act)
            theme_menu.addAction(act)
            self._theme_actions[name] = act

        # ---- Help ----
        help_menu = bar.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    def _set_theme(self, name: str) -> None:
        """Apply *name* and remember it for the next launch."""
        applied = apply_theme(QApplication.instance(), name)
        self.settings_panel.theme = applied
        # Charts paint themselves rather than through QSS, so they need telling.
        self.metrics_panel.apply_theme(applied)
        act = self._theme_actions.get(applied)
        if act is not None and not act.isChecked():
            act.setChecked(True)

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About Image Classifier",
            "<b>Image Classifier</b><br>"
            "Training GUI for HDF5 image datasets.<br><br>"
            "See docs/h5_format.md for the expected dataset layout.",
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
        # Charts belong to one run — carrying the previous run's curves into a
        # new one would silently splice two different experiments together.
        self.metrics_panel.clear(settings.get("target_metric", ""))
        self._status.showMessage("Training started…")

    def closeEvent(self, event):
        self.settings_panel.save_settings()
        self.tensorboard_panel.cleanup()
        super().closeEvent(event)
