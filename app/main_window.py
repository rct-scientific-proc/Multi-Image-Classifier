"""
Main window — QMainWindow with dockable panels.

Layout:
    Left dock   — Settings panel   (scrollable)
    Center      — Console panel    (expands)
    Right dock  — Tabbed: Train / Inference / Checkpoints / TensorBoard
"""

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAction, QActionGroup, QApplication, QMainWindow, QDockWidget,
    QFileDialog, QMessageBox, QStatusBar, QTabWidget,
)

from src.model import (
    PRETRAINED_BACKBONES, download_pretrained_weights, find_local_weights,
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


class _WeightsDownloadWorker(QThread):
    """Downloads ImageNet weight files without freezing the window.

    Even the smallest backbone is ~10 MB; the largest ~100 MB. Inline, that is
    a frozen GUI for however long the connection takes.
    """

    sig_log  = pyqtSignal(str)
    sig_done = pyqtSignal(int, int, str)     # succeeded, attempted, folder

    def __init__(self, backbones: list[str], folder: str, parent=None):
        super().__init__(parent)
        self._backbones = list(backbones)
        self._folder    = folder

    def run(self) -> None:
        ok = 0
        for name in self._backbones:
            try:
                path = download_pretrained_weights(name, self._folder)
                ok += 1
                size_mb = path.stat().st_size / (1024 * 1024)
                self.sig_log.emit(f"  {name}: {path.name} ({size_mb:.1f} MB)")
            except Exception as exc:         # one failure must not stop the rest
                self.sig_log.emit(f"[ERROR] {name}: {exc}")
        self.sig_done.emit(ok, len(self._backbones), self._folder)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier")
        # The tabbed right dock only needs ~220px now, so height is driven by the
        # settings form (~1025px natural — it scrolls at any realistic size) and
        # by how much log you want visible. 820 is a comfortable default that
        # still fits a 1080p screen once the taskbar is accounted for.
        self.resize(1200, 820)

        self._weights_worker: _WeightsDownloadWorker | None = None

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

        # ---- Models ----
        # The offline workflow: download weight files here on a connected
        # machine, copy the folder to the offline machine (hard drive, share,
        # whatever), then "Set Weights Folder…" there. Training reads the
        # files from that folder instead of downloading.
        models_menu = bar.addMenu("&Models")

        self._act_dl_current = QAction("&Download Weights (Current Backbone)…", self)
        self._act_dl_current.setStatusTip(
            "Download ImageNet weights for the backbone selected in Settings")
        self._act_dl_current.triggered.connect(
            lambda: self._on_download_weights(all_backbones=False))
        models_menu.addAction(self._act_dl_current)

        self._act_dl_all = QAction("Download Weights (&All Backbones)…", self)
        self._act_dl_all.setStatusTip(
            "Download ImageNet weights for every supported backbone (~290 MB)")
        self._act_dl_all.triggered.connect(
            lambda: self._on_download_weights(all_backbones=True))
        models_menu.addAction(self._act_dl_all)

        models_menu.addSeparator()

        act_set_weights = QAction("&Set Weights Folder…", self)
        act_set_weights.setStatusTip(
            "Point at a copied weights folder so training loads it offline")
        act_set_weights.triggered.connect(self._on_set_weights_folder)
        models_menu.addAction(act_set_weights)

        act_clear_weights = QAction("&Clear Weights Folder", self)
        act_clear_weights.setStatusTip(
            "Forget the folder — pretrained weights are downloaded again")
        act_clear_weights.triggered.connect(self._on_clear_weights_folder)
        models_menu.addAction(act_clear_weights)

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

    # ------------------------------------------------------------------
    # Models menu — pretrained weights for offline machines

    def _on_download_weights(self, all_backbones: bool) -> None:
        if self._weights_worker is not None and self._weights_worker.isRunning():
            QMessageBox.information(self, "Download weights",
                                    "A download is already running — see the log.")
            return

        if all_backbones:
            backbones = list(PRETRAINED_BACKBONES)
        else:
            backbone = self.settings_panel.get_settings().get("backbone", "")
            if backbone not in PRETRAINED_BACKBONES:
                QMessageBox.information(
                    self, "Download weights",
                    f"'{backbone}' has no pretrained weights to download.\n\n"
                    f"Pick a torchvision backbone in Settings ▸ Model, or use "
                    f"'Download Weights (All Backbones)…'.")
                return
            backbones = [backbone]

        folder = QFileDialog.getExistingDirectory(
            self, "Folder to download weights into — copy it to the offline machine",
            self.settings_panel.weights_dir or "pretrained_weights")
        if not folder:
            return

        self.console_panel.append_message(
            f"Downloading ImageNet weights for {len(backbones)} backbone(s) "
            f"into: {folder}")
        self._act_dl_current.setEnabled(False)
        self._act_dl_all.setEnabled(False)
        self._weights_worker = _WeightsDownloadWorker(backbones, folder)
        self._weights_worker.sig_log.connect(self.console_panel.append_message)
        self._weights_worker.sig_done.connect(self._on_weights_downloaded)
        self._weights_worker.start()

    def _on_weights_downloaded(self, ok: int, attempted: int, folder: str) -> None:
        self._act_dl_current.setEnabled(True)
        self._act_dl_all.setEnabled(True)
        self.console_panel.append_message(
            f"Weight download finished: {ok} of {attempted} succeeded.")
        if ok:
            # Using the folder right away is what you want on a connected
            # machine too, and it means the copied settings.json already points
            # at the right place on the offline one.
            self.settings_panel.weights_dir = folder
            self.console_panel.append_message(
                f"[INFO] Weights folder set to: {folder}")
        QMessageBox.information(
            self, "Download weights",
            f"Downloaded {ok} of {attempted} weight file(s) to:\n{folder}\n\n"
            f"Copy this folder to the offline machine and select it there via "
            f"Models ▸ Set Weights Folder…")

    def _on_set_weights_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select the copied weights folder",
            self.settings_panel.weights_dir or "pretrained_weights")
        if not folder:
            return
        self.settings_panel.weights_dir = folder
        # Say what the folder actually provides — the point of this action on
        # the offline machine is confirming the copy worked.
        found = [n for n in PRETRAINED_BACKBONES
                 if find_local_weights(n, folder) is not None]
        if found:
            self.console_panel.append_message(
                f"[INFO] Weights folder set to: {folder}\n"
                f"       Found weights for: {', '.join(found)}")
        else:
            self.console_panel.append_message(
                f"[WARN] Weights folder set to: {folder} — but it contains no "
                f"known weight files. Training with pretrained weights will "
                f"fall back to downloading.")
            QMessageBox.warning(
                self, "Set weights folder",
                f"No known weight files found in:\n{folder}\n\n"
                f"Expected torchvision files like resnet18-f37072fd.pth — "
                f"download them via Models ▸ Download Weights… on a connected "
                f"machine and copy the folder here.")

    def _on_clear_weights_folder(self) -> None:
        self.settings_panel.weights_dir = ""
        self.console_panel.append_message(
            "[INFO] Weights folder cleared — pretrained weights will be "
            "downloaded when needed.")

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
        # Qt aborts if a QThread is destroyed while running. The largest weight
        # file is ~100 MB, so waiting out an in-flight download beats crashing
        # on exit.
        if self._weights_worker is not None and self._weights_worker.isRunning():
            self._weights_worker.wait()
        super().closeEvent(event)
