"""
Checkpoint panel — list .pt checkpoints, resume from selected, export best.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CheckpointPanel(QWidget):
    """Shows all epoch_*.pt files in the current checkpoint directory.

    Public signals
    --------------
    sig_resume_requested(str)
        Emitted when the user clicks "Resume" with the path to the chosen .pt file.
    """

    sig_resume_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ck_dir: str = ""

        box = QGroupBox("Checkpoints")
        box_lay = QVBoxLayout(box)

        # ── list ──────────────────────────────────────────────────────────
        self._list = QListWidget()
        self._list.setToolTip("Double-click a checkpoint to inspect its metrics")
        self._list.itemDoubleClicked.connect(self._on_inspect)
        box_lay.addWidget(self._list)

        # ── info label ────────────────────────────────────────────────────
        self._info = QLabel("No checkpoints yet.")
        self._info.setAlignment(Qt.AlignCenter)
        self._info.setWordWrap(True)
        box_lay.addWidget(self._info)

        # ── buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._btn_resume = QPushButton("Resume from selected")
        self._btn_export = QPushButton("Export best…")
        self._btn_refresh = QPushButton("↻")
        self._btn_refresh.setFixedWidth(32)
        self._btn_resume.setEnabled(False)
        self._btn_export.setEnabled(False)
        btn_row.addWidget(self._btn_resume)
        btn_row.addWidget(self._btn_export)
        btn_row.addWidget(self._btn_refresh)
        box_lay.addLayout(btn_row)

        self._btn_resume.clicked.connect(self._on_resume)
        self._btn_export.clicked.connect(self._on_export)
        self._btn_refresh.clicked.connect(lambda: self.refresh(self._ck_dir))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)

        self._list.itemSelectionChanged.connect(self._on_selection_changed)

    # ── Public API ────────────────────────────────────────────────────────────

    def refresh(self, checkpoint_dir: str = "") -> None:
        """Reload the checkpoint list from *checkpoint_dir*."""
        if checkpoint_dir:
            self._ck_dir = checkpoint_dir
        self._list.clear()

        ck_path = Path(self._ck_dir)
        if not ck_path.is_dir():
            self._info.setText("Checkpoint directory not found.")
            self._btn_export.setEnabled(False)
            return

        files = sorted(ck_path.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
        best  = ck_path / "best.pt"

        for f in files:
            item = QListWidgetItem(f.name)
            item.setData(Qt.UserRole, str(f))
            if best.exists() and f.stat().st_size == best.stat().st_size:
                item.setText(f"★ {f.name}")
            self._list.addItem(item)

        if files:
            self._list.scrollToBottom()
            self._info.setText(f"{len(files)} checkpoint(s)  ·  dir: {self._ck_dir}")
        else:
            self._info.setText("No epoch checkpoints found.")

        self._btn_export.setEnabled(best.exists())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _on_selection_changed(self) -> None:
        self._btn_resume.setEnabled(bool(self._list.selectedItems()))

    def _on_resume(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        path = items[0].data(Qt.UserRole)
        self.sig_resume_requested.emit(path)

    def _on_export(self) -> None:
        best = Path(self._ck_dir) / "best.pt"
        if not best.exists():
            QMessageBox.warning(self, "Export", "best.pt not found.")
            return
        dst, _ = QFileDialog.getSaveFileName(
            self, "Export best model", "best_model.pt", "PyTorch checkpoint (*.pt)"
        )
        if dst:
            shutil.copy2(best, dst)
            QMessageBox.information(self, "Export", f"Saved to:\n{dst}")

    def _on_inspect(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        try:
            ckpt = torch.load(path, weights_only=True)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        metrics   = ckpt.get("metrics", {})
        epoch     = ckpt.get("epoch", "?")
        hp        = ckpt.get("hyperparams", {})

        lines = [f"<b>File:</b> {Path(path).name}", f"<b>Epoch:</b> {epoch}"]
        if hp.get("backbone"):
            lines.append(f"<b>Backbone:</b> {hp['backbone']}")
        lines.append("<b>Metrics:</b>")
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"  &nbsp;&nbsp;{k} = {v:.4f}")

        QMessageBox.information(self, "Checkpoint info", "<br>".join(lines))
