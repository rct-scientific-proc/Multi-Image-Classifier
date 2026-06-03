"""
Checkpoint panel — list .pt checkpoints, resume from selected, export best.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
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


def _json_default(obj):
    """Fallback for json.dump — handles types _to_jsonable misses."""
    if isinstance(obj, torch.Tensor):
        return _to_jsonable(obj.detach().cpu().tolist())
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _to_jsonable(obj):
    """Recursively convert numpy / NaN / tensor values to JSON-safe primitives."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return _to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


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
        self._btn_export_metrics = QPushButton("Export metrics…")
        self._btn_refresh = QPushButton("↻")
        self._btn_refresh.setFixedWidth(32)
        self._btn_resume.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._btn_export_metrics.setEnabled(False)
        self._btn_export_metrics.setToolTip(
            "Export the full metrics dict (including per-class thresholds at "
            "target recalls) of the selected checkpoint — or best.pt if none "
            "selected — to a JSON file."
        )
        btn_row.addWidget(self._btn_resume)
        btn_row.addWidget(self._btn_export)
        btn_row.addWidget(self._btn_export_metrics)
        btn_row.addWidget(self._btn_refresh)
        box_lay.addLayout(btn_row)

        self._btn_resume.clicked.connect(self._on_resume)
        self._btn_export.clicked.connect(self._on_export)
        self._btn_export_metrics.clicked.connect(self._on_export_metrics)
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

        files = sorted(ck_path.glob("*epoch_*.pt"), key=lambda p: p.stat().st_mtime)
        best  = ck_path / "best.pt"

        # Determine which epoch_*.pt corresponds to best.pt by reading its epoch field
        best_epoch: int | None = None
        if best.exists():
            try:
                meta = torch.load(best, weights_only=True, map_location="cpu")
                best_epoch = meta.get("epoch")
            except Exception:
                best_epoch = None

        for f in files:
            item = QListWidgetItem(f.name)
            item.setData(Qt.UserRole, str(f))
            # Filenames are epoch_{epoch:03d}_acc..., parse the leading epoch number
            try:
                epoch_in_name = int(f.name.split("_")[1])
            except (IndexError, ValueError):
                epoch_in_name = None
            if best_epoch is not None and epoch_in_name == best_epoch:
                item.setText(f"★ {f.name}")
            self._list.addItem(item)

        if files:
            self._list.scrollToBottom()
            self._info.setText(f"{len(files)} checkpoint(s)  ·  dir: {self._ck_dir}")
        else:
            self._info.setText("No epoch checkpoints found.")

        self._btn_export.setEnabled(best.exists())
        self._btn_export_metrics.setEnabled(best.exists() or bool(files))

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

    def _on_export_metrics(self) -> None:
        # Prefer the selected checkpoint; fall back to best.pt.
        items = self._list.selectedItems()
        if items:
            src = Path(items[0].data(Qt.UserRole))
            default_name = src.stem + "_metrics.json"
        else:
            src = Path(self._ck_dir) / "best.pt"
            default_name = "best_metrics.json"

        if not src.exists():
            QMessageBox.warning(self, "Export metrics", f"{src.name} not found.")
            return

        try:
            ckpt = torch.load(src, weights_only=True, map_location="cpu")
        except Exception as exc:
            QMessageBox.critical(self, "Export metrics", f"Failed to load checkpoint:\n{exc}")
            return

        payload = {
            "source_checkpoint": str(src),
            "epoch":             ckpt.get("epoch"),
            "hyperparams":       ckpt.get("hyperparams", {}),
            "metrics":           ckpt.get("metrics", {}),
        }

        dst, _ = QFileDialog.getSaveFileName(
            self, "Export metrics to JSON", default_name, "JSON (*.json);;All files (*)"
        )
        if not dst:
            return
        try:
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(payload), f, indent=2,
                          default=_json_default, allow_nan=False)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Export metrics", f"Write failed:\n{exc}")
            return
        QMessageBox.information(self, "Export metrics", f"Saved to:\n{dst}")

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
