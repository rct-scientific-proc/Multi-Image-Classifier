"""
Augmentation preview — see what the current settings actually do to images.

Misconfigured augmentation is close to undiagnosable from a loss curve and
obvious at a glance, which is the whole reason this panel exists. It draws two
rows from the train split: the images as stored, and the same images after the
configured augmentation.

Shown deliberately WITHOUT normalisation. Normalised pixels are centred near
zero and clip to nonsense when displayed, so a preview of them tells you
nothing; augmentation is what you are inspecting here, and it operates in
[0, 1] before normalisation runs.
"""

from __future__ import annotations

# PyQt5 before torch-adjacent imports, consistent with the rest of app/.
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_N_SAMPLES = 8
_CELL_PX = 96          # each thumbnail is upscaled to this, nearest-neighbour
_GAP_PX = 6


class _PreviewWorker(QThread):
    """Loads a batch and augments it off the GUI thread."""

    sig_ready = pyqtSignal(object, object, list)   # original, augmented, ops
    sig_error = pyqtSignal(str)

    def __init__(self, settings: dict, n: int, parent=None):
        super().__init__(parent)
        self._s = settings
        self._n = n

    def run(self) -> None:
        try:
            import numpy as np
            import torch
            from src.augment import GpuAugment
            from src.dataset import H5Dataset, SPLIT_TRAIN

            s = self._s
            ds = H5Dataset(s["h5_path"], split=SPLIT_TRAIN)
            if len(ds) == 0:
                raise ValueError("Train split contains 0 samples.")

            n = min(self._n, len(ds))
            # Random picks, so repeated refreshes show different images rather
            # than the same first N every time.
            idx = torch.randperm(len(ds))[:n].tolist()
            batch = torch.stack([
                (img.float() / 255.0 if img.dtype == torch.uint8 else img.float())
                for img, _lbl, _gt in (ds[i] for i in idx)
            ])

            aug = GpuAugment(s, in_channels=batch.shape[1])
            aug.train()
            out = aug(batch.clone())

            self.sig_ready.emit(
                (batch.clamp(0, 1) * 255).to(torch.uint8).numpy(),
                (out.clamp(0, 1) * 255).to(torch.uint8).numpy(),
                aug.active_ops if aug.enabled else [],
            )
        except Exception as exc:
            self.sig_error.emit(f"{type(exc).__name__}: {exc}")


class PreviewPanel(QWidget):
    """Two rows of thumbnails: stored images above, augmented below."""

    sig_log_message = pyqtSignal(str)

    def __init__(self, settings_panel, parent=None):
        super().__init__(parent)
        self._settings = settings_panel
        self._worker: _PreviewWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        row = QHBoxLayout()
        self._btn = QPushButton("↻  Refresh preview")
        self._btn.clicked.connect(self.refresh)
        row.addWidget(self._btn)

        row.addWidget(QLabel("Samples:"))
        self._count = QSpinBox()
        self._count.setRange(1, 16)
        self._count.setValue(_N_SAMPLES)
        row.addWidget(self._count)
        row.addStretch()
        layout.addLayout(row)

        self._status = QLabel(
            "Set an H5 file in Settings ▸ Data, then refresh.\n"
            "Top row: images as stored.  Bottom row: after augmentation."
        )
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._canvas, 1)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        s = self._settings.get_settings()
        h5 = s.get("h5_path", "").strip()
        if not h5:
            self._status.setText("No H5 file set — Settings ▸ Data ▸ H5 file.")
            return

        self._btn.setEnabled(False)
        self._status.setText("Loading…")
        self._worker = _PreviewWorker(s, self._count.value())
        self._worker.sig_ready.connect(self._on_ready)
        self._worker.sig_error.connect(self._on_error)
        self._worker.start()

    # ------------------------------------------------------------------
    def _on_ready(self, original, augmented, ops: list) -> None:
        self._btn.setEnabled(True)
        pix = self._compose(original, augmented)
        if pix is None:
            self._status.setText("Nothing to show.")
            return
        self._canvas.setPixmap(pix)
        self._status.setText(
            (f"Active ops: {', '.join(ops)}" if ops
             else "Augmentation is disabled — both rows are identical.")
            + "   (shown before normalisation)"
        )

    def _on_error(self, msg: str) -> None:
        self._btn.setEnabled(True)
        self._status.setText(f"Preview failed — {msg}")
        self.sig_log_message.emit(f"[PREVIEW ERROR] {msg}")

    @staticmethod
    def _compose(original, augmented) -> QPixmap | None:
        """Tile (N,C,H,W) uint8 arrays into one two-row RGB QPixmap."""
        import numpy as np

        n, c, h, w = original.shape
        if n == 0:
            return None

        rows, cell = 2, _CELL_PX
        sheet_w = n * cell + (n - 1) * _GAP_PX
        sheet_h = rows * cell + _GAP_PX
        # Mid-grey gutters read as deliberate separators against both themes.
        sheet = np.full((sheet_h, sheet_w, 3), 128, dtype=np.uint8)

        scale = max(1, cell // max(h, w))
        for r, arr in enumerate((original, augmented)):
            for i in range(n):
                img = arr[i].transpose(1, 2, 0)                  # (H,W,C)
                if c == 1:
                    img = np.repeat(img, 3, axis=2)
                # Nearest-neighbour upscale — bilinear would blur exactly the
                # pixel-level artefacts (erase edges, crop borders) worth seeing.
                img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
                ih, iw = img.shape[:2]
                y0 = r * (cell + _GAP_PX) + max(0, (cell - ih) // 2)
                x0 = i * (cell + _GAP_PX) + max(0, (cell - iw) // 2)
                sheet[y0:y0 + min(ih, cell), x0:x0 + min(iw, cell)] = \
                    img[:cell, :cell]

        sheet = np.ascontiguousarray(sheet)
        qimg = QImage(sheet.data, sheet_w, sheet_h, 3 * sheet_w,
                      QImage.Format_RGB888)
        # copy(): QImage does not own the numpy buffer, which is freed on return.
        return QPixmap.fromImage(qimg.copy())
