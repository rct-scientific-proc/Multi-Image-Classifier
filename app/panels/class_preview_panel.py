"""
Dataset preview — browse the H5 file's snippets one class at a time.

The complement of the augmentation preview: that tab shows what the pipeline
does to images, this one shows what is actually *in* the dataset. Pick a class
(and optionally a split), get a page of its snippets as a grid, and page
through the rest — the quickest way to spot mislabelled chips, duplicate
scenes, or a class whose examples look nothing like you expected.

Images are shown as stored (just /255 for display) — no normalisation, no
augmentation.
"""

from __future__ import annotations

import math

# PyQt5 before torch-adjacent imports, consistent with the rest of app/.
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.panels.common import scrollable

_PER_PAGE = 24
_COLS = 6
_CELL_PX = 96          # thumbnail size, nearest-neighbour upscaled
_GAP_PX = 6

# Combo entries mapped to H5 split values (None = no filter).
_SPLITS = (("All splits", None), ("Train", 0), ("Validate", 1), ("Test", 2))
_ALL_CLASSES = "(all classes)"


class _PageWorker(QThread):
    """Reads one page of snippets off the GUI thread.

    A page is at most ~100 chip reads, but on a cold disk or a network share
    that is still long enough to freeze the window if done inline.
    """

    # thumbs (N,H,W,C) uint8, class names, total matches, page, page count
    sig_ready = pyqtSignal(object, list, int, int, int)
    sig_error = pyqtSignal(str)

    def __init__(self, h5_path: str, class_index: int | None,
                 split: int | None, page: int, per_page: int, parent=None):
        super().__init__(parent)
        self._h5 = h5_path
        self._class = class_index
        self._split = split
        self._page = page
        self._per_page = per_page

    def run(self) -> None:
        try:
            import h5py
            import numpy as np

            with h5py.File(self._h5, "r") as f:
                classes = list(f["classes"].asstr()[:])
                labels = f["labels"][:]

                mask = np.ones(len(labels), dtype=bool)
                if self._split is not None:
                    mask &= f["split"][:] == self._split
                if self._class is not None:
                    mask &= labels == self._class
                matches = np.where(mask)[0]

                total = len(matches)
                pages = max(1, math.ceil(total / self._per_page))
                page = min(max(0, self._page), pages - 1)
                sel = matches[page * self._per_page:(page + 1) * self._per_page]

                if len(sel):
                    # np.where output is ascending, which is exactly what
                    # h5py's fancy selection requires.
                    thumbs = f["images"][sel.tolist()]
                else:
                    thumbs = np.empty((0, 1, 1, 1), dtype=np.uint8)

            if thumbs.ndim == 3:                       # (N,H,W) — grayscale
                thumbs = thumbs[..., None]
            if thumbs.dtype != np.uint8:
                # Float files hold [0,1] by this project's convention.
                thumbs = np.clip(thumbs * 255.0, 0, 255).astype(np.uint8)

            self.sig_ready.emit(thumbs, classes, total, page, pages)
        except Exception as exc:
            self.sig_error.emit(f"{type(exc).__name__}: {exc}")


class ClassPreviewPanel(QWidget):
    """A pageable grid of the snippets belonging to one class."""

    sig_log_message = pyqtSignal(str)

    def __init__(self, settings_panel, parent=None):
        super().__init__(parent)
        self._settings = settings_panel
        self._worker: _PageWorker | None = None
        self._page = 0
        self._pages = 1
        # Combo repopulation must not fire the change handler and reset paging.
        self._loading_classes = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # ── filter row ────────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Class:"))
        self._class_combo = QComboBox()
        self._class_combo.addItem(_ALL_CLASSES)
        self._class_combo.setMinimumWidth(160)
        self._class_combo.setToolTip(
            "Class names are read from the H5 file on the first refresh.")
        self._class_combo.currentIndexChanged.connect(self._on_filter_changed)
        row.addWidget(self._class_combo)

        row.addWidget(QLabel("Split:"))
        self._split_combo = QComboBox()
        for label, value in _SPLITS:
            self._split_combo.addItem(label, value)
        self._split_combo.currentIndexChanged.connect(self._on_filter_changed)
        row.addWidget(self._split_combo)

        row.addWidget(QLabel("Per page:"))
        self._per_page = QSpinBox()
        self._per_page.setRange(6, 96)
        self._per_page.setSingleStep(6)
        self._per_page.setValue(_PER_PAGE)
        self._per_page.valueChanged.connect(self._on_filter_changed)
        row.addWidget(self._per_page)

        self._btn_refresh = QPushButton("↻  Refresh")
        self._btn_refresh.clicked.connect(self.refresh)
        row.addWidget(self._btn_refresh)
        row.addStretch()
        layout.addLayout(row)

        # ── pager row ─────────────────────────────────────────────────────
        pager = QHBoxLayout()
        self._btn_prev = QPushButton("◀  Prev")
        self._btn_prev.clicked.connect(lambda: self._go(self._page - 1))
        self._btn_next = QPushButton("Next  ▶")
        self._btn_next.clicked.connect(lambda: self._go(self._page + 1))
        self._btn_prev.setEnabled(False)
        self._btn_next.setEnabled(False)
        self._page_label = QLabel("—")
        self._page_label.setAlignment(Qt.AlignCenter)
        pager.addWidget(self._btn_prev)
        pager.addWidget(self._page_label, 1)
        pager.addWidget(self._btn_next)
        layout.addLayout(pager)

        self._status = QLabel(
            "Set an H5 file in Settings ▸ Data, then refresh.\n"
            "Images are shown as stored — no augmentation, no normalisation."
        )
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # The grid grows downward with page size; scrolling beats clipping.
        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        layout.addWidget(scrollable(self._canvas), 1)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Reload the current page (and the class list) from the H5 file."""
        self._load_page(self._page)

    def _on_filter_changed(self, *_a) -> None:
        if self._loading_classes:
            return
        # New filter, new result set — page numbers from the old one are
        # meaningless.
        self._load_page(0)

    def _go(self, page: int) -> None:
        self._load_page(min(max(0, page), self._pages - 1))

    def _load_page(self, page: int) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        h5 = self._settings.get_settings().get("h5_path", "").strip()
        if not h5:
            self._status.setText("No H5 file set — Settings ▸ Data ▸ H5 file.")
            return

        idx = self._class_combo.currentIndex()
        class_index = None if idx <= 0 else idx - 1     # 0 is "(all classes)"

        self._set_busy(True)
        self._status.setText("Loading…")
        self._worker = _PageWorker(
            h5, class_index, self._split_combo.currentData(),
            page, self._per_page.value())
        self._worker.sig_ready.connect(self._on_ready)
        self._worker.sig_error.connect(self._on_error)
        self._worker.start()

    def _set_busy(self, busy: bool) -> None:
        self._btn_refresh.setEnabled(not busy)
        self._btn_prev.setEnabled(not busy and self._page > 0)
        self._btn_next.setEnabled(not busy and self._page < self._pages - 1)

    # ------------------------------------------------------------------
    def _on_ready(self, thumbs, classes: list, total: int,
                  page: int, pages: int) -> None:
        self._page, self._pages = page, pages
        self._sync_class_combo(classes)

        label = (self._class_combo.currentText()
                 if self._class_combo.currentIndex() > 0 else "all classes")
        if total == 0:
            self._canvas.clear()
            self._page_label.setText("—")
            self._status.setText(
                f"No samples for '{label}' in "
                f"{self._split_combo.currentText().lower()}.")
        else:
            pix = self._compose_grid(thumbs)
            if pix is not None:
                self._canvas.setPixmap(pix)
            first = page * self._per_page.value() + 1
            self._page_label.setText(f"Page {page + 1} / {pages}")
            self._status.setText(
                f"{label}: {total} sample(s) in "
                f"{self._split_combo.currentText().lower()} — showing "
                f"{first}–{first + len(thumbs) - 1}.")
        self._set_busy(False)

    def _on_error(self, msg: str) -> None:
        self._set_busy(False)
        self._status.setText(f"Preview failed — {msg}")
        self.sig_log_message.emit(f"[PREVIEW ERROR] {msg}")

    def _sync_class_combo(self, classes: list) -> None:
        """Fill the combo from the file, preserving the current selection.

        Classes come back with every page load, so this also tracks a changed
        H5 path without any extra wiring.
        """
        current = [self._class_combo.itemText(i)
                   for i in range(1, self._class_combo.count())]
        if current == list(classes):
            return
        selected = self._class_combo.currentText()
        self._loading_classes = True
        try:
            self._class_combo.clear()
            self._class_combo.addItem(_ALL_CLASSES)
            self._class_combo.addItems(list(classes))
            idx = self._class_combo.findText(selected)
            self._class_combo.setCurrentIndex(max(0, idx))
        finally:
            self._loading_classes = False

    # ------------------------------------------------------------------
    @staticmethod
    def _compose_grid(thumbs) -> QPixmap | None:
        """Tile (N,H,W,C) uint8 into a _COLS-wide grid pixmap."""
        import numpy as np

        n = len(thumbs)
        if n == 0:
            return None
        h, w, c = thumbs.shape[1:]

        cols = min(_COLS, n)
        rows = math.ceil(n / cols)
        cell = _CELL_PX
        sheet_w = cols * cell + (cols - 1) * _GAP_PX
        sheet_h = rows * cell + (rows - 1) * _GAP_PX
        # Mid-grey gutters read as separators against both themes.
        sheet = np.full((sheet_h, sheet_w, 3), 128, dtype=np.uint8)

        # Nearest-neighbour only — blurring would hide exactly the pixel-level
        # detail (compression blocks, sensor noise) worth inspecting. Chips
        # larger than a cell are cropped rather than shrunk for the same reason.
        scale = max(1, cell // max(h, w))
        for i, img in enumerate(thumbs):
            if c == 1:
                img = np.repeat(img, 3, axis=2)
            img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
            ih, iw = img.shape[:2]
            r, col = divmod(i, cols)
            y0 = r * (cell + _GAP_PX) + max(0, (cell - ih) // 2)
            x0 = col * (cell + _GAP_PX) + max(0, (cell - iw) // 2)
            sheet[y0:y0 + min(ih, cell), x0:x0 + min(iw, cell)] = \
                img[:cell, :cell]

        sheet = np.ascontiguousarray(sheet)
        qimg = QImage(sheet.data, sheet_w, sheet_h, 3 * sheet_w,
                      QImage.Format_RGB888)
        # copy(): QImage does not own the numpy buffer, which is freed on return.
        return QPixmap.fromImage(qimg.copy())
