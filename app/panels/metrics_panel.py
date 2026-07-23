"""
Metrics panel — live per-epoch training curves.

Four linked plots in a 2x2 grid, fed from the same on_epoch_end payload the
console line is built from:

    Loss              train vs validation
    Accuracy          train vs validation
    <target metric>   validation only — whichever metric best.pt tracks
    Learning rate     log-scaled

Deliberately four separate plots rather than fewer with twin axes: two y-scales
on one plot make the alignment of the scales arbitrary and invent a correlation
that is not in the data.

Colour encodes the *entity*, consistently across every plot — blue is always
train, orange is always validation, aqua is optimizer state. So a reader who
learns the mapping on the loss plot keeps it everywhere, and adding or removing
a series never repaints the others.

The palette is the validated categorical set (slots 1-3), stepped separately for
each surface rather than flipped: worst all-pairs CVD ΔE 9.2 light / 9.4 dark
(target ≥8), normal-vision ΔE 24.0 / 20.9 (floor ≥15). Aqua sits at 2.7:1 on the
light surface, under the 3:1 contrast guide — it is the sole series on a titled
plot, and the Log tab carries every value as text, so identity is never left to
colour alone.
"""

from __future__ import annotations

# PyQt5 must be imported before pyqtgraph. pyqtgraph binds to whichever Qt
# binding is already in sys.modules and otherwise picks PyQt6 when both are
# installed, which would tear the app across two bindings at runtime.
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True)
    _IMPORT_ERROR: str | None = None
except ImportError as exc:                                  # pragma: no cover
    pg = None
    _IMPORT_ERROR = str(exc)


# Surfaces match the app theme; series colours are the validated slots 1-3.
_PALETTE = {
    "light": {
        "surface":  "#fafafa",
        "text":     "#0b0b0b",
        "text_dim": "#52514e",
        "train":    "#2a78d6",
        "val":      "#eb6834",
        "optim":    "#1baf7a",
    },
    "dark": {
        "surface":  "#1e1e1e",
        "text":     "#ffffff",
        "text_dim": "#c3c2b7",
        "train":    "#3987e5",
        "val":      "#d95926",
        "optim":    "#199e70",
    },
}

_LINE_WIDTH = 2
_MARKER_SIZE = 8
# Past this many epochs the per-point markers become noise and the line alone
# carries the shape. Below it they matter — a single epoch has no line to draw.
_MARKER_LIMIT = 30


class MetricsPanel(QWidget):
    """Live training charts. Degrades to a message if pyqtgraph is missing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._theme = "light"
        self._epochs: list[int] = []
        self._data: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "target":     [], "lr":       [],
        }
        self._target_name = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        if pg is None:                                      # pragma: no cover
            msg = QLabel(
                "Live charts need pyqtgraph.\n\n"
                f"    pip install pyqtgraph\n\n({_IMPORT_ERROR})"
            )
            msg.setAlignment(Qt.AlignCenter)
            msg.setWordWrap(True)
            layout.addWidget(msg)
            self._glw = None
            return

        self._readout = QLabel("Hover a chart to read values at an epoch.")
        self._readout.setWordWrap(True)
        layout.addWidget(self._readout)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        self._p_loss = self._glw.addPlot(row=0, col=0)
        self._p_acc  = self._glw.addPlot(row=0, col=1)
        self._p_tgt  = self._glw.addPlot(row=1, col=0)
        self._p_lr   = self._glw.addPlot(row=1, col=1)
        self._plots  = [self._p_loss, self._p_acc, self._p_tgt, self._p_lr]

        # Two series per plot get a legend; the single-series plots are named by
        # their title instead, so no legend box competes with the data.
        self._legends = [
            self._p_loss.addLegend(offset=(-10, 10)),
            self._p_acc.addLegend(offset=(-10, 10)),
        ]

        self._p_lr.setLogMode(y=True)
        for p in self._plots[1:]:
            p.setXLink(self._p_loss)

        self._curves = {
            "train_loss": self._p_loss.plot(name="Train"),
            "val_loss":   self._p_loss.plot(name="Validation"),
            "train_acc":  self._p_acc.plot(name="Train"),
            "val_acc":    self._p_acc.plot(name="Validation"),
            "target":     self._p_tgt.plot(),
            "lr":         self._p_lr.plot(),
        }
        self._curve_role = {
            "train_loss": "train", "val_loss": "val",
            "train_acc":  "train", "val_acc":  "val",
            "target":     "val",   "lr":       "optim",
        }

        for p, left in ((self._p_loss, "Loss"), (self._p_acc, "Accuracy"),
                        (self._p_tgt, "Score"), (self._p_lr, "Learning rate")):
            p.setLabel("bottom", "Epoch")
            p.setLabel("left", left)
            p.showGrid(x=True, y=True, alpha=0.15)

        # One crosshair per plot, all driven from the shared scene so hovering
        # any plot reads the same epoch on all four.
        self._crosshairs = []
        for p in self._plots:
            line = pg.InfiniteLine(angle=90, movable=False)
            line.setVisible(False)
            p.addItem(line, ignoreBounds=True)
            self._crosshairs.append(line)
        self._glw.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self.apply_theme("light")
        self._set_titles()

    # ── Public API ────────────────────────────────────────────────────────────

    def clear(self, target_metric: str = "") -> None:
        """Drop all series. Called when a training run starts."""
        if self._glw is None:                               # pragma: no cover
            return
        self._epochs.clear()
        for v in self._data.values():
            v.clear()
        for c in self._curves.values():
            c.setData([], [])
        for line in self._crosshairs:
            line.setVisible(False)
        self._target_name = target_metric
        self._set_titles()
        self._readout.setText("Hover a chart to read values at an epoch.")

    def add_epoch(self, info: dict) -> None:
        """Append one epoch from the Trainer's on_epoch_end payload."""
        if self._glw is None:                               # pragma: no cover
            return

        name = info.get("target_metric", "")
        if name and name != self._target_name:
            self._target_name = name
            self._set_titles()

        self._epochs.append(int(info.get("epoch", len(self._epochs))))
        self._data["train_loss"].append(float(info.get("train_loss", float("nan"))))
        self._data["val_loss"].append(float(info.get("val_loss", float("nan"))))
        self._data["train_acc"].append(float(info.get("train_accuracy", float("nan"))))
        self._data["val_acc"].append(float(info.get("val_accuracy", float("nan"))))
        self._data["target"].append(float(info.get("target_val", float("nan"))))
        self._data["lr"].append(float(info.get("lr", float("nan"))))
        self._redraw()

    def apply_theme(self, name: str) -> None:
        """Restyle for "light" or "dark". Charts follow the app theme."""
        if self._glw is None:                               # pragma: no cover
            return
        self._theme = name if name in _PALETTE else "light"
        c = _PALETTE[self._theme]

        self._glw.setBackground(c["surface"])
        self._readout.setStyleSheet(f"color: {c['text_dim']};")

        axis_pen = pg.mkPen(c["text_dim"])
        for p in self._plots:
            for side in ("left", "bottom"):
                ax = p.getAxis(side)
                ax.setPen(axis_pen)
                ax.setTextPen(axis_pen)
        for legend in self._legends:
            legend.setLabelTextColor(c["text"])
        for line in self._crosshairs:
            line.setPen(pg.mkPen(c["text_dim"], width=1, style=Qt.DashLine))

        self._redraw()
        self._set_titles()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _set_titles(self) -> None:
        c = _PALETTE[self._theme]
        target = self._target_name or "target metric"
        # Single-series plots name their series in the title, since they carry
        # no legend.
        for p, title in (
            (self._p_loss, "Loss"),
            (self._p_acc,  "Accuracy"),
            (self._p_tgt,  f"Validation {target}"),
            (self._p_lr,   "Learning rate"),
        ):
            p.setTitle(title, color=c["text"])
        self._p_tgt.setLabel("left", target)

    def _redraw(self) -> None:
        c = _PALETTE[self._theme]
        symbol = "o" if len(self._epochs) <= _MARKER_LIMIT else None
        for key, curve in self._curves.items():
            colour = c[self._curve_role[key]]
            curve.setData(
                self._epochs, self._data[key],
                pen=pg.mkPen(colour, width=_LINE_WIDTH),
                symbol=symbol,
                symbolSize=_MARKER_SIZE,
                symbolBrush=colour,
                symbolPen=None,
            )

    def _on_mouse_moved(self, pos) -> None:
        if not self._epochs:
            return
        for plot in self._plots:
            if plot.sceneBoundingRect().contains(pos):
                x = plot.vb.mapSceneToView(pos).x()
                break
        else:
            return

        idx = min(range(len(self._epochs)),
                  key=lambda i: abs(self._epochs[i] - x))
        epoch = self._epochs[idx]
        for line in self._crosshairs:
            line.setPos(epoch)
            line.setVisible(True)

        d = self._data
        target = self._target_name or "target"
        self._readout.setText(
            f"epoch {epoch}   "
            f"train_loss {d['train_loss'][idx]:.4f}   "
            f"val_loss {d['val_loss'][idx]:.4f}   "
            f"val_acc {d['val_acc'][idx]:.4f}   "
            f"{target} {d['target'][idx]:.4f}   "
            f"lr {d['lr'][idx]:.2e}"
        )
