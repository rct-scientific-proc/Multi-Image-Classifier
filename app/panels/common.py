"""
Shared layout helpers for the panel widgets.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollArea, QTabWidget, QWidget


def fit_tabs(tabs: QTabWidget, min_width: int = 240) -> None:
    """Let a tab bar elide its labels instead of sprouting scroll arrows.

    The explicit minimum width matters as much as the elide mode: left alone,
    QTabWidget reports a minimumSizeHint wide enough to spell out every tab
    label, and a QDockWidget cannot go narrower than its content's minimum — so
    the tab bar silently dictates how wide the whole dock must be, which grows
    with the user's UI font. Pinning it lets the labels elide under pressure and
    keeps the dock width a layout decision rather than a font side effect.
    """
    bar = tabs.tabBar()
    bar.setExpanding(False)
    bar.setElideMode(Qt.ElideRight)
    tabs.setUsesScrollButtons(False)
    tabs.setMinimumWidth(min_width)


def scrollable(widget: QWidget) -> QScrollArea:
    """Wrap *widget* in a vertical-only scroll area.

    Applied per tab page so any one page can grow past the dock height without
    clipping — the headroom that makes it safe to keep adding options.
    """
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area
