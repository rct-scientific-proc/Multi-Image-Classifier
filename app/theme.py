"""
Light / dark theming.

Two layers, deliberately:

  * QPalette carries every colour. Fusion derives most of its own painting from
    the palette, so getting this right is what makes dark mode look native
    rather than like a stylesheet bolted on top. The disabled-state entries
    matter more than usual here — this app leaves a lot of buttons disabled
    (Pause/Stop before a run, Export before a checkpoint exists), and washed-out
    disabled text is the usual giveaway of a half-done dark theme.

  * QSS carries geometry — padding, radii, spacing — plus the few borders the
    palette cannot express. It is written against tokens rather than literal
    colours so both themes share one sheet.

Usage:
    from app.theme import apply_theme
    apply_theme(QApplication.instance(), "dark")
"""

from __future__ import annotations

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

THEMES = ("light", "dark")
DEFAULT_THEME = "light"

# Accent is shared by both themes so selection colour stays recognisable.
_ACCENT = "#2f7fd1"

_TOKENS = {
    "light": {
        "accent":     _ACCENT,
        "border":     "#c4c4c4",
        "border_hi":  "#9a9a9a",
        "panel":      "#f0f0f0",
        "surface":    "#fafafa",
        "hover":      "#e6effa",
        "text_dim":   "#8a8a8a",
    },
    "dark": {
        "accent":     _ACCENT,
        "border":     "#3c3c3c",
        "border_hi":  "#565656",
        "panel":      "#252526",
        "surface":    "#1e1e1e",
        "hover":      "#2a3f55",
        "text_dim":   "#6b6b6b",
    },
}

_QSS = """
QWidget {{ font-size: 9pt; }}

QMainWindow::separator {{ background: {border}; width: 3px; height: 3px; }}

QDockWidget::title {{
    background: {panel};
    padding: 5px 8px;
    border-bottom: 1px solid {border};
}}

QTabWidget::pane {{ border: 1px solid {border}; border-radius: 3px; top: -1px; }}
QTabBar::tab {{
    padding: 6px 12px;
    border: 1px solid {border};
    border-bottom: none;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
    background: {panel};
}}
QTabBar::tab:selected {{ background: {surface}; border-bottom: 2px solid {accent}; }}
QTabBar::tab:hover:!selected {{ background: {hover}; }}

QPushButton {{
    padding: 5px 12px;
    min-height: 18px;
    border: 1px solid {border};
    border-radius: 3px;
    background: {panel};
}}
QPushButton:hover:!disabled {{ border-color: {accent}; background: {hover}; }}
QPushButton:pressed {{ background: {border}; }}
QPushButton:disabled {{ color: {text_dim}; border-color: {border}; }}
/* Square glyph buttons are fixed-width; the default padding would clip them. */
QPushButton#iconButton {{ padding: 2px; min-height: 18px; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    padding: 4px 6px;
    border: 1px solid {border};
    border-radius: 3px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {accent};
}}
QLineEdit:disabled, QComboBox:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {text_dim}; }}

QProgressBar {{
    border: 1px solid {border};
    border-radius: 3px;
    text-align: center;
    min-height: 18px;
}}
QProgressBar::chunk {{ background: {accent}; border-radius: 2px; }}

QListWidget {{ border: 1px solid {border}; border-radius: 3px; }}
QListWidget::item {{ padding: 4px 6px; }}
QListWidget::item:selected {{ background: {accent}; color: white; }}
QListWidget::item:hover:!selected {{ background: {hover}; }}

QPlainTextEdit {{ border: 1px solid {border}; border-radius: 3px; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {border_hi}; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {accent}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

QMenuBar {{ background: {panel}; border-bottom: 1px solid {border}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {hover}; }}
QMenu {{ border: 1px solid {border}; }}
QMenu::item {{ padding: 5px 22px; }}
QMenu::item:selected {{ background: {accent}; color: white; }}

QStatusBar {{ border-top: 1px solid {border}; }}
QToolTip {{ border: 1px solid {border}; padding: 3px; }}
"""


def _dark_palette() -> QPalette:
    p = QPalette()
    win, base, text = QColor("#1e1e1e"), QColor("#252526"), QColor("#e4e4e4")

    p.setColor(QPalette.Window, win)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, QColor("#2d2d30"))
    p.setColor(QPalette.ToolTipBase, QColor("#2d2d30"))
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, QColor("#2d2d30"))
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.BrightText, QColor("#ff6b6b"))
    p.setColor(QPalette.Link, QColor(_ACCENT))
    p.setColor(QPalette.Highlight, QColor(_ACCENT))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))

    # Disabled states — without these, dark mode looks broken wherever the UI
    # greys a control out, which for this app is most of the time.
    dim = QColor("#6b6b6b")
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        p.setColor(QPalette.Disabled, role, dim)
    p.setColor(QPalette.Disabled, QPalette.Highlight, QColor("#37373a"))
    p.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor("#9b9b9b"))
    p.setColor(QPalette.Disabled, QPalette.Base, win)
    return p


def apply_theme(app: QApplication, name: str) -> str:
    """Apply *name* ("light" or "dark") to *app*. Returns the theme applied.

    Falls back to DEFAULT_THEME for an unknown name so a hand-edited settings
    file cannot leave the app unstyled.
    """
    if name not in THEMES:
        name = DEFAULT_THEME

    if name == "dark":
        app.setPalette(_dark_palette())
    else:
        # standardPalette() is Fusion's own light palette — cleaner than
        # hand-rolling one and stays correct if the style changes.
        app.setPalette(app.style().standardPalette())

    app.setStyleSheet(_QSS.format(**_TOKENS[name]))
    return name
