"""
Entry point — QApplication and main window.

Run with:
    python -m app.main
or via the project entry point:
    image-classifier
"""

import sys
# torch must be imported before PyQt5 on Windows to avoid DLL init failure
import torch  # noqa: F401
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QApplication
from app.main_window import MainWindow


def _configure_hidpi() -> None:
    """Opt in to high-DPI scaling.

    Qt5 does not enable this by default (Qt6 does), so on a scaled Windows
    display the app is DPI-virtualised by the OS and renders blurry at the
    wrong effective size. Every call here must happen before the QApplication
    is constructed, otherwise it is silently ignored.
    """
    # PassThrough preserves fractional scale factors — 150% stays 1.5 instead
    # of being rounded up to 2.0. Qt 5.14+, so guard it rather than assume.
    policy = getattr(Qt, "HighDpiScaleFactorRoundingPolicy", None)
    if policy is not None:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(policy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


def main():
    _configure_hidpi()

    app = QApplication(sys.argv)
    # Fusion is consistent across platforms and honours stylesheets properly;
    # the native Windows Qt5 style ignores much of QSS and looks dated.
    app.setStyle("Fusion")
    app.setApplicationName("Image Classifier")
    app.setOrganizationName("image_classifier")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
