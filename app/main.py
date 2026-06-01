"""
Entry point — QApplication and main window.

Run with:
    python -m app.main
or via the project entry point:
    image-classifier
"""

import sys
from PyQt5.QtWidgets import QApplication
from app.main_window import MainWindow


def main():
    app    = QApplication(sys.argv)
    app.setApplicationName("Image Classifier")
    app.setOrganizationName("image_classifier")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
