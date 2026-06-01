"""
Console panel — read-only QPlainTextEdit for epoch/metric output.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


class ConsolePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        label = QLabel("Training Log")
        layout.addWidget(label)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Courier New", 9))
        self._text.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self._text)

    def append_message(self, msg: str):
        self._text.appendPlainText(msg)
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear(self):
        self._text.clear()
