"""
Control panel — Start / Pause / Stop buttons, QThread worker.
(Full implementation in Phase 7)
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel


class ControlPanel(QWidget):
    sig_log_message      = pyqtSignal(str)
    sig_epoch_complete   = pyqtSignal(dict)
    sig_batch_complete   = pyqtSignal(dict)
    sig_training_finished = pyqtSignal()
    sig_checkpoint_saved  = pyqtSignal(str)  # path to checkpoint dir

    def __init__(self, settings_panel, parent=None):
        super().__init__(parent)
        self._settings = settings_panel
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Controls — Phase 7"))
        layout.addStretch()
