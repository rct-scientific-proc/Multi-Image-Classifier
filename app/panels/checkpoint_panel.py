"""
Checkpoint panel — list checkpoints, resume, export best model.
(Full implementation in Phase 9)
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel


class CheckpointPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Checkpoints — Phase 9"))
        layout.addStretch()

    def refresh(self, checkpoint_dir: str = ""):
        """Reload the checkpoint list from disk. Stub for Phase 9."""
        pass
