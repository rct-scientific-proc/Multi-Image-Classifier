"""
TensorBoard panel — launch/stop tensorboard subprocess, open in browser.
"""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class TensorBoardPanel(QWidget):
    """Manages a tensorboard subprocess and exposes Start / Stop / Open buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: subprocess.Popen | None = None
        self._log_dir  = "runs"
        self._port     = 6006

        box = QGroupBox("TensorBoard")
        box_lay = QVBoxLayout(box)

        # ── status label ──────────────────────────────────────────────────
        self._status_label = QLabel("Not running")
        box_lay.addWidget(self._status_label)

        # ── buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("▶  Start TB")
        self._btn_stop  = QPushButton("⏹  Stop TB")
        self._btn_open  = QPushButton("🌐  Open in browser")
        self._btn_stop.setEnabled(False)
        self._btn_open.setEnabled(False)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        btn_row.addWidget(self._btn_open)
        box_lay.addLayout(btn_row)

        self._btn_start.clicked.connect(self.start)
        self._btn_stop.clicked.connect(self.stop)
        self._btn_open.clicked.connect(self.open_browser)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)

        # Poll every 2 s to detect if the process has died unexpectedly
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._poll)

    # ── Public API ────────────────────────────────────────────────────────────

    def configure(self, log_dir: str, port: int) -> None:
        """Update log_dir and port from settings (called before Start)."""
        self._log_dir = log_dir or "runs"
        self._port    = int(port) if port else 6006

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return  # already running

        tb_exe = _find_tensorboard()
        if tb_exe is None:
            self._status_label.setText("tensorboard not found — is it installed?")
            return

        try:
            self._proc = subprocess.Popen(
                [tb_exe, "--logdir", self._log_dir, "--port", str(self._port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            self._status_label.setText(f"Failed to start: {exc}")
            self._proc = None
            return

        self._status_label.setText(
            f"Running on http://localhost:{self._port}  (logdir: {self._log_dir})"
        )
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_open.setEnabled(True)
        self._poll_timer.start()

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        self._poll_timer.stop()
        self._status_label.setText("Stopped")
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_open.setEnabled(False)

    def open_browser(self) -> None:
        webbrowser.open(f"http://localhost:{self._port}")

    def cleanup(self) -> None:
        """Call on app exit to ensure the subprocess is killed."""
        if self._proc is not None and self._proc.poll() is None:
            self.stop()

    # ── Private ───────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        """Detect if tensorboard died on its own."""
        if self._proc is not None and self._proc.poll() is not None:
            # Read stderr to surface the real error
            stderr = ""
            try:
                if self._proc.stderr is not None:
                    stderr = self._proc.stderr.read().decode(errors="replace").strip()
            except Exception:
                pass
            self._proc = None
            self._poll_timer.stop()
            msg = "Process exited unexpectedly"
            if stderr:
                # Keep label short — show last line
                last = stderr.splitlines()[-1][:140]
                msg = f"Exited: {last}"
            self._status_label.setText(msg)
            self._btn_start.setEnabled(True)
            self._btn_stop.setEnabled(False)
            self._btn_open.setEnabled(False)


def _find_tensorboard() -> str | None:
    """Return the path to the tensorboard executable in the current venv/PATH."""
    import shutil
    # Prefer the venv-local tensorboard next to the current python
    python_dir = os.path.dirname(sys.executable)
    for name in ("tensorboard.exe", "tensorboard"):
        candidate = os.path.join(python_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("tensorboard")
