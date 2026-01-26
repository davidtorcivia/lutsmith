"""Scrollable diagnostic log viewer with severity coloring."""

from __future__ import annotations

from datetime import datetime

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QTextCursor, QColor, QFont
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit,
        QPushButton, QGroupBox, QLabel,
    )
except ImportError:
    raise ImportError("PySide6 is required for the GUI.")

from chromaforge.gui.styles.theme import PALETTE, FONT_MONO, SPACING_SM


# Severity levels with associated colors
_SEVERITY_COLORS = {
    "debug": PALETTE.text_disabled,
    "info": PALETTE.text_secondary,
    "stage": PALETTE.accent,
    "warning": PALETTE.warning,
    "error": PALETTE.error,
    "success": PALETTE.success,
}


class LogViewer(QWidget):
    """Timestamped scrolling log with severity-based coloring."""

    MAX_LINES = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._line_count = 0
        self._auto_scroll = True
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        group = QGroupBox("Log")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(SPACING_SM)

        # Log text area
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(self.MAX_LINES)
        self._text.setFont(QFont(FONT_MONO, 11))
        self._text.setStyleSheet(
            f"QPlainTextEdit {{"
            f"  background-color: {PALETTE.bg_dark};"
            f"  color: {PALETTE.text_secondary};"
            f"  border: 1px solid {PALETTE.border_subtle};"
            f"  padding: 4px;"
            f"}}"
        )
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        group_layout.addWidget(self._text)

        # Bottom controls
        controls = QHBoxLayout()
        controls.setSpacing(SPACING_SM)

        self._line_count_label = QLabel("0 lines")
        self._line_count_label.setStyleSheet(
            f"color: {PALETTE.text_disabled}; font-size: 11px;"
        )
        controls.addWidget(self._line_count_label)

        controls.addStretch()

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedWidth(60)
        self._clear_btn.clicked.connect(self.clear)
        controls.addWidget(self._clear_btn)

        group_layout.addLayout(controls)
        layout.addWidget(group)

    def append(self, message: str, severity: str = "info"):
        """Append a log message with timestamp and severity coloring.

        Args:
            message: The log message text.
            severity: One of 'debug', 'info', 'stage', 'warning', 'error', 'success'.
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        color = _SEVERITY_COLORS.get(severity, PALETTE.text_secondary)

        # Build formatted line with HTML coloring
        severity_tag = severity.upper()[:5].ljust(5)
        html = (
            f'<span style="color: {PALETTE.text_disabled};">{timestamp}</span> '
            f'<span style="color: {color};">[{severity_tag}]</span> '
            f'<span style="color: {color};">{_escape_html(message)}</span>'
        )

        self._text.appendHtml(html)
        self._line_count += 1
        self._line_count_label.setText(f"{self._line_count} lines")

        # Auto-scroll to bottom
        if self._auto_scroll:
            scrollbar = self._text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def append_separator(self):
        """Add a visual separator line."""
        sep = "\u2500" * 60  # Box drawing horizontal line
        self._text.appendHtml(
            f'<span style="color: {PALETTE.border};">{sep}</span>'
        )

    def clear(self):
        """Clear all log messages."""
        self._text.clear()
        self._line_count = 0
        self._line_count_label.setText("0 lines")


def _escape_html(text: str) -> str:
    """Escape HTML special characters in log messages."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
