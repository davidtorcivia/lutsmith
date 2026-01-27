"""Multi-stage progress display for pipeline execution."""

from __future__ import annotations

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QFrame,
    )
except ImportError:
    raise ImportError("PySide6 is required for the GUI.")

from lutsmith.gui.styles.theme import PALETTE, FONT_MONO, SPACING_SM


class StageIndicator(QFrame):
    """Single pipeline stage progress indicator."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING_SM, 2, SPACING_SM, 2)
        layout.setSpacing(SPACING_SM)

        self._dot = QLabel()
        self._dot.setFixedSize(8, 8)
        self._dot.setStyleSheet(
            f"background-color: {PALETTE.text_disabled}; "
            f"border-radius: 4px;"
        )
        layout.addWidget(self._dot)

        self._name = QLabel(name)
        self._name.setStyleSheet(f"color: {PALETTE.text_secondary}; font-size: 12px;")
        layout.addWidget(self._name, 1)

        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color: {PALETTE.text_disabled}; "
            f"font-family: {FONT_MONO}; font-size: 11px;"
        )
        layout.addWidget(self._status)

    def set_pending(self):
        self._dot.setStyleSheet(
            f"background-color: {PALETTE.text_disabled}; border-radius: 4px;"
        )
        self._name.setStyleSheet(f"color: {PALETTE.text_secondary}; font-size: 12px;")
        self._status.setText("")

    def set_active(self, message: str = ""):
        self._dot.setStyleSheet(
            f"background-color: {PALETTE.accent}; border-radius: 4px;"
        )
        self._name.setStyleSheet(f"color: {PALETTE.text_primary}; font-size: 12px;")
        self._status.setText(message)
        self._status.setStyleSheet(
            f"color: {PALETTE.accent}; font-family: {FONT_MONO}; font-size: 11px;"
        )

    def set_complete(self, message: str = ""):
        self._dot.setStyleSheet(
            f"background-color: {PALETTE.success}; border-radius: 4px;"
        )
        self._name.setStyleSheet(f"color: {PALETTE.text_primary}; font-size: 12px;")
        self._status.setText(message)
        self._status.setStyleSheet(
            f"color: {PALETTE.success}; font-family: {FONT_MONO}; font-size: 11px;"
        )

    def set_error(self, message: str = ""):
        self._dot.setStyleSheet(
            f"background-color: {PALETTE.error}; border-radius: 4px;"
        )
        self._status.setText(message)
        self._status.setStyleSheet(
            f"color: {PALETTE.error}; font-family: {FONT_MONO}; font-size: 11px;"
        )


class PipelineProgress(QWidget):
    """Multi-stage pipeline progress display."""

    STAGES = ["Preprocess", "Sampling", "Solving", "Refinement", "Validation", "Export"]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._overall = QProgressBar()
        self._overall.setRange(0, 100)
        self._overall.setValue(0)
        self._overall.setTextVisible(False)
        self._overall.setFixedHeight(6)
        layout.addWidget(self._overall)

        self._indicators = {}
        for name in self.STAGES:
            indicator = StageIndicator(name)
            self._indicators[name.lower()] = indicator
            layout.addWidget(indicator)

    def reset(self):
        self._overall.setValue(0)
        for indicator in self._indicators.values():
            indicator.set_pending()

    def update_stage(self, stage: str, fraction: float, message: str = ""):
        stage_key = stage.lower()
        if stage_key not in self._indicators:
            return

        # Mark previous stages as complete
        stage_keys = [s.lower() for s in self.STAGES]
        current_idx = stage_keys.index(stage_key) if stage_key in stage_keys else -1

        for i, key in enumerate(stage_keys):
            if i < current_idx:
                self._indicators[key].set_complete()
            elif i == current_idx:
                if fraction >= 1.0:
                    self._indicators[key].set_complete(message)
                else:
                    self._indicators[key].set_active(message)

        # Overall progress
        overall = (current_idx * 100 // len(self.STAGES)) + int(
            fraction * 100 / len(self.STAGES)
        )
        self._overall.setValue(min(overall, 100))

    def set_complete(self):
        self._overall.setValue(100)
        for indicator in self._indicators.values():
            indicator.set_complete()

    def set_error(self, stage: str, message: str):
        stage_key = stage.lower()
        if stage_key in self._indicators:
            self._indicators[stage_key].set_error(message)
