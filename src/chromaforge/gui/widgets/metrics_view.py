"""Quality metrics display widget."""

from __future__ import annotations

from typing import Optional

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QFrame, QGroupBox,
    )
except ImportError:
    raise ImportError("PySide6 is required for the GUI.")

from chromaforge.core.types import QualityMetrics
from chromaforge.gui.styles.theme import PALETTE, FONT_MONO, SPACING_SM, SPACING_MD


class MetricRow(QFrame):
    """Single metric display with label, value, and status indicator."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING_SM, 2, SPACING_SM, 2)
        layout.setSpacing(SPACING_SM)

        self._label = QLabel(label)
        self._label.setStyleSheet(f"color: {PALETTE.text_secondary}; font-size: 12px;")
        layout.addWidget(self._label, 1)

        self._value = QLabel("--")
        self._value.setStyleSheet(
            f"color: {PALETTE.text_primary}; "
            f"font-family: {FONT_MONO}; font-size: 13px;"
        )
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._value)

        self._dot = QLabel()
        self._dot.setFixedSize(8, 8)
        self._dot.setStyleSheet(
            f"background-color: {PALETTE.text_disabled}; border-radius: 4px;"
        )
        layout.addWidget(self._dot)

    def set_value(self, text: str, status: str = "neutral"):
        self._value.setText(text)
        colors = {
            "good": PALETTE.success,
            "warning": PALETTE.warning,
            "bad": PALETTE.error,
            "neutral": PALETTE.text_disabled,
        }
        color = colors.get(status, PALETTE.text_disabled)
        self._dot.setStyleSheet(f"background-color: {color}; border-radius: 4px;")

    def clear(self):
        self._value.setText("--")
        self._dot.setStyleSheet(
            f"background-color: {PALETTE.text_disabled}; border-radius: 4px;"
        )


class MetricsDisplay(QWidget):
    """Display panel for LUT quality metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        group = QGroupBox("Quality Metrics")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(2)

        self._mean_de = MetricRow("Mean dE2000")
        self._median_de = MetricRow("Median dE2000")
        self._p95_de = MetricRow("P95 dE2000")
        self._max_de = MetricRow("Max dE2000")
        self._tv = MetricRow("Total Variation")
        self._mono = MetricRow("Neutral Monotonic")
        self._oog = MetricRow("Out-of-Gamut")
        self._coverage = MetricRow("Coverage")

        for row in [self._mean_de, self._median_de, self._p95_de,
                    self._max_de, self._tv, self._mono, self._oog,
                    self._coverage]:
            group_layout.addWidget(row)

        layout.addWidget(group)

    def _de_status(self, val: float) -> str:
        if val < 1.0:
            return "good"
        elif val < 3.0:
            return "good"
        elif val < 5.0:
            return "warning"
        return "bad"

    def update_metrics(self, metrics: QualityMetrics):
        self._mean_de.set_value(f"{metrics.mean_delta_e:.2f}", self._de_status(metrics.mean_delta_e))
        self._median_de.set_value(f"{metrics.median_delta_e:.2f}", self._de_status(metrics.median_delta_e))
        self._p95_de.set_value(f"{metrics.p95_delta_e:.2f}", self._de_status(metrics.p95_delta_e))
        self._max_de.set_value(f"{metrics.max_delta_e:.2f}", self._de_status(metrics.max_delta_e))
        self._tv.set_value(f"{metrics.total_variation:.4f}", "neutral")
        self._mono.set_value(
            "Yes" if metrics.neutral_monotonic else f"No ({metrics.neutral_mono_violations})",
            "good" if metrics.neutral_monotonic else "warning",
        )
        self._oog.set_value(
            f"{metrics.oog_percentage:.2f}%",
            "good" if metrics.oog_percentage < 1.0 else "warning",
        )
        self._coverage.set_value(
            f"{metrics.coverage_percentage:.1f}%",
            "good" if metrics.coverage_percentage > 10 else "bad",
        )

    def clear(self):
        for row in [self._mean_de, self._median_de, self._p95_de,
                    self._max_de, self._tv, self._mono, self._oog,
                    self._coverage]:
            row.clear()
