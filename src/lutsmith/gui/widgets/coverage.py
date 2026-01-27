"""Coverage visualization widget -- 2D slices through the LUT grid."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from PySide6.QtCore import Qt, Signal, QRectF
    from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QSlider, QGroupBox, QComboBox, QFrame,
    )
except ImportError:
    raise ImportError("PySide6 is required for the GUI.")

from lutsmith.gui.styles.theme import PALETTE, FONT_MONO, SPACING_SM, SPACING_MD


class SliceView(QLabel):
    """Renders a single 2D slice of the coverage/distance grid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(128, 128)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"background-color: {PALETTE.bg_dark}; "
            f"border: 1px solid {PALETTE.border};"
        )
        self._slice_data: Optional[np.ndarray] = None

    def set_slice(self, data: np.ndarray, colormap: str = "viridis"):
        """Set the 2D slice data to display.

        Args:
            data: 2D array of values in [0, 1] range.
            colormap: Color mapping to use.
        """
        self._slice_data = data
        h, w = data.shape

        # Map values to RGB using a simple colormap
        rgb = _apply_colormap(data, colormap)

        # Scale up for display
        scale = max(1, min(256 // max(h, w, 1), 8))
        display_w = w * scale
        display_h = h * scale

        # Create QImage from RGB data
        rgb_scaled = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        rgb_bytes = np.ascontiguousarray(rgb_scaled, dtype=np.uint8)
        qimg = QImage(
            rgb_bytes.data, display_w, display_h, display_w * 3,
            QImage.Format.Format_RGB888,
        )
        # Must copy since rgb_bytes may be garbage collected
        pixmap = QPixmap.fromImage(qimg.copy())
        self.setPixmap(pixmap)

    def clear_slice(self):
        self._slice_data = None
        self.clear()


def _apply_colormap(data: np.ndarray, name: str = "viridis") -> np.ndarray:
    """Apply a simple colormap to normalized 2D data.

    Returns (H, W, 3) uint8 array.
    """
    clamped = np.clip(data, 0.0, 1.0)

    if name == "viridis":
        # Simplified viridis-like: dark purple -> teal -> yellow
        r = np.clip(0.28 + 0.72 * clamped**0.5, 0, 1)
        g = np.clip(0.08 + 0.85 * clamped, 0, 1)
        b = np.clip(0.65 - 0.55 * clamped, 0, 1)
    elif name == "heat":
        # Black -> red -> yellow -> white
        r = np.clip(3.0 * clamped, 0, 1)
        g = np.clip(3.0 * clamped - 1.0, 0, 1)
        b = np.clip(3.0 * clamped - 2.0, 0, 1)
    else:
        # Grayscale fallback
        r = g = b = clamped

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


class CoverageViewer(QWidget):
    """Interactive coverage visualization with axis selection and slice slider."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._distance_grid: Optional[np.ndarray] = None
        self._grid_size: int = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING_SM)

        group = QGroupBox("Coverage Map")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(SPACING_SM)

        # Controls row
        controls = QHBoxLayout()
        controls.setSpacing(SPACING_SM)

        axis_label = QLabel("Slice Axis:")
        axis_label.setStyleSheet(f"color: {PALETTE.text_secondary}; font-size: 12px;")
        controls.addWidget(axis_label)

        self._axis_combo = QComboBox()
        self._axis_combo.addItems(["R (Red)", "G (Green)", "B (Blue)"])
        self._axis_combo.setCurrentIndex(2)  # Default: B axis
        self._axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        controls.addWidget(self._axis_combo)

        controls.addStretch()

        self._slice_label = QLabel("Slice: --")
        self._slice_label.setStyleSheet(
            f"color: {PALETTE.text_secondary}; "
            f"font-family: {FONT_MONO}; font-size: 12px;"
        )
        controls.addWidget(self._slice_label)

        group_layout.addLayout(controls)

        # Slice slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slice_changed)
        group_layout.addWidget(self._slider)

        # Slice view
        self._view = SliceView()
        group_layout.addWidget(self._view)

        # Legend
        legend = QHBoxLayout()
        legend.setSpacing(SPACING_MD)

        near_label = QLabel("Near data")
        near_label.setStyleSheet(f"color: {PALETTE.text_secondary}; font-size: 11px;")
        legend.addWidget(near_label)

        legend.addStretch()

        far_label = QLabel("Far from data")
        far_label.setStyleSheet(f"color: {PALETTE.text_secondary}; font-size: 11px;")
        legend.addWidget(far_label)

        group_layout.addLayout(legend)

        layout.addWidget(group)

    def set_distance_grid(self, distances: np.ndarray, grid_size: int):
        """Set the 3D distance-to-data grid for visualization.

        Args:
            distances: Flat array of length N^3, or (N, N, N) 3D array.
            grid_size: LUT grid size N.
        """
        self._grid_size = grid_size

        if distances.ndim == 1:
            # Reshape flat (b, g, r order) to 3D
            self._distance_grid = distances.reshape(
                grid_size, grid_size, grid_size
            )
        else:
            self._distance_grid = distances

        # Normalize to [0, 1]
        dmax = self._distance_grid.max()
        if dmax > 0:
            self._distance_grid = self._distance_grid / dmax

        self._slider.setMaximum(grid_size - 1)
        self._slider.setValue(grid_size // 2)
        self._slider.setEnabled(True)
        self._update_view()

    def _on_axis_changed(self, _index: int):
        self._update_view()

    def _on_slice_changed(self, value: int):
        self._update_view()

    def _update_view(self):
        if self._distance_grid is None:
            return

        axis = self._axis_combo.currentIndex()
        idx = self._slider.value()
        N = self._grid_size

        axis_names = ["R", "G", "B"]
        self._slice_label.setText(
            f"Slice: {axis_names[axis]}={idx}/{N - 1}"
        )

        # Extract 2D slice
        if axis == 0:  # R axis
            slice_2d = self._distance_grid[idx, :, :]
        elif axis == 1:  # G axis
            slice_2d = self._distance_grid[:, idx, :]
        else:  # B axis
            slice_2d = self._distance_grid[:, :, idx]

        self._view.set_slice(slice_2d, colormap="viridis")

    def clear(self):
        self._distance_grid = None
        self._grid_size = 0
        self._slider.setMaximum(0)
        self._slider.setEnabled(False)
        self._slice_label.setText("Slice: --")
        self._view.clear_slice()
