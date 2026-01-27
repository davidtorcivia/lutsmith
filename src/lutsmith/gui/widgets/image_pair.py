"""Side-by-side image viewer with synchronized zoom and pan."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QSettings
    from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QWheelEvent, QMouseEvent
    from PySide6.QtWidgets import (
        QWidget, QHBoxLayout, QVBoxLayout, QGraphicsView,
        QGraphicsScene, QGraphicsPixmapItem, QLabel, QPushButton,
        QFileDialog, QSizePolicy, QFrame,
    )
except ImportError:
    raise ImportError("PySide6 is required for the GUI. Install with: pip install PySide6")

from lutsmith.gui.styles.theme import PALETTE, FONT_MONO, SPACING_SM, SPACING_MD


class SyncGraphicsView(QGraphicsView):
    """Graphics view with zoom/pan that can sync with a sibling."""

    zoom_changed = Signal(float)
    pan_changed = Signal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QColor(PALETTE.bg_dark))
        self._zoom_level = 1.0
        self._syncing = False

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._zoom_level *= factor
        self.scale(factor, factor)
        if not self._syncing:
            self.zoom_changed.emit(self._zoom_level)

    def sync_zoom(self, level: float):
        if self._syncing:
            return
        self._syncing = True
        factor = level / self._zoom_level
        self._zoom_level = level
        self.scale(factor, factor)
        self._syncing = False

    def fit_in_view_custom(self):
        if self.scene() and self.scene().items():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_level = self.transform().m11()


class ImagePairViewer(QWidget):
    """Side-by-side image viewer for source/target comparison."""

    source_loaded = Signal(str)
    target_loaded = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_path: Optional[Path] = None
        self._target_path: Optional[Path] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING_SM)

        # Image views
        views_layout = QHBoxLayout()
        views_layout.setSpacing(1)

        # Source view
        source_container = QVBoxLayout()
        self._source_label = QLabel("Source (ungraded)")
        self._source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._source_label.setStyleSheet(f"color: {PALETTE.text_secondary}; font-size: 11px;")
        self._source_scene = QGraphicsScene()
        self._source_view = SyncGraphicsView()
        self._source_view.setScene(self._source_scene)
        self._source_view.setMinimumHeight(200)
        source_container.addWidget(self._source_label)
        source_container.addWidget(self._source_view, 1)
        views_layout.addLayout(source_container, 1)

        # Target view
        target_container = QVBoxLayout()
        self._target_label = QLabel("Target (graded)")
        self._target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._target_label.setStyleSheet(f"color: {PALETTE.text_secondary}; font-size: 11px;")
        self._target_scene = QGraphicsScene()
        self._target_view = SyncGraphicsView()
        self._target_view.setScene(self._target_scene)
        self._target_view.setMinimumHeight(200)
        target_container.addWidget(self._target_label)
        target_container.addWidget(self._target_view, 1)
        views_layout.addLayout(target_container, 1)

        layout.addLayout(views_layout, 1)

        # Sync zoom between views
        self._source_view.zoom_changed.connect(self._target_view.sync_zoom)
        self._target_view.zoom_changed.connect(self._source_view.sync_zoom)

        # Controls
        controls = QHBoxLayout()
        controls.setSpacing(SPACING_SM)

        self._btn_source = QPushButton("Load Source")
        self._btn_source.clicked.connect(self._load_source)
        controls.addWidget(self._btn_source)

        self._btn_target = QPushButton("Load Target")
        self._btn_target.clicked.connect(self._load_target)
        controls.addWidget(self._btn_target)

        controls.addStretch()

        self._btn_fit = QPushButton("Fit")
        self._btn_fit.setToolTip("Fit images to view")
        self._btn_fit.clicked.connect(self._fit_views)
        controls.addWidget(self._btn_fit)

        self._pixel_info = QLabel("")
        self._pixel_info.setStyleSheet(f"font-family: {FONT_MONO}; font-size: 11px; color: {PALETTE.text_secondary};")
        controls.addWidget(self._pixel_info)

        layout.addLayout(controls)

    def _load_source(self):
        settings = QSettings("LutSmith", "LutSmith")
        last_dir = settings.value("last_image_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Source Image", last_dir,
            "Images (*.png *.tiff *.tif *.exr *.jpg *.jpeg *.bmp *.dpx *.hdr)"
        )
        if path:
            settings.setValue("last_image_dir", str(Path(path).parent))
            self._source_path = Path(path)
            self._display_image(path, self._source_scene)
            self._source_label.setText(f"Source: {Path(path).name}")
            self._source_view.fit_in_view_custom()
            self.source_loaded.emit(path)

    def _load_target(self):
        settings = QSettings("LutSmith", "LutSmith")
        last_dir = settings.value("last_image_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Target Image", last_dir,
            "Images (*.png *.tiff *.tif *.exr *.jpg *.jpeg *.bmp *.dpx *.hdr)"
        )
        if path:
            settings.setValue("last_image_dir", str(Path(path).parent))
            self._target_path = Path(path)
            self._display_image(path, self._target_scene)
            self._target_label.setText(f"Target: {Path(path).name}")
            self._target_view.fit_in_view_custom()
            self.target_loaded.emit(path)

    def _display_image(self, path: str, scene: QGraphicsScene):
        """Load and display an image in a scene."""
        scene.clear()
        try:
            from lutsmith.io.image import load_image
            img, _ = load_image(path)
            # Convert float32 to uint8 for display
            display = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            h, w, c = display.shape
            bytes_per_line = w * c
            qimg = QImage(display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scene.addPixmap(pixmap)
        except Exception as e:
            scene.addText(f"Error: {e}")

    def _fit_views(self):
        self._source_view.fit_in_view_custom()
        self._target_view.fit_in_view_custom()

    @property
    def source_path(self) -> Optional[Path]:
        return self._source_path

    @property
    def target_path(self) -> Optional[Path]:
        return self._target_path

    def set_source(self, path: str):
        self._source_path = Path(path)
        self._display_image(path, self._source_scene)
        self._source_label.setText(f"Source: {Path(path).name}")
        self._source_view.fit_in_view_custom()

    def set_target(self, path: str):
        self._target_path = Path(path)
        self._display_image(path, self._target_scene)
        self._target_label.setText(f"Target: {Path(path).name}")
        self._target_view.fit_in_view_custom()
