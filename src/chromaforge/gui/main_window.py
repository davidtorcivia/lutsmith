"""Main window for ChromaForge GUI."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Optional

try:
    from PySide6.QtCore import Qt, Slot, QSettings
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QMainWindow, QTabWidget, QWidget,
        QVBoxLayout, QHBoxLayout, QSplitter,
        QFileDialog, QMessageBox, QStatusBar,
        QPushButton, QLabel, QGroupBox,
        QFormLayout, QLineEdit, QSpinBox,
        QComboBox, QCheckBox, QScrollArea, QFrame,
    )
except ImportError:
    raise ImportError("PySide6 is required for the GUI.")

from chromaforge import __version__
from chromaforge.core.types import (
    ExportFormat,
    InterpolationKernel,
    PipelineConfig,
    PipelineResult,
    RobustLoss,
    TransferFunction,
)
from chromaforge.gui.styles.theme import PALETTE, SPACING_SM, SPACING_MD
from chromaforge.gui.widgets.coverage import CoverageViewer
from chromaforge.gui.widgets.image_pair import ImagePairViewer
from chromaforge.gui.widgets.log_viewer import LogViewer
from chromaforge.gui.widgets.metrics_view import MetricsDisplay
from chromaforge.gui.widgets.parameters import ParameterPanel
from chromaforge.gui.widgets.progress import PipelineProgress
from chromaforge.gui.workers import PipelineWorker, HaldWorker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """ChromaForge main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"ChromaForge v{__version__}")
        self.setMinimumSize(1100, 720)
        self.resize(1400, 860)

        self._settings = QSettings("ChromaForge", "ChromaForge")
        self._worker: Optional[PipelineWorker] = None
        self._hald_worker: Optional[HaldWorker] = None
        self._source_path: Optional[Path] = None
        self._target_path: Optional[Path] = None
        self._last_result: Optional[PipelineResult] = None

        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Create the main layout with tab bar."""
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # Tab 1: Image Pair workflow
        self._tabs.addTab(self._build_image_pair_tab(), "Image Pair")

        # Tab 2: Hald CLUT workflow
        self._tabs.addTab(self._build_hald_tab(), "Hald CLUT")

        # Tab 3: Settings
        self._tabs.addTab(self._build_settings_tab(), "Settings")

    def _build_image_pair_tab(self) -> QWidget:
        """Build the main image pair extraction workflow tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Image viewer + progress + log
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(SPACING_SM, SPACING_SM, 0, SPACING_SM)
        left_layout.setSpacing(SPACING_SM)

        self._image_viewer = ImagePairViewer()
        left_layout.addWidget(self._image_viewer, 3)

        self._progress = PipelineProgress()
        left_layout.addWidget(self._progress)

        self._log = LogViewer()
        left_layout.addWidget(self._log, 1)

        splitter.addWidget(left)

        # Right: Parameters + metrics + coverage + action buttons (scrollable)
        right = QWidget()
        right.setMaximumWidth(380)
        right.setMinimumWidth(280)
        right_outer = QVBoxLayout(right)
        right_outer.setContentsMargins(0, 0, 0, 0)
        right_outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        right_layout = QVBoxLayout(scroll_content)
        right_layout.setContentsMargins(0, SPACING_SM, SPACING_SM, SPACING_SM)
        right_layout.setSpacing(SPACING_SM)

        self._params = ParameterPanel()
        right_layout.addWidget(self._params)

        self._metrics = MetricsDisplay()
        right_layout.addWidget(self._metrics)

        self._coverage = CoverageViewer()
        right_layout.addWidget(self._coverage)

        right_layout.addStretch()

        scroll.setWidget(scroll_content)
        right_outer.addWidget(scroll, 1)

        # Action buttons (outside scroll, always visible)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, SPACING_SM, SPACING_SM, SPACING_SM)
        btn_row.setSpacing(SPACING_SM)

        self._btn_extract = QPushButton("Extract LUT")
        self._btn_extract.setProperty("class", "primary")
        self._btn_extract.setMinimumHeight(36)
        self._btn_extract.setEnabled(False)
        self._btn_extract.setToolTip("Start LUT extraction from loaded image pair")
        btn_row.addWidget(self._btn_extract)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setMinimumHeight(36)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setToolTip("Cancel running extraction")
        btn_row.addWidget(self._btn_cancel)

        right_outer.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setSizes([700, 350])

        layout.addWidget(splitter)
        return tab

    def _build_hald_tab(self) -> QWidget:
        """Build the Hald CLUT workflow tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        layout.setSpacing(SPACING_MD)

        # Step 1: Generate identity
        gen_group = QGroupBox("1. Generate Hald Identity")
        gen_layout = QFormLayout(gen_group)
        gen_layout.setSpacing(SPACING_SM)

        self._hald_level = QSpinBox()
        self._hald_level.setRange(4, 12)
        self._hald_level.setValue(8)
        self._hald_level.setToolTip("Hald level (8 = 512x512, 64^3 LUT)")
        gen_layout.addRow("Level:", self._hald_level)

        self._btn_gen_hald = QPushButton("Generate Identity Image")
        self._btn_gen_hald.setProperty("class", "primary")
        gen_layout.addRow("", self._btn_gen_hald)

        layout.addWidget(gen_group)

        # Step 2: Reconstruct
        recon_group = QGroupBox("2. Reconstruct LUT from Processed Hald")
        recon_layout = QFormLayout(recon_group)
        recon_layout.setSpacing(SPACING_SM)

        hald_file_row = QHBoxLayout()
        self._hald_processed_path = QLineEdit()
        self._hald_processed_path.setPlaceholderText("Processed Hald image...")
        self._hald_processed_path.setReadOnly(True)
        hald_file_row.addWidget(self._hald_processed_path)
        self._btn_browse_hald = QPushButton("Browse...")
        self._btn_browse_hald.setFixedWidth(80)
        hald_file_row.addWidget(self._btn_browse_hald)
        recon_layout.addRow("Input:", hald_file_row)

        self._hald_target_size = QSpinBox()
        self._hald_target_size.setRange(0, 129)
        self._hald_target_size.setValue(0)
        self._hald_target_size.setSpecialValueText("Native")
        self._hald_target_size.setToolTip("Target LUT size (0 = native from Hald level)")
        recon_layout.addRow("Target Size:", self._hald_target_size)

        self._btn_reconstruct = QPushButton("Reconstruct LUT")
        self._btn_reconstruct.setProperty("class", "primary")
        self._btn_reconstruct.setEnabled(False)
        recon_layout.addRow("", self._btn_reconstruct)

        layout.addWidget(recon_group)

        # Hald log
        self._hald_log = LogViewer()
        layout.addWidget(self._hald_log, 1)

        layout.addStretch()
        return tab

    def _build_settings_tab(self) -> QWidget:
        """Build the settings/preferences tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        layout.setSpacing(SPACING_MD)

        # Output directory
        out_group = QGroupBox("Output")
        out_layout = QFormLayout(out_group)
        out_layout.setSpacing(SPACING_SM)

        out_row = QHBoxLayout()
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Current directory")
        out_row.addWidget(self._output_dir)
        self._btn_browse_out = QPushButton("Browse...")
        self._btn_browse_out.setFixedWidth(80)
        self._btn_browse_out.clicked.connect(self._browse_output_dir)
        out_row.addWidget(self._btn_browse_out)
        out_layout.addRow("Output Dir:", out_row)

        self._output_title = QLineEdit("ChromaForge LUT")
        self._output_title.setToolTip("Title embedded in LUT metadata")
        out_layout.addRow("LUT Title:", self._output_title)

        layout.addWidget(out_group)

        # I/O Backend
        io_group = QGroupBox("I/O Backend")
        io_layout = QFormLayout(io_group)
        io_layout.setSpacing(SPACING_SM)

        self._io_backend_label = QLabel()
        self._io_backend_label.setStyleSheet(f"color: {PALETTE.text_secondary};")
        self._detect_io_backend()
        io_layout.addRow("Image I/O:", self._io_backend_label)

        layout.addWidget(io_group)

        # About
        about_group = QGroupBox("About")
        about_layout = QVBoxLayout(about_group)
        about_label = QLabel(
            f"ChromaForge v{__version__}\n"
            f"Image-derived 3D LUT generation tool.\n\n"
            f"Regularized lattice regression with Laplacian smoothing."
        )
        about_label.setStyleSheet(f"color: {PALETTE.text_secondary}; font-size: 12px;")
        about_label.setWordWrap(True)
        about_layout.addWidget(about_label)
        layout.addWidget(about_group)

        layout.addStretch()
        return tab

    def _setup_menu(self):
        """Create the menu bar."""
        menu = self.menuBar()

        file_menu = menu.addMenu("File")

        open_source = QAction("Open Source Image...", self)
        open_source.setShortcut("Ctrl+O")
        open_source.triggered.connect(lambda: self._open_image("source"))
        file_menu.addAction(open_source)

        open_target = QAction("Open Target Image...", self)
        open_target.setShortcut("Ctrl+Shift+O")
        open_target.triggered.connect(lambda: self._open_image("target"))
        file_menu.addAction(open_target)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _setup_statusbar(self):
        """Create the status bar."""
        status = QStatusBar()
        self.setStatusBar(status)
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(f"color: {PALETTE.text_secondary};")
        status.addPermanentWidget(self._status_label)

    def _connect_signals(self):
        """Wire up widget signals to slots."""
        self._btn_extract.clicked.connect(self._start_extraction)
        self._btn_cancel.clicked.connect(self._cancel_extraction)

        # Image viewer load buttons
        self._image_viewer.source_loaded.connect(self._on_viewer_source_loaded)
        self._image_viewer.target_loaded.connect(self._on_viewer_target_loaded)

        # Hald tab
        self._btn_gen_hald.clicked.connect(self._generate_hald_identity)
        self._btn_browse_hald.clicked.connect(self._browse_hald_input)
        self._btn_reconstruct.clicked.connect(self._start_hald_reconstruction)
        self._hald_processed_path.textChanged.connect(
            lambda t: self._btn_reconstruct.setEnabled(bool(t))
        )

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _open_image(self, role: str):
        """Open file dialog to load a source or target image."""
        caption = f"Open {'Source' if role == 'source' else 'Target'} Image"
        last_dir = self._settings.value("last_image_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, caption, last_dir,
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.exr *.dpx *.hdr);;All Files (*)",
        )
        if not path:
            return

        self._settings.setValue("last_image_dir", str(Path(path).parent))
        p = Path(path)
        if role == "source":
            self._source_path = p
            self._image_viewer.set_source(str(p))
            self._log.append(f"Source loaded: {p.name}", "info")
        else:
            self._target_path = p
            self._image_viewer.set_target(str(p))
            self._log.append(f"Target loaded: {p.name}", "info")

        # Enable extract button when both images loaded
        self._btn_extract.setEnabled(
            self._source_path is not None and self._target_path is not None
        )
        self._status_label.setText(f"Loaded: {p.name}")

    @Slot(str)
    def _on_viewer_source_loaded(self, path: str):
        """Handle source image loaded via the ImagePairViewer buttons."""
        self._source_path = Path(path)
        self._log.append(f"Source loaded: {self._source_path.name}", "info")
        self._btn_extract.setEnabled(
            self._source_path is not None and self._target_path is not None
        )
        self._status_label.setText(f"Loaded: {self._source_path.name}")

    @Slot(str)
    def _on_viewer_target_loaded(self, path: str):
        """Handle target image loaded via the ImagePairViewer buttons."""
        self._target_path = Path(path)
        self._log.append(f"Target loaded: {self._target_path.name}", "info")
        self._btn_extract.setEnabled(
            self._source_path is not None and self._target_path is not None
        )
        self._status_label.setText(f"Loaded: {self._target_path.name}")

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    @Slot()
    def _start_extraction(self):
        """Launch the pipeline worker thread."""
        if self._source_path is None or self._target_path is None:
            return

        # Build config from parameter panel
        config = self._build_config()

        if config.include_shaper is True and config.format != ExportFormat.CUBE:
            self._log.append(
                "Shaper requested, but only .cube export supports 1D shapers. "
                "Shaper will not be exported.",
                "warning",
            )

        self._log.append_separator()
        self._progress.reset()
        self._metrics.clear()
        self._coverage.clear()

        self._btn_extract.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._status_label.setText("Extracting LUT...")

        self._worker = PipelineWorker(config, parent=self)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.log_message.connect(self._on_log)
        self._worker.finished_ok.connect(self._on_pipeline_done)
        self._worker.finished_error.connect(self._on_pipeline_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _build_config(self) -> PipelineConfig:
        """Assemble PipelineConfig from current GUI state."""
        output_dir = self._output_dir.text() or "."
        fmt_str = self._params.get_format_string()
        ext = {"cube": ".cube", "aml": ".aml", "alf4": ".alf4"}.get(fmt_str, ".cube")
        output_path, resolved_title = self._resolve_output_path_and_title(
            output_dir,
            self._output_title.text(),
            ext,
            "ChromaForge LUT",
        )
        shaper_text = self._params.shaper_mode.currentText()
        include_shaper = None
        if shaper_text == "on":
            include_shaper = True
        elif shaper_text == "off":
            include_shaper = False
        shadow_auto = self._params.shadow_auto.isChecked()
        shadow_threshold = None
        deep_shadow_threshold = None
        if not shadow_auto:
            shadow_threshold = self._params.shadow_threshold.value()
            deep_shadow_threshold = self._params.deep_shadow_threshold.value()

        return PipelineConfig(
            source_path=self._source_path,
            target_path=self._target_path,
            output_path=output_path,
            format=ExportFormat(fmt_str),
            title=resolved_title,
            lut_size=int(self._params.lut_size.currentText()),
            smoothness=self._params.smoothness.value(),
            prior_strength=self._params.prior_strength.value(),
            kernel=InterpolationKernel(self._params.kernel.currentText()),
            bin_resolution=self._params.bin_res.value(),
            min_samples_per_bin=self._params.min_samples.value(),
            robust_loss=RobustLoss(self._params.robust_loss.currentText()),
            irls_iterations=self._params.irls_iter.value(),
            transfer_function=TransferFunction(self._params.transfer_fn.currentText()),
            enable_refinement=self._params.enable_refine.isChecked(),
            include_shaper=include_shaper,
            shadow_auto=shadow_auto,
            shadow_threshold=shadow_threshold,
            deep_shadow_threshold=deep_shadow_threshold,
        )

    @Slot()
    def _cancel_extraction(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._log.append("Cancellation requested...", "warning")
            self._status_label.setText("Cancelling...")

    @Slot(str, float, str)
    def _on_progress(self, stage: str, fraction: float, message: str):
        self._progress.update_stage(stage, fraction, message)

    @Slot(str, str)
    def _on_log(self, message: str, severity: str):
        self._log.append(message, severity)

    @Slot(object)
    def _on_pipeline_done(self, result: PipelineResult):
        self._last_result = result
        self._progress.set_complete()
        self._metrics.update_metrics(result.metrics)

        if result.coverage_map is not None:
            self._coverage.set_distance_grid(result.coverage_map, result.lut.size)

        if result.output_path:
            self._log.append(f"Output: {result.output_path}", "success")
            self._status_label.setText(f"Complete: {result.output_path.name}")

        total_time = result.diagnostics.get("total_time", 0)
        if total_time > 0:
            self._log.append(f"Total time: {total_time:.2f}s", "info")

    @Slot(str)
    def _on_pipeline_error(self, error_msg: str):
        self._progress.set_error("solving", error_msg[:40])
        self._status_label.setText("Error")
        QMessageBox.warning(self, "Pipeline Error", error_msg)

    @Slot()
    def _on_worker_finished(self):
        self._btn_extract.setEnabled(
            self._source_path is not None and self._target_path is not None
        )
        self._btn_cancel.setEnabled(False)

    # ------------------------------------------------------------------
    # Hald CLUT workflow
    # ------------------------------------------------------------------

    @Slot()
    def _generate_hald_identity(self):
        """Generate and save a Hald identity image."""
        level = self._hald_level.value()

        last_out = self._settings.value("last_output_dir", "")
        default_name = Path(last_out) / f"hald_identity_L{level}.tiff" if last_out else f"hald_identity_L{level}.tiff"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Hald Identity Image",
            str(default_name),
            "TIFF (*.tiff *.tif);;PNG (*.png)",
        )
        if not path:
            return

        self._settings.setValue("last_output_dir", str(Path(path).parent))

        try:
            from chromaforge.hald.identity import (
                generate_hald_identity,
                hald_image_size,
                hald_lut_size,
            )
            from chromaforge.io.image import save_image

            identity = generate_hald_identity(level)
            save_image(identity, Path(path), bit_depth=16)

            img_sz = hald_image_size(level)
            lut_sz = hald_lut_size(level)
            self._hald_log.append(
                f"Generated Hald identity: {img_sz}x{img_sz}, "
                f"LUT size {lut_sz}^3",
                "success",
            )
            self._hald_log.append(f"Saved: {path}", "info")
            self._hald_log.append(
                "Process this image through your grading pipeline, "
                "then load the result below.",
                "info",
            )
        except Exception as e:
            self._hald_log.append(f"Error: {e}", "error")
            QMessageBox.warning(self, "Error", str(e))

    @Slot()
    def _browse_hald_input(self):
        last_dir = self._settings.value("last_image_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Processed Hald Image", last_dir,
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.exr);;All Files (*)",
        )
        if path:
            self._settings.setValue("last_image_dir", str(Path(path).parent))
            self._hald_processed_path.setText(path)

    @Slot()
    def _start_hald_reconstruction(self):
        processed = self._hald_processed_path.text()
        if not processed:
            return

        out_dir = self._output_dir.text() or "."
        output_path, resolved_title = self._resolve_output_path_and_title(
            out_dir,
            self._output_title.text(),
            ".cube",
            "ChromaForge Hald LUT",
        )

        level = self._hald_level.value()
        target_size = self._hald_target_size.value()

        self._btn_reconstruct.setEnabled(False)
        self._hald_log.append_separator()

        self._hald_worker = HaldWorker(
            processed_path=Path(processed),
            output_path=output_path,
            level=level,
            target_size=target_size,
            title=resolved_title,
            parent=self,
        )
        self._hald_worker.log_message.connect(
            lambda msg, sev: self._hald_log.append(msg, sev)
        )
        self._hald_worker.finished_ok.connect(self._on_hald_done)
        self._hald_worker.finished_error.connect(self._on_hald_error)
        self._hald_worker.finished.connect(
            lambda: self._btn_reconstruct.setEnabled(bool(self._hald_processed_path.text()))
        )
        self._hald_worker.start()

    @Slot(object)
    def _on_hald_done(self, result: dict):
        self._status_label.setText(f"Hald complete: {result.get('output_path', '')}")

    @Slot(str)
    def _on_hald_error(self, error_msg: str):
        self._status_label.setText("Hald error")
        QMessageBox.warning(self, "Hald Error", error_msg)

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------

    def _browse_output_dir(self):
        last_out = self._settings.value("last_output_dir", "")
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", last_out)
        if d:
            self._settings.setValue("last_output_dir", d)
            self._output_dir.setText(d)

    def _sanitize_title_for_filename(self, title: str, fallback: str) -> str:
        """Return a filesystem-safe title for output naming."""
        candidate = (title or "").strip()
        if not candidate:
            candidate = fallback
        candidate = re.sub(r'[<>:"/\\\\|?*]', "_", candidate)
        candidate = candidate.strip()
        if not candidate:
            candidate = fallback
        return candidate

    def _resolve_output_path_and_title(
        self,
        output_dir: str,
        title: str,
        ext: str,
        fallback_title: str,
    ) -> tuple[Path, str]:
        """Resolve a unique output path and matching title."""
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        safe_title = self._sanitize_title_for_filename(title, fallback_title)
        if safe_title.lower().endswith(ext.lower()):
            safe_title = safe_title[: -len(ext)].strip()

        out_dir = Path(output_dir) if output_dir else Path(".")
        candidate = out_dir / f"{safe_title}{ext}"
        if not candidate.exists():
            return candidate, safe_title

        match = re.match(r"^(.*?)(\d+)$", safe_title)
        if match:
            base = match.group(1)
            num = int(match.group(2))
        else:
            base = safe_title
            num = 0

        while True:
            num += 1
            new_stem = f"{base}{num}"
            candidate = out_dir / f"{new_stem}{ext}"
            if not candidate.exists():
                return candidate, new_stem

    def _detect_io_backend(self):
        """Detect available image I/O backends."""
        backends = []
        try:
            import OpenImageIO  # noqa: F401
            backends.append("OpenImageIO")
        except ImportError:
            pass
        try:
            import imageio  # noqa: F401
            backends.append("imageio")
        except ImportError:
            pass

        if backends:
            self._io_backend_label.setText(", ".join(backends))
        else:
            self._io_backend_label.setText("None detected")
            self._io_backend_label.setStyleSheet(f"color: {PALETTE.error};")
