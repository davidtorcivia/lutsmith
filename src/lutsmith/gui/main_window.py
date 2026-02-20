"""Main window for LutSmith GUI."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Optional

try:
    from PySide6.QtCore import Qt, Slot, QSettings, QUrl
    from PySide6.QtGui import QAction, QDesktopServices
    from PySide6.QtWidgets import (
        QMainWindow, QTabWidget, QWidget,
        QVBoxLayout, QHBoxLayout, QSplitter,
        QFileDialog, QMessageBox, QStatusBar,
        QPushButton, QLabel, QGroupBox,
        QFormLayout, QLineEdit, QSpinBox,
        QComboBox, QCheckBox, QScrollArea, QFrame, QDoubleSpinBox,
        QTableWidget, QTableWidgetItem, QHeaderView,
    )
except ImportError:
    raise ImportError("PySide6 is required for the GUI.")

from lutsmith import __version__
from lutsmith.core.types import (
    ColorBasis,
    ExportFormat,
    InterpolationKernel,
    PipelineConfig,
    PipelineResult,
    PriorModel,
    RobustLoss,
    TransferFunction,
)
from lutsmith.gui.styles.theme import PALETTE, SPACING_SM, SPACING_MD
from lutsmith.gui.widgets.coverage import CoverageViewer
from lutsmith.gui.widgets.image_pair import ImagePairViewer
from lutsmith.gui.widgets.log_viewer import LogViewer
from lutsmith.gui.widgets.metrics_view import MetricsDisplay
from lutsmith.gui.widgets.parameters import ParameterPanel
from lutsmith.gui.widgets.progress import PipelineProgress
from lutsmith.gui.workers import PipelineWorker, HaldWorker, BatchPipelineWorker
from lutsmith.pipeline.batch_manifest import parse_pair_manifest
from lutsmith.pipeline.reporting import build_batch_metrics_rows, write_batch_metrics_csv

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """LutSmith main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"LutSmith v{__version__}")
        self.setMinimumSize(1100, 720)
        self.resize(1400, 860)

        self._settings = QSettings("LutSmith", "LutSmith")
        self._worker: Optional[PipelineWorker] = None
        self._batch_worker: Optional[BatchPipelineWorker] = None
        self._hald_worker: Optional[HaldWorker] = None
        self._source_path: Optional[Path] = None
        self._target_path: Optional[Path] = None
        self._last_result: Optional[PipelineResult] = None
        self._batch_metrics_csv_output: Optional[Path] = None

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

        # Tab 2: Batch workflow
        self._tabs.addTab(self._build_batch_tab(), "Batch")

        # Tab 3: Hald CLUT workflow
        self._tabs.addTab(self._build_hald_tab(), "Hald CLUT")

        # Tab 4: Settings
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

    def _build_batch_tab(self) -> QWidget:
        """Build batch extraction tab with optional scene clustering."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        layout.setSpacing(SPACING_SM)

        manifest_group = QGroupBox("Batch Manifest")
        manifest_form = QFormLayout(manifest_group)
        manifest_form.setSpacing(SPACING_SM)

        manifest_row = QHBoxLayout()
        self._batch_manifest_path = QLineEdit()
        self._batch_manifest_path.setPlaceholderText(
            "pairs.csv (source,target[,weight][,cluster][,transfer_fn][,normalization])"
        )
        manifest_row.addWidget(self._batch_manifest_path, 1)
        self._btn_browse_batch_manifest = QPushButton("Browse...")
        self._btn_browse_batch_manifest.setFixedWidth(80)
        manifest_row.addWidget(self._btn_browse_batch_manifest)
        self._btn_batch_manifest_template = QPushButton("Template...")
        self._btn_batch_manifest_template.setFixedWidth(90)
        manifest_row.addWidget(self._btn_batch_manifest_template)
        manifest_form.addRow("Manifest:", manifest_row)

        self._batch_manifest_hint = QLabel(
            "Use one CSV line per pair. Optional columns: weight, cluster, transfer_fn, normalization."
        )
        self._batch_manifest_hint.setWordWrap(True)
        self._batch_manifest_hint.setStyleSheet(f"color: {PALETTE.text_secondary}; font-size: 11px;")
        manifest_form.addRow("", self._batch_manifest_hint)
        layout.addWidget(manifest_group)

        cluster_group = QGroupBox("Scene Clustering")
        cluster_form = QFormLayout(cluster_group)
        cluster_form.setSpacing(SPACING_SM)

        self._batch_cluster_mode = QComboBox()
        self._batch_cluster_mode.addItems(["none", "manual", "auto"])
        self._batch_cluster_mode.setToolTip("none: single LUT, manual: use manifest cluster column, auto: feature-based clustering")
        cluster_form.addRow("Mode:", self._batch_cluster_mode)

        self._batch_cluster_count = QSpinBox()
        self._batch_cluster_count.setRange(0, 20)
        self._batch_cluster_count.setSpecialValueText("Auto")
        self._batch_cluster_count.setToolTip("Fixed clusters for auto mode (0 chooses automatically)")
        cluster_form.addRow("Cluster Count:", self._batch_cluster_count)

        self._batch_max_clusters = QSpinBox()
        self._batch_max_clusters.setRange(2, 20)
        self._batch_max_clusters.setValue(6)
        self._batch_max_clusters.setToolTip("Maximum clusters considered when Cluster Count is Auto")
        cluster_form.addRow("Max Clusters:", self._batch_max_clusters)

        self._batch_cluster_seed = QSpinBox()
        self._batch_cluster_seed.setRange(0, 999999)
        self._batch_cluster_seed.setValue(42)
        self._batch_cluster_seed.setToolTip("Random seed for auto clustering")
        cluster_form.addRow("Cluster Seed:", self._batch_cluster_seed)

        self._batch_export_master = QCheckBox("Export master LUT (all pairs)")
        self._batch_export_master.setChecked(True)
        cluster_form.addRow("", self._batch_export_master)

        layout.addWidget(cluster_group)

        robustness_group = QGroupBox("Robustness")
        robust_form = QFormLayout(robustness_group)
        robust_form.setSpacing(SPACING_SM)

        self._batch_pair_balance = QComboBox()
        self._batch_pair_balance.addItems(["equal", "by_bins", "by_pixels"])
        self._batch_pair_balance.setToolTip("How each pair contributes to the aggregate fit")
        robust_form.addRow("Pair Balance:", self._batch_pair_balance)

        self._batch_outlier_sigma = QDoubleSpinBox()
        self._batch_outlier_sigma.setRange(0.0, 10.0)
        self._batch_outlier_sigma.setSingleStep(0.1)
        self._batch_outlier_sigma.setValue(0.0)
        self._batch_outlier_sigma.setToolTip("Outlier pair rejection threshold (median + sigma*MAD). 0 disables")
        robust_form.addRow("Outlier Sigma:", self._batch_outlier_sigma)

        self._batch_min_pairs = QSpinBox()
        self._batch_min_pairs.setRange(1, 1000)
        self._batch_min_pairs.setValue(3)
        self._batch_min_pairs.setToolTip("Minimum pairs retained after outlier rejection")
        robust_form.addRow("Min Pairs:", self._batch_min_pairs)

        self._batch_allow_mixed_tf = QCheckBox("Allow mixed transfer-function detections")
        robust_form.addRow("", self._batch_allow_mixed_tf)
        layout.addWidget(robustness_group)

        metrics_group = QGroupBox("Batch Metrics Export")
        metrics_form = QFormLayout(metrics_group)
        metrics_form.setSpacing(SPACING_SM)

        self._batch_export_metrics_csv = QCheckBox("Export metrics CSV")
        self._batch_export_metrics_csv.setChecked(True)
        self._batch_export_metrics_csv.setToolTip("Write master/cluster quality summary to CSV")
        metrics_form.addRow("", self._batch_export_metrics_csv)

        csv_row = QHBoxLayout()
        self._batch_metrics_csv_path = QLineEdit()
        self._batch_metrics_csv_path.setPlaceholderText("Auto: <output_stem>_batch_metrics.csv")
        csv_row.addWidget(self._batch_metrics_csv_path, 1)
        self._btn_browse_batch_metrics_csv = QPushButton("Browse...")
        self._btn_browse_batch_metrics_csv.setFixedWidth(80)
        csv_row.addWidget(self._btn_browse_batch_metrics_csv)
        metrics_form.addRow("CSV Path:", csv_row)

        layout.addWidget(metrics_group)

        self._batch_hint = QLabel(
            "Batch uses the same solver parameters as the Image Pair tab. "
            "Adjust LUT size, prior model, basis, and smoothing there."
        )
        self._batch_hint.setWordWrap(True)
        self._batch_hint.setStyleSheet(f"color: {PALETTE.text_secondary}; font-size: 11px;")
        layout.addWidget(self._batch_hint)

        self._batch_progress = PipelineProgress()
        layout.addWidget(self._batch_progress)

        self._batch_metrics = MetricsDisplay()
        layout.addWidget(self._batch_metrics)

        summary_group = QGroupBox("Batch Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setSpacing(SPACING_SM)

        self._batch_summary_table = QTableWidget(0, 6)
        self._batch_summary_table.setHorizontalHeaderLabels(
            ["Label", "Mean dE", "Coverage %", "Time (s)", "Pairs", "Output"]
        )
        self._batch_summary_table.setSortingEnabled(True)
        self._batch_summary_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._batch_summary_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._batch_summary_table.verticalHeader().setVisible(False)
        header = self._batch_summary_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        summary_layout.addWidget(self._batch_summary_table)

        summary_btn_row = QHBoxLayout()
        summary_btn_row.addStretch()
        self._btn_open_batch_output_folder = QPushButton("Open Output Folder")
        self._btn_open_batch_output_folder.setEnabled(False)
        summary_btn_row.addWidget(self._btn_open_batch_output_folder)
        summary_layout.addLayout(summary_btn_row)
        layout.addWidget(summary_group)

        self._batch_log = LogViewer()
        layout.addWidget(self._batch_log, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(SPACING_SM)
        self._btn_batch_extract = QPushButton("Extract Batch LUTs")
        self._btn_batch_extract.setProperty("class", "primary")
        self._btn_batch_extract.setMinimumHeight(36)
        self._btn_batch_cancel = QPushButton("Cancel")
        self._btn_batch_cancel.setEnabled(False)
        self._btn_batch_cancel.setMinimumHeight(36)
        btn_row.addWidget(self._btn_batch_extract)
        btn_row.addWidget(self._btn_batch_cancel)
        layout.addLayout(btn_row)

        self._update_batch_cluster_controls()
        self._update_batch_metrics_controls()
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

        self._output_title = QLineEdit("LutSmith LUT")
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
            f"LutSmith v{__version__}\n"
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

        # Batch tab
        self._btn_browse_batch_manifest.clicked.connect(self._browse_batch_manifest)
        self._btn_batch_manifest_template.clicked.connect(self._save_batch_manifest_template)
        self._btn_browse_batch_metrics_csv.clicked.connect(self._browse_batch_metrics_csv)
        self._btn_batch_extract.clicked.connect(self._start_batch_extraction)
        self._btn_batch_cancel.clicked.connect(self._cancel_batch_extraction)
        self._batch_cluster_mode.currentTextChanged.connect(self._update_batch_cluster_controls)
        self._batch_export_metrics_csv.stateChanged.connect(self._update_batch_metrics_controls)
        self._batch_summary_table.itemSelectionChanged.connect(self._update_batch_output_folder_button)
        self._btn_open_batch_output_folder.clicked.connect(self._open_selected_batch_output_folder)

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
        config = self._build_config(self._source_path, self._target_path)

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

    def _build_config(
        self,
        source_path: Optional[Path],
        target_path: Optional[Path],
    ) -> PipelineConfig:
        """Assemble PipelineConfig from current GUI state."""
        output_dir = self._output_dir.text() or "."
        fmt_str = self._params.get_format_string()
        ext = {"cube": ".cube", "aml": ".aml", "alf4": ".alf4"}.get(fmt_str, ".cube")
        output_path, resolved_title = self._resolve_output_path_and_title(
            output_dir,
            self._output_title.text(),
            ext,
            "LutSmith LUT",
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
            source_path=source_path,
            target_path=target_path,
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
            prior_model=PriorModel(self._params.prior_model.currentText()),
            color_basis=ColorBasis(self._params.color_basis.currentText()),
            chroma_smoothness_ratio=self._params.chroma_ratio.value(),
            laplacian_connectivity=int(self._params.laplacian_connectivity.currentText()),
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
    # Batch execution
    # ------------------------------------------------------------------

    def _update_batch_cluster_controls(self):
        mode = self._batch_cluster_mode.currentText().strip().lower()
        auto = mode == "auto"
        self._batch_cluster_count.setEnabled(auto)
        self._batch_max_clusters.setEnabled(auto)
        self._batch_cluster_seed.setEnabled(auto)

    def _update_batch_metrics_controls(self):
        enabled = self._batch_export_metrics_csv.isChecked()
        self._batch_metrics_csv_path.setEnabled(enabled)
        self._btn_browse_batch_metrics_csv.setEnabled(enabled)

    def _default_batch_metrics_csv_path(self, config: PipelineConfig) -> Path:
        output_path = Path(config.output_path) if config.output_path is not None else Path("output.cube")
        return output_path.with_name(f"{output_path.stem}_batch_metrics.csv")

    @Slot()
    def _browse_batch_manifest(self):
        last_dir = self._settings.value("last_image_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Batch Manifest",
            last_dir,
            "CSV (*.csv);;All Files (*)",
        )
        if path:
            self._settings.setValue("last_image_dir", str(Path(path).parent))
            self._batch_manifest_path.setText(path)

    @Slot()
    def _save_batch_manifest_template(self):
        last_dir = self._settings.value("last_output_dir", "")
        default_name = Path(last_dir) / "pairs_template.csv" if last_dir else "pairs_template.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Batch Manifest Template",
            str(default_name),
            "CSV (*.csv);;All Files (*)",
        )
        if not path:
            return

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        template = (
            "source,target,weight,cluster,transfer_fn,normalization\n"
            "restored/frame_0001.png,original/frame_0001.png,1.0,scene_a,log_c4,none\n"
            "restored/frame_0002.png,original/frame_0002.png,1.0,scene_a,log_c4,luma_affine\n"
            "restored/frame_0003.png,original/frame_0003.png,0.8,scene_b,auto,rgb_affine\n"
        )
        out.write_text(template, encoding="utf-8")
        self._settings.setValue("last_output_dir", str(out.parent))
        self._batch_log.append(f"Manifest template saved: {out}", "success")

    @Slot()
    def _browse_batch_metrics_csv(self):
        last_dir = self._settings.value("last_output_dir", "")
        default_name = Path(last_dir) / "batch_metrics.csv" if last_dir else "batch_metrics.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Batch Metrics CSV",
            str(default_name),
            "CSV (*.csv);;All Files (*)",
        )
        if path:
            self._settings.setValue("last_output_dir", str(Path(path).parent))
            self._batch_metrics_csv_path.setText(path)

    @Slot()
    def _start_batch_extraction(self):
        manifest_text = self._batch_manifest_path.text().strip()
        if not manifest_text:
            QMessageBox.warning(self, "Batch Extraction", "Select a manifest CSV first.")
            return

        manifest_path = Path(manifest_text)
        try:
            entries = parse_pair_manifest(manifest_path)
        except Exception as e:
            QMessageBox.warning(self, "Manifest Error", str(e))
            return

        if len(entries) < 1:
            QMessageBox.warning(self, "Batch Extraction", "Manifest has no valid pairs.")
            return

        config = self._build_config(None, None)
        mode = self._batch_cluster_mode.currentText().strip().lower()
        cluster_count = self._batch_cluster_count.value()
        max_clusters = self._batch_max_clusters.value()
        cluster_seed = self._batch_cluster_seed.value()
        export_master = self._batch_export_master.isChecked()
        pair_balance = self._batch_pair_balance.currentText()
        outlier_sigma = self._batch_outlier_sigma.value()
        min_pairs = self._batch_min_pairs.value()
        allow_mixed = self._batch_allow_mixed_tf.isChecked()
        export_metrics_csv = self._batch_export_metrics_csv.isChecked()
        metrics_path_text = self._batch_metrics_csv_path.text().strip()
        self._batch_metrics_csv_output = None
        if export_metrics_csv:
            self._batch_metrics_csv_output = (
                Path(metrics_path_text)
                if metrics_path_text
                else self._default_batch_metrics_csv_path(config)
            )
            self._batch_log.append(f"Metrics CSV: {self._batch_metrics_csv_output}", "info")

        self._batch_log.append_separator()
        self._batch_log.append(f"Manifest: {manifest_path}", "info")
        self._batch_log.append(f"Pairs: {len(entries)}", "info")
        self._batch_log.append(f"Cluster mode: {mode}", "info")
        self._batch_progress.reset()
        self._batch_metrics.clear()
        self._batch_summary_table.setRowCount(0)
        self._update_batch_output_folder_button()

        self._btn_batch_extract.setEnabled(False)
        self._btn_batch_cancel.setEnabled(True)
        self._status_label.setText("Running batch extraction...")

        self._batch_worker = BatchPipelineWorker(
            entries,
            config,
            cluster_mode=mode,
            cluster_count=cluster_count,
            max_clusters=max_clusters,
            export_master=export_master,
            cluster_seed=cluster_seed,
            pair_balance=pair_balance,
            outlier_sigma=outlier_sigma,
            min_pairs_after_outlier=min_pairs,
            allow_mixed_transfer=allow_mixed,
            parent=self,
        )
        self._batch_worker.progress_updated.connect(self._on_batch_progress)
        self._batch_worker.log_message.connect(self._on_batch_log)
        self._batch_worker.finished_ok.connect(self._on_batch_done)
        self._batch_worker.finished_error.connect(self._on_batch_error)
        self._batch_worker.finished.connect(self._on_batch_worker_finished)
        self._batch_worker.start()

    @Slot()
    def _cancel_batch_extraction(self):
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.cancel()
            self._batch_log.append("Cancellation requested...", "warning")
            self._status_label.setText("Cancelling batch...")

    @Slot(str, float, str)
    def _on_batch_progress(self, stage: str, fraction: float, message: str):
        self._batch_progress.update_stage(stage, fraction, message)

    @Slot(str, str)
    def _on_batch_log(self, message: str, severity: str):
        self._batch_log.append(message, severity)

    def _populate_batch_summary_table(self, master, clusters: list):
        rows = build_batch_metrics_rows(master, clusters)
        self._batch_summary_table.setSortingEnabled(False)
        self._batch_summary_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            label_item = QTableWidgetItem(row.get("label", ""))
            self._batch_summary_table.setItem(r, 0, label_item)

            mean_text = row.get("mean_de2000", "")
            mean_item = QTableWidgetItem(mean_text)
            try:
                mean_item.setData(Qt.ItemDataRole.UserRole, float(mean_text))
            except Exception:
                pass
            self._batch_summary_table.setItem(r, 1, mean_item)

            cov_text = row.get("coverage_pct", "")
            cov_item = QTableWidgetItem(cov_text)
            try:
                cov_item.setData(Qt.ItemDataRole.UserRole, float(cov_text))
            except Exception:
                pass
            self._batch_summary_table.setItem(r, 2, cov_item)

            time_text = row.get("total_time_s", "")
            time_item = QTableWidgetItem(time_text)
            try:
                time_item.setData(Qt.ItemDataRole.UserRole, float(time_text))
            except Exception:
                pass
            self._batch_summary_table.setItem(r, 3, time_item)

            pairs_item = QTableWidgetItem(row.get("num_pairs_used", "") or row.get("num_pairs", ""))
            self._batch_summary_table.setItem(r, 4, pairs_item)

            output_path = row.get("output_path", "")
            output_item = QTableWidgetItem(output_path)
            output_item.setData(Qt.ItemDataRole.UserRole, output_path)
            self._batch_summary_table.setItem(r, 5, output_item)

        self._batch_summary_table.setSortingEnabled(True)
        if rows:
            self._batch_summary_table.selectRow(0)
        self._update_batch_output_folder_button()

    def _selected_batch_output_path(self) -> Optional[Path]:
        row = self._batch_summary_table.currentRow()
        if row < 0:
            return None
        item = self._batch_summary_table.item(row, 5)
        if item is None:
            return None
        path_text = item.data(Qt.ItemDataRole.UserRole) or item.text()
        if not path_text:
            return None
        return Path(str(path_text))

    def _update_batch_output_folder_button(self):
        selected = self._selected_batch_output_path()
        self._btn_open_batch_output_folder.setEnabled(selected is not None)

    @Slot()
    def _open_selected_batch_output_folder(self):
        selected = self._selected_batch_output_path()
        if selected is None:
            return
        folder = selected.parent if selected.suffix else selected
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    @Slot(object)
    def _on_batch_done(self, payload: dict):
        self._batch_progress.set_complete()
        master = payload.get("master")
        clusters = payload.get("clusters", [])
        self._populate_batch_summary_table(master, clusters)

        chosen = master
        if chosen is None and clusters:
            chosen = clusters[0]

        if chosen is not None:
            self._batch_metrics.update_metrics(chosen.metrics)

        if master and master.output_path:
            self._batch_log.append(f"Master LUT: {master.output_path}", "success")
            self._status_label.setText(f"Batch complete: {master.output_path.name}")

        for cluster_result in clusters:
            label = cluster_result.diagnostics.get("cluster_label", "cluster")
            if cluster_result.output_path:
                self._batch_log.append(f"{label}: {cluster_result.output_path}", "success")
                if master is None:
                    self._status_label.setText(f"Batch complete: {cluster_result.output_path.name}")
            self._batch_log.append(
                f"{label} mean dE={cluster_result.metrics.mean_delta_e:.2f}",
                "info",
            )

        clustering = payload.get("clustering", {})
        if clustering:
            k = clustering.get("k")
            mode = clustering.get("mode", "n/a")
            self._batch_log.append(f"Clustering summary: mode={mode}, k={k}", "info")

        if self._batch_metrics_csv_output is not None:
            try:
                rows = build_batch_metrics_rows(master, clusters)
                csv_path = write_batch_metrics_csv(rows, self._batch_metrics_csv_output)
                self._batch_log.append(f"Metrics CSV saved: {csv_path}", "success")
            except Exception as e:
                self._batch_log.append(f"Failed to write metrics CSV: {e}", "error")

    @Slot(str)
    def _on_batch_error(self, error_msg: str):
        self._batch_progress.set_error("solving", error_msg[:40])
        self._status_label.setText("Batch error")
        self._update_batch_output_folder_button()
        QMessageBox.warning(self, "Batch Pipeline Error", error_msg)

    @Slot()
    def _on_batch_worker_finished(self):
        self._btn_batch_extract.setEnabled(True)
        self._btn_batch_cancel.setEnabled(False)
        self._batch_metrics_csv_output = None

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
            from lutsmith.hald.identity import (
                generate_hald_identity,
                hald_image_size,
                hald_lut_size,
            )
            from lutsmith.io.image import save_image

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
            "LutSmith Hald LUT",
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
