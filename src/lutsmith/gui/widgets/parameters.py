"""Parameter panel with grouped controls for LUT extraction settings."""

from __future__ import annotations

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
        QGroupBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
        QSlider, QCheckBox, QPushButton,
    )
except ImportError:
    raise ImportError("PySide6 is required for the GUI.")

from lutsmith.gui.styles.theme import PALETTE, SPACING_SM, SPACING_MD


class ParameterPanel(QWidget):
    """Parameter controls for LUT extraction configuration."""

    parameters_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _tip_label(self, text: str, tooltip: str) -> QLabel:
        """Create a form row label with a detailed tooltip."""
        label = QLabel(text)
        label.setToolTip(tooltip)
        label.setCursor(Qt.CursorShape.WhatsThisCursor)
        return label

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_MD, SPACING_SM, SPACING_MD, SPACING_SM)
        layout.setSpacing(SPACING_SM)

        # Essential parameters
        essential = QGroupBox("Parameters")
        form = QFormLayout(essential)
        form.setSpacing(SPACING_SM)

        # LUT Size
        lut_size_tip = (
            "Number of nodes per axis in the output 3D LUT.\n\n"
            "17: Smaller file, faster processing. Good for subtle grades\n"
            "or previews. 4,913 total nodes.\n\n"
            "33: Standard quality. Balances accuracy and file size.\n"
            "35,937 total nodes. Recommended for most workflows.\n\n"
            "65: Maximum precision. Best for complex transforms with\n"
            "strong color shifts, but 4x larger files. 274,625 total nodes."
        )
        self.lut_size = QComboBox()
        self.lut_size.addItems(["17", "33", "65"])
        self.lut_size.setCurrentText("33")
        self.lut_size.setToolTip(lut_size_tip)
        form.addRow(self._tip_label("LUT Size:", lut_size_tip), self.lut_size)

        # Kernel
        kernel_tip = (
            "Interpolation method used to map pixels to LUT nodes\n"
            "during both fitting and validation.\n\n"
            "Tetrahedral: Splits each cube cell into 6 tetrahedra.\n"
            "Uses 4 vertices per lookup instead of 8, producing\n"
            "smoother results with fewer artifacts. Industry standard\n"
            "for color grading. Recommended.\n\n"
            "Trilinear: Classic 8-corner cube interpolation. Slightly\n"
            "faster but can show subtle banding on smooth gradients."
        )
        self.kernel = QComboBox()
        self.kernel.addItems(["tetrahedral", "trilinear"])
        self.kernel.setToolTip(kernel_tip)
        form.addRow(self._tip_label("Kernel:", kernel_tip), self.kernel)

        # Smoothness
        smooth_tip = (
            "Controls how much the solver penalizes abrupt changes\n"
            "between neighboring LUT nodes (Laplacian regularization).\n\n"
            "Higher values produce smoother, more gradual LUTs that\n"
            "are less likely to introduce banding or artifacts, but\n"
            "may lose fine detail in the color transform.\n\n"
            "Lower values allow the LUT to follow the data more\n"
            "closely, capturing subtle color variations but risking\n"
            "noise in areas with sparse pixel coverage.\n\n"
            "0.01 - 0.05: Aggressive fit, minimal smoothing\n"
            "0.1: Balanced (default)\n"
            "0.5 - 1.0: Very smooth, good for noisy or sparse data"
        )
        smooth_row = QHBoxLayout()
        self.smoothness = QDoubleSpinBox()
        self.smoothness.setRange(0.001, 10.0)
        self.smoothness.setSingleStep(0.01)
        self.smoothness.setValue(0.1)
        self.smoothness.setDecimals(3)
        self.smoothness.setToolTip(smooth_tip)
        self.smoothness_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothness_slider.setRange(1, 1000)
        self.smoothness_slider.setValue(100)
        self.smoothness_slider.setToolTip(smooth_tip)
        self.smoothness_slider.valueChanged.connect(
            lambda v: self.smoothness.setValue(v / 1000.0)
        )
        self.smoothness.valueChanged.connect(
            lambda v: self.smoothness_slider.setValue(int(v * 1000))
        )
        smooth_row.addWidget(self.smoothness_slider, 1)
        smooth_row.addWidget(self.smoothness)
        form.addRow(self._tip_label("Smoothness:", smooth_tip), smooth_row)

        # Format
        format_tip = (
            "Output file format for the generated LUT.\n\n"
            ".cube: Industry-standard format supported by DaVinci\n"
            "Resolve, Adobe Premiere, Final Cut Pro, and most color\n"
            "grading software. Supports optional 1D shaper LUTs.\n\n"
            ".aml (ARRI ALF2): ARRI Look Metadata format for use\n"
            "with ARRI cameras and ARRI workflows.\n\n"
            ".alf4 (ARRI ALF4): Newer ARRI Look File format with\n"
            "extended metadata support."
        )
        self.format = QComboBox()
        self.format.addItems([".cube", ".aml (ARRI ALF2)", ".alf4 (ARRI ALF4)"])
        self.format.setToolTip(format_tip)
        form.addRow(self._tip_label("Format:", format_tip), self.format)

        layout.addWidget(essential)

        # Advanced parameters (collapsible)
        self._advanced_visible = False
        self._advanced_toggle = QPushButton("> Advanced")
        self._advanced_toggle.setFlat(True)
        self._advanced_toggle.setStyleSheet(
            f"text-align: left; color: {PALETTE.text_secondary}; "
            f"font-size: 12px; padding: 4px;"
        )
        self._advanced_toggle.clicked.connect(self._toggle_advanced)
        layout.addWidget(self._advanced_toggle)

        self._advanced_group = QGroupBox()
        self._advanced_group.setVisible(False)
        adv_form = QFormLayout(self._advanced_group)
        adv_form.setSpacing(SPACING_SM)

        # Prior strength
        prior_tip = (
            "How strongly unobserved LUT nodes are pulled toward\n"
            "the identity (no-change) transform.\n\n"
            "In regions of color space where your images have no\n"
            "pixels (e.g., pure saturated colors), the solver has\n"
            "no data to work with. The prior fills these gaps by\n"
            "blending toward a neutral identity mapping.\n\n"
            "Nodes close to observed data are barely affected;\n"
            "nodes far from any data are pulled more strongly.\n\n"
            "0.001: Minimal prior, trust extrapolation\n"
            "0.01: Balanced (default)\n"
            "0.1 - 1.0: Strong prior, safer for sparse data"
        )
        self.prior_strength = QDoubleSpinBox()
        self.prior_strength.setRange(0.0, 1.0)
        self.prior_strength.setSingleStep(0.001)
        self.prior_strength.setValue(0.01)
        self.prior_strength.setDecimals(3)
        self.prior_strength.setToolTip(prior_tip)
        adv_form.addRow(self._tip_label("Prior Strength:", prior_tip), self.prior_strength)

        # Robust loss
        loss_tip = (
            "Loss function for the regression solver. Determines\n"
            "how pixel mismatches are penalized.\n\n"
            "Huber: Combines L2 (squared error) for small residuals\n"
            "with L1 (absolute error) for large ones. This makes\n"
            "the solver robust to outliers - pixels that don't fit\n"
            "the color transform (lens flares, text overlays, etc.)\n"
            "won't distort the LUT. Recommended.\n\n"
            "L2: Pure least-squares. Faster (no IRLS iterations\n"
            "needed) but sensitive to outliers. Use when you're\n"
            "confident the image pair has no artifacts."
        )
        self.robust_loss = QComboBox()
        self.robust_loss.addItems(["huber", "l2"])
        self.robust_loss.setToolTip(loss_tip)
        adv_form.addRow(self._tip_label("Robust Loss:", loss_tip), self.robust_loss)

        # IRLS iterations
        irls_tip = (
            "Number of Iteratively Reweighted Least Squares passes\n"
            "when using Huber loss.\n\n"
            "Each iteration identifies pixels with large errors,\n"
            "downweights them, and re-solves. This progressively\n"
            "reduces the influence of outliers.\n\n"
            "0: Skip IRLS entirely (equivalent to L2 loss)\n"
            "1 - 2: Light outlier rejection\n"
            "3: Standard (default). Usually sufficient.\n"
            "5 - 10: Aggressive outlier rejection for noisy data.\n"
            "More iterations increase processing time linearly."
        )
        self.irls_iter = QSpinBox()
        self.irls_iter.setRange(0, 10)
        self.irls_iter.setValue(3)
        self.irls_iter.setToolTip(irls_tip)
        adv_form.addRow(self._tip_label("IRLS Iterations:", irls_tip), self.irls_iter)

        # Bin resolution
        bin_tip = (
            "Resolution of the 3D histogram used to aggregate pixels\n"
            "before solving. Source pixels are sorted into a grid of\n"
            "this size per color axis.\n\n"
            "Each bin accumulates the mean input/output color and\n"
            "variance of all pixels that fall within it. This reduces\n"
            "millions of pixels to a manageable number of samples.\n\n"
            "32: Coarser binning, faster. May lose subtle gradations.\n"
            "64: Standard (default). Good balance of detail and speed.\n"
            "128: Fine binning, captures subtle color variations but\n"
            "uses more memory and may produce noisier estimates\n"
            "in bins with few pixels."
        )
        self.bin_res = QSpinBox()
        self.bin_res.setRange(16, 128)
        self.bin_res.setValue(64)
        self.bin_res.setToolTip(bin_tip)
        adv_form.addRow(self._tip_label("Bin Resolution:", bin_tip), self.bin_res)

        # Min samples
        min_samples_tip = (
            "Minimum number of pixels required in a bin for it to\n"
            "be used as a data point in the solver.\n\n"
            "Bins with fewer pixels than this threshold are discarded.\n"
            "This prevents noisy single-pixel observations from\n"
            "distorting the LUT.\n\n"
            "1: Use every bin, even with a single pixel. Maximizes\n"
            "coverage but may introduce noise.\n"
            "3: Standard (default). Filters the noisiest bins.\n"
            "10 - 50: Conservative. Only uses well-sampled bins.\n"
            "Good for very noisy images."
        )
        self.min_samples = QSpinBox()
        self.min_samples.setRange(1, 100)
        self.min_samples.setValue(3)
        self.min_samples.setToolTip(min_samples_tip)
        adv_form.addRow(self._tip_label("Min Samples:", min_samples_tip), self.min_samples)

        # Shadow thresholds
        shadow_tip = (
            "Adaptive thresholds that control shadow weighting and\n"
            "post-solve shadow smoothing. Useful for low-contrast\n"
            "or log-like inputs where noise concentrates in darks.\n\n"
            "Auto: Estimate thresholds from input luminance.\n"
            "Manual: Use the values below.\n\n"
            "Shadow Threshold: Upper luminance for shadow smoothing.\n"
            "Deep Threshold: Boundary for strongest smoothing.\n"
            "Typical defaults: shadow=0.25, deep=0.08."
        )
        self.shadow_auto = QCheckBox("Adaptive shadow thresholds")
        self.shadow_auto.setChecked(True)
        self.shadow_auto.setToolTip(shadow_tip)
        adv_form.addRow(self._tip_label("", shadow_tip), self.shadow_auto)

        self.shadow_threshold = QDoubleSpinBox()
        self.shadow_threshold.setRange(0.0, 1.0)
        self.shadow_threshold.setSingleStep(0.01)
        self.shadow_threshold.setValue(0.25)
        self.shadow_threshold.setDecimals(3)
        self.shadow_threshold.setToolTip(shadow_tip)
        adv_form.addRow(self._tip_label("Shadow Threshold:", shadow_tip), self.shadow_threshold)

        self.deep_shadow_threshold = QDoubleSpinBox()
        self.deep_shadow_threshold.setRange(0.0, 1.0)
        self.deep_shadow_threshold.setSingleStep(0.01)
        self.deep_shadow_threshold.setValue(0.08)
        self.deep_shadow_threshold.setDecimals(3)
        self.deep_shadow_threshold.setToolTip(shadow_tip)
        adv_form.addRow(self._tip_label("Deep Threshold:", shadow_tip), self.deep_shadow_threshold)

        self.shadow_auto.stateChanged.connect(self._toggle_shadow_controls)
        self._toggle_shadow_controls()

        # Refinement
        refine_tip = (
            "After the initial solve, apply the LUT to the source\n"
            "samples, measure residual errors, downweight the worst\n"
            "outliers, and re-solve.\n\n"
            "This can improve accuracy when the image pair contains\n"
            "localized artifacts (reflections, motion blur, text)\n"
            "that the initial IRLS pass didn't fully suppress.\n\n"
            "Increases processing time roughly proportional to the\n"
            "number of refinement passes."
        )
        self.enable_refine = QCheckBox("Enable iterative refinement")
        self.enable_refine.setToolTip(refine_tip)
        adv_form.addRow(self._tip_label("", refine_tip), self.enable_refine)

        # Transfer function
        transfer_tip = (
            "The encoding curve (OETF) of the source image. This\n"
            "tells LutSmith how pixel values map to light levels.\n\n"
            "Auto: Detect from image metadata and value distribution.\n"
            "Recommended for most workflows.\n\n"
            "Linear: Scene-linear / raw sensor data (e.g., EXR files\n"
            "from CG renders or camera RAW). Values can exceed 1.0.\n\n"
            "Log C3 / Log C4: ARRI LogC encoding. Common in\n"
            "professional cinema workflows.\n\n"
            "S-Log3: Sony's log encoding for cinema cameras.\n\n"
            "V-Log: Panasonic's log encoding.\n\n"
            "When a log encoding is detected, a 1D shaper LUT may\n"
            "be generated to linearize the input before the 3D LUT."
        )
        self.transfer_fn = QComboBox()
        self.transfer_fn.addItems(["auto", "linear", "log_c3", "log_c4", "slog3", "vlog"])
        self.transfer_fn.setToolTip(transfer_tip)
        adv_form.addRow(self._tip_label("Transfer Fn:", transfer_tip), self.transfer_fn)

        # Shaper control
        shaper_tip = (
            "Controls whether a 1D shaper (pre-LUT) curve is included\n"
            "in the output file.\n\n"
            "A shaper linearizes non-linear input encodings (like log)\n"
            "before the 3D LUT, which improves interpolation accuracy\n"
            "for log-encoded footage.\n\n"
            "Auto: Include a shaper only when a log transfer function\n"
            "is detected. Recommended.\n\n"
            "On: Always generate a shaper. Useful if you know your\n"
            "input is log-encoded but auto-detection failed.\n\n"
            "Off: Never include a shaper. Use when your source is\n"
            "already linear or standard gamma.\n\n"
            "Note: Only the .cube format supports embedded shapers."
        )
        self.shaper_mode = QComboBox()
        self.shaper_mode.addItems(["auto", "on", "off"])
        self.shaper_mode.setToolTip(shaper_tip)
        adv_form.addRow(self._tip_label("Shaper:", shaper_tip), self.shaper_mode)

        layout.addWidget(self._advanced_group)

        # Reset button
        reset_row = QHBoxLayout()
        reset_row.addStretch()
        self._btn_reset = QPushButton("Reset Defaults")
        self._btn_reset.clicked.connect(self.reset_defaults)
        reset_row.addWidget(self._btn_reset)
        layout.addLayout(reset_row)

    def _toggle_advanced(self):
        self._advanced_visible = not self._advanced_visible
        self._advanced_group.setVisible(self._advanced_visible)
        self._advanced_toggle.setText(
            "v Advanced" if self._advanced_visible else "> Advanced"
        )

    def reset_defaults(self):
        self.lut_size.setCurrentText("33")
        self.kernel.setCurrentIndex(0)
        self.smoothness.setValue(0.1)
        self.format.setCurrentIndex(0)
        self.prior_strength.setValue(0.01)
        self.robust_loss.setCurrentIndex(0)
        self.irls_iter.setValue(3)
        self.bin_res.setValue(64)
        self.min_samples.setValue(3)
        self.shadow_auto.setChecked(True)
        self.shadow_threshold.setValue(0.25)
        self.deep_shadow_threshold.setValue(0.08)
        self.enable_refine.setChecked(False)
        self.transfer_fn.setCurrentIndex(0)
        self.shaper_mode.setCurrentIndex(0)

    def get_format_string(self) -> str:
        text = self.format.currentText()
        if ".cube" in text:
            return "cube"
        elif ".aml" in text:
            return "aml"
        elif ".alf4" in text:
            return "alf4"
        return "cube"

    def _toggle_shadow_controls(self):
        manual = not self.shadow_auto.isChecked()
        self.shadow_threshold.setEnabled(manual)
        self.deep_shadow_threshold.setEnabled(manual)
