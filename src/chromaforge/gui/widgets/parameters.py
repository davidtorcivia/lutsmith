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

from chromaforge.gui.styles.theme import PALETTE, SPACING_SM, SPACING_MD


class ParameterPanel(QWidget):
    """Parameter controls for LUT extraction configuration."""

    parameters_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_MD, SPACING_SM, SPACING_MD, SPACING_SM)
        layout.setSpacing(SPACING_SM)

        # Essential parameters
        essential = QGroupBox("Parameters")
        form = QFormLayout(essential)
        form.setSpacing(SPACING_SM)

        # LUT Size
        self.lut_size = QComboBox()
        self.lut_size.addItems(["17", "33", "65"])
        self.lut_size.setCurrentText("33")
        self.lut_size.setToolTip("Output LUT resolution (nodes per axis)")
        form.addRow("LUT Size:", self.lut_size)

        # Kernel
        self.kernel = QComboBox()
        self.kernel.addItems(["tetrahedral", "trilinear"])
        self.kernel.setToolTip("Interpolation kernel for fitting and validation")
        form.addRow("Kernel:", self.kernel)

        # Smoothness
        smooth_row = QHBoxLayout()
        self.smoothness = QDoubleSpinBox()
        self.smoothness.setRange(0.001, 10.0)
        self.smoothness.setSingleStep(0.01)
        self.smoothness.setValue(0.1)
        self.smoothness.setDecimals(3)
        self.smoothness.setToolTip("Smoothness regularization (lambda_s)")
        self.smoothness_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothness_slider.setRange(1, 1000)
        self.smoothness_slider.setValue(100)
        self.smoothness_slider.valueChanged.connect(
            lambda v: self.smoothness.setValue(v / 1000.0)
        )
        self.smoothness.valueChanged.connect(
            lambda v: self.smoothness_slider.setValue(int(v * 1000))
        )
        smooth_row.addWidget(self.smoothness_slider, 1)
        smooth_row.addWidget(self.smoothness)
        form.addRow("Smoothness:", smooth_row)

        # Format
        self.format = QComboBox()
        self.format.addItems([".cube", ".aml (ARRI ALF2)", ".alf4 (ARRI ALF4)"])
        self.format.setToolTip("Output file format")
        form.addRow("Format:", self.format)

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
        self.prior_strength = QDoubleSpinBox()
        self.prior_strength.setRange(0.0, 1.0)
        self.prior_strength.setSingleStep(0.001)
        self.prior_strength.setValue(0.01)
        self.prior_strength.setDecimals(3)
        self.prior_strength.setToolTip("Identity prior strength (lambda_r)")
        adv_form.addRow("Prior Strength:", self.prior_strength)

        # Robust loss
        self.robust_loss = QComboBox()
        self.robust_loss.addItems(["huber", "l2"])
        self.robust_loss.setToolTip("Loss function for IRLS")
        adv_form.addRow("Robust Loss:", self.robust_loss)

        # IRLS iterations
        self.irls_iter = QSpinBox()
        self.irls_iter.setRange(0, 10)
        self.irls_iter.setValue(3)
        self.irls_iter.setToolTip("IRLS outer loop iterations (0 = L2 only)")
        adv_form.addRow("IRLS Iterations:", self.irls_iter)

        # Bin resolution
        self.bin_res = QSpinBox()
        self.bin_res.setRange(16, 128)
        self.bin_res.setValue(64)
        self.bin_res.setToolTip("Sampling bin resolution per axis")
        adv_form.addRow("Bin Resolution:", self.bin_res)

        # Min samples
        self.min_samples = QSpinBox()
        self.min_samples.setRange(1, 100)
        self.min_samples.setValue(3)
        self.min_samples.setToolTip("Minimum pixel samples per bin")
        adv_form.addRow("Min Samples:", self.min_samples)

        # Refinement
        self.enable_refine = QCheckBox("Enable iterative refinement")
        self.enable_refine.setToolTip("Run additional refinement passes after initial solve")
        adv_form.addRow("", self.enable_refine)

        # Transfer function
        self.transfer_fn = QComboBox()
        self.transfer_fn.addItems(["auto", "linear", "log_c3", "log_c4", "slog3", "vlog"])
        self.transfer_fn.setToolTip("Input transfer function / encoding")
        adv_form.addRow("Transfer Fn:", self.transfer_fn)

        layout.addWidget(self._advanced_group)

        # Reset button
        reset_row = QHBoxLayout()
        reset_row.addStretch()
        self._btn_reset = QPushButton("Reset Defaults")
        self._btn_reset.clicked.connect(self.reset_defaults)
        reset_row.addWidget(self._btn_reset)
        layout.addLayout(reset_row)

        layout.addStretch()

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
        self.enable_refine.setChecked(False)
        self.transfer_fn.setCurrentIndex(0)

    def get_format_string(self) -> str:
        text = self.format.currentText()
        if ".cube" in text:
            return "cube"
        elif ".aml" in text:
            return "aml"
        elif ".alf4" in text:
            return "alf4"
        return "cube"
