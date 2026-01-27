"""Qt stylesheet (QSS) generation for the LutSmith dark theme."""

from __future__ import annotations

from lutsmith.gui.styles.theme import (
    PALETTE,
    FONT_FAMILY,
    FONT_MONO,
    FONT_SIZE,
    FONT_SIZE_SMALL,
    FONT_SIZE_HEADER,
    RADIUS_MD,
    RADIUS_SM,
    SPACING_SM,
    SPACING_MD,
)

p = PALETTE


def build_stylesheet() -> str:
    """Generate the complete application stylesheet."""
    return f"""
/* ============================================================ */
/* LutSmith Dark Theme                                        */
/* ============================================================ */

/* -- Global -- */
QWidget {{
    background-color: {p.bg_dark};
    color: {p.text_primary};
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE}px;
    selection-background-color: {p.accent_dim};
    selection-color: {p.text_primary};
}}

/* -- Main Window -- */
QMainWindow {{
    background-color: {p.bg_dark};
}}

QMainWindow::separator {{
    width: 1px;
    height: 1px;
    background: {p.border};
}}

/* -- Tab Bar -- */
QTabWidget::pane {{
    border: 1px solid {p.border};
    border-top: none;
    background-color: {p.bg_dark};
}}

QTabBar::tab {{
    background-color: {p.bg_panel};
    color: {p.text_secondary};
    border: 1px solid {p.border};
    border-bottom: none;
    padding: {SPACING_SM}px {SPACING_MD}px;
    margin-right: 1px;
    min-width: 100px;
    font-size: {FONT_SIZE}px;
}}

QTabBar::tab:selected {{
    background-color: {p.bg_dark};
    color: {p.accent};
    border-bottom: 2px solid {p.accent};
}}

QTabBar::tab:hover:!selected {{
    background-color: {p.bg_hover};
    color: {p.text_primary};
}}

/* -- Group Box -- */
QGroupBox {{
    background-color: {p.bg_panel};
    border: 1px solid {p.border};
    border-radius: {RADIUS_MD}px;
    margin-top: 12px;
    padding: {SPACING_MD}px;
    padding-top: 24px;
    font-weight: bold;
    color: {p.text_secondary};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: {SPACING_MD}px;
    padding: 0 {SPACING_SM}px;
    color: {p.text_accent};
}}

/* -- Labels -- */
QLabel {{
    background-color: transparent;
    color: {p.text_primary};
}}

QLabel[class="secondary"] {{
    color: {p.text_secondary};
    font-size: {FONT_SIZE_SMALL}px;
}}

QLabel[class="header"] {{
    font-size: {FONT_SIZE_HEADER}px;
    font-weight: bold;
    color: {p.text_primary};
}}

QLabel[class="mono"] {{
    font-family: {FONT_MONO};
}}

/* -- Push Buttons -- */
QPushButton {{
    background-color: {p.bg_input};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {RADIUS_MD}px;
    padding: {SPACING_SM}px {SPACING_MD}px;
    min-height: 28px;
    font-size: {FONT_SIZE}px;
}}

QPushButton:hover {{
    background-color: {p.bg_hover};
    border-color: {p.text_secondary};
}}

QPushButton:pressed {{
    background-color: {p.bg_selected};
}}

QPushButton:disabled {{
    color: {p.text_disabled};
    border-color: {p.border_subtle};
}}

QPushButton[class="primary"] {{
    background-color: {p.accent};
    color: #1a1a1e;
    border: none;
    font-weight: bold;
}}

QPushButton[class="primary"]:hover {{
    background-color: {p.accent_hover};
}}

QPushButton[class="primary"]:pressed {{
    background-color: {p.accent_pressed};
}}

QPushButton[class="primary"]:disabled {{
    background-color: {p.accent_dim};
    color: {p.text_disabled};
}}

/* -- Combo Box -- */
QComboBox {{
    background-color: {p.bg_input};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {RADIUS_SM}px;
    padding: 4px 8px;
    min-height: 24px;
}}

QComboBox:hover {{
    border-color: {p.text_secondary};
}}

QComboBox:focus {{
    border-color: {p.accent};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {p.text_secondary};
    margin-right: 6px;
}}

QComboBox QAbstractItemView {{
    background-color: {p.bg_panel};
    color: {p.text_primary};
    border: 1px solid {p.border};
    selection-background-color: {p.accent_dim};
    outline: none;
}}

/* -- Spin Box -- */
QSpinBox, QDoubleSpinBox {{
    background-color: {p.bg_input};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {RADIUS_SM}px;
    padding: 4px 8px;
    min-height: 24px;
    font-family: {FONT_MONO};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {p.accent};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {p.bg_hover};
    border: none;
    width: 16px;
}}

/* -- Slider -- */
QSlider::groove:horizontal {{
    height: 4px;
    background: {p.bg_input};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {p.accent};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    background: {p.accent_hover};
}}

QSlider::sub-page:horizontal {{
    background: {p.accent_dim};
    border-radius: 2px;
}}

/* -- Progress Bar -- */
QProgressBar {{
    background-color: {p.bg_input};
    border: 1px solid {p.border};
    border-radius: {RADIUS_SM}px;
    height: 8px;
    text-align: center;
    font-size: {FONT_SIZE_SMALL}px;
    color: {p.text_secondary};
}}

QProgressBar::chunk {{
    background-color: {p.accent};
    border-radius: {RADIUS_SM}px;
}}

/* -- Scroll Bar -- */
QScrollBar:vertical {{
    background: {p.scrollbar_bg};
    width: 10px;
    margin: 0;
    border: none;
}}

QScrollBar::handle:vertical {{
    background: {p.scrollbar_handle};
    min-height: 30px;
    border-radius: 4px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background: {p.scrollbar_hover};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {p.scrollbar_bg};
    height: 10px;
    margin: 0;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background: {p.scrollbar_handle};
    min-width: 30px;
    border-radius: 4px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {p.scrollbar_hover};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* -- Text Edit / Plain Text -- */
QTextEdit, QPlainTextEdit {{
    background-color: {p.bg_panel};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {RADIUS_SM}px;
    font-family: {FONT_MONO};
    font-size: {FONT_SIZE_SMALL}px;
    padding: {SPACING_SM}px;
}}

/* -- Line Edit -- */
QLineEdit {{
    background-color: {p.bg_input};
    color: {p.text_primary};
    border: 1px solid {p.border};
    border-radius: {RADIUS_SM}px;
    padding: 4px 8px;
    min-height: 24px;
}}

QLineEdit:focus {{
    border-color: {p.accent};
}}

/* -- Check Box -- */
QCheckBox {{
    spacing: 8px;
    color: {p.text_primary};
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {p.border};
    border-radius: {RADIUS_SM}px;
    background-color: {p.bg_input};
}}

QCheckBox::indicator:checked {{
    background-color: {p.accent};
    border-color: {p.accent};
}}

/* -- Splitter -- */
QSplitter::handle {{
    background-color: {p.border};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

/* -- Status Bar -- */
QStatusBar {{
    background-color: {p.bg_panel};
    color: {p.text_secondary};
    border-top: 1px solid {p.border};
    font-size: {FONT_SIZE_SMALL}px;
    padding: 2px {SPACING_SM}px;
}}

/* -- Menu Bar -- */
QMenuBar {{
    background-color: {p.bg_panel};
    color: {p.text_primary};
    border-bottom: 1px solid {p.border};
}}

QMenuBar::item:selected {{
    background-color: {p.bg_hover};
}}

QMenu {{
    background-color: {p.bg_panel};
    color: {p.text_primary};
    border: 1px solid {p.border};
}}

QMenu::item:selected {{
    background-color: {p.accent_dim};
}}

/* -- Tool Tip -- */
QToolTip {{
    background-color: {p.bg_panel};
    color: {p.text_primary};
    border: 1px solid {p.border};
    padding: 4px 8px;
    font-size: {FONT_SIZE_SMALL}px;
}}

/* -- Header View (Tables) -- */
QHeaderView::section {{
    background-color: {p.bg_panel};
    color: {p.text_secondary};
    border: none;
    border-right: 1px solid {p.border};
    border-bottom: 1px solid {p.border};
    padding: 4px 8px;
    font-weight: bold;
}}
"""
