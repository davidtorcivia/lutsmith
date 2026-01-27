"""Dark theme color palette and typography for LutSmith GUI.

Design language:
    - Professional dark theme inspired by DaVinci Resolve / Substance Designer
    - Amber/gold accent (#d4a03c)
    - Clean typography, monospace for values
    - 8px grid spacing, max 4px border radius
    - Zero emojis
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorPalette:
    """Application color palette."""
    # Backgrounds
    bg_dark: str = "#1a1a1e"
    bg_panel: str = "#232328"
    bg_input: str = "#2a2a30"
    bg_hover: str = "#2e2e34"
    bg_selected: str = "#353540"

    # Accent
    accent: str = "#d4a03c"
    accent_hover: str = "#e0b050"
    accent_pressed: str = "#c08a2c"
    accent_dim: str = "#8a6a28"

    # Text
    text_primary: str = "#e8e8e8"
    text_secondary: str = "#888890"
    text_disabled: str = "#555560"
    text_accent: str = "#d4a03c"

    # Borders
    border: str = "#333338"
    border_focus: str = "#d4a03c"
    border_subtle: str = "#2a2a30"

    # Status
    success: str = "#4a9e4a"
    warning: str = "#c98030"
    error: str = "#c94040"
    info: str = "#5088b0"

    # Scrollbar
    scrollbar_bg: str = "#1a1a1e"
    scrollbar_handle: str = "#404048"
    scrollbar_hover: str = "#505058"


# Default palette instance
PALETTE = ColorPalette()

# Typography
FONT_FAMILY = "Segoe UI, Inter, -apple-system, sans-serif"
FONT_MONO = "Consolas, Cascadia Code, Fira Code, monospace"
FONT_SIZE = 13
FONT_SIZE_SMALL = 11
FONT_SIZE_HEADER = 16
FONT_SIZE_TITLE = 20

# Spacing (8px grid)
SPACING_XS = 4
SPACING_SM = 8
SPACING_MD = 16
SPACING_LG = 24
SPACING_XL = 32

# Border radius
RADIUS_SM = 2
RADIUS_MD = 4
RADIUS_LG = 6
