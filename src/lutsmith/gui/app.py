"""LutSmith GUI application entry point."""

from __future__ import annotations

import sys


def run():
    """Launch the LutSmith GUI application."""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
    except ImportError:
        print(
            "PySide6 is required for the GUI.\n"
            "Install it with: pip install lutsmith[gui]",
            file=sys.stderr,
        )
        sys.exit(1)

    from lutsmith.gui.styles.qss import build_stylesheet
    from lutsmith.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("LutSmith")
    app.setOrganizationName("LutSmith")

    # Apply dark theme
    app.setStyleSheet(build_stylesheet())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run()
