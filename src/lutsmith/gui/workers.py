"""Background worker threads for pipeline execution."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Optional

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:
    raise ImportError("PySide6 is required for the GUI.")

from lutsmith.core.types import (
    ExportFormat,
    InterpolationKernel,
    PipelineConfig,
    PipelineResult,
    RobustLoss,
    TransferFunction,
)


class PipelineWorker(QThread):
    """Runs the LUT extraction pipeline on a background thread.

    Signals:
        progress_updated(str, float, str): (stage, fraction, message)
        log_message(str, str): (message, severity)
        finished_ok(PipelineResult): Emitted on successful completion.
        finished_error(str): Emitted on error with traceback string.
    """

    progress_updated = Signal(str, float, str)
    log_message = Signal(str, str)
    finished_ok = Signal(object)
    finished_error = Signal(str)

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._cancel_event = threading.Event()

    def cancel(self):
        """Request cancellation of the running pipeline."""
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def run(self):
        """Execute the pipeline (called by QThread.start())."""
        try:
            from lutsmith.pipeline.runner import run_pipeline

            self.log_message.emit("Pipeline started", "stage")
            self.log_message.emit(
                f"Source: {self._config.source_path}", "info"
            )
            self.log_message.emit(
                f"Target: {self._config.target_path}", "info"
            )
            self.log_message.emit(
                f"LUT size: {self._config.lut_size}^3 "
                f"({self._config.lut_size**3:,} nodes)",
                "info",
            )

            def on_progress(stage: str, fraction: float, message: str):
                self.progress_updated.emit(stage, fraction, message)
                if message:
                    self.log_message.emit(f"[{stage}] {message}", "info")

            def cancel_check() -> bool:
                return self._cancel_event.is_set()

            result = run_pipeline(
                self._config,
                progress_callback=on_progress,
                cancel_check=cancel_check,
            )

            if self._cancel_event.is_set():
                self.log_message.emit("Pipeline cancelled", "warning")
                self.finished_error.emit("Pipeline cancelled by user.")
                return

            self.log_message.emit("Pipeline complete", "success")
            self.finished_ok.emit(result)

        except Exception as e:
            tb = traceback.format_exc()
            self.log_message.emit(f"Error: {e}", "error")
            self.finished_error.emit(str(e))


class HaldWorker(QThread):
    """Runs Hald CLUT reconstruction on a background thread.

    Signals:
        progress_updated(str, float, str): (stage, fraction, message)
        log_message(str, str): (message, severity)
        finished_ok(object): Emitted with result dict on success.
        finished_error(str): Emitted on error.
    """

    progress_updated = Signal(str, float, str)
    log_message = Signal(str, str)
    finished_ok = Signal(object)
    finished_error = Signal(str)

    def __init__(
        self,
        processed_path: Path,
        output_path: Path,
        level: int = 8,
        target_size: int = 0,
        title: str = "LutSmith Hald LUT",
        parent=None,
    ):
        super().__init__(parent)
        self._processed_path = processed_path
        self._output_path = output_path
        self._level = level
        self._target_size = target_size
        self._title = title

    def run(self):
        try:
            from lutsmith.hald.identity import hald_lut_size
            from lutsmith.hald.reconstruct import reconstruct_from_hald
            from lutsmith.hald.resample import resample_lut
            from lutsmith.io.cube import write_cube
            from lutsmith.io.image import load_image

            self.log_message.emit("Hald reconstruction started", "stage")

            # Load processed image
            self.progress_updated.emit("preprocess", 0.0, "Loading image...")
            img, _ = load_image(self._processed_path)
            self.progress_updated.emit("preprocess", 1.0, "Loaded")

            # Reconstruct LUT
            self.progress_updated.emit("solving", 0.0, "Reconstructing...")
            lut = reconstruct_from_hald(img, self._level)
            native_size = hald_lut_size(self._level)
            self.log_message.emit(
                f"Reconstructed {native_size}^3 LUT from Hald level {self._level}",
                "info",
            )
            self.progress_updated.emit("solving", 1.0, "Done")

            # Resample if needed
            final_size = native_size
            if self._target_size > 0 and self._target_size != native_size:
                self.progress_updated.emit("refinement", 0.0, "Resampling...")
                lut = resample_lut(lut, self._target_size)
                final_size = self._target_size
                self.log_message.emit(
                    f"Resampled: {native_size}^3 -> {final_size}^3", "info"
                )
                self.progress_updated.emit("refinement", 1.0, "Done")

            # Export
            self.progress_updated.emit("export", 0.0, "Writing .cube...")
            write_cube(self._output_path, lut, title=self._title)
            self.progress_updated.emit("export", 1.0, "Done")

            self.log_message.emit(
                f"Saved: {self._output_path} ({final_size}^3)", "success"
            )
            self.finished_ok.emit({
                "output_path": self._output_path,
                "lut_size": final_size,
            })

        except Exception as e:
            self.log_message.emit(f"Error: {e}", "error")
            self.finished_error.emit(str(e))
