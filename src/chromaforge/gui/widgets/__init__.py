"""Reusable widgets for the ChromaForge GUI."""

from chromaforge.gui.widgets.coverage import CoverageViewer
from chromaforge.gui.widgets.image_pair import ImagePairViewer
from chromaforge.gui.widgets.log_viewer import LogViewer
from chromaforge.gui.widgets.metrics_view import MetricsDisplay
from chromaforge.gui.widgets.parameters import ParameterPanel
from chromaforge.gui.widgets.progress import PipelineProgress

__all__ = [
    "CoverageViewer",
    "ImagePairViewer",
    "LogViewer",
    "MetricsDisplay",
    "ParameterPanel",
    "PipelineProgress",
]
