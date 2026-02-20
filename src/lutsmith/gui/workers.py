"""Background worker threads for pipeline execution."""

from __future__ import annotations

import threading
import traceback
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
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
from lutsmith.pipeline.batch_manifest import ManifestEntry


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


class BatchPipelineWorker(QThread):
    """Runs batch/clustered LUT extraction in the background."""

    progress_updated = Signal(str, float, str)
    log_message = Signal(str, str)
    finished_ok = Signal(object)  # dict with master/clusters/clustering
    finished_error = Signal(str)

    def __init__(
        self,
        manifest_entries: list[ManifestEntry],
        config: PipelineConfig,
        cluster_mode: str = "none",
        cluster_count: int = 0,
        max_clusters: int = 6,
        export_master: bool = True,
        cluster_seed: int = 42,
        pair_balance: str = "equal",
        outlier_sigma: float = 0.0,
        min_pairs_after_outlier: int = 3,
        allow_mixed_transfer: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._entries = manifest_entries
        self._config = config
        self._cluster_mode = cluster_mode
        self._cluster_count = cluster_count
        self._max_clusters = max_clusters
        self._export_master = export_master
        self._cluster_seed = cluster_seed
        self._pair_balance = pair_balance
        self._outlier_sigma = outlier_sigma
        self._min_pairs_after_outlier = min_pairs_after_outlier
        self._allow_mixed_transfer = allow_mixed_transfer
        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel_event.set()

    def _cluster_output_path(self, base_output: Path, cluster_name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in cluster_name.strip())
        safe = safe.strip("_") or "cluster"
        return base_output.with_name(f"{base_output.stem}_{safe}{base_output.suffix}")

    def _run_job(
        self,
        label: str,
        pair_indices: list[int],
        all_pairs: list[tuple[Path, Path]],
        all_weights: list[float],
        all_transfer_fns: list[Optional[str]],
        all_normalizations: list[Optional[str]],
        config: PipelineConfig,
    ):
        from lutsmith.pipeline.runner import run_multi_pipeline

        pairs = [all_pairs[i] for i in pair_indices]
        weights = [all_weights[i] for i in pair_indices]
        transfer_fns = [all_transfer_fns[i] for i in pair_indices]
        normalizations = [all_normalizations[i] for i in pair_indices]

        self.log_message.emit(
            f"Running {label}: {len(pairs)} pairs, output={config.output_path}",
            "stage",
        )

        def on_progress(stage: str, fraction: float, message: str):
            msg = f"{label}: {message}" if message else label
            self.progress_updated.emit(stage, fraction, msg)

        result = run_multi_pipeline(
            pairs,
            config,
            progress_callback=on_progress,
            cancel_check=lambda: self._cancel_event.is_set(),
            pair_weights=weights,
            pair_transfer_fns=transfer_fns,
            pair_normalization_modes=normalizations,
            pair_balance=self._pair_balance,
            outlier_sigma=self._outlier_sigma,
            min_pairs_after_outlier=self._min_pairs_after_outlier,
            allow_mixed_transfer=self._allow_mixed_transfer,
        )
        return result

    def run(self):
        try:
            if not self._entries:
                raise ValueError("No manifest entries provided")

            from lutsmith.pipeline.clustering import (
                auto_cluster_features,
                compute_pair_signatures,
                kmeans_cluster_features,
            )

            pairs = [(e.source, e.target) for e in self._entries]
            weights = [float(e.weight) for e in self._entries]
            manual_clusters = [e.cluster for e in self._entries]
            transfer_fns = [e.transfer_fn for e in self._entries]
            normalizations = [e.normalization for e in self._entries]

            self.log_message.emit(
                f"Batch extraction started: pairs={len(pairs)}, mode={self._cluster_mode}",
                "stage",
            )

            mode = self._cluster_mode.strip().lower()
            if mode not in {"none", "manual", "auto"}:
                raise ValueError(f"Invalid cluster mode: {self._cluster_mode}")

            outputs = {
                "master": None,
                "clusters": [],
                "clustering": {},
            }

            if mode == "none":
                result = self._run_job(
                    "master",
                    list(range(len(pairs))),
                    pairs,
                    weights,
                    transfer_fns,
                    normalizations,
                    self._config,
                )
                outputs["master"] = result
                self.log_message.emit("Batch extraction complete", "success")
                self.finished_ok.emit(outputs)
                return

            if mode == "manual":
                if any(c is None or not str(c).strip() for c in manual_clusters):
                    raise ValueError(
                        "Manual clustering requires a cluster label for every manifest row."
                    )
                label_to_id = {}
                assignments = []
                next_id = 0
                for c in manual_clusters:
                    key = str(c).strip()
                    if key not in label_to_id:
                        label_to_id[key] = next_id
                        next_id += 1
                    assignments.append(label_to_id[key])
                assignments = np.array(assignments, dtype=np.int64)
                id_to_label = {v: k for k, v in label_to_id.items()}
                outputs["clustering"] = {"mode": "manual", "k": len(label_to_id)}
            else:
                self.log_message.emit("Computing pair signatures for auto clustering...", "info")

                def on_sig_progress(fraction: float, message: str):
                    self.progress_updated.emit("preprocess", fraction, f"cluster: {message}")

                signatures, _ = compute_pair_signatures(
                    pairs,
                    self._config,
                    progress_callback=on_sig_progress,
                    cancel_check=lambda: self._cancel_event.is_set(),
                    pair_transfer_fns=transfer_fns,
                    pair_normalization_modes=normalizations,
                )
                if self._cluster_count > 0:
                    assignments, diag = kmeans_cluster_features(
                        signatures,
                        k=self._cluster_count,
                        random_seed=self._cluster_seed,
                    )
                    diag["mode"] = "auto_fixed"
                else:
                    assignments, diag = auto_cluster_features(
                        signatures,
                        max_clusters=self._max_clusters,
                        random_seed=self._cluster_seed,
                    )
                    diag["mode"] = "auto"
                outputs["clustering"] = diag
                id_to_label = {i: f"cluster_{i + 1:02d}" for i in sorted(set(assignments.tolist()))}

            groups: dict[int, list[int]] = defaultdict(list)
            for i, cid in enumerate(assignments.tolist()):
                groups[int(cid)].append(i)

            self.log_message.emit(f"Clustering complete: {len(groups)} cluster(s)", "info")
            for cid in sorted(groups):
                label = id_to_label.get(cid, f"cluster_{cid + 1:02d}")
                self.log_message.emit(f"  {label}: {len(groups[cid])} pairs", "info")

            if self._export_master:
                outputs["master"] = self._run_job(
                    "master",
                    list(range(len(pairs))),
                    pairs,
                    weights,
                    transfer_fns,
                    normalizations,
                    self._config,
                )

            for cid in sorted(groups):
                if self._cancel_event.is_set():
                    raise RuntimeError("Cancelled")
                label = id_to_label.get(cid, f"cluster_{cid + 1:02d}")
                idxs = groups[cid]
                if self._config.output_path is None:
                    raise ValueError("Batch config missing output_path")
                c_output = self._cluster_output_path(Path(self._config.output_path), label)
                c_config = replace(
                    self._config,
                    output_path=c_output,
                    title=f"{self._config.title} [{label}]",
                )
                c_result = self._run_job(
                    label,
                    idxs,
                    pairs,
                    weights,
                    transfer_fns,
                    normalizations,
                    c_config,
                )
                c_result.diagnostics["cluster_label"] = label
                c_result.diagnostics["cluster_indices"] = [i + 1 for i in idxs]
                outputs["clusters"].append(c_result)

            self.log_message.emit("Batch extraction complete", "success")
            self.finished_ok.emit(outputs)
        except Exception as e:
            self.log_message.emit(f"Error: {e}", "error")
            self.finished_error.emit(str(e))
