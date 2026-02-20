"""LutSmith CLI application.

Commands:
    extract     - Extract LUT from source/target image pair
    extract-batch - Extract LUT from many matched source/target pairs
    route-shots - Route shots to clustered LUTs using centroid signatures
    hald-gen    - Generate Hald CLUT identity image
    hald-recon  - Reconstruct LUT from processed Hald image
    validate    - Validate an existing LUT against image pair
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from lutsmith import __version__
from lutsmith.core.types import (
    ColorBasis,
    ExportFormat,
    InterpolationKernel,
    PipelineConfig,
    PriorModel,
    RobustLoss,
    TransferFunction,
)
from lutsmith.pipeline.batch_manifest import ManifestEntry, parse_pair_manifest
from lutsmith.pipeline.reporting import (
    build_batch_metrics_rows,
    build_cluster_assignment_rows,
    build_cluster_centroid_rows,
    read_cluster_centroids_csv,
    write_batch_metrics_csv,
    write_cluster_assignments_csv,
    write_cluster_centroids_csv,
)

app = typer.Typer(
    name="lutsmith",
    help="Image-derived 3D LUT generation tool.",
    no_args_is_help=True,
)
console = Console()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def version_callback(value: bool):
    if value:
        console.print(f"LutSmith v{__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit.",
        callback=version_callback, is_eager=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
):
    if verbose:
        logging.getLogger("lutsmith").setLevel(logging.DEBUG)


_STAGE_WEIGHTS = {
    "preprocess": 10,
    "sampling": 20,
    "solving": 50,
    "refinement": 10,
    "validation": 5,
    "export": 5,
}


def _resolve_shaper_mode(shaper: str) -> Optional[bool]:
    """Resolve --shaper option to include_shaper flag."""
    shaper_mode = shaper.strip().lower()
    if shaper_mode not in {"auto", "on", "off"}:
        raise typer.BadParameter("Shaper must be one of: auto, on, off.")
    if shaper_mode == "on":
        return True
    if shaper_mode == "off":
        return False
    return None


def _build_pipeline_config(
    *,
    source: Optional[Path],
    target: Optional[Path],
    output: Path,
    size: int,
    kernel: str,
    smoothness: float,
    prior: float,
    loss: str,
    irls_iter: int,
    bin_res: int,
    min_samples: int,
    title: str,
    refine: bool,
    format: str,
    shadow_auto: bool,
    shadow_threshold: Optional[float],
    deep_shadow_threshold: Optional[float],
    prior_model: str,
    color_basis: str,
    chroma_ratio: float,
    laplacian_connectivity: int,
    shaper: str,
) -> PipelineConfig:
    """Create PipelineConfig from shared CLI options."""
    return PipelineConfig(
        source_path=source,
        target_path=target,
        output_path=output,
        lut_size=size,
        kernel=InterpolationKernel(kernel),
        smoothness=smoothness,
        prior_strength=prior,
        robust_loss=RobustLoss(loss),
        irls_iterations=irls_iter,
        bin_resolution=bin_res,
        min_samples_per_bin=min_samples,
        title=title,
        enable_refinement=refine,
        format=ExportFormat(format),
        include_shaper=_resolve_shaper_mode(shaper),
        shadow_auto=shadow_auto,
        shadow_threshold=shadow_threshold,
        deep_shadow_threshold=deep_shadow_threshold,
        prior_model=PriorModel(prior_model),
        color_basis=ColorBasis(color_basis),
        chroma_smoothness_ratio=chroma_ratio,
        laplacian_connectivity=laplacian_connectivity,
    )


def _parse_pair_manifest(manifest: Path) -> list[ManifestEntry]:
    """Parse batch manifest and map parser errors to CLI-friendly exceptions."""
    try:
        return parse_pair_manifest(manifest)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _build_progress_callback(progress: Progress, task_id: int):
    """Create weighted stage-progress callback for pipeline runs."""

    stage_order = list(_STAGE_WEIGHTS.keys())

    def on_progress(stage: str, fraction: float, message: str):
        base = sum(
            _STAGE_WEIGHTS[s]
            for s in stage_order
            if stage in _STAGE_WEIGHTS and stage_order.index(s) < stage_order.index(stage)
        )
        weight = _STAGE_WEIGHTS.get(stage, 0)
        pct = base + weight * fraction
        progress.update(
            task_id,
            completed=pct,
            description=f"{stage}: {message}" if message else stage,
        )

    return on_progress


def _cluster_output_path(base_output: Path, cluster_name: str) -> Path:
    """Generate per-cluster output path based on base output stem."""
    safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in cluster_name.strip())
    safe = safe.strip("_") or "cluster"
    return base_output.with_name(f"{base_output.stem}_{safe}{base_output.suffix}")


def _print_batch_result_summary(result, fallback_pairs: int) -> None:
    """Print metrics + diagnostics summary for batch run result."""
    console.print()
    _print_metrics(result.metrics)

    if result.output_path:
        console.print(f"\n[green]Output:[/green] {result.output_path}")

    pair_count = result.diagnostics.get("num_pairs", fallback_pairs)
    bins = result.diagnostics.get("occupied_bins", 0)
    dropped = result.diagnostics.get("outlier_rejection", {}).get("dropped_pair_indices", [])
    total_time = result.diagnostics.get("total_time", 0)
    console.print(f"[dim]Pairs: {pair_count}, aggregated bins: {bins}[/dim]")
    if dropped:
        console.print(f"[dim]Outliers dropped: {dropped}[/dim]")
    console.print(f"[dim]Total time: {total_time:.2f}s[/dim]\n")


@app.command()
def extract(
    source: Path = typer.Argument(..., help="Source (ungraded) image path."),
    target: Path = typer.Argument(..., help="Target (graded) image path."),
    output: Path = typer.Option("output.cube", "-o", "--output", help="Output LUT path."),
    size: int = typer.Option(33, "-s", "--size", help="LUT grid size (17, 33, 65)."),
    kernel: str = typer.Option("tetrahedral", "-k", "--kernel", help="Interpolation kernel."),
    smoothness: float = typer.Option(0.1, "--smoothness", help="Smoothness (lambda_s)."),
    prior: float = typer.Option(0.01, "--prior", help="Prior strength (lambda_r)."),
    loss: str = typer.Option("huber", "--loss", help="Loss function (l2, huber)."),
    irls_iter: int = typer.Option(3, "--irls-iter", help="IRLS iterations."),
    bin_res: int = typer.Option(64, "--bin-res", help="Bin resolution per axis."),
    min_samples: int = typer.Option(3, "--min-samples", help="Min samples per bin."),
    title: str = typer.Option("LutSmith LUT", "--title", help="LUT title."),
    refine: bool = typer.Option(False, "--refine", help="Enable iterative refinement."),
    format: str = typer.Option("cube", "-f", "--format", help="Output format (cube, aml, alf4)."),
    shadow_auto: bool = typer.Option(
        True,
        "--shadow-auto/--shadow-manual",
        help="Use adaptive shadow thresholds for smoothing/weighting.",
    ),
    shadow_threshold: Optional[float] = typer.Option(
        None,
        "--shadow-threshold",
        min=0.0,
        max=1.0,
        help="Shadow smoothing threshold (0-1). Overrides auto if set.",
    ),
    deep_shadow_threshold: Optional[float] = typer.Option(
        None,
        "--deep-shadow-threshold",
        min=0.0,
        max=1.0,
        help="Deep shadow threshold (0-1). Overrides auto if set.",
    ),
    prior_model: str = typer.Option(
        "identity",
        "--prior-model",
        help="Prior model: identity, baseline_residual, baseline_multigrid_residual.",
    ),
    color_basis: str = typer.Option(
        "rgb",
        "--color-basis",
        help="Regularization color space: rgb, opponent.",
    ),
    chroma_ratio: float = typer.Option(
        4.0,
        "--chroma-ratio",
        help="Chroma-to-luma smoothness ratio (opponent mode).",
    ),
    laplacian_connectivity: int = typer.Option(
        6,
        "--laplacian-connectivity",
        help="Laplacian connectivity: 6, 18, or 26.",
    ),
    shaper: str = typer.Option(
        "auto",
        "--shaper",
        help="1D shaper LUT: auto, on, or off.",
    ),
):
    """Extract a 3D LUT from a source/target image pair."""
    config = _build_pipeline_config(
        source=source,
        target=target,
        output=output,
        size=size,
        kernel=kernel,
        smoothness=smoothness,
        prior=prior,
        loss=loss,
        irls_iter=irls_iter,
        bin_res=bin_res,
        min_samples=min_samples,
        title=title,
        refine=refine,
        format=format,
        shadow_auto=shadow_auto,
        shadow_threshold=shadow_threshold,
        deep_shadow_threshold=deep_shadow_threshold,
        prior_model=prior_model,
        color_basis=color_basis,
        chroma_ratio=chroma_ratio,
        laplacian_connectivity=laplacian_connectivity,
        shaper=shaper,
    )

    from lutsmith.pipeline.runner import run_pipeline

    console.print(f"\n[bold]LutSmith LUT Extraction[/bold]")
    console.print(f"  Source: {source}")
    console.print(f"  Target: {target}")
    console.print(f"  Size:   {size}^3 = {size**3:,} nodes")
    console.print(f"  Kernel: {kernel}")
    console.print(f"  Loss:   {loss}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting LUT...", total=100)
        on_progress = _build_progress_callback(progress, task)

        try:
            result = run_pipeline(config, progress_callback=on_progress)
            progress.update(task, completed=100, description="Complete")
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(code=1)

    # Display results
    console.print()
    _print_metrics(result.metrics)

    if result.output_path:
        console.print(f"\n[green]Output:[/green] {result.output_path}")

    total_time = result.diagnostics.get("total_time", 0)
    console.print(f"[dim]Total time: {total_time:.2f}s[/dim]\n")


@app.command("extract-batch")
def extract_batch(
    manifest: Path = typer.Argument(
        ...,
        help="CSV file with source,target[,weight][,cluster][,transfer_fn][,normalization] rows.",
    ),
    output: Path = typer.Option("output.cube", "-o", "--output", help="Output LUT path."),
    size: int = typer.Option(33, "-s", "--size", help="LUT grid size (17, 33, 65)."),
    kernel: str = typer.Option("tetrahedral", "-k", "--kernel", help="Interpolation kernel."),
    smoothness: float = typer.Option(0.1, "--smoothness", help="Smoothness (lambda_s)."),
    prior: float = typer.Option(0.01, "--prior", help="Prior strength (lambda_r)."),
    loss: str = typer.Option("huber", "--loss", help="Loss function (l2, huber)."),
    irls_iter: int = typer.Option(3, "--irls-iter", help="IRLS iterations."),
    bin_res: int = typer.Option(64, "--bin-res", help="Bin resolution per axis."),
    min_samples: int = typer.Option(3, "--min-samples", help="Min samples per bin."),
    title: str = typer.Option("LutSmith LUT", "--title", help="LUT title."),
    refine: bool = typer.Option(False, "--refine", help="Enable iterative refinement."),
    format: str = typer.Option("cube", "-f", "--format", help="Output format (cube, aml, alf4)."),
    shadow_auto: bool = typer.Option(
        True,
        "--shadow-auto/--shadow-manual",
        help="Use adaptive shadow thresholds for smoothing/weighting.",
    ),
    shadow_threshold: Optional[float] = typer.Option(
        None,
        "--shadow-threshold",
        min=0.0,
        max=1.0,
        help="Shadow smoothing threshold (0-1). Overrides auto if set.",
    ),
    deep_shadow_threshold: Optional[float] = typer.Option(
        None,
        "--deep-shadow-threshold",
        min=0.0,
        max=1.0,
        help="Deep shadow threshold (0-1). Overrides auto if set.",
    ),
    prior_model: str = typer.Option(
        "identity",
        "--prior-model",
        help="Prior model: identity, baseline_residual, baseline_multigrid_residual.",
    ),
    color_basis: str = typer.Option(
        "rgb",
        "--color-basis",
        help="Regularization color space: rgb, opponent.",
    ),
    chroma_ratio: float = typer.Option(
        4.0,
        "--chroma-ratio",
        help="Chroma-to-luma smoothness ratio (opponent mode).",
    ),
    laplacian_connectivity: int = typer.Option(
        6,
        "--laplacian-connectivity",
        help="Laplacian connectivity: 6, 18, or 26.",
    ),
    shaper: str = typer.Option(
        "auto",
        "--shaper",
        help="1D shaper LUT: auto, on, or off.",
    ),
    pair_balance: str = typer.Option(
        "equal",
        "--pair-balance",
        help="Pair contribution mode: equal, by_bins, by_pixels.",
    ),
    outlier_sigma: float = typer.Option(
        0.0,
        "--outlier-sigma",
        min=0.0,
        help="Drop high-error outlier pairs using median+sigma*MAD (0 disables).",
    ),
    min_pairs_after_outlier: int = typer.Option(
        3,
        "--min-pairs-after-outlier",
        min=1,
        help="Minimum number of pairs to keep after outlier rejection.",
    ),
    allow_mixed_transfer: bool = typer.Option(
        False,
        "--allow-mixed-transfer",
        help="Allow mixed transfer-function detections across pairs.",
    ),
    cluster_mode: str = typer.Option(
        "none",
        "--cluster-mode",
        help="Scene clustering mode: none, manual, auto.",
    ),
    cluster_count: int = typer.Option(
        0,
        "--cluster-count",
        min=0,
        help="Fixed number of clusters for auto mode (0 = choose automatically).",
    ),
    max_clusters: int = typer.Option(
        6,
        "--max-clusters",
        min=2,
        help="Maximum clusters considered when --cluster-count=0 in auto mode.",
    ),
    export_master: bool = typer.Option(
        True,
        "--export-master/--no-export-master",
        help="Also export one LUT fit across all pairs when clustering is enabled.",
    ),
    cluster_seed: int = typer.Option(
        42,
        "--cluster-seed",
        help="Random seed for auto clustering initialization.",
    ),
    metrics_csv: Optional[Path] = typer.Option(
        None,
        "--metrics-csv",
        help="Optional CSV output path for batch/cluster metrics summary.",
    ),
    export_cluster_artifacts: bool = typer.Option(
        True,
        "--export-cluster-artifacts/--no-export-cluster-artifacts",
        help="Export pair assignments and cluster centroid/signature CSVs when clustering.",
    ),
    cluster_artifacts_prefix: Optional[Path] = typer.Option(
        None,
        "--cluster-artifacts-prefix",
        help="Optional output prefix for cluster artifact CSVs (without extension).",
    ),
):
    """Extract one LUT from many matched source/target frame pairs."""
    manifest_entries = _parse_pair_manifest(manifest)
    pairs = [(entry.source, entry.target) for entry in manifest_entries]
    pair_weights = [entry.weight for entry in manifest_entries]
    manual_clusters = [entry.cluster for entry in manifest_entries]
    pair_transfer_fns = [entry.transfer_fn for entry in manifest_entries]
    pair_normalizations = [entry.normalization for entry in manifest_entries]
    config = _build_pipeline_config(
        source=None,
        target=None,
        output=output,
        size=size,
        kernel=kernel,
        smoothness=smoothness,
        prior=prior,
        loss=loss,
        irls_iter=irls_iter,
        bin_res=bin_res,
        min_samples=min_samples,
        title=title,
        refine=refine,
        format=format,
        shadow_auto=shadow_auto,
        shadow_threshold=shadow_threshold,
        deep_shadow_threshold=deep_shadow_threshold,
        prior_model=prior_model,
        color_basis=color_basis,
        chroma_ratio=chroma_ratio,
        laplacian_connectivity=laplacian_connectivity,
        shaper=shaper,
    )

    from lutsmith.pipeline.runner import run_multi_pipeline
    from dataclasses import replace
    from lutsmith.pipeline.clustering import (
        auto_cluster_features,
        compute_pair_signatures,
        kmeans_cluster_features,
    )

    cluster_mode = cluster_mode.strip().lower()
    if cluster_mode not in {"none", "manual", "auto"}:
        raise typer.BadParameter("cluster-mode must be one of: none, manual, auto")
    if cluster_count > 0 and cluster_mode != "auto":
        raise typer.BadParameter("--cluster-count is only valid with --cluster-mode auto")

    console.print(f"\n[bold]LutSmith Batch LUT Extraction[/bold]")
    console.print(f"  Manifest: {manifest}")
    console.print(f"  Pairs:    {len(pairs)}")
    console.print(f"  Clusters: {cluster_mode}")
    console.print(f"  Balance:  {pair_balance}")
    if outlier_sigma > 0:
        console.print(f"  Outliers: sigma={outlier_sigma:.2f} (min keep={min_pairs_after_outlier})")
    console.print(f"  Size:     {size}^3 = {size**3:,} nodes")
    console.print(f"  Kernel:   {kernel}")
    console.print(f"  Loss:     {loss}")
    console.print()

    def _run_batch_job(
        job_label: str,
        pair_indices: list[int],
        job_config: PipelineConfig,
    ):
        job_pairs = [pairs[i] for i in pair_indices]
        job_weights = [pair_weights[i] for i in pair_indices]
        job_transfer_fns = [pair_transfer_fns[i] for i in pair_indices]
        job_normalizations = [pair_normalizations[i] for i in pair_indices]
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Extracting {job_label}...", total=100)
            on_progress = _build_progress_callback(progress, task)
            result = run_multi_pipeline(
                job_pairs,
                job_config,
                progress_callback=on_progress,
                pair_weights=job_weights,
                pair_transfer_fns=job_transfer_fns,
                pair_normalization_modes=job_normalizations,
                pair_balance=pair_balance,
                outlier_sigma=outlier_sigma,
                min_pairs_after_outlier=min_pairs_after_outlier,
                allow_mixed_transfer=allow_mixed_transfer,
            )
            progress.update(task, completed=100, description=f"{job_label} complete")
            return result

    try:
        master_result = None
        cluster_results = []
        signature_matrix = None
        signature_meta = None
        assignments = None
        id_to_label = None
        if cluster_mode == "none":
            master_result = _run_batch_job("master", list(range(len(pairs))), config)
            _print_batch_result_summary(master_result, fallback_pairs=len(pairs))
        else:
            # Build assignments for clustered runs.
            clustering_diag = {}
            need_signatures = (cluster_mode == "auto") or export_cluster_artifacts

            if need_signatures:
                console.print("[dim]Computing pair signatures...[/dim]")
                signature_matrix, signature_meta = compute_pair_signatures(
                    pairs,
                    config,
                    pair_transfer_fns=pair_transfer_fns,
                    pair_normalization_modes=pair_normalizations,
                )

            if cluster_mode == "manual":
                if any(c is None or not str(c).strip() for c in manual_clusters):
                    raise typer.BadParameter(
                        "Manual clustering requires a cluster label for every manifest row "
                        "(4th CSV column)."
                    )
                label_to_id = {}
                next_id = 0
                assignments = []
                for label in manual_clusters:
                    key = str(label).strip()
                    if key not in label_to_id:
                        label_to_id[key] = next_id
                        next_id += 1
                    assignments.append(label_to_id[key])
                assignments = np.array(assignments, dtype=np.int64)
                id_to_label = {v: k for k, v in label_to_id.items()}
                clustering_diag = {"mode": "manual", "k": len(label_to_id)}
            else:
                signatures = signature_matrix
                if cluster_count > 0:
                    assignments, clustering_diag = kmeans_cluster_features(
                        signatures, k=cluster_count, random_seed=cluster_seed,
                    )
                    clustering_diag["mode"] = "auto_fixed"
                else:
                    assignments, clustering_diag = auto_cluster_features(
                        signatures,
                        max_clusters=max_clusters,
                        random_seed=cluster_seed,
                    )
                    clustering_diag["mode"] = "auto"
                id_to_label = {i: f"cluster_{i + 1:02d}" for i in sorted(set(assignments.tolist()))}

            groups: dict[int, list[int]] = defaultdict(list)
            for i, cid in enumerate(assignments.tolist()):
                groups[int(cid)].append(i)

            console.print(
                f"[cyan]Clustering:[/cyan] {clustering_diag.get('mode', cluster_mode)}, "
                f"k={len(groups)}"
            )
            for cid in sorted(groups):
                label = id_to_label.get(cid, f"cluster_{cid + 1:02d}")
                console.print(f"  - {label}: {len(groups[cid])} pairs")

            results = []
            if export_master:
                master_result = _run_batch_job("master", list(range(len(pairs))), config)
                results.append(("master", master_result))

            for cid in sorted(groups):
                idxs = groups[cid]
                cluster_label = id_to_label.get(cid, f"cluster_{cid + 1:02d}")
                cluster_output = _cluster_output_path(output, cluster_label)
                cluster_title = f"{title} [{cluster_label}]"
                cluster_config = replace(config, output_path=cluster_output, title=cluster_title)
                cluster_result = _run_batch_job(cluster_label, idxs, cluster_config)
                cluster_result.diagnostics["cluster_label"] = cluster_label
                cluster_result.diagnostics["cluster_indices"] = [i + 1 for i in idxs]
                cluster_results.append(cluster_result)
                results.append((cluster_label, cluster_result))

            for label, result in results:
                console.print(f"\n[bold]{label}[/bold]")
                _print_batch_result_summary(result, fallback_pairs=len(pairs))

            if export_cluster_artifacts and assignments is not None and id_to_label is not None:
                # Use explicit prefix if provided, otherwise derive from output stem.
                base_prefix = cluster_artifacts_prefix or output.with_suffix("")
                assignments_path = base_prefix.parent / f"{base_prefix.name}_cluster_assignments.csv"
                centroids_path = base_prefix.parent / f"{base_prefix.name}_cluster_centroids.csv"

                cluster_lut_paths = {}
                if master_result is not None and master_result.output_path is not None:
                    cluster_lut_paths["master"] = str(master_result.output_path)
                for cr in cluster_results:
                    lbl = str(cr.diagnostics.get("cluster_label", ""))
                    if lbl and cr.output_path is not None:
                        cluster_lut_paths[lbl] = str(cr.output_path)

                if signature_matrix is None or signature_meta is None:
                    signature_matrix, signature_meta = compute_pair_signatures(
                        pairs,
                        config,
                        pair_transfer_fns=pair_transfer_fns,
                        pair_normalization_modes=pair_normalizations,
                    )
                app_matrix = np.asarray(
                    [m.get("appearance_signature", [0.0] * 8) for m in signature_meta],
                    dtype=np.float64,
                )

                assignment_rows = build_cluster_assignment_rows(
                    manifest_entries,
                    assignments,
                    id_to_label,
                    signature_meta=signature_meta,
                )
                centroid_rows = build_cluster_centroid_rows(
                    assignments,
                    signature_matrix,
                    app_matrix,
                    id_to_label,
                    cluster_lut_paths=cluster_lut_paths,
                )
                write_cluster_assignments_csv(assignment_rows, assignments_path)
                write_cluster_centroids_csv(centroid_rows, centroids_path)
                console.print(f"[green]Cluster Assignments CSV:[/green] {assignments_path}")
                console.print(f"[green]Cluster Centroids CSV:[/green] {centroids_path}")

        if metrics_csv is not None:
            rows = build_batch_metrics_rows(master_result, cluster_results)
            write_batch_metrics_csv(rows, metrics_csv)
            console.print(f"[green]Metrics CSV:[/green] {metrics_csv}")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("route-shots")
def route_shots(
    shots_manifest: Path = typer.Argument(
        ...,
        help="CSV with frame_path or shot_id,frame_path[,transfer_fn] rows.",
    ),
    cluster_centroids_csv: Path = typer.Argument(
        ...,
        help="Cluster centroid/signature CSV from extract-batch.",
    ),
    output: Path = typer.Option(
        "shot_routing.csv",
        "-o",
        "--output",
        help="Output routing CSV path.",
    ),
    transfer_fn: str = typer.Option(
        "auto",
        "--transfer-fn",
        help="Default transfer function for shot frames (auto|linear|log_c3|log_c4|slog3|vlog|unknown).",
    ),
    shaper: str = typer.Option(
        "auto",
        "--shaper",
        help="Shaper handling for signature preprocessing: auto, on, off.",
    ),
    temporal_window: int = typer.Option(
        1,
        "--temporal-window",
        min=1,
        help="Moving-average window on shot-to-cluster distance tracks.",
    ),
    switch_penalty: float = typer.Option(
        0.0,
        "--switch-penalty",
        min=0.0,
        help="Penalty for changing cluster between adjacent shots (temporal Viterbi smoothing).",
    ),
):
    """Route shots to clustered LUTs using source appearance signatures."""
    from lutsmith.pipeline.routing import parse_shot_manifest, run_shot_routing

    try:
        shot_entries = parse_shot_manifest(shots_manifest)
        centroid_rows, centroid_appearance = read_cluster_centroids_csv(cluster_centroids_csv)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    try:
        tf_enum = TransferFunction(transfer_fn.strip().lower())
    except ValueError as e:
        console.print(f"\n[red]Error:[/red] Invalid transfer-fn: {transfer_fn}")
        raise typer.Exit(code=1) from e

    route_config = PipelineConfig(
        transfer_function=tf_enum,
        include_shaper=_resolve_shaper_mode(shaper),
    )

    console.print(f"\n[bold]LutSmith Shot Routing[/bold]")
    console.print(f"  Shots manifest: {shots_manifest}")
    console.print(f"  Centroids:      {cluster_centroids_csv}")
    console.print(f"  Shots rows:     {len(shot_entries)}")
    console.print(f"  Clusters:       {len(centroid_rows)}")
    console.print(f"  Temporal win:   {temporal_window}")
    console.print(f"  Switch penalty: {switch_penalty:.3f}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Routing shots...", total=100)

        stage_offset = {"preprocess": 0.0, "solving": 80.0, "export": 95.0}
        stage_scale = {"preprocess": 80.0, "solving": 15.0, "export": 5.0}

        def on_progress(stage: str, fraction: float, message: str):
            base = stage_offset.get(stage, 0.0)
            scale = stage_scale.get(stage, 0.0)
            completed = min(base + scale * max(0.0, min(1.0, fraction)), 100.0)
            progress.update(task, completed=completed, description=message or stage)

        try:
            routing_result = run_shot_routing(
                shot_entries,
                centroid_rows,
                centroid_appearance,
                route_config,
                output_path=output,
                temporal_window=temporal_window,
                switch_penalty=switch_penalty,
                progress_callback=on_progress,
            )
            progress.update(task, completed=100, description="Complete")
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(code=1)

    console.print(f"[green]Routing CSV:[/green] {routing_result.output_path or output}")
    console.print(
        f"[dim]Shots: {len(routing_result.rows)}, "
        f"cluster switches: {routing_result.cluster_switches}[/dim]\n"
    )


@app.command("hald-gen")
def hald_generate(
    output: Path = typer.Option("hald_identity.tiff", "-o", "--output", help="Output image path."),
    level: int = typer.Option(8, "-l", "--level", help="Hald level (8 = 512x512, 64^3 LUT)."),
    bit_depth: int = typer.Option(16, "-b", "--bit-depth", help="Bit depth (8 or 16)."),
):
    """Generate a Hald CLUT identity image."""
    from lutsmith.hald.identity import generate_hald_identity, hald_image_size, hald_lut_size
    from lutsmith.io.image import save_image

    img_size = hald_image_size(level)
    lut_size = hald_lut_size(level)

    console.print(f"\n[bold]Hald Identity Generation[/bold]")
    console.print(f"  Level:    {level}")
    console.print(f"  Image:    {img_size}x{img_size}")
    console.print(f"  LUT size: {lut_size}^3 = {lut_size**3:,} nodes")
    console.print(f"  Depth:    {bit_depth}-bit")
    console.print()

    identity = generate_hald_identity(level)
    save_image(identity, output, bit_depth=bit_depth)

    console.print(f"[green]Saved:[/green] {output}")
    console.print(f"\n[dim]Process this image through your grading pipeline,")
    console.print(f"then use 'lutsmith hald-recon' to extract the LUT.[/dim]\n")


@app.command("hald-recon")
def hald_reconstruct(
    processed: Path = typer.Argument(..., help="Processed Hald image path."),
    output: Path = typer.Option("output.cube", "-o", "--output", help="Output LUT path."),
    level: int = typer.Option(8, "-l", "--level", help="Hald level used for identity."),
    target_size: int = typer.Option(0, "-s", "--target-size",
                                    help="Target LUT size (0 = native Hald size)."),
    title: str = typer.Option("LutSmith Hald LUT", "--title", help="LUT title."),
):
    """Reconstruct a 3D LUT from a processed Hald CLUT image."""
    from lutsmith.hald.identity import hald_lut_size
    from lutsmith.hald.reconstruct import reconstruct_from_hald
    from lutsmith.hald.resample import resample_lut
    from lutsmith.io.cube import write_cube
    from lutsmith.io.image import load_image

    native_size = hald_lut_size(level)

    console.print(f"\n[bold]Hald LUT Reconstruction[/bold]")
    console.print(f"  Input:       {processed}")
    console.print(f"  Hald level:  {level}")
    console.print(f"  Native size: {native_size}^3")

    img, _ = load_image(processed)

    lut = reconstruct_from_hald(img, level)

    # Resample if requested
    if target_size > 0 and target_size != native_size:
        console.print(f"  Resampling:  {native_size}^3 -> {target_size}^3")
        lut = resample_lut(lut, target_size)
        final_size = target_size
    else:
        final_size = native_size

    write_cube(output, lut, title=title)
    console.print(f"\n[green]Saved:[/green] {output} ({final_size}^3)\n")


@app.command()
def validate(
    source: Path = typer.Argument(..., help="Source image path."),
    target: Path = typer.Argument(..., help="Target image path."),
    lut_file: Path = typer.Argument(..., help="LUT file to validate (.cube)."),
    kernel: str = typer.Option("tetrahedral", "-k", "--kernel", help="Interpolation kernel."),
):
    """Validate an existing LUT against an image pair."""
    from lutsmith.io.cube import read_cube
    from lutsmith.io.image import load_image
    from lutsmith.pipeline.preprocess import sanitize_image
    from lutsmith.pipeline.validation import validate_lut

    console.print(f"\n[bold]LUT Validation[/bold]")
    console.print(f"  Source: {source}")
    console.print(f"  Target: {target}")
    console.print(f"  LUT:    {lut_file}")
    console.print()

    src_img, _ = load_image(source)
    tgt_img, _ = load_image(target)

    src_img = sanitize_image(src_img)
    tgt_img = sanitize_image(tgt_img)

    lut, meta = read_cube(lut_file)
    N = meta["size"]

    # Flatten for validation
    src_flat = src_img.reshape(-1, 3)
    tgt_flat = tgt_img.reshape(-1, 3)

    metrics = validate_lut(src_flat, tgt_flat, lut, N, kernel)
    _print_metrics(metrics)
    console.print()


def _print_metrics(metrics):
    """Display quality metrics in a formatted table."""
    table = Table(title="Quality Metrics", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Status", justify="center")

    # DeltaE metrics
    def de_status(val):
        if val < 1.0:
            return "[green]Excellent[/green]"
        elif val < 3.0:
            return "[green]Good[/green]"
        elif val < 5.0:
            return "[yellow]Fair[/yellow]"
        else:
            return "[red]Poor[/red]"

    table.add_row("Mean dE2000", f"{metrics.mean_delta_e:.2f}", de_status(metrics.mean_delta_e))
    table.add_row("Median dE2000", f"{metrics.median_delta_e:.2f}", de_status(metrics.median_delta_e))
    table.add_row("P95 dE2000", f"{metrics.p95_delta_e:.2f}", de_status(metrics.p95_delta_e))
    table.add_row("Max dE2000", f"{metrics.max_delta_e:.2f}", de_status(metrics.max_delta_e))
    table.add_row("", "", "")
    table.add_row("Total Variation", f"{metrics.total_variation:.4f}", "")
    table.add_row("Neutral Monotonic",
                  "Yes" if metrics.neutral_monotonic else f"No ({metrics.neutral_mono_violations} violations)",
                  "[green]OK[/green]" if metrics.neutral_monotonic else "[yellow]Warning[/yellow]")
    table.add_row("Out-of-Gamut", f"{metrics.oog_percentage:.2f}%",
                  "[green]OK[/green]" if metrics.oog_percentage < 1.0 else "[yellow]Warning[/yellow]")

    if metrics.num_total_bins > 0:
        table.add_row("Coverage", f"{metrics.coverage_percentage:.1f}%",
                      "[green]OK[/green]" if metrics.coverage_percentage > 10 else "[red]Low[/red]")
        table.add_row("Occupied Bins", f"{metrics.num_occupied_bins:,} / {metrics.num_total_bins:,}", "")

    console.print(table)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
