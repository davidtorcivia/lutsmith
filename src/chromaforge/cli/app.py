"""ChromaForge CLI application.

Commands:
    extract     - Extract LUT from source/target image pair
    hald-gen    - Generate Hald CLUT identity image
    hald-recon  - Reconstruct LUT from processed Hald image
    validate    - Validate an existing LUT against image pair
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from chromaforge import __version__
from chromaforge.core.types import (
    ExportFormat,
    InterpolationKernel,
    PipelineConfig,
    RobustLoss,
    TransferFunction,
)

app = typer.Typer(
    name="chromaforge",
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
        console.print(f"ChromaForge v{__version__}")
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
        logging.getLogger("chromaforge").setLevel(logging.DEBUG)


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
    title: str = typer.Option("ChromaForge LUT", "--title", help="LUT title."),
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
    shaper: str = typer.Option(
        "auto",
        "--shaper",
        help="1D shaper LUT: auto, on, or off.",
    ),
):
    """Extract a 3D LUT from a source/target image pair."""
    shaper_mode = shaper.strip().lower()
    if shaper_mode not in {"auto", "on", "off"}:
        raise typer.BadParameter("Shaper must be one of: auto, on, off.")
    include_shaper = None
    if shaper_mode == "on":
        include_shaper = True
    elif shaper_mode == "off":
        include_shaper = False

    config = PipelineConfig(
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
        include_shaper=include_shaper,
        shadow_auto=shadow_auto,
        shadow_threshold=shadow_threshold,
        deep_shadow_threshold=deep_shadow_threshold,
    )

    from chromaforge.pipeline.runner import run_pipeline

    console.print(f"\n[bold]ChromaForge LUT Extraction[/bold]")
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

        def on_progress(stage: str, fraction: float, message: str):
            stage_weights = {
                "preprocess": 10,
                "sampling": 20,
                "solving": 50,
                "refinement": 10,
                "validation": 5,
                "export": 5,
            }
            base = sum(
                v for k, v in stage_weights.items()
                if list(stage_weights.keys()).index(k) < list(stage_weights.keys()).index(stage)
            ) if stage in stage_weights else 0
            weight = stage_weights.get(stage, 0)
            pct = base + weight * fraction
            progress.update(task, completed=pct,
                            description=f"{stage}: {message}" if message else stage)

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


@app.command("hald-gen")
def hald_generate(
    output: Path = typer.Option("hald_identity.tiff", "-o", "--output", help="Output image path."),
    level: int = typer.Option(8, "-l", "--level", help="Hald level (8 = 512x512, 64^3 LUT)."),
    bit_depth: int = typer.Option(16, "-b", "--bit-depth", help="Bit depth (8 or 16)."),
):
    """Generate a Hald CLUT identity image."""
    from chromaforge.hald.identity import generate_hald_identity, hald_image_size, hald_lut_size
    from chromaforge.io.image import save_image

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
    console.print(f"then use 'chromaforge hald-recon' to extract the LUT.[/dim]\n")


@app.command("hald-recon")
def hald_reconstruct(
    processed: Path = typer.Argument(..., help="Processed Hald image path."),
    output: Path = typer.Option("output.cube", "-o", "--output", help="Output LUT path."),
    level: int = typer.Option(8, "-l", "--level", help="Hald level used for identity."),
    target_size: int = typer.Option(0, "-s", "--target-size",
                                    help="Target LUT size (0 = native Hald size)."),
    title: str = typer.Option("ChromaForge Hald LUT", "--title", help="LUT title."),
):
    """Reconstruct a 3D LUT from a processed Hald CLUT image."""
    from chromaforge.hald.identity import hald_lut_size
    from chromaforge.hald.reconstruct import reconstruct_from_hald
    from chromaforge.hald.resample import resample_lut
    from chromaforge.io.cube import write_cube
    from chromaforge.io.image import load_image

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
    from chromaforge.io.cube import read_cube
    from chromaforge.io.image import load_image
    from chromaforge.pipeline.preprocess import sanitize_image
    from chromaforge.pipeline.validation import validate_lut

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
