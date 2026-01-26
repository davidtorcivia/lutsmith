"""ARRI Look File export support.

Primary strategy: export .cube and invoke ARRI Reference Tool's look-builder
CLI if available. Falls back to documentation with conversion instructions.

ALF2 (.aml): LogC3 -> target (e.g., Rec709). 28-char filename limit.
ALF4 (.alf4): Log-to-Log creative modification. More flexible.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from chromaforge.errors import ExportError
from chromaforge.io.cube import write_cube

logger = logging.getLogger(__name__)

# Maximum filename length for ARRI camera filesystem
ALF2_MAX_FILENAME = 28


def find_arri_reference_tool(custom_path: Optional[str] = None) -> Optional[Path]:
    """Locate the ARRI Reference Tool's look-builder executable.

    Args:
        custom_path: User-specified path to the tool.

    Returns:
        Path to the executable, or None if not found.
    """
    if custom_path:
        p = Path(custom_path)
        if p.exists() and p.is_file():
            return p

    # Try common installation locations
    candidates = [
        "look-builder",
        "look-builder.exe",
    ]

    for name in candidates:
        found = shutil.which(name)
        if found:
            return Path(found)

    return None


def validate_alf2_filename(filepath: Path) -> None:
    """Validate filename length for ALF2 camera filesystem constraint."""
    if len(filepath.name) > ALF2_MAX_FILENAME:
        raise ExportError(
            f"ALF2 filename '{filepath.name}' is {len(filepath.name)} chars, "
            f"exceeds ARRI camera limit of {ALF2_MAX_FILENAME} characters "
            f"(including extension)"
        )


def export_arri(
    lut: "np.ndarray",
    output_path: str | Path,
    format: str = "aml",
    arri_tool_path: Optional[str] = None,
    cube_path: Optional[str | Path] = None,
) -> Path:
    """Export a LUT in ARRI format.

    If the ARRI Reference Tool is available, converts a .cube file to
    the requested ARRI format. Otherwise, exports .cube with instructions.

    Args:
        lut: (N, N, N, 3) LUT array.
        output_path: Desired output path (.aml or .alf4).
        format: "aml" (ALF2) or "alf4" (ALF4).
        arri_tool_path: Optional path to look-builder executable.
        cube_path: Optional pre-existing .cube file to convert.

    Returns:
        Path to the output file.
    """
    output = Path(output_path).resolve()

    if format == "aml":
        validate_alf2_filename(output)

    tool = find_arri_reference_tool(arri_tool_path)

    # Ensure we have a .cube file to convert
    if cube_path is None:
        cube_path = output.with_suffix(".cube")
        write_cube(cube_path, lut, title=output.stem)

    cube_path = Path(cube_path).resolve()

    if tool is not None:
        logger.info("Converting via ARRI Reference Tool: %s", tool)
        try:
            # Use list-form subprocess to prevent shell injection
            cmd = [str(tool), str(cube_path), "-o", str(output)]
            if format == "alf4":
                cmd.extend(["--format", "alf4"])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise ExportError(
                    f"ARRI look-builder failed (exit {result.returncode}): "
                    f"{result.stderr}"
                )

            logger.info("ARRI export successful: %s", output)
            return output

        except FileNotFoundError:
            logger.warning("ARRI Reference Tool not found at: %s", tool)
        except subprocess.TimeoutExpired:
            raise ExportError("ARRI Reference Tool timed out")

    # Fallback: export .cube with instructions
    logger.info(
        "ARRI Reference Tool not found. Exported .cube file: %s\n"
        "To convert to .%s format:\n"
        "  1. Download ARRI Reference Tool from arri.com\n"
        "  2. Run: look-builder %s -o %s",
        cube_path, format, cube_path, output,
    )
    return cube_path
