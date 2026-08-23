"""Simple development runner for micro2damask.

This script provides a convenient way to run the complete micro2damask
pipeline directly from an IDE.

Usage
-----
1. Place this file in the repository root, next to the ``src`` directory.
2. Adjust the values in the "USER SETTINGS" section below.
3. Run the script from PyCharm or another Python IDE.

The script can also be executed directly with:

    python run_micro2damask.py

No command-line arguments are required.

Notes
-----
The phase configuration is read from a separate YAML file. The software uses
the following generic image-phase convention:

    phase_id = 0 -> dark phase
    phase_id = 1 -> light phase

The physical names and DAMASK material definitions of these phases are defined
in ``phase_config_example.yaml`` or another user-specified YAML file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml


# ============================================================================
# PROJECT PATH AND PACKAGE IMPORT
# ============================================================================

# Repository root directory.
PROJECT_DIR = Path(__file__).resolve().parent

# Source directory used by the src-layout of the Python package.
SRC_DIR = PROJECT_DIR / "src"

# Add the local source directory to the Python search path.
#
# This allows the script to be executed directly from the repository without
# requiring an editable installation such as:
#
#     python -m pip install -e .
#
# For regular package use, installing the project is still recommended.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from micro2damask import Config, run_pipeline  # noqa: E402


# ============================================================================
# USER SETTINGS
# ============================================================================
# ---------------------------------------------------------------------------
# Input image
# ---------------------------------------------------------------------------

IMAGE_PATH = PROJECT_DIR / "examples" / "AlSi7Mg_1 Mitte 100x.tif"

# ---------------------------------------------------------------------------
# Phase configuration
# ---------------------------------------------------------------------------

PHASE_CONFIG_PATH = PROJECT_DIR / "phase_config_example.yaml"

# ---------------------------------------------------------------------------
# Image scale
# ---------------------------------------------------------------------------

# Physical resolution of the original image in micrometres per pixel.
UM_PER_PIXEL = 0.35

# ---------------------------------------------------------------------------
# Representative volume element (RVE)
# ---------------------------------------------------------------------------

# RVE dimensions in pixels of the original input image.
RVE_WIDTH = 512
RVE_HEIGHT = 512

# Optional position of the upper-left corner of the RVE.
#
# Set both values to None to let micro2damask select the RVE automatically.
RVE_X = None
RVE_Y = None

# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------

# Downsampling factor in the x-y plane.
# Example:
#   DOWNSAMPLE_FACTOR = 4 means that 4 x 4 image pixels are represented by one voxel.
DOWNSAMPLE_FACTOR = 4

# Number of voxel layers used to extrude the 2D microstructure in z direction.
NZ_LAYERS = 1

# ---------------------------------------------------------------------------
# Crystal orientations
# ---------------------------------------------------------------------------

# Allowed values:
#   "random"
#   "mtex_odf"
#
# Each phase can use its own orientation source.

DARK_PHASE_ORIENTATION_MODE = "mtex_odf"
LIGHT_PHASE_ORIENTATION_MODE = "random"

# MTEX ODF files.
# Only required if the corresponding orientation mode is "mtex_odf".

DARK_PHASE_ODF_PATH = PROJECT_DIR / "examples" / "ODF" / "odf_ferrite.txt"
LIGHT_PHASE_ODF_PATH = None

# Reproducible orientation sampling
RANDOM_SEED = 420

# Save RD / TD / ND IPF maps
SAVE_IPF_MAPS = True


# ---------------------------------------------------------------------------
# Morphological processing
# ---------------------------------------------------------------------------

# Select which image phase is affected by the configured morphological cleanup.
# Allowed values:
#   "dark"
#   "light"
MORPHOLOGY_TARGET_PHASE = "light"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# Root directory in which timestamped simulation/output directories are created.
OUTPUT_DIR = PROJECT_DIR / "output"

# Save diagnostic plots generated during image processing.
SAVE_DEBUG_PLOTS = True

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_phase_config(path: Path) -> Dict[str, Any]:
    """Load and validate the two-phase material configuration.

    Parameters
    ----------
    path
        Path to the YAML file containing the dark and light phase definitions.

    Returns
    -------
    dict
        Parsed phase configuration.

    Raises
    ------
    ValueError
        If the YAML file does not contain valid ``dark`` and ``light`` phase
        definitions.
    """

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "The phase configuration must contain a YAML mapping."
        )

    for phase in ("dark", "light"):
        if phase not in data or not isinstance(data[phase], dict):
            raise ValueError(
                f"The phase configuration is missing the '{phase}' section."
            )

        if not data[phase].get("name"):
            raise ValueError(
                f"The '{phase}' phase does not define a phase name."
            )

        if not isinstance(data[phase].get("material"), dict):
            raise ValueError(
                f"The '{phase}' phase does not contain a valid DAMASK "
                f"material definition."
            )

    return data


def build_config(phase_cfg: Dict[str, Any]) -> Config:
    return Config(
        # Input image and physical scale
        image_path=str(IMAGE_PATH),
        um_per_pixel=UM_PER_PIXEL,

        # Phase definitions
        dark_phase_name=str(phase_cfg["dark"]["name"]),
        light_phase_name=str(phase_cfg["light"]["name"]),
        dark_phase_material=phase_cfg["dark"]["material"],
        light_phase_material=phase_cfg["light"]["material"],

        # Morphological processing
        morphology_target_phase=MORPHOLOGY_TARGET_PHASE,

        # RVE definition
        rve_w=RVE_WIDTH,
        rve_h=RVE_HEIGHT,
        rve_x=RVE_X,
        rve_y=RVE_Y,

        # 2D-to-3D voxelization
        downsample_factor=DOWNSAMPLE_FACTOR,
        nz_layers=NZ_LAYERS,

        # Crystal orientations
        dark_phase_orientation_mode=DARK_PHASE_ORIENTATION_MODE,
        light_phase_orientation_mode=LIGHT_PHASE_ORIENTATION_MODE,

        dark_phase_odf_path=(
            str(DARK_PHASE_ODF_PATH)
            if DARK_PHASE_ODF_PATH is not None
            else None
        ),
        light_phase_odf_path=(
            str(LIGHT_PHASE_ODF_PATH)
            if LIGHT_PHASE_ODF_PATH is not None
            else None
        ),

        random_seed=RANDOM_SEED,
        save_ipf_maps=SAVE_IPF_MAPS,

        # Output settings
        output_root=str(OUTPUT_DIR),
        save_debug_plots=SAVE_DEBUG_PLOTS,
    )


# ============================================================================
# MAIN PROGRAM
# ============================================================================


def main() -> None:
    """Run the complete micro2damask processing pipeline."""

    print("=" * 70)
    print("micro2damask")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Validate required files and directories
    # ------------------------------------------------------------------------

    if not SRC_DIR.is_dir():
        raise FileNotFoundError(
            f"Source directory not found:\n{SRC_DIR}\n\n"
            "Place run_micro2damask.py in the repository root next to 'src'."
        )

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Input image not found:\n{IMAGE_PATH}\n\n"
            "Adjust IMAGE_PATH in the USER SETTINGS section."
        )

    if not PHASE_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Phase configuration not found:\n{PHASE_CONFIG_PATH}\n\n"
            "Adjust PHASE_CONFIG_PATH in the USER SETTINGS section."
        )

    # ------------------------------------------------------------------------
    # Load phase definitions and construct the main configuration
    # ------------------------------------------------------------------------

    phase_cfg = load_phase_config(PHASE_CONFIG_PATH)
    cfg = build_config(phase_cfg)

    # ------------------------------------------------------------------------
    # Print run configuration
    # ------------------------------------------------------------------------

    effective_voxel_size_xy = (
        cfg.um_per_pixel * cfg.downsample_factor
    )

    print(f"Input image:        {IMAGE_PATH}")
    print(f"Dark phase (0):     {cfg.dark_phase_name}")
    print(f"Light phase (1):    {cfg.light_phase_name}")
    print(f"Image resolution:   {cfg.um_per_pixel} µm/pixel")
    print(f"RVE size:           {cfg.rve_w} x {cfg.rve_h} pixels")
    print(f"Downsampling:       {cfg.downsample_factor}")
    print(
        f"Voxel size in x/y:  "
        f"{effective_voxel_size_xy:.6g} µm"
    )
    print(f"Number of z layers: {cfg.nz_layers}")
    print(f"Output directory:   {OUTPUT_DIR}")

    # ------------------------------------------------------------------------
    # Execute the processing pipeline
    # ------------------------------------------------------------------------

    print("\nRunning micro2damask pipeline ...\n")

    result = run_pipeline(cfg)

    # ------------------------------------------------------------------------
    # Validation report
    # ------------------------------------------------------------------------

    validation = result["validation"]

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(f"Model valid: {validation['ok']}")

    if validation["warnings"]:
        print("\nWarnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    if validation["errors"]:
        print("\nErrors:")
        for error in validation["errors"]:
            print(f"  - {error}")

    # ------------------------------------------------------------------------
    # Generated files
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Output directory: {result['output_dir']}")
    print(
        f"Material file:   "
        f"{result['material']['material_file']}"
    )
    print(
        f"Geometry file:   "
        f"{result['damask'].get('vti_path')}"
    )

    # ------------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------------

    if validation["ok"]:
        print(
            "\nPipeline completed successfully and the generated "
            "model passed validation."
        )
    else:
        print(
            "\nPipeline completed, but the generated model contains "
            "validation errors."
        )


if __name__ == "__main__":
    main()