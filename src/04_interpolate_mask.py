# -*- coding: utf-8 -*-
"""
===============================================================================
PROJECT
    Pacific Warm Pool (PWP) Centroid and Area Analysis

SCRIPT
    04_interpolate_mask.py

PURPOSE
    Read the legacy IndonesiaNew0.msk mask, reconstruct its original
    global 1° × 1° grid (180 latitudes × 360 longitudes), and remap it
    to the NOAA OISST 0.25° × 0.25° grid (720 × 1440) using exact
    nearest-neighbour replication.

INPUT
    data/masks/IndonesiaNew0.msk
    data/processed/grid_lat.npy
    data/processed/grid_lon.npy

OUTPUT
    data/processed/pacific_mask_1deg.npy
    data/processed/pacific_mask_oisst.npy
    outputs/figures/pacific_mask_1deg.png
    outputs/figures/pacific_mask_oisst.png

MASK VALUES
    1 = Pacific Ocean cell included
    0 = Land or excluded cell

IMPORTANT
    The textual shape of IndonesiaNew0.msk is not its geographic shape.

    The file contains exactly 64,800 binary values:

        64,800 = 180 latitudes × 360 longitudes

    According to the legacy C-code convention:

        longitude varies fastest;
        longitude centres:   0.5° to 359.5°E;
        latitude centres:  -89.5° to  89.5°.

    Therefore, the correct reconstruction is:

        vector(64,800)
            -> reshape(180, 360)
            -> replicate each cell as a 4 × 4 block
            -> OISST mask (720, 1440)

AUTHOR
    Fabio Vieira Machado
===============================================================================
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PATHS
# =============================================================================

MASK_FILE = (
    PROJECT_ROOT
    / "data"
    / "masks"
    / "IndonesiaNew0.msk"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

GRID_LAT_FILE = PROCESSED_DIR / "grid_lat.npy"
GRID_LON_FILE = PROCESSED_DIR / "grid_lon.npy"

MASK_1DEG_FILE = PROCESSED_DIR / "pacific_mask_1deg.npy"
MASK_OISST_FILE = PROCESSED_DIR / "pacific_mask_oisst.npy"

FIGURE_1DEG_FILE = FIGURE_DIR / "pacific_mask_1deg.png"
FIGURE_OISST_FILE = FIGURE_DIR / "pacific_mask_oisst.png"


# =============================================================================
# GRID CONSTANTS
# =============================================================================

ORIGINAL_NLAT = 180
ORIGINAL_NLON = 360

EXPECTED_OISST_NLAT = 720
EXPECTED_OISST_NLON = 1440

EXPECTED_VALUES = ORIGINAL_NLAT * ORIGINAL_NLON

ORIGINAL_LAT = np.arange(
    -89.5,
    90.0,
    1.0,
    dtype=np.float64,
)

ORIGINAL_LON = np.arange(
    0.5,
    360.0,
    1.0,
    dtype=np.float64,
)


# =============================================================================
# FUNCTIONS
# =============================================================================

def validate_input_files() -> None:
    """Check whether all required input files exist."""

    required_files = [
        MASK_FILE,
        GRID_LAT_FILE,
        GRID_LON_FILE,
    ]

    missing_files = [
        file_path
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"  - {file_path}"
            for file_path in missing_files
        )

        raise FileNotFoundError(
            "Required input file(s) not found:\n"
            f"{missing_text}"
        )


def read_legacy_mask(mask_file: Path) -> np.ndarray:
    """
    Read all binary values from IndonesiaNew0.msk.

    Whitespace and line breaks are ignored. Every non-whitespace
    character must be either 0 or 1.

    Returns
    -------
    numpy.ndarray
        One-dimensional uint8 array containing 64,800 values.
    """

    text = mask_file.read_text(encoding="ascii")

    invalid_characters = sorted(
        {
            character
            for character in text
            if not character.isspace()
            and character not in {"0", "1"}
        }
    )

    if invalid_characters:
        raise ValueError(
            "Invalid character(s) found in legacy mask:\n"
            f"{invalid_characters}"
        )

    mask_vector = np.fromiter(
        (
            int(character)
            for character in text
            if character in {"0", "1"}
        ),
        dtype=np.uint8,
    )

    if mask_vector.size != EXPECTED_VALUES:
        raise ValueError(
            "Unexpected number of mask values.\n"
            f"Expected : {EXPECTED_VALUES:,}\n"
            f"Found    : {mask_vector.size:,}"
        )

    unique_values = np.unique(mask_vector)

    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            "The legacy mask contains non-binary values:\n"
            f"{unique_values}"
        )

    return mask_vector


def reconstruct_1deg_mask(
    mask_vector: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the original 180 × 360 geographic mask.

    Longitude varies fastest, corresponding to C-order reshaping.

    Returns
    -------
    numpy.ndarray
        Binary uint8 array with shape (180, 360).
    """

    mask_1deg = mask_vector.reshape(
        ORIGINAL_NLAT,
        ORIGINAL_NLON,
        order="C",
    )

    if mask_1deg.shape != (
        ORIGINAL_NLAT,
        ORIGINAL_NLON,
    ):
        raise ValueError(
            "Incorrect reconstructed mask shape:\n"
            f"{mask_1deg.shape}"
        )

    return mask_1deg.astype(np.uint8)


def load_oisst_grid() -> tuple[np.ndarray, np.ndarray]:
    """
    Load and validate the target OISST latitude and longitude vectors.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Latitude and longitude vectors.
    """

    latitude = np.asarray(
        np.load(GRID_LAT_FILE),
        dtype=np.float64,
    ).squeeze()

    longitude = np.asarray(
        np.load(GRID_LON_FILE),
        dtype=np.float64,
    ).squeeze()

    if latitude.ndim != 1:
        raise ValueError(
            "grid_lat.npy must contain a one-dimensional vector.\n"
            f"Found shape: {latitude.shape}"
        )

    if longitude.ndim != 1:
        raise ValueError(
            "grid_lon.npy must contain a one-dimensional vector.\n"
            f"Found shape: {longitude.shape}"
        )

    if latitude.size != EXPECTED_OISST_NLAT:
        raise ValueError(
            "Unexpected OISST latitude dimension.\n"
            f"Expected : {EXPECTED_OISST_NLAT}\n"
            f"Found    : {latitude.size}"
        )

    if longitude.size != EXPECTED_OISST_NLON:
        raise ValueError(
            "Unexpected OISST longitude dimension.\n"
            f"Expected : {EXPECTED_OISST_NLON}\n"
            f"Found    : {longitude.size}"
        )

    if np.any(~np.isfinite(latitude)):
        raise ValueError(
            "The OISST latitude vector contains invalid values."
        )

    if np.any(~np.isfinite(longitude)):
        raise ValueError(
            "The OISST longitude vector contains invalid values."
        )

    latitude_difference = np.diff(latitude)
    longitude_difference = np.diff(longitude)

    latitude_is_ascending = np.all(
        latitude_difference > 0
    )

    latitude_is_descending = np.all(
        latitude_difference < 0
    )

    if not (
        latitude_is_ascending
        or latitude_is_descending
    ):
        raise ValueError(
            "The OISST latitude vector is not monotonic."
        )

    if not np.all(longitude_difference > 0):
        raise ValueError(
            "The OISST longitude vector must be monotonically "
            "increasing."
        )

    return latitude, longitude


def expand_to_oisst(
    mask_1deg: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> np.ndarray:
    """
    Expand the 1° mask to the OISST 0.25° grid.

    Each original grid cell becomes a 4 × 4 block. The latitude
    orientation is adjusted to match the OISST coordinate vector.

    Parameters
    ----------
    mask_1deg
        Original binary mask with shape (180, 360).

    target_latitude
        OISST latitude coordinate vector.

    target_longitude
        OISST longitude coordinate vector.

    Returns
    -------
    numpy.ndarray
        Binary uint8 mask with shape (720, 1440).
    """

    latitude_factor = (
        target_latitude.size // ORIGINAL_NLAT
    )

    longitude_factor = (
        target_longitude.size // ORIGINAL_NLON
    )

    if latitude_factor != 4:
        raise ValueError(
            "Expected latitude expansion factor 4.\n"
            f"Found: {latitude_factor}"
        )

    if longitude_factor != 4:
        raise ValueError(
            "Expected longitude expansion factor 4.\n"
            f"Found: {longitude_factor}"
        )

    mask_oisst = np.repeat(
        np.repeat(
            mask_1deg,
            latitude_factor,
            axis=0,
        ),
        longitude_factor,
        axis=1,
    )

    # The legacy mask is ordered from south to north.
    # Reverse it only when the OISST latitude vector is north to south.
    if target_latitude[0] > target_latitude[-1]:
        mask_oisst = np.flipud(mask_oisst)

    expected_shape = (
        target_latitude.size,
        target_longitude.size,
    )

    if mask_oisst.shape != expected_shape:
        raise ValueError(
            "Unexpected expanded mask shape.\n"
            f"Expected : {expected_shape}\n"
            f"Found    : {mask_oisst.shape}"
        )

    original_active_cells = int(
        np.count_nonzero(mask_1deg)
    )

    expected_active_cells = (
        original_active_cells
        * latitude_factor
        * longitude_factor
    )

    expanded_active_cells = int(
        np.count_nonzero(mask_oisst)
    )

    if expanded_active_cells != expected_active_cells:
        raise ValueError(
            "Mask-cell conservation failed during expansion.\n"
            f"Expected active cells : "
            f"{expected_active_cells:,}\n"
            f"Found active cells    : "
            f"{expanded_active_cells:,}"
        )

    return mask_oisst.astype(np.uint8)


def save_mask_figure(
    mask: np.ndarray,
    longitude_min: float,
    longitude_max: float,
    latitude_min: float,
    latitude_max: float,
    output_file: Path,
    title: str,
    origin: str,
) -> None:
    """Save a geographic diagnostic figure of a binary mask."""

    figure, axis = plt.subplots(
        figsize=(14, 6),
    )

    image = axis.imshow(
        mask,
        origin=origin,
        extent=[
            longitude_min,
            longitude_max,
            latitude_min,
            latitude_max,
        ],
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="Blues",
    )

    axis.set_title(title)
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°)")

    axis.set_xlim(
        longitude_min,
        longitude_max,
    )

    axis.set_ylim(
        latitude_min,
        latitude_max,
    )

    axis.grid(
        linestyle=":",
        linewidth=0.5,
        alpha=0.4,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        pad=0.12,
        shrink=0.75,
        ticks=[0, 1],
    )

    colorbar.ax.set_xticklabels(
        ["Excluded", "Pacific"],
    )

    figure.tight_layout()

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Build and validate Pacific masks on 1° and OISST grids."""

    print("=" * 70)
    print("BUILDING PACIFIC MASK ON OISST GRID")
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_input_files()

    mask_vector = read_legacy_mask(
        MASK_FILE,
    )

    mask_1deg = reconstruct_1deg_mask(
        mask_vector,
    )

    oisst_latitude, oisst_longitude = (
        load_oisst_grid()
    )

    mask_oisst = expand_to_oisst(
        mask_1deg=mask_1deg,
        target_latitude=oisst_latitude,
        target_longitude=oisst_longitude,
    )

    np.save(
        MASK_1DEG_FILE,
        mask_1deg,
    )

    np.save(
        MASK_OISST_FILE,
        mask_oisst,
    )

    save_mask_figure(
        mask=mask_1deg,
        longitude_min=0.0,
        longitude_max=360.0,
        latitude_min=-90.0,
        latitude_max=90.0,
        output_file=FIGURE_1DEG_FILE,
        title="Legacy Pacific Mask — Original 1° Grid",
        origin="lower",
    )

    oisst_origin = (
        "lower"
        if oisst_latitude[0] < oisst_latitude[-1]
        else "upper"
    )

    save_mask_figure(
        mask=mask_oisst,
        longitude_min=float(
            oisst_longitude.min()
        ),
        longitude_max=float(
            oisst_longitude.max()
        ),
        latitude_min=float(
            oisst_latitude.min()
        ),
        latitude_max=float(
            oisst_latitude.max()
        ),
        output_file=FIGURE_OISST_FILE,
        title=(
            "Pacific Mask Expanded to "
            "NOAA OISST 0.25° Grid"
        ),
        origin=oisst_origin,
    )

    active_1deg = int(
        np.count_nonzero(mask_1deg)
    )

    active_oisst = int(
        np.count_nonzero(mask_oisst)
    )

    print()
    print("LEGACY MASK")
    print("-" * 40)
    print(f"Shape               : {mask_1deg.shape}")
    print(f"Total cells         : {mask_1deg.size:,}")
    print(f"Pacific cells       : {active_1deg:,}")
    print(
        f"Excluded cells      : "
        f"{mask_1deg.size - active_1deg:,}"
    )
    print(
        f"Latitude range      : "
        f"{ORIGINAL_LAT[0]:.1f} to "
        f"{ORIGINAL_LAT[-1]:.1f}"
    )
    print(
        f"Longitude range     : "
        f"{ORIGINAL_LON[0]:.1f} to "
        f"{ORIGINAL_LON[-1]:.1f}"
    )

    print()
    print("OISST MASK")
    print("-" * 40)
    print(f"Shape               : {mask_oisst.shape}")
    print(f"Total cells         : {mask_oisst.size:,}")
    print(f"Pacific cells       : {active_oisst:,}")
    print(
        f"Excluded cells      : "
        f"{mask_oisst.size - active_oisst:,}"
    )
    print(
        f"Latitude first/last : "
        f"{oisst_latitude[0]:.3f}, "
        f"{oisst_latitude[-1]:.3f}"
    )
    print(
        f"Longitude first/last: "
        f"{oisst_longitude[0]:.3f}, "
        f"{oisst_longitude[-1]:.3f}"
    )
    print("Latitude expansion  : 4×")
    print("Longitude expansion : 4×")

    print()
    print("FILES CREATED")
    print("-" * 40)
    print(MASK_1DEG_FILE)
    print(MASK_OISST_FILE)
    print(FIGURE_1DEG_FILE)
    print(FIGURE_OISST_FILE)

    print()
    print("=" * 70)
    print("Finished successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()