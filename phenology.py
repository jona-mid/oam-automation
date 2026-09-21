#!/usr/bin/env python3
"""
Extract MODIS phenology data for bounding boxes.

This script extracts leaf-on phenology timing (start and end Day of Year)
from the MODIS phenology dataset for TIF file bounding boxes and adds
this information to the metadata CSV.
"""

import xarray as xr
import pandas as pd
import numpy as np
import rasterio
from rasterio import crs
from rasterio.warp import transform
from pathlib import Path
from typing import Tuple, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define MODIS Sinusoidal CRS
MODIS_CRS = crs.CRS.from_string("""PROJCS["unnamed",
GEOGCS["Unknown datum based upon the custom spheroid",
DATUM["Not specified (based on custom spheroid)",
SPHEROID["Custom spheroid",6371007.181,0]],
PRIMEM["Greenwich",0],
UNIT["degree",0.0174532925199433]],
PROJECTION["Sinusoidal"],
PARAMETER["longitude_of_center",0],
PARAMETER["false_easting",0],
PARAMETER["false_northing",0],
UNIT["Meter",1]]""")

# WGS84 CRS
WGS84_CRS = crs.CRS.from_epsg(4326)

# Default paths: repo-relative
_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PHENOLOGY_PATH = str(
    _REPO_ROOT
    / "phenology"
    / "modis_pheno_processed_v3.zarr"
)
DEFAULT_METADATA_CSV = str(_REPO_ROOT / "tif_metadata.csv")
DEFAULT_TIF_DIR = str(_REPO_ROOT / "tifs")


def load_phenology_data(zarr_path: str = DEFAULT_PHENOLOGY_PATH) -> xr.DataArray:
    """
    Load MODIS phenology data from zarr archive.

    Args:
        zarr_path: Path to the zarr phenology dataset

    Returns:
        xarray DataArray with phenology data
    """
    logger.info(f"Loading phenology data from {zarr_path}")
    pheno = xr.open_zarr(zarr_path).phenology40km.load()
    logger.info(f"Loaded phenology data with shape: {pheno.shape}")
    logger.info(f"Available variables: {pheno.coords['var'].values}")
    return pheno


def transform_bbox_to_modis(
    bbox_min_lon: float, bbox_min_lat: float, bbox_max_lon: float, bbox_max_lat: float
) -> Tuple[float, float]:
    """
    Transform bounding box from WGS84 to MODIS Sinusoidal CRS.
    Returns the centroid coordinates in MODIS projection.

    Args:
        bbox_min_lon: Minimum longitude (WGS84)
        bbox_min_lat: Minimum latitude (WGS84)
        bbox_max_lon: Maximum longitude (WGS84)
        bbox_max_lat: Maximum latitude (WGS84)

    Returns:
        Tuple of (x, y) coordinates in MODIS Sinusoidal CRS
    """
    # Calculate centroid in WGS84
    centroid_lon = (bbox_min_lon + bbox_max_lon) / 2.0
    centroid_lat = (bbox_min_lat + bbox_max_lat) / 2.0

    # Transform to MODIS Sinusoidal
    modis_coords = transform(WGS84_CRS, MODIS_CRS, [centroid_lon], [centroid_lat])

    return modis_coords[0][0], modis_coords[1][0]


def extract_phenology_for_bbox(
    bbox_min_lon: float,
    bbox_min_lat: float,
    bbox_max_lon: float,
    bbox_max_lat: float,
    pheno_data: xr.DataArray,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract phenology start and end DOY for a given bounding box.

    Args:
        bbox_min_lon: Minimum longitude (WGS84)
        bbox_min_lat: Minimum latitude (WGS84)
        bbox_max_lon: Maximum longitude (WGS84)
        bbox_max_lat: Maximum latitude (WGS84)
        pheno_data: xarray DataArray with phenology data

    Returns:
        Tuple of (start_doy, end_doy) or (None, None) if extraction fails
    """
    try:
        # Transform bbox centroid to MODIS coordinates
        x_modis, y_modis = transform_bbox_to_modis(
            bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat
        )

        # Select nearest phenology pixel
        pheno_val = pheno_data.sel(x=x_modis, y=y_modis, method="nearest")

        # Extract start and end DOY (scalar after selecting nearest x/y)
        start_val = np.asarray(pheno_val.sel(var="start_interp").values).squeeze()
        end_val = np.asarray(pheno_val.sel(var="end_interp").values).squeeze()

        pheno_start = (
            float(start_val.item()) if start_val.size == 1 else float(start_val.flat[0])
        )
        pheno_end = (
            float(end_val.item()) if end_val.size == 1 else float(end_val.flat[0])
        )

        # Handle NaN values
        if np.isnan(pheno_start) or np.isnan(pheno_end):
            logger.warning(
                f"NaN phenology values for bbox ({bbox_min_lon}, {bbox_min_lat}, {bbox_max_lon}, {bbox_max_lat})"
            )
            return None, None

        return float(pheno_start), float(pheno_end)

    except Exception as e:
        logger.error(
            f"Error extracting phenology for bbox ({bbox_min_lon}, {bbox_min_lat}, {bbox_max_lon}, {bbox_max_lat}): {e}"
        )
        return None, None


def extract_bbox_from_tif(tif_path: str) -> Tuple[float, float, float, float]:
    """
    Extract bounding box from TIF file metadata.

    Args:
        tif_path: Path to TIF file

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84
    """
    with rasterio.open(tif_path) as src:
        bounds = src.bounds  # (left, bottom, right, top)

        # Transform to WGS84 if needed
        if src.crs != WGS84_CRS:
            # Transform corners to WGS84
            min_lon, min_lat = transform(
                src.crs, WGS84_CRS, [bounds.left], [bounds.bottom]
            )
            max_lon, max_lat = transform(
                src.crs, WGS84_CRS, [bounds.right], [bounds.top]
            )
            return min_lon[0], min_lat[0], max_lon[0], max_lat[0]
        else:
            return bounds.left, bounds.bottom, bounds.right, bounds.top


def parse_date_to_doy(date_str: Optional[str]) -> Optional[int]:
    """Parse date string to 0-indexed Day of Year (0-365) to match MODIS. Supports ISO 8601 with timezone."""
    if not date_str:
        return None
    date_str = str(date_str).strip()
    if not date_str:
        return None

    # Strip timezone suffixes (Z, +00:00, etc.) for parsing
    if date_str.endswith("Z"):
        date_str = date_str[:-1]
    elif "+" in date_str:
        date_str = date_str.split("+")[0]

    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.timetuple().tm_yday - 1
        except ValueError:
            continue

    return None


def classify_season(
    capture_doy: Optional[int],
    pheno_start: Optional[float],
    pheno_end: Optional[float],
    pad_days: int = 0,
) -> str:
    """Classify whether capture is in season, out of season, or unknown."""
    if capture_doy is None or pheno_start is None or pheno_end is None:
        return "unknown"

    start = int(round(pheno_start)) - pad_days
    end = int(round(pheno_end)) + pad_days

    if start <= end:
        in_season = start <= capture_doy <= end
    else:
        in_season = capture_doy >= start or capture_doy <= end

    return "in_season" if in_season else "out_of_season"


def process_bboxes_from_csv(
    csv_path: str = DEFAULT_METADATA_CSV,
    output_path: Optional[str] = None,
    phenology_path: str = DEFAULT_PHENOLOGY_PATH,
    pad_days: int = 0,
) -> pd.DataFrame:
    """
    Process bounding boxes from CSV and add phenology data.

    Args:
        csv_path: Path to input CSV with bounding box data
        output_path: Path to output CSV (if None, overwrites input)
        phenology_path: Path to phenology zarr dataset
        pad_days: Number of days to expand phenology window on each side

    Returns:
        DataFrame with added phenology columns
    """
    logger.info(f"Processing bounding boxes from CSV: {csv_path}")
    logger.info(f"Using pad_days: {pad_days}")

    # Load metadata CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from CSV")

    # Check required columns
    required_cols = ["bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for date column (acquisition_start or capture_date)
    date_col = (
        "acquisition_start" if "acquisition_start" in df.columns else "capture_date"
    )
    if date_col not in df.columns:
        logger.warning(
            f"No date column found (expected 'acquisition_start' or 'capture_date'). Season classification will be 'unknown'."
        )
    else:
        logger.info(f"Using date column: {date_col}")

    # Load phenology data
    pheno_data = load_phenology_data(phenology_path)

    # Extract phenology for each bounding box
    logger.info("Extracting phenology data for each bounding box...")
    pheno_starts = []
    pheno_ends = []
    pheno_seasons = []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Processing row {idx}/{len(df)}")

        start_doy, end_doy = extract_phenology_for_bbox(
            row["bbox_min_lon"],
            row["bbox_min_lat"],
            row["bbox_max_lon"],
            row["bbox_max_lat"],
            pheno_data,
        )

        # Get capture date and convert to DOY
        capture_doy = None
        if date_col in df.columns:
            capture_doy = parse_date_to_doy(row.get(date_col))

        # Classify season
        season = classify_season(capture_doy, start_doy, end_doy, pad_days)

        pheno_starts.append(start_doy)
        pheno_ends.append(end_doy)
        pheno_seasons.append(season)

    # Add phenology columns to dataframe
    df["pheno_start_doy"] = pheno_starts
    df["pheno_end_doy"] = pheno_ends
    df["pheno_season"] = pheno_seasons

    # Save output
    if output_path is None:
        output_path = csv_path

    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    # Print summary statistics
    valid_count = df["pheno_start_doy"].notna().sum()
    logger.info(
        f"Successfully extracted phenology for {valid_count}/{len(df)} bounding boxes"
    )

    if valid_count > 0:
        logger.info(
            f"Phenology start DOY range: {df['pheno_start_doy'].min():.1f} - {df['pheno_start_doy'].max():.1f}"
        )
        logger.info(
            f"Phenology end DOY range: {df['pheno_end_doy'].min():.1f} - {df['pheno_end_doy'].max():.1f}"
        )

    # Season distribution
    season_counts = df["pheno_season"].value_counts()
    logger.info(f"Season classification: {season_counts.to_dict()}")

    return df


def process_bboxes_from_tifs(
    tif_dir: str = DEFAULT_TIF_DIR,
    output_csv: str = "tif_phenology.csv",
    phenology_path: str = DEFAULT_PHENOLOGY_PATH,
) -> pd.DataFrame:
    """
    Extract bounding boxes from TIF files and add phenology data.

    Args:
        tif_dir: Directory containing TIF files
        output_csv: Path to output CSV file
        phenology_path: Path to phenology zarr dataset

    Returns:
        DataFrame with TIF filenames, bounding boxes, and phenology data
    """
    logger.info(f"Processing TIF files from directory: {tif_dir}")

    # Find all TIF files
    tif_files = list(Path(tif_dir).glob("*.tif")) + list(Path(tif_dir).glob("*.tiff"))
    logger.info(f"Found {len(tif_files)} TIF files")

    if len(tif_files) == 0:
        logger.warning(f"No TIF files found in {tif_dir}")
        return pd.DataFrame()

    # Load phenology data
    pheno_data = load_phenology_data(phenology_path)

    # Process each TIF file
    results = []
    for idx, tif_path in enumerate(tif_files):
        if idx % 100 == 0:
            logger.info(f"Processing TIF {idx}/{len(tif_files)}")

        try:
            # Extract bounding box from TIF
            min_lon, min_lat, max_lon, max_lat = extract_bbox_from_tif(str(tif_path))

            # Extract phenology
            start_doy, end_doy = extract_phenology_for_bbox(
                min_lon, min_lat, max_lon, max_lat, pheno_data
            )

            results.append(
                {
                    "filename": tif_path.name,
                    "bbox_min_lon": min_lon,
                    "bbox_min_lat": min_lat,
                    "bbox_max_lon": max_lon,
                    "bbox_max_lat": max_lat,
                    "pheno_start_doy": start_doy,
                    "pheno_end_doy": end_doy,
                }
            )

        except Exception as e:
            logger.error(f"Error processing {tif_path.name}: {e}")

    # Create dataframe and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved results to {output_csv}")

    # Print summary statistics
    valid_count = df["pheno_start_doy"].notna().sum()
    logger.info(
        f"Successfully extracted phenology for {valid_count}/{len(df)} TIF files"
    )

    if valid_count > 0:
        logger.info(
            f"Phenology start DOY range: {df['pheno_start_doy'].min():.1f} - {df['pheno_start_doy'].max():.1f}"
        )
        logger.info(
            f"Phenology end DOY range: {df['pheno_end_doy'].min():.1f} - {df['pheno_end_doy'].max():.1f}"
        )

    return df


def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract MODIS phenology data for bounding boxes"
    )
    parser.add_argument(
        "--mode",
        choices=["csv", "tif"],
        default="csv",
        help="Processing mode: csv (from metadata CSV) or tif (from TIF files)",
    )
    parser.add_argument(
        "--csv", default=DEFAULT_METADATA_CSV, help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--tif-dir", default=DEFAULT_TIF_DIR, help="Directory containing TIF files"
    )
    parser.add_argument(
        "--output",
        help="Path to output CSV file (optional, defaults to overwriting input)",
    )
    parser.add_argument(
        "--phenology",
        default=DEFAULT_PHENOLOGY_PATH,
        help="Path to phenology zarr dataset",
    )
    parser.add_argument(
        "--pad-days",
        type=int,
        default=0,
        help="Expand phenology window by N days on each side (default: 0)",
    )

    args = parser.parse_args()

    if args.mode == "csv":
        process_bboxes_from_csv(
            csv_path=args.csv,
            output_path=args.output,
            phenology_path=args.phenology,
            pad_days=args.pad_days,
        )
    elif args.mode == "tif":
        output_csv = args.output if args.output else "tif_phenology.csv"
        process_bboxes_from_tifs(
            tif_dir=args.tif_dir, output_csv=output_csv, phenology_path=args.phenology
        )


if __name__ == "__main__":
    main()
