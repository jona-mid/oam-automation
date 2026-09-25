#!/usr/bin/env python3
"""
Convert TIF files to JPEG at 20 cm (0.2 m) resolution.

Handles both meter-based CRS (e.g. UTM) and degree-based CRS (e.g. WGS84):
- Meter CRS: resample in same CRS to 0.2 m per pixel.
- Degree CRS: reproject to a meter CRS (EPSG:3857) then resample to 0.2 m.
Output is written to a dedicated directory (default: jpegs/).

Uses WarpedVRT for memory-efficient processing of large rasters.
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, Resampling
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Target GSD in meters
TARGET_RESOLUTION_M = 0.2

# Meter CRS used when source is geographic (WGS84). EPSG:3857 is global.
DEFAULT_METER_CRS = CRS.from_epsg(3857)


def _convert_one(args_tuple):
    """Worker: convert a single TIF to 20 cm JPEG. Used with ThreadPoolExecutor."""
    input_path, output_path, target_crs = args_tuple
    return _convert_tif_to_20cm_jpeg(str(input_path), str(output_path), target_crs)


def _convert_tif_to_20cm_jpeg(tif_path: str, output_path: str, target_meter_crs: CRS):
    """
    Convert one TIF to JPEG at 0.2 m resolution using WarpedVRT.

    Uses GDAL's internal tiling via WarpedVRT for memory-efficient processing
    of large rasters. Processes in windows to avoid loading entire image.
    """
    try:
        with rasterio.open(tif_path) as src:
            src_crs = src.crs
            if src_crs is None:
                return False, Path(tif_path).name, "Missing CRS"

            is_geographic = src_crs.is_geographic

            # Determine destination CRS
            if is_geographic:
                dst_crs = target_meter_crs
            else:
                dst_crs = src_crs

            # Calculate output dimensions at target resolution
            dst_transform, width, height = calculate_default_transform(
                src_crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=(TARGET_RESOLUTION_M, TARGET_RESOLUTION_M),
            )

            num_bands = min(src.count, 3)

            # Create WarpedVRT for on-the-fly reprojection and resampling
            # GDAL handles this internally with efficient tiling
            vrt_options = {
                "crs": dst_crs,
                "resampling": Resampling.bilinear,
                "width": width,
                "height": height,
                "transform": dst_transform,
            }

            with WarpedVRT(src, **vrt_options) as vrt:
                # Prepare output JPEG profile
                kwargs = {
                    "driver": "JPEG",
                    "height": height,
                    "width": width,
                    "count": num_bands,
                    "dtype": "uint8",
                    "transform": dst_transform,
                    "crs": dst_crs,
                }

                with rasterio.open(output_path, "w", **kwargs) as dst:
                    # Process in windows/blocks - GDAL handles I/O efficiently
                    for _, window in vrt.block_windows(1):
                        # Read window from VRT (GDAL does the resampling)
                        # Only the colour bands: RGBA orthophotos (ODM) carry an alpha band
                        data = vrt.read(
                            indexes=list(range(1, num_bands + 1)), window=window, out_dtype=np.uint8
                        )

                        # Clip to valid uint8 range
                        data = np.clip(data, 0, 255).astype(np.uint8)

                        # Write window to output
                        dst.write(data, window=window)

            return True, Path(tif_path).name, None

    except Exception as e:
        return False, Path(tif_path).name, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Convert TIF files to JPEG at 20 cm resolution (meter and degree CRS supported)"
    )
    parser.add_argument(
        "--input-dir",
        default="tifs",
        help="Directory containing TIF files (default: tifs)",
    )
    parser.add_argument(
        "--output-dir",
        default="jpegs",
        help="Output directory for JPEGs (default: jpegs)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    if not tif_files:
        logger.warning("No TIF files found in %s", input_dir)
        return

    logger.info("Found %d TIF files; output to %s", len(tif_files), output_dir)

    # Output filename: base.tif -> base.jpeg (same base name, in output_dir)
    tasks = []
    for tif_path in tif_files:
        base = tif_path.stem
        out_path = output_dir / f"{base}.jpeg"
        tasks.append((tif_path, out_path, DEFAULT_METER_CRS))

    ok = 0
    fail = 0
    if args.workers <= 1:
        for tif_path, out_path, crs in tasks:
            success, name, err = _convert_tif_to_20cm_jpeg(
                str(tif_path), str(out_path), crs
            )
            if success:
                ok += 1
                logger.info("Converted %s", name)
            else:
                fail += 1
                logger.error("Error %s: %s", name, err)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_convert_one, task): task for task in tasks}
            for future in as_completed(futures):
                success, name, err = future.result()
                if success:
                    ok += 1
                    logger.info("Converted %s", name)
                else:
                    fail += 1
                    logger.error("Error %s: %s", name, err)

    logger.info("Done: %d ok, %d failed", ok, fail)
    if ok == 0:
        logger.error("All TIF conversions failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
