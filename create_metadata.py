#!/usr/bin/env python3
"""
Unified metadata creation tool for OpenAerialMap scraper.

Usage:
    python create_metadata.py tif --csv <file> --output-dir <dir> [--add-image-size]
    python create_metadata.py jpeg --source-metadata <file> --jpeg-folder <dir> [options]
    python create_metadata.py filter --selected <file> --metadata <file> --phenology <file> [options]
"""

import argparse
import os
import shutil
from pathlib import Path

import utils


def cmd_tif(args):
    """Create metadata for downloaded TIF files."""
    logger = utils.configure_logging()
    utils.configure_utf8_stdio()

    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        return

    if not os.path.exists(args.output_dir):
        logger.error(f"Output directory not found: {args.output_dir}")
        return

    logger.info(f"Reading CSV file: {args.csv}")
    csv_rows = utils.read_csv(args.csv)
    logger.info(f"Loaded {len(csv_rows)} rows from {args.csv}")

    if "uuid" not in csv_rows[0]:
        logger.error("Column 'uuid' not found in CSV")
        return

    tif_files = [f for f in os.listdir(args.output_dir) if f.endswith(".tif")]
    logger.info(f"Found {len(tif_files)} TIF files in {args.output_dir}")

    if not tif_files:
        logger.warning("No TIF files found")
        return

    metadata_output = args.metadata_output or os.path.join(
        args.output_dir, "tif_metadata.csv"
    )

    existing_metadata = {}
    if os.path.exists(metadata_output):
        try:
            existing_rows = utils.read_csv(metadata_output)
            for row in existing_rows:
                existing_metadata[row["filename"]] = row
            logger.info(f"Loaded {len(existing_metadata)} existing metadata entries")
        except Exception as e:
            logger.warning(f"Could not load existing metadata: {e}")

    metadata_list = []
    matched_count = 0
    not_found_count = 0

    for tif_filename in tif_files:
        base_name = tif_filename[:-4] if tif_filename.endswith(".tif") else tif_filename

        matching_rows = [r for r in csv_rows if base_name in r.get("uuid", "")]

        if not matching_rows:
            logger.warning(f"No matching CSV row for TIF file: {tif_filename}")
            not_found_count += 1
            continue

        row = matching_rows[0]

        image_dims = None
        if args.add_image_size:
            tif_path = os.path.join(args.output_dir, tif_filename)
            image_dims = utils.get_tif_dimensions(tif_path)

        authors = utils.extract_author(row)
        capture_date = ""
        if row.get("acquisition_start"):
            date = utils.parse_iso_date(row["acquisition_start"])
            if date:
                capture_date = date.isoformat()

        platform = utils.remap_platform(row.get("platform", ""))
        long_campaign = utils.is_long_campaign(row)

        width = height = pixels = ""
        if image_dims:
            if image_dims.get("width") is not None:
                width = str(image_dims["width"])
            if image_dims.get("height") is not None:
                height = str(image_dims["height"])
            if image_dims.get("pixels") is not None:
                pixels = str(image_dims["pixels"])

        entry = {
            "filename": tif_filename,
            "authors": authors,
            "origin": "openaerialmap.org",
            "capture_date": capture_date,
            "is_long_campaign": str(long_campaign),
            "platform": platform,
            "gsd": row.get("gsd", ""),
            "forest_cover_percentage": row.get("forest_percentage_gee", ""),
            "width": width,
            "height": height,
            "pixels": pixels,
        }

        for col in row:
            if col not in entry:
                entry[col] = row[col]

        metadata_list.append(entry)
        matched_count += 1

    logger.info(f"Matched {matched_count} TIF files to CSV rows")
    logger.info(f"Could not match {not_found_count} TIF files")

    all_metadata = list(existing_metadata.values()) + metadata_list

    if all_metadata:
        utils.write_csv(all_metadata, metadata_output)
        logger.info(
            f"Metadata CSV saved: {metadata_output} ({len(all_metadata)} entries)"
        )


def cmd_jpeg(args):
    """Create metadata for JPEG files based on TIF metadata."""
    logger = utils.configure_logging()
    utils.configure_utf8_stdio()

    if not os.path.exists(args.source_metadata):
        logger.error(f"Source metadata not found: {args.source_metadata}")
        return

    if not os.path.exists(args.jpeg_folder):
        logger.error(f"JPEG folder not found: {args.jpeg_folder}")
        return

    logger.info(f"Reading source metadata: {args.source_metadata}")
    csv_rows = utils.read_csv(args.source_metadata)
    logger.info(f"Loaded {len(csv_rows)} rows from source metadata")

    image_dims_map = {}
    if args.image_size_csv and os.path.exists(args.image_size_csv):
        logger.info(f"Reading image size CSV: {args.image_size_csv}")
        image_dims_map = utils.load_image_size_csv(args.image_size_csv)
        logger.info(f"Loaded dimensions for {len(image_dims_map)} images")

    uploaded_files = set()
    if args.uploaded_folder and os.path.exists(args.uploaded_folder):
        logger.info(f"Scanning uploaded folder: {args.uploaded_folder}")
        uploaded_files = utils.load_uploaded_filenames(args.uploaded_folder)
        logger.info(f"Found {len(uploaded_files)} uploaded files to exclude")

    jpeg_files = [
        f for f in os.listdir(args.jpeg_folder) if f.lower().endswith((".jpg", ".jpeg"))
    ]
    jpeg_files.sort()

    if args.prioritize_small_files:

        def size_key(name):
            if name in image_dims_map:
                return image_dims_map[name].get("pixels", float("inf"))
            try:
                return os.path.getsize(os.path.join(args.jpeg_folder, name))
            except OSError:
                return float("inf")

        jpeg_files.sort(key=lambda n: (size_key(n), n.lower()))

    logger.info(f"Found {len(jpeg_files)} JPEG files")

    metadata_list = []
    matched_jpeg_names = []
    matched = 0
    skipped_uploaded = 0
    not_found = 0

    for jpeg_filename in jpeg_files:
        potential_tif = os.path.splitext(jpeg_filename)[0]

        if potential_tif in uploaded_files:
            skipped_uploaded += 1
            continue

        match = None
        for row in csv_rows:
            if row.get("filename") == potential_tif:
                match = row
                break
            if (
                os.path.splitext(row.get("filename", ""))[0]
                == os.path.splitext(potential_tif)[0]
            ):
                match = row
                break

        if not match:
            logger.warning(f"No matching CSV row for JPEG: {jpeg_filename}")
            not_found += 1
            continue

        dims = image_dims_map.get(potential_tif)
        if not dims and match.get("width") and match.get("height"):
            try:
                w, h = int(match["width"]), int(match["height"])
                dims = {"width": w, "height": h, "pixels": w * h}
            except (ValueError, TypeError):
                pass

        acquisition_date = ""
        if match.get("capture_date"):
            date = utils.parse_iso_date(match["capture_date"])
            if date:
                acquisition_date = date.isoformat()

        platform = utils.remap_platform(match.get("platform", ""))

        additional_info = ""
        if match.get("_id"):
            additional_info = (
                f"This orthophoto data is available through OpenAerialMap, provided by Contributors of Open Imagery Network. "
                f"More information: https://api.openaerialmap.org/meta?_id={match['_id']} Accessed December 4, 2025."
            )

        entry = {
            "filename": match.get("filename", potential_tif),
            "acquisition_date": acquisition_date,
            "platform": platform,
            "licence": "CC BY",
            "authors": "Contributors of Open Imagery Network",
            "additional_information": additional_info,
            "gsd": match.get("gsd", ""),
            "pheno_start_doy": match.get("pheno_start_doy", ""),
            "pheno_end_doy": match.get("pheno_end_doy", ""),
        }

        if dims:
            entry["height"] = str(dims.get("height", ""))
            entry["width"] = str(dims.get("width", ""))
            entry["pixels"] = str(dims.get("pixels", ""))

        metadata_list.append(entry)
        matched_jpeg_names.append(jpeg_filename)
        matched += 1

    logger.info(
        f"Matched {matched} JPEG files, skipped {skipped_uploaded} uploaded, not found {not_found}"
    )

    if not metadata_list:
        logger.warning("No metadata entries created")
        return

    utils.write_csv(metadata_list, args.output_metadata)
    logger.info(
        f"Metadata CSV saved: {args.output_metadata} ({len(metadata_list)} entries)"
    )

    if args.batch_size > 0:
        _batch_jpeg_output(args, metadata_list, matched_jpeg_names, logger)


def _batch_jpeg_output(args, metadata_list, matched_jpeg_names, logger):
    """Handle batched output for JPEG metadata."""
    batch_size = args.batch_size
    total = len(metadata_list)
    num_batches = (total + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, total)
        batch_rows = metadata_list[start:end]
        batch_jpegs = matched_jpeg_names[start:end]

        batch_path = utils.batched_output_path(args.output_metadata, i + 1)
        utils.write_csv(batch_rows, batch_path)
        logger.info(f"Batch saved: {batch_path} ({len(batch_rows)} entries)")

        if args.jpeg_folder:
            _copy_batch_images(args, batch_path, batch_jpegs, logger)


def _copy_batch_images(args, batch_csv_path, jpeg_filenames, logger):
    """Copy images for a batch."""
    batch_csv = Path(batch_csv_path)
    out_dir = batch_csv.parent / "batch_images_from_csv" / batch_csv.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    for name in jpeg_filenames:
        src = Path(args.jpeg_folder) / name
        if not src.exists():
            missing += 1
            continue
        dst = out_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    logger.info(f"Batch images ready: {out_dir} (copied={copied}, missing={missing})")


def cmd_filter(args):
    """Filter metadata by selection, phenology, and campaign length."""
    logger = utils.configure_logging()
    utils.configure_utf8_stdio()

    selected = utils.load_selected_filenames(args.selected)
    pheno_map = utils.load_phenology_map(args.phenology)

    logger.info(f"Loaded {len(selected)} selected filenames")
    logger.info(f"Loaded {len(pheno_map)} phenology entries")

    metadata_rows = utils.read_csv(args.metadata)
    logger.info(f"Reading {len(metadata_rows)} rows from {args.metadata}")

    output_rows = []
    total = 0
    kept = 0
    dropped = {}

    for row in metadata_rows:
        total += 1
        fname = utils.norm_filename(row.get("filename", ""))

        if fname not in selected:
            dropped["not_selected"] = dropped.get("not_selected", 0) + 1
            continue

        if utils.parse_bool(row.get("is_long_campaign", "")):
            dropped["long_campaign"] = dropped.get("long_campaign", 0) + 1
            continue

        pixels = row.get("pixels", "")
        if pixels:
            try:
                pixel_count = int(pixels)
                if args.min_pixels is not None and pixel_count < args.min_pixels:
                    dropped["min_pixels"] = dropped.get("min_pixels", 0) + 1
                    continue
                if args.max_pixels is not None and pixel_count > args.max_pixels:
                    dropped["max_pixels"] = dropped.get("max_pixels", 0) + 1
                    continue
            except (ValueError, TypeError):
                pass

        capture_date = utils.parse_iso_date(row.get("capture_date", ""))
        if not capture_date:
            dropped["missing_date"] = dropped.get("missing_date", 0) + 1
            continue

        capture_doy = utils.date_to_doy(capture_date)

        pheno = pheno_map.get(fname)
        if not pheno or pheno[0] is None or pheno[1] is None:
            dropped["missing_phenology"] = dropped.get("missing_phenology", 0) + 1
            continue

        pheno_start, pheno_end = pheno
        in_season = utils.in_leaf_on(capture_doy, pheno_start, pheno_end)
        in_padded = utils.in_leaf_on_padded(
            capture_doy, pheno_start, pheno_end, pad_days=args.pad_days
        )

        if args.shoulder_only:
            if not in_padded:
                dropped["outside_window"] = dropped.get("outside_window", 0) + 1
                continue
            if in_season:
                dropped["in_season"] = dropped.get("in_season", 0) + 1
                continue
        else:
            if not in_padded:
                dropped["outside_window"] = dropped.get("outside_window", 0) + 1
                continue

        if args.include_debug_columns:
            row["pheno_start_doy"] = str(pheno_start)
            row["pheno_end_doy"] = str(pheno_end)
            row["capture_doy"] = str(capture_doy)

        output_rows.append(row)
        kept += 1

    utils.write_csv(output_rows, args.output)
    logger.info(f"Filtered metadata saved: {args.output} ({kept}/{total} kept)")

    for reason, count in sorted(dropped.items()):
        logger.info(f"  Dropped ({reason}): {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified metadata creation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s tif --csv results.csv --output-dir tifs
  %(prog)s jpeg --source-metadata tif_metadata.csv --jpeg-folder jpegs
  %(prog)s filter --selected selected.csv --metadata tif_metadata.csv --phenology phenology.csv
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_tif = subparsers.add_parser("tif", help="Create metadata for TIF files")
    p_tif.add_argument(
        "--csv",
        default="openaerial_data_filtered.csv",
        help="Input CSV with uuid column",
    )
    p_tif.add_argument(
        "--output-dir", default="tifs", help="Directory containing TIF files"
    )
    p_tif.add_argument(
        "--metadata-output", default=None, help="Output metadata CSV path"
    )
    p_tif.add_argument(
        "--add-image-size", action="store_true", help="Read dimensions from TIF files"
    )

    p_jpeg = subparsers.add_parser("jpeg", help="Create metadata for JPEG files")
    p_jpeg.add_argument(
        "--source-metadata", default="tif_metadata.csv", help="Source TIF metadata CSV"
    )
    p_jpeg.add_argument(
        "--jpeg-folder", default="jpegs", help="Folder containing JPEG files"
    )
    p_jpeg.add_argument(
        "--output-metadata",
        default="jpegs/jpeg_metadata.csv",
        help="Output CSV path",
    )
    p_jpeg.add_argument(
        "--image-size-csv", default="", help="Optional CSV with image dimensions"
    )
    p_jpeg.add_argument(
        "--uploaded-folder", default="", help="Folder with uploaded CSVs to exclude"
    )
    p_jpeg.add_argument(
        "--prioritize-small-files",
        action="store_true",
        help="Process smallest files first",
    )
    p_jpeg.add_argument(
        "--batch-size", type=int, default=0, help="Split output into batches"
    )

    p_filter = subparsers.add_parser(
        "filter", help="Filter metadata by selection/phenology"
    )
    p_filter.add_argument(
        "--selected", default="selected_stripped.csv", help="Selected filenames"
    )
    p_filter.add_argument(
        "--metadata", default="tif_metadata.csv", help="Input metadata CSV"
    )
    p_filter.add_argument(
        "--phenology", default="tif_phenology.csv", help="Phenology data CSV"
    )
    p_filter.add_argument(
        "--output", default="tif_metadata_filtered.csv", help="Output CSV path"
    )
    p_filter.add_argument(
        "--pad-days", type=int, default=0, help="Expand phenology window"
    )
    p_filter.add_argument(
        "--shoulder-only", action="store_true", help="Keep only shoulder season"
    )
    p_filter.add_argument(
        "--include-debug-columns",
        action="store_true",
        default=True,
        help="Add debug columns",
    )
    p_filter.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Minimum pixel count (height x width)",
    )
    p_filter.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Maximum pixel count (height x width)",
    )

    args = parser.parse_args()

    if args.command == "tif":
        cmd_tif(args)
    elif args.command == "jpeg":
        cmd_jpeg(args)
    elif args.command == "filter":
        cmd_filter(args)


if __name__ == "__main__":
    main()
