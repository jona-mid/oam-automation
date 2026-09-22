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
            "oam_id": match.get("_id", ""),
            "property_license": match.get("property_license", ""),
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


def main():
    parser = argparse.ArgumentParser(
        description="Unified metadata creation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s tif --csv results.csv --output-dir tifs
  %(prog)s jpeg --source-metadata tif_metadata.csv --jpeg-folder jpegs
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

    args = parser.parse_args()

    if args.command == "tif":
        cmd_tif(args)
    elif args.command == "jpeg":
        cmd_jpeg(args)


if __name__ == "__main__":
    main()
