#!/usr/bin/env python3
"""
Unified download script for OpenAerialMap data.
Supports downloading thumbnails and TIF files.
"""

import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from utils import (
    configure_download_logging,
    download_parallel,
    extract_filename_from_url,
    load_download_state,
    save_download_state,
)


def calculate_file_hash(filepath):
    """Calculate MD5 hash of file content."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        return None


def find_and_remove_duplicate_thumbnails(thumbnails_dir):
    """
    Find exact duplicate thumbnails by comparing file content.
    Removes duplicates, keeping only the first occurrence.

    Returns:
        tuple: (sets_of_duplicates, files_removed)
    """
    if not os.path.exists(thumbnails_dir):
        return {}, 0

    png_files = [f for f in os.listdir(thumbnails_dir) if f.endswith(".png")]
    if not png_files:
        return {}, 0

    hash_to_files = defaultdict(list)

    for filename in png_files:
        filepath = os.path.join(thumbnails_dir, filename)
        file_hash = calculate_file_hash(filepath)
        if file_hash:
            hash_to_files[file_hash].append(filepath)

    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}

    removed_count = 0
    for file_hash, files in duplicates.items():
        for filepath in files[1:]:
            try:
                os.remove(filepath)
                removed_count += 1
            except Exception as e:
                pass

    return duplicates, removed_count


def cmd_thumbnails(args):
    """Download thumbnails from CSV."""
    logger = configure_download_logging("thumbnail_download.log")

    try:
        df = pd.read_csv(args.csv)
        logger.info(f"Loaded {len(df)} rows from {args.csv}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return 1

    if "property_thumbnail" not in df.columns:
        logger.error("Column 'property_thumbnail' not found in CSV")
        return 1

    # Apply season filter if specified
    if args.season:
        if "pheno_season" not in df.columns:
            logger.warning(
                f"--season={args.season} specified but 'pheno_season' column not found in CSV. "
                f"Downloading all thumbnails. Run phenology.py first to add season data."
            )
        else:
            original_count = len(df)
            df = df[df["pheno_season"] == args.season]
            filtered_count = len(df)
            logger.info(
                f"Filtered by season '{args.season}': {filtered_count}/{original_count} thumbnails match"
            )
            if filtered_count == 0:
                logger.warning(f"No thumbnails match season '{args.season}'")
                return 0

    Path(args.folder).mkdir(parents=True, exist_ok=True)

    existing = set()
    if not args.no_skip and os.path.exists(args.folder):
        existing = {f for f in os.listdir(args.folder) if f.endswith(".png")}
        logger.info(f"Found {len(existing)} existing thumbnails")

    tasks = []
    skipped = 0

    for idx, row in df.iterrows():
        url = row.get("property_thumbnail")
        if pd.isna(url) or not str(url).strip():
            logger.warning(f"Row {idx}: Empty thumbnail URL")
            continue

        filename = extract_filename_from_url(url, ".png")
        filepath = os.path.join(args.folder, filename)

        if not args.no_skip and filename in existing:
            skipped += 1
            continue

        tasks.append((url, filepath, filename))

    if not tasks:
        logger.info("No files to download")
        logger.info(f"Skipped (already existed): {skipped}")
        return 0

    success, failed = download_parallel(tasks, workers=args.workers, desc="Thumbnails")

    logger.info(f"\nThumbnail Download Summary:")
    logger.info(f"Successfully downloaded: {success}")
    logger.info(f"Failed to download: {failed}")
    logger.info(f"Skipped (already existed): {skipped}")
    logger.info(f"Total: {success + failed + skipped}")

    if args.dedupe:
        logger.info("\nFinding and removing duplicate thumbnails...")
        duplicates, removed = find_and_remove_duplicate_thumbnails(args.folder)
        if duplicates:
            logger.info(
                f"Found {len(duplicates)} sets of duplicates ({removed} files removed)"
            )
        else:
            logger.info("No duplicates found")

    return 0 if failed == 0 else 1


def cmd_tifs(args):
    """Download TIF files based on thumbnail filenames.

    Downloads directly to the output directory. JPEG conversion should only
    be run after all TIF downloads are complete."""
    logger = configure_download_logging("tif_download.log")

    if not args.all and not os.path.exists(args.thumbnails_dir):
        logger.error(f"Thumbnails directory not found: {args.thumbnails_dir}")
        return 1

    try:
        with open(args.csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        df = pd.DataFrame(rows)
        logger.info(f"Loaded {len(df)} rows from {args.csv}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return 1

    if "uuid" not in df.columns:
        logger.error("Column 'uuid' not found in CSV")
        return 1

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    state_file = os.path.join(args.output_dir, ".download_state.json")
    downloaded_files = load_download_state(state_file)
    logger.info(f"Loaded state: {len(downloaded_files)} files previously downloaded")

    if args.all:
        thumbnail_bases = None
        logger.info("All-download mode: using every row in the filtered CSV")
    else:
        thumbnail_files = [f for f in os.listdir(args.thumbnails_dir) if f.endswith(".png")]
        logger.info(f"Found {len(thumbnail_files)} thumbnail files")
        thumbnail_bases = {f[:-4]: f for f in thumbnail_files}

    tasks = []
    skipped = 0
    not_found = 0

    rows_to_download = df.to_dict("records") if thumbnail_bases is None else []
    iterator = rows_to_download if thumbnail_bases is None else thumbnail_bases.items()
    for item in iterator:
        if thumbnail_bases is None:
            thumbnail_file = ""
            row = item
            matching = pd.DataFrame([row])
        else:
            base_name, thumbnail_file = item
            matching = df[df["uuid"].str.contains(base_name, na=False)]

        if matching.empty:
            logger.warning(f"No matching CSV row for thumbnail: {thumbnail_file}")
            not_found += 1
            continue

        row = matching.iloc[0]
        tif_url = str(row["uuid"])

        tif_filename = extract_filename_from_url(tif_url, ".tif")
        output_path = os.path.join(args.output_dir, tif_filename)

        if tif_filename in downloaded_files and os.path.exists(output_path):
            skipped += 1
            continue

        tasks.append((tif_url, output_path, tif_filename))

    if not tasks:
        logger.info("No files to download")
        logger.info(f"Skipped (already existed): {skipped}")
        logger.info(f"Not found in CSV: {not_found}")
        return 0

    success, failed = download_parallel(
        tasks,
        workers=args.workers,
        desc="TIFs",
        state_file=state_file,
        max_retries=5,
    )

    logger.info(f"\nTIF Download Summary:")
    logger.info(f"Successfully downloaded: {success}")
    logger.info(f"Failed to download: {failed}")
    logger.info(f"Skipped (already existed): {skipped}")
    logger.info(f"Not found in CSV: {not_found}")
    logger.info(f"Total: {success + failed + skipped + not_found}")

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Download files from OpenAerialMap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Download command")

    p_thumb = subparsers.add_parser("thumbnails", help="Download thumbnail PNGs")
    p_thumb.add_argument(
        "--csv", default="openaerial_data_filtered.csv", help="CSV file"
    )
    p_thumb.add_argument("--folder", default="thumbnails", help="Output folder")
    p_thumb.add_argument("--workers", type=int, default=8, help="Parallel workers")
    p_thumb.add_argument(
        "--no-skip", action="store_true", help="Don't skip existing files"
    )
    p_thumb.add_argument(
        "--dedupe",
        action="store_true",
        help="Find and remove duplicate thumbnails after download",
    )
    p_thumb.add_argument(
        "--season",
        choices=["in_season", "out_of_season"],
        default=None,
        help="Filter thumbnails by phenology season (requires pheno_season column in CSV)",
    )
    p_thumb.set_defaults(func=cmd_thumbnails)

    p_tif = subparsers.add_parser("tifs", help="Download TIF files")
    p_tif.add_argument("--csv", default="openaerial_data_filtered.csv", help="CSV file")
    p_tif.add_argument(
        "--thumbnails-dir", default="thumbnails", help="Thumbnails directory"
    )
    p_tif.add_argument("--output-dir", default="tifs", help="Output directory")
    p_tif.add_argument("--workers", type=int, default=8, help="Parallel workers")
    p_tif.add_argument(
        "--all",
        action="store_true",
        help="Download every row in the CSV instead of only rows with local thumbnails",
    )
    p_tif.set_defaults(func=cmd_tifs)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
