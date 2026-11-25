#!/usr/bin/env python3
"""Download the GeoTIFFs listed in a CSV (the `uuid` column holds each file's URL)."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from utils import (
    configure_download_logging,
    download_parallel,
    extract_filename_from_url,
    load_download_state,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV with a 'uuid' column")
    parser.add_argument("--output-dir", default="tifs", help="Output directory")
    parser.add_argument(
        "--season",
        choices=["in_season", "out_of_season"],
        default=None,
        help="Only rows with this pheno_season (column written by phenology.py)",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument(
        "--skip-csv",
        action="append",
        default=[],
        help="CSV with a 'filename' column whose files are not downloaded (upload ledger, rejection list); repeatable",
    )
    args = parser.parse_args(argv)

    logger = configure_download_logging("tif_download.log")
    df = pd.read_csv(args.csv, dtype=str)
    logger.info(f"Loaded {len(df)} rows from {args.csv}")
    if "uuid" not in df.columns:
        logger.error("Column 'uuid' not found in CSV")
        return 1
    if args.season:
        if "pheno_season" not in df.columns:
            logger.error(f"--season {args.season} needs a 'pheno_season' column; run phenology.py first")
            return 1
        df = df[df["pheno_season"] == args.season]
        logger.info(f"{len(df)} rows with pheno_season '{args.season}'")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Files are recorded here only after a complete download, so an
    # interrupted one is fetched again on the next run.
    state_file = os.path.join(args.output_dir, ".download_state.json")
    downloaded_files = load_download_state(state_file)

    known = set()
    for path in args.skip_csv:
        if os.path.exists(path):
            known |= set(pd.read_csv(path, dtype=str)["filename"].dropna().str.strip().str.lower())

    tasks = []
    skipped = known_skipped = 0
    rows = df.dropna(subset=["uuid"]).drop_duplicates("uuid")
    for url, name in zip(rows["uuid"], rows.get("filename", [None] * len(rows))):
        # phenology.py's `filename` column wins (STAC uploads are named by item id).
        tif_filename = name if isinstance(name, str) and name else extract_filename_from_url(url, ".tif")
        if tif_filename.lower() in known:
            known_skipped += 1  # already uploaded or rejected; the upload dedup would drop it anyway
            continue
        output_path = os.path.join(args.output_dir, tif_filename)
        if tif_filename in downloaded_files and os.path.exists(output_path):
            skipped += 1
            continue
        tasks.append((url, output_path, tif_filename))

    if known_skipped:
        logger.info(f"{known_skipped} files skipped: already in the ledger or on the rejection list")
    if not tasks:
        logger.info(f"No files to download ({skipped} already downloaded)")
        return 0

    success, failed = download_parallel(
        tasks,
        workers=args.workers,
        desc="TIFs",
        state_file=state_file,
        max_retries=5,
    )
    logger.info(f"TIF download: {success} downloaded, {failed} failed, {skipped} already downloaded")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
