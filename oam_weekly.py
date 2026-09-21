#!/usr/bin/env python3
"""Weekly wrapper: run the pipeline, gate candidates, upload via deadtrees-cli, update the ledger.

Steps: pipeline subprocess -> VLM gate (in_season AND leaf_on AND review success)
-> two-legged diff (ledger CSV + server-side file_name check) -> upload each
candidate -> append canonical rows to the ledger -> status file.
"""

import argparse
import csv
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd

import deadtrees_seam

ROOT = Path(__file__).resolve().parent

LEDGER_COLUMNS = [
    "filename",
    "license",
    "platform",
    "authors",
    "acquisition_year",
    "acquisition_month",
    "acquisition_day",
    "data_access",
    "additional_information",
    "citation_doi",
]
VALID_LICENSES = {"CC BY", "CC BY-SA", "CC BY-NC-SA", "CC BY-NC", "MIT"}
VALID_PLATFORMS = {"drone", "airborne"}


def run_pipeline(repo_root: Path, run_dir: Path) -> None:
    command = [sys.executable, str(repo_root / "pipeline.py"), "--output-dir", str(run_dir)]
    print("+", " ".join(command))
    subprocess.run(command, cwd=repo_root, check=True)


def load_gate_candidates(run_dir: Path) -> pd.DataFrame:
    """Join the VLM report with the canonical metadata rows and apply the upload gate."""
    report = pd.read_csv(run_dir / "metadata" / "phenology_report.csv")
    jpeg_meta = pd.read_csv(run_dir / "metadata" / "jpeg_metadata.csv")
    # Both sides carry a `platform` column; keep the jpeg-metadata names clean.
    joined = report.merge(jpeg_meta, on="filename", how="inner", suffixes=("_report", ""))
    return joined[
        (joined["modis_category"] == "in_season")
        & (joined["tree_canopy_leaf_state"] == "leaf_on")
        & (joined["review_status"] == "success")
    ]


def load_ledger_filenames(csv_path: Path) -> Set[str]:
    """Normalized (lowercase) filenames already recorded as uploaded."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, dtype=str)
    return {str(name).strip().lower() for name in df["filename"].dropna()}


def format_en_date(value: date) -> str:
    """'September 21, 2026' without platform-specific strftime padding flags."""
    return f"{value:%B} {value.day}, {value:%Y}"


def build_upload_kwargs(row: pd.Series, run_date: date) -> dict:
    """Build deadtrees-cli upload kwargs from a joined candidate row. Fails closed on unknown values."""
    additional = row["additional_information"]
    additional = "" if pd.isna(additional) else str(additional)
    additional = re.sub(
        r"Accessed [A-Za-z]+ \d{1,2}, \d{4}",
        f"Accessed {format_en_date(run_date)}",
        additional,
    )
    acquisition_raw = row["acquisition_date"]
    if pd.isna(acquisition_raw) or not str(acquisition_raw).strip():
        raise ValueError("missing acquisition_date")
    acquisition = datetime.strptime(str(acquisition_raw).strip(), "%Y-%m-%d").date()
    platform = str(row["platform"]).strip().lower()
    license_str = str(row["licence"]).strip()
    authors = row["authors"]
    if pd.isna(authors) or not str(authors).strip():
        raise ValueError("missing authors")
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"unknown platform value: {platform!r}")
    if license_str not in VALID_LICENSES:
        raise ValueError(f"unknown license value: {license_str!r}")
    return {
        "authors": [str(authors)],
        "platform": platform,
        "license": license_str,
        "data_access": "public",
        "acquisition_year": acquisition.year,
        "acquisition_month": acquisition.month,
        "acquisition_day": acquisition.day,
        "additional_information": additional,
        "citation_doi": None,
    }


def append_ledger_row(csv_path: Path, filename: str, kwargs: dict) -> None:
    """Append one canonical-schema row for a successful upload."""
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(LEDGER_COLUMNS)
        writer.writerow(
            [
                filename,
                kwargs["license"],
                kwargs["platform"],
                kwargs["authors"][0],
                kwargs["acquisition_year"],
                kwargs["acquisition_month"],
                kwargs["acquisition_day"],
                kwargs["data_access"],
                kwargs["additional_information"],
                kwargs["citation_doi"] or "",
            ]
        )


def write_status(path: Path, run_dir: Path, dry_run: bool, counts: dict, exit_status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"run_dir: {run_dir}",
        f"dry_run: {dry_run}",
        f"candidates: {counts['candidates']}",
        f"uploaded: {counts['uploaded']}",
        f"failed: {counts['failed']}",
        f"exit: {exit_status}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Weekly run dir (default: runs/<today>; pass the same dir to resume)",
    )
    parser.add_argument(
        "--uploaded-csv",
        type=Path,
        default=Path(r"H:\projects\deadtrees\data_openaerialmap\metadata_uploaded.csv"),
        help="Canonical upload ledger (diff leg 1 and append target)",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="Status file path (default: <output-dir parent>/last_run.txt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List candidates and upload kwargs; no uploads, no ledger appends",
    )
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Skip the server-side duplicate check (diff leg 2)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = (args.output_dir or ROOT / "runs" / datetime.now().strftime("%Y-%m-%d")).resolve()
    status_file = (args.status_file or run_dir.parent / "last_run.txt").resolve()

    counts = {"candidates": 0, "uploaded": 0, "failed": 0}
    try:
        if args.dry_run and not (run_dir / "metadata" / "phenology_report.csv").exists():
            print(f"Dry run needs an existing run dir with pipeline outputs; none at {run_dir}")
            return 1

        if not args.dry_run:
            run_pipeline(ROOT, run_dir)

        gate = load_gate_candidates(run_dir)
        ledger = load_ledger_filenames(args.uploaded_csv)
        candidates = [
            row for _, row in gate.iterrows() if str(row["filename"]).strip().lower() not in ledger
        ]
        counts["candidates"] = len(candidates)
        print(f"Gate passed {len(gate)} images; {counts['candidates']} not in the ledger.")

        server_check_enabled = not args.skip_server_check
        upload_specs: List[Tuple[str, Path, dict]] = []
        for row in candidates:
            filename = str(row["filename"])
            if server_check_enabled:
                try:
                    if deadtrees_seam.file_exists_on_platform(filename):
                        print(f"  = {filename} already on the platform, skipping")
                        continue
                except Exception as error:
                    print(f"  ! Server-side check unavailable ({error}); continuing without it")
                    server_check_enabled = False
            tif_path = run_dir / "tifs" / filename
            if not tif_path.exists():
                print(f"  ! {filename}: TIFF missing at {tif_path}, skipping")
                counts["failed"] += 1
                continue
            try:
                kwargs = build_upload_kwargs(row, date.today())
            except Exception as error:
                print(f"  ! {filename}: candidate rejected ({error}), skipping")
                counts["failed"] += 1
                continue
            upload_specs.append((filename, tif_path, kwargs))

        if args.dry_run:
            print(f"Dry run: {len(upload_specs)} would-be uploads, zero upload calls.")
            for filename, tif_path, kwargs in upload_specs:
                print(f"  * {filename} <- {tif_path}")
                for key, value in kwargs.items():
                    print(f"      {key}: {value}")
            write_status(status_file, run_dir, True, counts, "dry-run")
            return 0

        for filename, tif_path, kwargs in upload_specs:
            try:
                dataset_id = deadtrees_seam.upload_and_process(tif_path, **kwargs)
            except Exception as error:
                print(f"  x {filename}: upload failed: {error}")
                counts["failed"] += 1
                continue
            append_ledger_row(args.uploaded_csv, filename, kwargs)
            counts["uploaded"] += 1
            print(f"  + {filename} uploaded (dataset {dataset_id}), ledger row appended")

        print(
            f"Summary: {counts['candidates']} candidates, "
            f"{counts['uploaded']} uploaded, {counts['failed']} failed."
        )
        exit_status = "success" if counts["failed"] == 0 else "failed"
        write_status(status_file, run_dir, False, counts, exit_status)
        return 0 if counts["failed"] == 0 else 1
    except Exception as error:
        print(f"Run aborted: {error}")
        write_status(status_file, run_dir, args.dry_run, counts, "failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
