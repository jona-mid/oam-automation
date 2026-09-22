#!/usr/bin/env python3
"""Weekly wrapper: run the pipeline, gate candidates, upload via deadtrees-cli, update the ledger.

Steps: pipeline subprocess -> VLM gate (in_season AND leaf_on AND review success)
-> two-legged diff (ledger CSV + server-side file_name check) -> upload each
candidate -> append canonical rows to the ledger -> status file.

Machine-specific configuration (ledger path, credentials) lives in the
gitignored `.env`; see OAM_UPLOADED_CSV below.
"""

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import List, NamedTuple, Optional, Set

import pandas as pd
from dotenv import load_dotenv

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
VALID_PLATFORMS = {"drone", "airborne"}


@dataclass
class RunCounts:
    candidates: int = 0
    uploaded: int = 0
    failed: int = 0


class UploadSpec(NamedTuple):
    filename: str
    tif_path: Path
    kwargs: dict


@dataclass
class Preparation:
    candidates: int = 0
    specs: List[UploadSpec] = field(default_factory=list)
    rejected: int = 0


def normalize_filename(value) -> str:
    return str(value).strip().lower()


def format_en_date(value: date) -> str:
    """'September 21, 2026' without platform-specific strftime padding flags."""
    return f"{value:%B} {value.day}, {value:%Y}"


def run_pipeline(repo_root: Path, run_dir: Path, uploaded_after: Optional[str] = None) -> None:
    command = [sys.executable, str(repo_root / "pipeline.py"), "--output-dir", str(run_dir)]
    if uploaded_after:
        command += ["--uploaded-after-date", uploaded_after]
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
    return {normalize_filename(name) for name in df["filename"].dropna()}


def build_upload_kwargs(row: pd.Series, run_date: date) -> dict:
    """Build deadtrees-cli upload kwargs from a joined candidate row. Fails closed on unknown values."""
    acquisition_raw = row["acquisition_date"]
    if pd.isna(acquisition_raw) or not str(acquisition_raw).strip():
        raise ValueError("missing acquisition_date")
    acquisition = datetime.strptime(str(acquisition_raw).strip(), "%Y-%m-%d").date()
    platform = str(row["platform"]).strip().lower()
    authors = row["authors"]
    if pd.isna(authors) or not str(authors).strip():
        raise ValueError("missing authors")
    if platform not in VALID_PLATFORMS:
        raise ValueError(f"unknown platform value: {platform!r}")

    raw_license = row["property_license"]
    if pd.isna(raw_license) or not str(raw_license).strip():
        raise ValueError("missing property_license")
    raw_license = str(raw_license).strip()
    if raw_license not in deadtrees_seam.OAM_LICENSE_MAP:
        raise ValueError(f"unknown OAM license value: {raw_license!r}")

    oam_id = row["oam_id"]
    if pd.isna(oam_id) or not str(oam_id).strip():
        raise ValueError("missing oam_id")
    additional = (
        "This orthophoto data is available through OpenAerialMap, provided by Contributors "
        f"of Open Imagery Network. Licensed under {raw_license}. More information about this "
        f"dataset: https://api.openaerialmap.org/meta?_id={str(oam_id).strip()} "
        f"Accessed {format_en_date(run_date)}."
    )
    return {
        "authors": [str(authors)],
        "platform": platform,
        "license": deadtrees_seam.OAM_LICENSE_MAP[raw_license],
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


def write_status(path: Path, run_dir: Path, dry_run: bool, counts: RunCounts, exit_status: str, uploaded_after: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"run_dir: {run_dir}",
        f"dry_run: {dry_run}",
        f"uploaded_after: {uploaded_after or ''}",
        f"candidates: {counts.candidates}",
        f"uploaded: {counts.uploaded}",
        f"failed: {counts.failed}",
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
        default=None,
        help="Canonical upload ledger (diff leg 1 and append target); defaults to $OAM_UPLOADED_CSV",
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
    parser.add_argument(
        "--uploaded-after",
        default=None,
        help="Scrape only OAM uploads on/after this date (YYYY-MM-DD); "
        "default: the date of the last run's status file; first run requires it",
    )
    args = parser.parse_args(argv)
    if args.uploaded_csv is None:
        env_value = os.environ.get("OAM_UPLOADED_CSV", "").strip()
        if env_value:
            args.uploaded_csv = Path(env_value)
    if args.uploaded_csv is None:
        parser.error("--uploaded-csv or OAM_UPLOADED_CSV is required")
    return args


def resolve_uploaded_after(explicit: Optional[str], status_file: Path) -> Optional[str]:
    """Explicit flag wins; else the date of the last run's status timestamp."""
    if explicit:
        return explicit
    if status_file.exists():
        for line in status_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("timestamp:"):
                stamp = line.split(":", 1)[1].strip()
                return stamp.split("T")[0]
    return None


def prepare_candidates(
    gate: pd.DataFrame,
    ledger: Set[str],
    run_dir: Path,
    server_check: bool,
) -> Preparation:
    """Diff gate rows against the ledger and the platform, then build upload specs."""
    candidates = [
        row for _, row in gate.iterrows() if normalize_filename(row["filename"]) not in ledger
    ]

    if server_check and candidates:
        kept = []
        try:
            for row in candidates:
                filename = normalize_filename(row["filename"])
                if deadtrees_seam.file_exists_on_platform(filename):
                    print(f"  = {filename} already on the platform, skipping")
                else:
                    kept.append(row)
            candidates = kept
        except Exception as error:
            checked = len(kept)
            print(
                f"  ! Server-side check unavailable ({error}); "
                f"skipping the check for the remaining {len(candidates) - checked} candidates"
            )
            candidates = kept + candidates[checked:]

    specs: List[UploadSpec] = []
    rejected = 0
    for row in candidates:
        filename = normalize_filename(row["filename"])
        tif_path = run_dir / "tifs" / filename
        if not tif_path.exists():
            print(f"  ! {filename}: TIFF missing at {tif_path}, skipping")
            rejected += 1
            continue
        try:
            kwargs = build_upload_kwargs(row, date.today())
        except Exception as error:
            print(f"  ! {filename}: candidate rejected ({error}), skipping")
            rejected += 1
            continue
        specs.append(UploadSpec(filename=filename, tif_path=tif_path, kwargs=kwargs))
    return Preparation(candidates=len(candidates), specs=specs, rejected=rejected)


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args(argv)
    run_dir = (args.output_dir or ROOT / "runs" / datetime.now().strftime("%Y-%m-%d")).resolve()
    status_file = (args.status_file or run_dir.parent / "last_run.txt").resolve()
    uploaded_after = resolve_uploaded_after(args.uploaded_after, status_file)

    counts = RunCounts()
    exit_status = "failed"
    try:
        if args.dry_run and not (run_dir / "metadata" / "phenology_report.csv").exists():
            raise RuntimeError(
                f"Dry run needs an existing run dir with pipeline outputs; none at {run_dir}"
            )

        if not args.dry_run:
            if uploaded_after is None:
                raise RuntimeError(
                    "--uploaded-after is required for the first run "
                    "(no status file found to derive the last run date from); "
                    "this prevents an accidental full-catalog VLM audit"
                )
            run_pipeline(ROOT, run_dir, uploaded_after)

        gate = load_gate_candidates(run_dir)
        ledger = load_ledger_filenames(args.uploaded_csv)
        prep = prepare_candidates(gate, ledger, run_dir, server_check=not args.skip_server_check)
        counts.candidates = prep.candidates
        counts.failed = prep.rejected
        print(f"Gate passed {len(gate)} images; {prep.candidates} not in the ledger.")

        if args.dry_run:
            print(f"Dry run: {len(prep.specs)} would-be uploads, zero upload calls.")
            for spec in prep.specs:
                print(f"  * {spec.filename} <- {spec.tif_path}")
                for key, value in spec.kwargs.items():
                    print(f"      {key}: {value}")
            exit_status = "dry-run"
        else:
            for spec in prep.specs:
                try:
                    dataset_id = deadtrees_seam.upload_and_process(spec.tif_path, **spec.kwargs)
                except Exception as error:
                    print(f"  x {spec.filename}: upload failed: {error}")
                    counts.failed += 1
                    continue
                append_ledger_row(args.uploaded_csv, spec.filename, spec.kwargs)
                counts.uploaded += 1
                print(f"  + {spec.filename} uploaded (dataset {dataset_id}), ledger row appended")
            print(
                f"Summary: {counts.candidates} candidates, "
                f"{counts.uploaded} uploaded, {counts.failed} failed."
            )
            exit_status = "success" if counts.failed == 0 else "failed"
    except Exception as error:
        print(f"Run aborted: {error}")
        exit_status = "failed"
    finally:
        write_status(status_file, run_dir, args.dry_run, counts, exit_status, uploaded_after)

    return 0 if exit_status in ("success", "dry-run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
