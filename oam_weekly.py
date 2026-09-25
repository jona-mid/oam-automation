#!/usr/bin/env python3
"""Weekly wrapper: run the pipeline, gate candidates, upload via deadtrees-cli, update the ledger.

Steps: pipeline subprocess -> VLM gate (in_season AND leaf_on AND review success)
-> dedup (ledger CSV, server-side file_name check, content hash) -> upload each
candidate -> append canonical rows to the ledger -> status file.

Machine-specific configuration (ledger path, credentials) lives in the
gitignored `.env`; see OAM_UPLOADED_CSV below.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Set

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
    rejected: int = 0


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


def run_pipeline(
    repo_root: Path,
    run_dir: Path,
    uploaded_after: Optional[str] = None,
    uploaded_before: Optional[str] = None,
) -> None:
    command = [sys.executable, str(repo_root / "pipeline.py"), "--output-dir", str(run_dir)]
    if uploaded_after:
        command += ["--uploaded-after-date", uploaded_after]
    if uploaded_before:
        command += ["--uploaded-before-date", uploaded_before]
    print("+", " ".join(command))
    subprocess.run(command, cwd=repo_root, check=True)


def pipeline_stopped_early(run_dir: Path) -> Optional[str]:
    """Reason the pipeline recorded for stopping before the VLM stage (nothing to review), else None."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("stopped_early")
    except (OSError, ValueError, AttributeError):
        return None


def load_gate_candidates(run_dir: Path) -> pd.DataFrame:
    """Join the VLM report with the canonical metadata rows and apply the upload gate."""
    if pipeline_stopped_early(run_dir):
        return pd.DataFrame(columns=["filename"])
    report = pd.read_csv(run_dir / "metadata" / "phenology_report.csv")
    jpeg_meta = pd.read_csv(run_dir / "metadata" / "jpeg_metadata.csv")
    # Both sides carry a `platform` column; keep the jpeg-metadata names clean.
    joined = report.merge(jpeg_meta, on="filename", how="inner", suffixes=("_report", ""))
    # OAM title and timestamps feed the capture-date checks in build_upload_kwargs.
    pheno_csv = run_dir / "metadata" / "phenology.csv"
    wanted = [c for c in ("title", "acquisition_start", "uploaded_at") if c not in joined.columns]
    if pheno_csv.exists() and wanted:
        extra = pd.read_csv(pheno_csv, usecols=lambda c: c in ["filename", *wanted], dtype=str)
        if len(extra.columns) > 1:
            joined = joined.merge(extra.drop_duplicates("filename"), on="filename", how="left")
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


def scrape_uploaded_at(run_dir: Path) -> Optional[str]:
    """Newest OAM upload date this run's scrape saw (uploaded_at column of the scrape CSV).

    Clamped to today's UTC date: one junk future-dated uploaded_at row must not
    poison the weekly chain.
    """
    csv_path = run_dir / "metadata" / "filtered.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, usecols=["uploaded_at"])
    parsed = pd.to_datetime(df["uploaded_at"], errors="coerce", utc=True)
    if not parsed.notna().any():
        return None
    newest = min(parsed.max(), pd.Timestamp(datetime.now(timezone.utc)))
    return newest.strftime("%Y-%m-%d")


def read_scrape_uploaded_at(status_file: Path) -> Optional[str]:
    """Stored weekly-chain value from the status file; None if missing or not a plain date."""
    if not status_file.exists():
        return None
    for line in status_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("scrape_uploaded_at:"):
            value = line.split(":", 1)[1].strip()
            if not value:
                return None
            try:
                date.fromisoformat(value)
            except ValueError:
                return None
            return value
    return None


def advance_scrape_uploaded_at(
    stored: Optional[str],
    scrape_at: Optional[str],
    uploaded_after: Optional[str],
) -> tuple:
    """Decide the weekly chain's next value from this run's outcome.

    The chain only advances when this run's window covered the chain point
    (uploaded_after <= stored); an ad-hoc window starting ahead of the chain
    would otherwise let every upload in the uncovered gap fall below all
    future bounds. Both values are clamped to today's UTC date so a junk
    future date can never enter the chain. Dry runs must not call this; they
    keep the stored value untouched.
    """
    today = datetime.now(timezone.utc).date()

    def clamp(value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            return None
        return min(parsed, today)

    stored_date = clamp(stored)
    scrape_date = clamp(scrape_at)
    if stored_date is None:
        return (scrape_date.isoformat() if scrape_date else None), None
    if scrape_date is None:
        return stored_date.isoformat(), None
    window_after = None
    if uploaded_after:
        try:
            window_after = date.fromisoformat(uploaded_after)
        except ValueError:
            window_after = None
    if window_after is not None and window_after > stored_date:
        warning = (
            f"this run's window starts {window_after.isoformat()}, after the chain point "
            f"{stored_date.isoformat()}; uploads in [{stored_date.isoformat()}, "
            f"{window_after.isoformat()}) were not covered; keeping the chain at "
            f"{stored_date.isoformat()} (re-run with --uploaded-after {stored_date.isoformat()} "
            "and a fresh --output-dir to cover the gap)"
        )
        return stored_date.isoformat(), warning
    return max(stored_date, scrape_date).isoformat(), None


MONTHS = {name: number for number, name in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1
)}
TITLE_DATE_TOLERANCE_DAYS = 30


def title_dates(title) -> List[date]:
    """Every calendar date a free-text OAM title can be read as.

    Numeric day/month dates are ambiguous (7/11/2015), so both orders are
    returned; a "Month YYYY" title is read as the 15th of that month.
    """
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return []
    text = str(title).lower()
    parts = set()
    for m in re.finditer(r"\b([a-z]{3})[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{2})\b", text):
        if m.group(1) in MONTHS:
            parts.add((int(m.group(3)), MONTHS[m.group(1)], int(m.group(2))))
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3})[a-z]*\.?,?\s+((?:19|20)\d{2})\b", text):
        if m.group(2) in MONTHS:
            parts.add((int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))))
    for m in re.finditer(r"\b((?:19|20)\d{2})[-_./](\d{1,2})[-_./](\d{1,2})\b", text):
        parts.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in re.finditer(r"\b(\d{1,2})\s*[-_./]\s*(\d{1,2})\s*[-_./]\s*((?:19|20)\d{2})\b", text):
        first, second, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        parts.update({(year, first, second), (year, second, first)})
    for m in re.finditer(r"\b(\d{1,2})\s+(\d{1,2})\s+((?:19|20)\d{2})\b", text):
        first, second, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        parts.update({(year, first, second), (year, second, first)})
    # Compact YYYYMMDD / six digits only right after "_" or "-" (WattleBay_260710),
    # so tile indices and other numbers are not read as dates. Six digits are
    # read both as YYMMDD and DDMMYY (Desp_220125 = 22 Jan 2025).
    for m in re.finditer(r"[_-](20\d{2})(\d{2})(\d{2})(?!\d)", text):
        parts.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in re.finditer(r"[_-](\d{2})(\d{2})(\d{2})(?!\d)", text):
        first, middle, last = int(m.group(1)), int(m.group(2)), int(m.group(3))
        parts.update({(2000 + first, middle, last), (2000 + last, middle, first)})
    if not parts:
        for m in re.finditer(r"\b([a-z]{3,9})\.?\s+((?:19|20)\d{2})\b", text):
            if m.group(1)[:3] in MONTHS:
                parts.add((int(m.group(2)), MONTHS[m.group(1)[:3]], 15))
    dates = []
    for year, month, day in parts:
        if not 2010 <= year <= date.today().year:
            continue
        try:
            dates.append(date(year, month, day))
        except ValueError:
            continue
    return dates


EVENT_WORDS = re.compile(
    r"fire|incendi|feu|brand|wildfire|tornado|hurrican|huracan|huracán|cyclon|typhoon|storm|tormenta"
    r"|flood|inund|earthquake|sismo|terremoto|quake|landslide|deslizamiento|eruption|disaster|desastre",
    re.IGNORECASE,
)


def check_title_date(title, acquisition: date, uploaded: Optional[date] = None) -> None:
    """Fail closed when the title names a capture date that contradicts acquisition_date.

    Some providers enter the upload date as the capture date; the title then
    still carries the real one. Every reading of the title must be more than
    TITLE_DATE_TOLERANCE_DAYS away for the image to be rejected. Readings after
    the upload cannot be the flight date and are ignored; in titles naming an
    event (fire, hurricane, ...) the date is the event's, and flying after it
    is expected.
    """
    dates = title_dates(title)
    if uploaded is not None:
        dates = [d for d in dates if (d - uploaded).days <= 1]
    if not dates:
        return
    closest = min(dates, key=lambda d: abs((d - acquisition).days))
    offset = (acquisition - closest).days
    if abs(offset) <= TITLE_DATE_TOLERANCE_DAYS:
        return
    if offset > 0 and EVENT_WORDS.search(str(title)):
        return
    raise ValueError(
        f"title date {closest.isoformat()} contradicts capture date {acquisition.isoformat()}"
    )


TIFF_DATE_TOLERANCE_DAYS = 30


def is_placeholder_timestamp(raw) -> bool:
    """A capture time of exactly local midnight on January 1: only the year was known.

    Stored in UTC, local midnight shows up as a whole hour on Jan 1 (west of
    Greenwich) or on Dec 31 from 10:00 (east of it).
    """
    stamp = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(stamp) or stamp.minute or stamp.second or stamp.microsecond:
        return False
    return (stamp.month == 1 and stamp.day == 1 and stamp.hour <= 14) or (
        stamp.month == 12 and stamp.day == 31 and stamp.hour >= 10
    )


def tiff_processing_date(tif_path: Path) -> Optional[date]:
    """TIFFTAG_DATETIME as written by ODM, Pix4D, etc. when the orthophoto was processed."""
    import rasterio

    # Extra evidence only: an unreadable tag means no signal, and a broken
    # file fails at upload anyway.
    try:
        with rasterio.open(tif_path) as src:
            value = src.tags().get("TIFFTAG_DATETIME") or ""
    except Exception:
        return None
    match = re.match(r"(\d{4}):(\d{2}):(\d{2})", value)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def check_capture_date(row: pd.Series, acquisition: date, tif_path: Optional[Path] = None) -> None:
    """Fail closed on capture dates that the metadata or the file itself contradicts.

    A wrong capture date both misleads the season check and lands on the
    platform, so each signal rejects the candidate (logged, counted as
    rejected, never uploaded):
    - capture after the OAM upload (impossible)
    - placeholder date (Jan 1, local midnight): the season is unknown
    - a date in the title that contradicts the capture date
    - a TIFF processing date well before the capture date (impossible)
    """
    uploaded = pd.to_datetime(row.get("uploaded_at"), utc=True, errors="coerce")
    if not pd.isna(uploaded) and (acquisition - uploaded.date()).days > 1:
        raise ValueError(
            f"capture date {acquisition.isoformat()} is after the OAM upload {uploaded.date().isoformat()}"
        )
    if is_placeholder_timestamp(row.get("acquisition_start")):
        raise ValueError(
            f"capture date {acquisition.isoformat()} is a placeholder (Jan 1, no time of day)"
        )
    check_title_date(row.get("title"), acquisition, None if pd.isna(uploaded) else uploaded.date())
    if tif_path is not None:
        processed = tiff_processing_date(tif_path)
        if processed is not None and (acquisition - processed).days > TIFF_DATE_TOLERANCE_DAYS:
            raise ValueError(
                f"TIFF was processed on {processed.isoformat()}, before the capture date {acquisition.isoformat()}"
            )


def build_upload_kwargs(row: pd.Series, run_date: date, tif_path: Optional[Path] = None) -> dict:
    """Build deadtrees-cli upload kwargs from a joined candidate row. Fails closed on unknown values."""
    acquisition_raw = row["acquisition_date"]
    if pd.isna(acquisition_raw) or not str(acquisition_raw).strip():
        raise ValueError("missing acquisition_date")
    acquisition = datetime.strptime(str(acquisition_raw).strip(), "%Y-%m-%d").date()
    check_capture_date(row, acquisition, tif_path)
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


def write_status(
    path: Path,
    run_dir: Path,
    dry_run: bool,
    counts: RunCounts,
    exit_status: str,
    uploaded_after: Optional[str] = None,
    scrape_uploaded_at: Optional[str] = None,
    uploaded_before: Optional[str] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # scrape_uploaded_at is the final chain value decided by main() via
    # advance_scrape_uploaded_at (or the stored value for dry runs); this
    # writer records it as given.
    lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"run_dir: {run_dir}",
        f"dry_run: {dry_run}",
        f"uploaded_after: {uploaded_after or ''}",
        f"scrape_uploaded_at: {scrape_uploaded_at or ''}",
        f"uploaded_before: {uploaded_before or ''}",
        f"candidates: {counts.candidates}",
        f"uploaded: {counts.uploaded}",
        f"failed: {counts.failed}",
        f"rejected: {counts.rejected}",
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
    parser.add_argument(
        "--uploaded-before",
        default=None,
        help="Scrape only OAM uploads up to and including this date (YYYY-MM-DD; "
        "the whole before day is included); "
        "ad-hoc windows only: requires --uploaded-after (or a resolvable date from the "
        "status file) with a common period in between; the weekly default never sets it",
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
    """Explicit flag wins; else the newest upload the last run's scrape saw; else its timestamp date."""
    if explicit:
        return explicit
    if status_file.exists():
        scrape_at = read_scrape_uploaded_at(status_file)
        if scrape_at:
            return scrape_at
        for line in status_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("timestamp:"):
                stamp_date = line.split(":", 1)[1].strip().split("T")[0]
                if stamp_date:
                    return stamp_date
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
        try:
            on_platform = deadtrees_seam.file_names_on_platform(
                [normalize_filename(row["filename"]) for row in candidates]
            )
        except Exception as error:
            print(
                f"  ! Server-side check unavailable ({error}); "
                f"skipping the check for {len(candidates)} candidates"
            )
        else:
            kept = []
            for row in candidates:
                filename = normalize_filename(row["filename"])
                if filename in on_platform:
                    print(f"  = {filename} already on the platform, skipping")
                else:
                    kept.append(row)
            candidates = kept

    if candidates:
        # Content-hash leg: skip identical files the platform already has,
        # and duplicates within this batch. Failure degrades like the
        # server-check leg.
        try:
            hashes: Any = {}
            for row in candidates:
                filename = normalize_filename(row["filename"])
                tif_path = run_dir / "tifs" / filename
                if tif_path.exists():
                    hashes[filename] = deadtrees_seam.file_hash(tif_path)
            known = deadtrees_seam.file_hashes_on_platform(list(hashes.values()))
        except Exception as error:
            print(f"  ! Content-hash check unavailable ({error}); skipping the leg")
        else:
            kept = []
            seen: Set[str] = set()
            for row in candidates:
                filename = normalize_filename(row["filename"])
                file_hash = hashes.get(filename)
                if file_hash and file_hash in known:
                    print(
                        f"  = {filename} same content already on the platform "
                        f"as dataset {known[file_hash]}, skipping"
                    )
                    continue
                if file_hash and file_hash in seen:
                    print(f"  = {filename} duplicate of an earlier candidate in this run, skipping")
                    continue
                if file_hash:
                    seen.add(file_hash)
                kept.append(row)
            candidates = kept

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
            kwargs = build_upload_kwargs(row, date.today(), tif_path)
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
    stored_scrape = read_scrape_uploaded_at(status_file)

    counts = RunCounts()
    exit_status = "failed"
    chain_value = stored_scrape
    try:
        if args.uploaded_before:
            if uploaded_after is None:
                raise RuntimeError(
                    "--uploaded-before requires an uploaded-after date "
                    "(explicit --uploaded-after or a previous run's status file); "
                    "the filter only applies date bounds when the after bound is present"
                )
            try:
                after_date = date.fromisoformat(uploaded_after)
                before_date = date.fromisoformat(args.uploaded_before)
            except ValueError as error:
                raise RuntimeError(
                    f"--uploaded-after/--uploaded-before must be YYYY-MM-DD dates: {error}"
                ) from error
            if after_date > before_date:
                raise RuntimeError(
                    f"--uploaded-before {args.uploaded_before} leaves no common period "
                    f"with --uploaded-after {uploaded_after}"
                )

        if (
            args.dry_run
            and not (run_dir / "metadata" / "phenology_report.csv").exists()
            and not pipeline_stopped_early(run_dir)
        ):
            raise RuntimeError(
                f"Dry run needs an existing run dir with pipeline outputs; none at {run_dir}"
            )

        if not args.dry_run and uploaded_after is None:
            raise RuntimeError(
                "--uploaded-after is required for the first run "
                "(no status file found to derive the last run date from); "
                "this prevents an accidental full-catalog VLM audit"
            )

        # A reused run dir silently keeps the previous window's scrape
        # outputs (the pipeline skips existing stages) while the status
        # file would record this run's window. Abort on mismatch. Dry runs
        # without explicit window flags just inspect whatever the dir
        # holds, so they skip the check.
        explicit_window = args.uploaded_after is not None or args.uploaded_before is not None
        if not args.dry_run or explicit_window:
            filtered_csv = run_dir / "metadata" / "filtered.csv"
            manifest_path = run_dir / "run_manifest.json"
            if filtered_csv.exists():
                recorded = None
                if manifest_path.exists():
                    try:
                        config = json.loads(manifest_path.read_text(encoding="utf-8"))["config"]
                        recorded = (config.get("uploaded_after_date"), config.get("uploaded_before_date"))
                    except (OSError, ValueError, KeyError, TypeError):
                        recorded = None
                if recorded is None:
                    print(
                        f"  ! {run_dir} holds scrape outputs without a readable "
                        f"{manifest_path.name}; its recorded window cannot be verified"
                    )
                else:
                    # The pipeline writes manifest config values with str(); normalize both sides.
                    def window_text(value):
                        return "None" if value is None else str(value)

                    recorded_window = tuple(window_text(value) for value in recorded)
                    requested_window = (window_text(uploaded_after), window_text(args.uploaded_before))
                    if recorded_window != requested_window:
                        raise RuntimeError(
                            f"{run_dir} already holds a different window "
                            f"(recorded after={recorded[0]!r}, before={recorded[1]!r}; "
                            f"requested after={uploaded_after!r}, before={args.uploaded_before!r}); "
                            "use a fresh --output-dir for each ad-hoc window"
                        )

        if not args.dry_run:
            run_pipeline(ROOT, run_dir, uploaded_after, args.uploaded_before)

        gate = load_gate_candidates(run_dir)
        ledger = load_ledger_filenames(args.uploaded_csv)
        prep = prepare_candidates(gate, ledger, run_dir, server_check=not args.skip_server_check)
        counts.candidates = prep.candidates
        counts.rejected = prep.rejected
        print(f"Gate passed {len(gate)} images; {prep.candidates} left after dedup (ledger, platform file_name, content hash).")

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
                f"Summary: {counts.candidates} candidates, {counts.uploaded} uploaded, "
                f"{counts.failed} failed, {counts.rejected} rejected (bad metadata, not retried)."
            )
            exit_status = "success" if counts.failed == 0 else "failed"
    except Exception as error:
        print(f"Run aborted: {error}")
        exit_status = "failed"
    finally:
        try:
            scrape_at = scrape_uploaded_at(run_dir)
        except Exception as error:
            print(f"  ! Could not derive scrape_uploaded_at ({error})")
            scrape_at = None
        # Dry runs never upload, so they must never advance the weekly chain.
        # A failed run holds the chain where it was, so the next scheduled run
        # re-covers its window (VLM errors, failed uploads, crashes); dedup
        # skips whatever did get uploaded. Only a successful run advances,
        # and only when its window covered the chain point.
        if args.dry_run:
            chain_value = stored_scrape
        elif exit_status != "success":
            chain_value = stored_scrape or uploaded_after
            if uploaded_after and chain_value:
                # The next scheduled run covers [chain, open end); only a
                # window reaching below the chain point is left uncovered.
                try:
                    rewind = date.fromisoformat(uploaded_after) < date.fromisoformat(chain_value)
                except ValueError:
                    rewind = False
                if rewind:
                    retry_flags = f"--uploaded-after {uploaded_after}"
                    if args.uploaded_before:
                        retry_flags += f" --uploaded-before {args.uploaded_before}"
                    print(
                        f"  ! run failed; the next scheduled run does not cover this ad-hoc window; "
                        f"to retry, re-run with {retry_flags} and the same --output-dir"
                    )
                else:
                    print(
                        f"  ! run failed; the weekly window stays at {chain_value}, so the next "
                        f"scheduled run retries it (or re-run now with --output-dir {run_dir})"
                    )
        else:
            chain_value, chain_warning = advance_scrape_uploaded_at(
                stored_scrape, scrape_at, uploaded_after
            )
            if chain_warning:
                print(f"  ! {chain_warning}")
        write_status(
            status_file,
            run_dir,
            args.dry_run,
            counts,
            exit_status,
            uploaded_after,
            chain_value,
            args.uploaded_before,
        )

    return 0 if exit_status in ("success", "dry-run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
