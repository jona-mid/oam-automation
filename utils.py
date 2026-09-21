#!/usr/bin/env python3
"""
Shared utilities for metadata processing scripts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
import rasterio
import requests
from tqdm import tqdm


Truthy = {"true", "1", "yes", "y", "t"}


def configure_logging(log_file: str = "metadata_creation.log") -> logging.Logger:
    """Configure standard logging with file and console handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def configure_utf8_stdio():
    """Ensure Chinese characters print correctly on Windows consoles."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def read_csv(path: str, encoding: str = "utf-8-sig") -> List[Dict[str, str]]:
    """Read CSV file and return list of dictionaries."""
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(
    rows: List[Dict[str, Any]], path: str, extrasaction: str = "ignore"
) -> None:
    """Write list of dictionaries to CSV file."""
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction=extrasaction)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_pandas(path: str, encoding: str = "utf-8-sig") -> "pd.DataFrame":
    """Read CSV file using pandas."""
    return pd.read_csv(path, encoding=encoding)


def write_csv_pandas(df: "pd.DataFrame", path: str, index: bool = False) -> None:
    """Write DataFrame to CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")


def norm_filename(v: Any) -> str:
    """Normalize filename for comparison."""
    return str(v).strip().lower()


def load_selected_filenames(path: str) -> Set[str]:
    """Load set of filenames from a text file (one per line)."""
    selected: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = norm_filename(line)
            if name:
                selected.add(name)
    return selected


def parse_bool(v: Any) -> bool:
    """Parse boolean value from various formats."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if not s:
        return False
    return s in Truthy


def parse_bbox_string(bbox_str: str) -> Optional[Any]:
    """Parse bbox string '[min_lon, min_lat, max_lon, max_lat]' or '[min_lon min_lat max_lon max_lat]' -> list of floats."""
    if bbox_str is None:
        return None
    try:
        cleaned = bbox_str.strip("[]")
        if "," in cleaned:
            coords = [float(c.strip()) for c in cleaned.split(",")]
        else:
            coords = [float(c) for c in cleaned.split()]
        if len(coords) == 4:
            return coords
    except (ValueError, AttributeError):
        return None
    return None


def parse_iso_date(v: Any) -> Optional[dt.date]:
    """Parse YYYY-MM-DD date, handling embedded timestamps."""
    s = str(v).strip()
    if not s:
        return None
    s10 = s[:10]
    try:
        return dt.date.fromisoformat(s10)
    except ValueError:
        return None


def date_to_doy(date: dt.date) -> int:
    """Convert date to day-of-year."""
    return date.timetuple().tm_yday


def normalize_doy(doy: int, year_len: int = 366) -> int:
    """Normalize day-of-year to range [1, year_len]."""
    if year_len <= 0:
        raise ValueError("year_len must be positive")
    return ((int(doy) - 1) % year_len) + 1


def in_interval_wrap(
    capture_doy: int, start_doy: int, end_doy: int, *, year_len: int = 366
) -> bool:
    """Check if capture_doy falls within [start_doy, end_doy] with wrap-around."""
    c = normalize_doy(capture_doy, year_len=year_len)
    start = normalize_doy(start_doy, year_len=year_len)
    end = normalize_doy(end_doy, year_len=year_len)
    if start <= end:
        return start <= c <= end
    return c >= start or c <= end


def in_leaf_on(
    capture_doy: int, pheno_start: Optional[float], pheno_end: Optional[float]
) -> bool:
    """Check if capture_doy is within phenology window (handles wrap-around)."""
    if pheno_start is None or pheno_end is None:
        return False
    start = int(round(pheno_start))
    end = int(round(pheno_end))
    if start <= end:
        return start <= capture_doy <= end
    return capture_doy >= start or capture_doy <= end


def in_leaf_on_padded(
    capture_doy: int,
    pheno_start: Optional[float],
    pheno_end: Optional[float],
    *,
    pad_days: int = 0,
) -> bool:
    """Like in_leaf_on but expands window by pad_days on both sides."""
    if pheno_start is None or pheno_end is None:
        return False
    """Like in_leaf_on but expands window by pad_days on both sides."""
    start = int(round(pheno_start)) - int(pad_days)
    end = int(round(pheno_end)) + int(pad_days)
    return in_interval_wrap(capture_doy, start, end)


def load_phenology_map(
    phenology_csv: str,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Load phenology data: filename -> (pheno_start_doy, pheno_end_doy)."""
    pheno: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    with open(phenology_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise ValueError(f"Expected 'filename' column in {phenology_csv}")
        if (
            "pheno_start_doy" not in reader.fieldnames
            or "pheno_end_doy" not in reader.fieldnames
        ):
            raise ValueError(f"Expected 'pheno_start_doy' and 'pheno_end_doy' columns")

        for row in reader:
            fname = norm_filename(row.get("filename", ""))
            if not fname:
                continue

            start_raw = (row.get("pheno_start_doy") or "").strip()
            end_raw = (row.get("pheno_end_doy") or "").strip()
            try:
                start = float(start_raw) if start_raw else None
            except ValueError:
                start = None
            try:
                end = float(end_raw) if end_raw else None
            except ValueError:
                end = None

            pheno[fname] = (start, end)
    return pheno


def remap_platform(platform_value: Any) -> str:
    """Remap platform: uav->drone, aircraft->airborne."""
    if pd.isna(platform_value):
        return ""
    platform_str = str(platform_value).strip().lower()
    if platform_str == "uav":
        return "drone"
    elif platform_str == "aircraft":
        return "airborne"
    return platform_str


def extract_author(row: Dict[str, Any]) -> str:
    """Extract author from CSV row. Priority: contact (name part) -> provider -> openaerialmap.org."""
    if "contact" in row and row["contact"]:
        contact_str = str(row["contact"]).strip()
        if "," in contact_str:
            name_part = contact_str.split(",")[0].strip()
            if name_part:
                return name_part
        return contact_str

    if "provider" in row and row["provider"]:
        return str(row["provider"]).strip()

    return "openaerialmap.org"


def is_long_campaign(row: Dict[str, Any]) -> bool:
    """Check if acquisition duration > 7 days."""
    try:
        if "acquisition_start" not in row or "acquisition_end" not in row:
            return False

        start_str = row.get("acquisition_start")
        end_str = row.get("acquisition_end")

        if not start_str or not end_str:
            return False

        start_date = pd.to_datetime(start_str, errors="coerce")
        end_date = pd.to_datetime(end_str, errors="coerce")
        if pd.isna(start_date) or pd.isna(end_date):
            return False
        duration = end_date - start_date
        return duration > pd.Timedelta(days=7)
    except Exception:
        return False


def get_tif_dimensions(tif_path: str) -> Dict[str, Optional[int]]:
    """Read width, height, pixels from TIF file using rasterio."""
    try:
        with rasterio.open(tif_path) as src:
            w, h = src.width, src.height
            return {"width": w, "height": h, "pixels": w * h}
    except Exception:
        return {"width": None, "height": None, "pixels": None}


def load_image_size_csv(path: str) -> Dict[str, Dict[str, int]]:
    """Load image dimensions from CSV (Filename, Height, Width)."""
    image_dims: Dict[str, Dict[str, int]] = {}
    if not os.path.exists(path):
        return image_dims

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fname = r.get("Filename", "").strip()
                if fname:
                    try:
                        h = int(r.get("Height", 0))
                        w = int(r.get("Width", 0))
                        image_dims[fname] = {"height": h, "width": w, "pixels": h * w}
                    except ValueError:
                        continue
    except Exception:
        pass
    return image_dims


def load_uploaded_filenames(folder: str) -> Set[str]:
    """Load filenames from uploaded CSVs in a folder."""
    uploaded_files: Set[str] = set()
    if not os.path.exists(folder):
        return uploaded_files

    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    for csv_file in csv_files:
        try:
            with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if "filename" in reader.fieldnames:
                    for r in reader:
                        fname = r.get("filename", "").strip()
                        if fname:
                            uploaded_files.add(fname)
        except Exception:
            continue
    return uploaded_files


def match_tif_to_csv(
    tif_name: str, csv_rows: List[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """Match TIF filename to CSV row. Returns first match or None."""
    potential_tif = os.path.splitext(tif_name)[0]

    for row in csv_rows:
        csv_filename = row.get("filename", "")
        if csv_filename == potential_tif:
            return row

        csv_base = os.path.splitext(csv_filename)[0]
        if csv_base == os.path.splitext(potential_tif)[0]:
            return row

    return None


def batched_output_path(output_csv: str, batch_index: int) -> str:
    """Generate batched output path like *_batch001.csv."""
    p = Path(output_csv)
    stem = p.stem
    suffix = p.suffix or ".csv"
    return str(p.with_name(f"{stem}_batch{batch_index:03d}{suffix}"))


def parse_args(description: str) -> "argparse.ArgumentParser":
    """Create argument parser with common options."""
    import argparse

    return argparse.ArgumentParser(description=description)


def configure_download_logging(log_file: str = "download.log") -> logging.Logger:
    """Configure logging for download scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def get_download_headers() -> Dict[str, str]:
    """Return standard headers for downloads."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }


def load_download_state(state_file: str) -> Set[str]:
    """Load set of successfully downloaded files from state file."""
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_download_state(state_file: str, downloaded_files: Set[str]) -> None:
    """Save set of successfully downloaded files to state file."""
    try:
        with open(state_file, "w") as f:
            json.dump(list(downloaded_files), f)
    except Exception:
        pass


def download_file(
    url: str,
    output_path: str,
    timeout: int = 60,
    show_progress: bool = False,
    headers: Optional[Dict[str, str]] = None,
    pbar: Optional[Any] = None,
    max_retries: int = 3,
) -> bool:
    """Download a file from URL to output path with retry logic."""
    if not url:
        return False

    logger = logging.getLogger(__name__)

    for attempt in range(max_retries):
        try:
            if headers is None:
                headers = get_download_headers()

            response = requests.get(url, timeout=timeout, headers=headers, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("Content-Length", 0))
            filename = os.path.basename(output_path)

            with open(output_path, "wb") as f:
                if show_progress and total_size > 0 and pbar is not None:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} for {url} after {wait_time}s: {e}"
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Failed to download {url} after {max_retries} attempts: {e}"
                )
                return False
    return False


def extract_filename_from_url(url: str, default_ext: str = ".tif") -> str:
    """Extract filename from URL, adding default extension if needed."""
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)

    if not filename or "." not in filename:
        filename = f"file_{int(time.time())}{default_ext}"

    return filename


def download_parallel(
    tasks: List[Tuple[str, str, str]],
    workers: int = 1,
    desc: str = "Downloading",
    state_file: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = 3,
) -> Tuple[int, int]:
    """
    Download files in parallel or sequential mode.

    Args:
        tasks: List of (url, output_path, filename) tuples
        workers: Number of parallel workers (1 = sequential)
        desc: Description for progress bar
        state_file: Optional state file to track completed downloads
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts per file

    Returns:
        Tuple of (success_count, failure_count)
    """
    if not tasks:
        return 0, 0

    logger = logging.getLogger(__name__)
    downloaded_files: Set[str] = set()

    if state_file:
        downloaded_files = load_download_state(state_file)

    headers = get_download_headers()

    if workers == 1:
        success_count = 0
        failure_count = 0

        for url, output_path, filename in tqdm(tasks, desc=desc, unit="file"):
            if download_file(
                url,
                output_path,
                timeout=timeout,
                show_progress=True,
                headers=headers,
                max_retries=max_retries,
            ):
                success_count += 1
                if state_file:
                    downloaded_files.add(filename)
                    save_download_state(state_file, downloaded_files)
            else:
                failure_count += 1

        return success_count, failure_count

    from concurrent.futures import ThreadPoolExecutor, as_completed

    lock = threading.Lock()

    def download_task(args: Tuple[str, str, str]) -> Tuple[str, bool]:
        url, output_path, filename = args
        success = download_file(
            url, output_path, timeout=timeout, headers=headers, max_retries=max_retries
        )
        if success and state_file:
            with lock:
                downloaded_files.add(filename)
                save_download_state(state_file, downloaded_files)
        return (filename, success)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pbar = tqdm(total=len(tasks), desc=desc, unit="file")

        futures = {executor.submit(download_task, task): task[2] for task in tasks}
        success_count = 0
        failure_count = 0

        for future in as_completed(futures):
            filename, success = future.result()
            if success:
                success_count += 1
            else:
                failure_count += 1

            pbar.update(1)

        pbar.close()

    return success_count, failure_count
