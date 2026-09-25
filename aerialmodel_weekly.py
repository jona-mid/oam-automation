#!/usr/bin/env python3
"""Weekly aerialmodel.com harvest: screen new projects, download in-season orthophotos, upload.

Steps: catalog (project IDs after the last run) -> screening from the public
ODM stats.json (capture date, GSD) + MODIS season -> download in-season
GeoTIFFs under 10 cm -> JPEGs -> VLM review -> gate (in_season AND leaf_on AND
review success) -> dedup (ledger, platform file_name, content hash) -> upload
-> ledger -> status file.

Project IDs grow monotonically, so the weekly chain is the highest project ID
a successful run covered (`last_project_id` in the status file). Projects
without a model yet are kept in `pending_ids.csv` and re-checked for
PENDING_DAYS days.
"""

import argparse
import csv
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

import aerialmodel
import deadtrees_seam
import oam_weekly
import phenology
import pipeline

ROOT = Path(__file__).resolve().parent
MAX_GSD_CM = 10.0
MIN_GSD_CM = 0.3  # below this the ODM scale is broken (e.g. 0.002 cm on unscaled models)
PAD_DAYS = 30
PENDING_DAYS = 28
VLM_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
VLM_MODEL = "google/gemini-3-flash-preview"
VLM_WORKERS = 4
JPEG_WORKERS = 8

LICENSE = "CC BY"
PLATFORM = "drone"
DATA_ACCESS = "private"
AUTHORS = ["aerialmodel.com"]

SCREEN_FIELDS = [
    "project_id", "filename", "name", "lat", "lng", "model_url", "storage_path",
    "capture_date", "gsd_cm", "pheno_start_doy", "pheno_end_doy", "pheno_season", "status",
]


def project_filename(project_id) -> str:
    return f"aerialmodel_{int(project_id)}.tif"


# --- status file and pending projects --------------------------------------


def read_status(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        values[key.strip()] = value.strip()
    return values


def read_chain(path: Path) -> Optional[int]:
    value = read_status(path).get("last_project_id", "")
    return int(value) if value.isdigit() else None


def write_status(path: Path, fields: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"timestamp: {datetime.now().isoformat(timespec='seconds')}"]
    lines += [f"{key}: {'' if value is None else value}" for key, value in fields.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_pending(path: Path) -> Dict[int, date]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["project_id"]): date.fromisoformat(row["first_seen"]) for row in csv.DictReader(handle)}


def write_pending(path: Path, pending: Dict[int, date]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["project_id", "first_seen"])
        for project_id in sorted(pending):
            writer.writerow([project_id, pending[project_id].isoformat()])


def select_window(items: List[dict], after_id: int, before_id: Optional[int], pending: Dict[int, date], today: date):
    """Projects to process now, the next pending set, and the highest ID in the window.

    Selected: projects with a model inside (after_id, before_id], plus pending
    projects that got their model since. Projects in the window without a
    model become pending; pending entries expire after PENDING_DAYS.
    """
    in_window = [
        item for item in items
        if item["id"] > after_id and (before_id is None or item["id"] <= before_id)
    ]
    selected = {item["id"]: item for item in in_window if item.get("modelUrl")}
    for item in items:
        if item["id"] in pending and item.get("modelUrl"):
            selected[item["id"]] = item
    next_pending = {
        project_id: first_seen for project_id, first_seen in pending.items()
        if project_id not in selected and today - first_seen <= timedelta(days=PENDING_DAYS)
    }
    for item in in_window:
        if not item.get("modelUrl"):
            next_pending.setdefault(item["id"], today)
    max_id = max((item["id"] for item in in_window), default=None)
    return sorted(selected.values(), key=lambda item: item["id"]), next_pending, max_id


def advance_chain(stored: Optional[int], after_id: int, max_window_id: Optional[int]):
    """Chain value after a successful run and an optional warning (same rules as the OAM chain)."""
    if stored is not None and after_id > stored:
        return stored, (
            f"this run's window starts after project {after_id}, beyond the chain point {stored}; "
            f"projects {stored + 1}..{after_id} were not covered; keeping the chain at {stored}"
        )
    if max_window_id is None:
        return stored if stored is not None else after_id, None
    return max(stored or 0, max_window_id), None


# --- stages -------------------------------------------------------------------


def read_manifest(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def write_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def screen_project(session, project: dict, pheno_data) -> dict:
    """One screened.csv row: storage path, capture date and GSD from stats.json, MODIS season."""
    row = {
        "project_id": project["id"],
        "filename": project_filename(project["id"]),
        "name": project.get("name", ""),
        "lat": project.get("lat"),
        "lng": project.get("lng"),
        "model_url": project.get("modelUrl", ""),
        "storage_path": "",
        "capture_date": "",
        "gsd_cm": "",
        "pheno_start_doy": "",
        "pheno_end_doy": "",
        "pheno_season": "",
        "status": "ok",
    }
    storage_path = aerialmodel.project_storage_path(session, project)
    if not storage_path:
        row["status"] = "no_storage_path"
        return row
    row["storage_path"] = storage_path
    capture, gsd = aerialmodel.parse_stats(aerialmodel.fetch_stats(session, storage_path))
    if capture is None:
        row["status"] = "no_capture_date"
        return row
    row["capture_date"] = capture.isoformat()
    row["gsd_cm"] = "" if gsd is None else round(gsd, 3)
    try:
        lat, lng = float(project["lat"]), float(project["lng"])
    except (KeyError, TypeError, ValueError):
        row["status"] = "no_location"
        return row
    start, end = phenology.extract_phenology_for_bbox(lng, lat, lng, lat, pheno_data)
    row["pheno_start_doy"] = "" if start is None else start
    row["pheno_end_doy"] = "" if end is None else end
    row["pheno_season"] = phenology.classify_season(
        phenology.parse_date_to_doy(row["capture_date"]), start, end, PAD_DAYS
    )
    return row


def screen(session, projects: List[dict], screened_csv: Path) -> None:
    """Append a screened row per project not yet in screened.csv (resumable)."""
    done = set()
    if screened_csv.exists():
        with screened_csv.open(newline="", encoding="utf-8") as handle:
            done = {int(row["project_id"]) for row in csv.DictReader(handle)}
    todo = [project for project in projects if project["id"] not in done]
    if not todo:
        return
    pheno_data = phenology.load_phenology_data()
    new_file = not screened_csv.exists()
    with screened_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCREEN_FIELDS)
        if new_file:
            writer.writeheader()
        for number, project in enumerate(todo, 1):
            writer.writerow(screen_project(session, project, pheno_data))
            handle.flush()
            if number % 50 == 0:
                print(f"  screened {number}/{len(todo)}")
            time.sleep(aerialmodel.REQUEST_DELAY)


def eligible_rows(screened: pd.DataFrame) -> pd.DataFrame:
    gsd = pd.to_numeric(screened["gsd_cm"], errors="coerce")
    return screened[
        (screened["status"] == "ok")
        & (screened["pheno_season"] == "in_season")
        & gsd.notna()
        & (gsd >= MIN_GSD_CM)
        & (gsd < MAX_GSD_CM)
    ]


def download_tifs(session, rows: pd.DataFrame, tifs: Path) -> None:
    """Download every eligible orthophoto not yet on disk; fail the run if any download fails."""
    missing = [row for _, row in rows.iterrows() if not (tifs / row["filename"]).exists()]
    if not missing:
        return
    aerialmodel.login(session, os.environ.get("AERIALMODEL_EMAIL", ""), os.environ.get("AERIALMODEL_PASSWORD", ""))
    failures = []
    for number, row in enumerate(missing, 1):
        print(f"  [{number}/{len(missing)}] downloading {row['filename']}")
        try:
            aerialmodel.download(session, aerialmodel.orthophoto_url(row["storage_path"]), tifs / row["filename"])
        except Exception as error:
            print(f"  x {row['filename']}: {error}")
            failures.append(row["filename"])
    if failures:
        raise RuntimeError(f"{len(failures)} orthophoto download(s) failed; re-run to retry")


def write_audit_inputs(rows: pd.DataFrame, tifs: Path, pheno_csv: Path, tif_metadata: Path) -> None:
    """The phenology and metadata CSVs aerial_phenology_audit's manifest reads."""
    import rasterio
    from rasterio.warp import transform_bounds

    pheno_rows, meta_rows = [], []
    for _, row in rows.iterrows():
        tif = tifs / row["filename"]
        if not tif.exists():
            continue
        with rasterio.open(tif) as src:
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds) if src.crs else src.bounds
            dimensions = f"[{src.width}, {src.height}]"
        pheno_rows.append({
            "filename": row["filename"],
            "classification": row["pheno_season"],
            "jpeg_filename": Path(row["filename"]).with_suffix(".jpeg").name,
            "capture_date": row["capture_date"],
        })
        meta_rows.append({
            "filename": row["filename"],
            "gsd": float(row["gsd_cm"]) / 100.0,
            "platform": PLATFORM,
            "bbox": "[" + ", ".join(f"{value:.6f}" for value in bounds) + "]",
            "properties.dimensions": dimensions,
            "acquisition_start": row["capture_date"],
        })
    pd.DataFrame(pheno_rows, columns=["filename", "classification", "jpeg_filename", "capture_date"]).to_csv(pheno_csv, index=False)
    pd.DataFrame(
        meta_rows, columns=["filename", "gsd", "platform", "bbox", "properties.dimensions", "acquisition_start"]
    ).to_csv(tif_metadata, index=False)


def load_gate(run_dir: Path) -> pd.DataFrame:
    """VLM report rows passing the gate, joined with their screened.csv rows."""
    report_csv = run_dir / "metadata" / "phenology_report.csv"
    if not report_csv.exists():
        return pd.DataFrame(columns=SCREEN_FIELDS)
    with report_csv.open(newline="", encoding="utf-8") as handle:
        report = pd.DataFrame(list(csv.DictReader(handle)))
    if report.empty:
        return pd.DataFrame(columns=SCREEN_FIELDS)
    screened = pd.read_csv(run_dir / "metadata" / "screened.csv", dtype=str)
    report = report[
        (report["modis_category"] == "in_season")
        & (report["tree_canopy_leaf_state"] == "leaf_on")
        & (report["review_status"] == "success")
    ]
    return report[["filename"]].merge(screened, on="filename", how="inner")


def build_aerialmodel_kwargs(row: pd.Series, run_date: date) -> dict:
    """Upload kwargs for one gate row; fails closed on a missing capture date or project ID."""
    raw_date = row.get("capture_date")
    if pd.isna(raw_date) or not str(raw_date).strip():
        raise ValueError("missing capture_date")
    capture = date.fromisoformat(str(raw_date).strip())
    project_id = str(row.get("project_id", "")).strip()
    if not project_id.isdigit():
        raise ValueError("missing project_id")
    url = aerialmodel.page_url(str(row.get("model_url") or ""))
    additional = (
        f"Drone orthophoto from aerialmodel.com, project {project_id}"
        + (f": {url}" if url else "")
        + f". Captured {oam_weekly.format_en_date(capture)}. Accessed {oam_weekly.format_en_date(run_date)}."
    )
    return {
        "authors": list(AUTHORS),
        "platform": PLATFORM,
        "license": LICENSE,
        "data_access": DATA_ACCESS,
        "acquisition_year": capture.year,
        "acquisition_month": capture.month,
        "acquisition_day": capture.day,
        "additional_information": additional,
        "citation_doi": None,
    }


# --- main -----------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=None, help="Run dir (default: runs_aerialmodel/<today>; pass the same dir to resume)")
    parser.add_argument("--uploaded-csv", type=Path, default=None, help="Upload ledger; defaults to $AERIALMODEL_UPLOADED_CSV")
    parser.add_argument("--status-file", type=Path, default=None, help="Status file (default: <output-dir parent>/last_run.txt)")
    parser.add_argument("--after-id", type=int, default=None, help="Process projects with ID > N (default: last_project_id from the status file; first run requires it)")
    parser.add_argument("--before-id", type=int, default=None, help="Only projects with ID <= N (ad-hoc windows / backlog chunks)")
    parser.add_argument("--screen-only", action="store_true", help="Catalog + screening only (public, no login, no downloads); prints what would be downloaded")
    parser.add_argument("--dry-run", action="store_true", help="Run every stage except the upload: list upload candidates; no uploads, no ledger appends, chain untouched")
    parser.add_argument("--skip-server-check", action="store_true", help="Skip the platform file_name duplicate check")
    args = parser.parse_args(argv)
    if args.uploaded_csv is None:
        env_value = os.environ.get("AERIALMODEL_UPLOADED_CSV", "").strip()
        if env_value:
            args.uploaded_csv = Path(env_value)
    if args.uploaded_csv is None and not args.screen_only:
        parser.error("--uploaded-csv or AERIALMODEL_UPLOADED_CSV is required")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args(argv)
    run_dir = (args.output_dir or ROOT / "runs_aerialmodel" / datetime.now().strftime("%Y-%m-%d")).resolve()
    status_file = (args.status_file or run_dir.parent / "last_run.txt").resolve()
    pending_file = status_file.with_name("pending_ids.csv")
    stored = read_chain(status_file)
    after_id = args.after_id if args.after_id is not None else stored
    metadata = run_dir / "metadata"
    tifs, jpegs = run_dir / "tifs", run_dir / "jpegs"
    screened_csv = metadata / "screened.csv"

    counts = oam_weekly.RunCounts()
    exit_status = "failed"
    chain_value, max_window_id, next_pending = stored, None, None
    try:
        if after_id is None:
            raise RuntimeError("--after-id is required for the first run (no status file with last_project_id)")
        if args.before_id is not None and args.before_id <= after_id:
            raise RuntimeError(f"--before-id {args.before_id} leaves no projects after ID {after_id}")
        manifest = read_manifest(run_dir)
        window = {"after_id": after_id, "before_id": args.before_id}
        if manifest.get("window") and manifest["window"] != window:
            raise RuntimeError(
                f"{run_dir} already holds a different window {manifest['window']} (requested {window}); "
                "use a fresh --output-dir for each window"
            )
        session = aerialmodel.make_session()
        for directory in (metadata, tifs, jpegs, run_dir / "logs"):
            directory.mkdir(parents=True, exist_ok=True)
        catalog_csv = metadata / "catalog.csv"
        if catalog_csv.exists():
            projects = pd.read_csv(catalog_csv).to_dict("records")
            max_window_id, next_pending = manifest.get("max_window_id"), manifest.get("next_pending")
            next_pending = {int(k): date.fromisoformat(v) for k, v in (next_pending or {}).items()}
        else:
            items = aerialmodel.fetch_catalog(session)
            projects, next_pending, max_window_id = select_window(
                items, after_id, args.before_id, read_pending(pending_file), date.today()
            )
            pd.DataFrame(projects, columns=["id", "name", "lat", "lng", "modelUrl"]).to_csv(catalog_csv, index=False)
            manifest = {
                "window": window,
                "max_window_id": max_window_id,
                "next_pending": {str(k): v.isoformat() for k, v in next_pending.items()},
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            write_manifest(run_dir, manifest)
        print(f"Catalog: {len(projects)} projects with a model to screen (window after {after_id}"
              + (f", up to {args.before_id}" if args.before_id else "") + ")")

        screen(session, projects, screened_csv)
        screened = pd.read_csv(screened_csv, dtype=str) if screened_csv.exists() else pd.DataFrame(columns=SCREEN_FIELDS)
        eligible = eligible_rows(screened)
        print(f"Screening: {len(screened)} screened, statuses {screened['status'].value_counts().to_dict()}; "
              f"{len(eligible)} in season with GSD {MIN_GSD_CM:g}-{MAX_GSD_CM:g} cm")

        if args.screen_only:
            for _, row in eligible.iterrows():
                print(f"  * {row['filename']}  captured {row['capture_date']}  GSD {row['gsd_cm']} cm  {aerialmodel.page_url(row['model_url'])}")
            exit_status = "screen-only"
            return 0

        if eligible.empty:
            manifest["stopped_early"] = "no in-season projects under the GSD limit"
            write_manifest(run_dir, manifest)
        else:
            download_tifs(session, eligible, tifs)
            tif_stems = {item.stem for item in tifs.glob("*.tif")}
            if tif_stems - {item.stem for item in jpegs.glob("*.jpeg")}:
                pipeline.run("tif_to_jpeg.py", "--input-dir", tifs, "--output-dir", jpegs, "--workers", JPEG_WORKERS, cwd=run_dir)
            pheno_csv, tif_metadata = metadata / "phenology.csv", metadata / "tif_metadata.csv"
            write_audit_inputs(eligible, tifs, pheno_csv, tif_metadata)
            pipeline.run_vlm_stages(run_dir, tifs, jpegs, pheno_csv, tif_metadata, VLM_ENDPOINT, VLM_MODEL, VLM_WORKERS)

        gate = load_gate(run_dir)
        ledger = oam_weekly.load_ledger_filenames(args.uploaded_csv)
        prep = oam_weekly.prepare_candidates(
            gate, ledger, run_dir, server_check=not args.skip_server_check, build_kwargs=build_aerialmodel_kwargs
        )
        counts.candidates, counts.rejected = prep.candidates, prep.rejected
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
                oam_weekly.append_ledger_row(args.uploaded_csv, spec.filename, spec.kwargs)
                counts.uploaded += 1
                print(f"  + {spec.filename} uploaded (dataset {dataset_id}), ledger row appended")
            print(f"Summary: {counts.candidates} candidates, {counts.uploaded} uploaded, "
                  f"{counts.failed} failed, {counts.rejected} rejected.")
            exit_status = "success" if counts.failed == 0 else "failed"
    except Exception as error:
        print(f"Run aborted: {error}")
        exit_status = "failed"
    finally:
        # Only upload runs touch the chain and the pending list: a successful
        # run advances (if its window covered the chain point), a failed run
        # holds the chain so the next run re-covers the window.
        if exit_status == "success":
            chain_value, warning = advance_chain(stored, after_id, max_window_id)
            if warning:
                print(f"  ! {warning}")
            if next_pending is not None:
                write_pending(pending_file, next_pending)
        elif exit_status == "failed" and after_id is not None:
            chain_value = stored if stored is not None else after_id
            print(f"  ! run failed; the chain stays at project {chain_value}, so the next run retries this window "
                  f"(or re-run now with --output-dir {run_dir})")
        if exit_status != "screen-only":
            write_status(status_file, {
                "run_dir": run_dir,
                "dry_run": args.dry_run,
                "after_id": after_id,
                "before_id": args.before_id,
                "last_project_id": chain_value,
                "candidates": counts.candidates,
                "uploaded": counts.uploaded,
                "failed": counts.failed,
                "rejected": counts.rejected,
                "exit": exit_status,
            })

    return 0 if exit_status in ("success", "dry-run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
