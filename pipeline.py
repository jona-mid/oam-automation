#!/usr/bin/env python3
"""Configurable, resumable OpenAerialMap processing pipeline."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script, *args, cwd):
    command = [sys.executable, str(ROOT / script), *map(str, args)]
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def count_files(path, suffixes):
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def csv_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("run"))
    parser.add_argument("--uploaded-after-date", default=None)
    parser.add_argument("--uploaded-before-date", default=None)
    parser.add_argument("--forest-min", type=float, default=None)
    parser.add_argument("--forest-max", type=float, default=None)
    parser.add_argument("--platform", nargs="+", default=["uav", "aircraft"])
    parser.add_argument("--pad-days", type=int, default=30)
    parser.add_argument("--thumbnail-workers", type=int, default=8)
    parser.add_argument("--tif-workers", type=int, default=8)
    parser.add_argument("--jpeg-workers", type=int, default=8)
    parser.add_argument("--vlm-endpoint", default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--vlm-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--vlm-workers", type=int, default=4)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    raw = output / "raw"
    metadata = output / "metadata"
    thumbnails = output / "thumbnails"
    tifs = output / "tifs"
    jpegs = output / "jpegs"
    logs = output / "logs"
    for directory in (raw, metadata, thumbnails, tifs, jpegs, logs):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": {key: str(value) for key, value in vars(args).items()},
        "llm_filtering": False,
        "stages": {},
    }
    manifest_path = output / "run_manifest.json"

    def finish(stopped_early=None):
        # An early stop is a successful run with nothing to upload; the
        # wrapper reads `stopped_early` instead of the VLM report.
        if stopped_early:
            manifest["stopped_early"] = stopped_early
            print(f"Nothing to review: {stopped_early}")
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Pipeline complete. Manifest: {manifest_path}")
        return 0

    raw_csv = raw / "openaerial_data.csv"
    filtered_csv = metadata / "filtered.csv"
    pheno_csv = metadata / "phenology.csv"
    tif_metadata = metadata / "tif_metadata.csv"
    jpeg_metadata = metadata / "jpeg_metadata.csv"

    if not raw_csv.exists():
        run("scrape.py", cwd=output)
        shutil.move(str(output / "openaerial_data.csv"), raw_csv)
    manifest["stages"]["scrape"] = {"output": str(raw_csv)}

    if not filtered_csv.exists():
        command = ["--input", raw_csv, "--output", filtered_csv, "--max_gsd_cm", "10", "--platform_type", *args.platform]
        if args.uploaded_after_date:
            command += ["--uploaded_after_date", args.uploaded_after_date]
        if args.uploaded_before_date:
            command += ["--uploaded_before_date", args.uploaded_before_date]
        if args.forest_min is not None and args.forest_max is not None:
            command += ["--forest_percentage_min", args.forest_min, "--forest_percentage_max", args.forest_max]
        run("filter.py", *command, cwd=output)
    if not filtered_csv.exists():
        raise SystemExit(f"filter.py produced no output at {filtered_csv}")
    manifest["stages"]["filter"] = {"output": str(filtered_csv)}
    if not csv_rows(filtered_csv):
        return finish("no filter-passing images in the upload window")

    if not pheno_csv.exists():
        run("phenology.py", "--csv", filtered_csv, "--output", pheno_csv, "--pad-days", args.pad_days, cwd=output)
    manifest["stages"]["phenology"] = {"output": str(pheno_csv), "pad_days": args.pad_days}
    if not any(row.get("pheno_season") == "in_season" for row in csv_rows(pheno_csv)):
        return finish("no in-season images in the upload window")

    if not any(thumbnails.iterdir()):
        run("download.py", "thumbnails", "--csv", pheno_csv, "--folder", thumbnails, "--workers", args.thumbnail_workers, cwd=output)
    manifest["stages"]["thumbnails"] = {"files": count_files(thumbnails, {".png", ".jpg", ".jpeg"}), "output": str(thumbnails)}

    if not any(tifs.iterdir()):
        run("download.py", "tifs", "--csv", pheno_csv, "--thumbnails-dir", thumbnails, "--output-dir", tifs, "--workers", args.tif_workers, cwd=output)
    manifest["stages"]["tifs"] = {"files": count_files(tifs, {".tif", ".tiff"}), "output": str(tifs)}
    if not manifest["stages"]["tifs"]["files"]:
        return finish("no GeoTIFFs downloaded")

    if not any(jpegs.iterdir()):
        run("tif_to_jpeg.py", "--input-dir", tifs, "--output-dir", jpegs, "--workers", args.jpeg_workers, cwd=output)
    manifest["stages"]["jpegs"] = {"files": count_files(jpegs, {".jpeg", ".jpg"}), "output": str(jpegs)}

    if not tif_metadata.exists():
        run("create_metadata.py", "tif", "--csv", pheno_csv, "--output-dir", tifs, "--metadata-output", tif_metadata, "--add-image-size", cwd=output)
    if not jpeg_metadata.exists():
        run("create_metadata.py", "jpeg", "--source-metadata", tif_metadata, "--jpeg-folder", jpegs, "--output-metadata", jpeg_metadata, cwd=output)
    manifest["stages"]["metadata"] = {"tif": str(tif_metadata), "jpeg": str(jpeg_metadata)}

    # VLM audit stages
    audit_manifest_csv = metadata / "audit_manifest.csv"
    vlm_attempts = logs / "phenology-attempts.jsonl"
    vlm_report_csv = metadata / "phenology_report.csv"
    vlm_images = output / "vlm_images"

    if not vlm_report_csv.exists():
        if not audit_manifest_csv.exists():
            run("aerial_phenology_audit.py", "manifest", "--source", tifs, "--jpegs", jpegs, "--phenology", pheno_csv, "--metadata", tif_metadata, "--output", audit_manifest_csv, cwd=output)

        run("aerial_phenology_audit.py", "phenology-run", "--manifest", audit_manifest_csv, "--attempts", vlm_attempts, "--endpoint", args.vlm_endpoint, "--model", args.vlm_model, "--workers", args.vlm_workers, "--priorities", "in_season", cwd=output)

        if vlm_images.exists() and any(vlm_images.iterdir()):
            shutil.rmtree(vlm_images)
        run("aerial_phenology_audit.py", "phenology-report", "--manifest", audit_manifest_csv, "--attempts", vlm_attempts, "--output", vlm_report_csv, "--images", vlm_images, cwd=output)

        # phenology-run records API errors (bad key, no credits, outages) and
        # exits 0. Those images would silently fall out of the upload gate, so
        # fail instead and drop the report: a re-run into this dir retries only
        # the failed images (successful attempts are kept), and the wrapper
        # keeps the weekly window in place until then.
        unreviewed = [
            row for row in csv_rows(vlm_report_csv)
            if row.get("modis_category") == "in_season" and row.get("review_status") != "success"
        ]
        if unreviewed:
            vlm_report_csv.unlink()
            errors = sorted({row.get("http_status") or row.get("error") or row.get("review_status") for row in unreviewed})
            raise SystemExit(
                f"VLM review failed for {len(unreviewed)} in-season image(s) (errors: {', '.join(map(str, errors))[:300]}); "
                f"check OPENROUTER_API_KEY and credits, then re-run into {output} to retry"
            )

    manifest["stages"]["audit_manifest"] = {"output": str(audit_manifest_csv)}
    manifest["stages"]["vlm_run"] = {"attempts": str(vlm_attempts), "model": args.vlm_model}
    manifest["stages"]["vlm_report"] = {"output": str(vlm_report_csv), "images": str(vlm_images)}
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
