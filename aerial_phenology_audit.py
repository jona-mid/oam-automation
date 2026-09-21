#!/usr/bin/env python3
"""Minimal hosted-vision audit of visible tree-canopy phenology.

The model sees one temporary resized JPEG and returns only a visible tree-canopy
leaf state. MODIS is joined after the blind visual review for comparison; this
tool never edits source TIFFs or JPEGs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import base64
import csv
import io
import json
import os
import random
import re
import shutil
import time
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator


Image.MAX_IMAGE_PIXELS = None
PROMPT_VERSION = "phenology-v2"
PROMPT = """You are classifying visible TREE-CANOPY phenology in one aerial image.
Judge only visible woody tree crowns. Do not infer date, location, season,
platform, metadata, or MODIS. Crops, grass, and other non-tree vegetation never
establish a tree-canopy leaf state.

Return JSON only with exactly these fields:
- tree_canopy_leaf_state: leaf_on, not_leaf_on, or not_assessable
- visual_cue: 3 to 12 words describing visible evidence

Definitions:
- leaf_on: clearly green, foliated tree canopy, with no obvious autumn colouring
  or leaf-off crowns.
- not_leaf_on: visible autumn-coloured, brown, sparse, or bare/leafless tree
  crowns.
- not_assessable: tree canopy is absent, too small, obscured, or cannot be
  judged reliably.
"""

MANIFEST_FIELDS = [
    "image_id", "filename", "source_path", "jpeg_path", "gsd_m", "platform",
    "capture_date", "bbox", "dimensions", "modis_category", "eligible_gsd",
    "jpeg_width", "jpeg_height", "jpeg_long_edge", "jpeg_bytes", "preview_status",
    "duplicate_of", "priority", "selection_status",
]


class PhenologyReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tree_canopy_leaf_state: Literal["leaf_on", "not_leaf_on", "not_assessable"]
    visual_cue: str = Field(min_length=3, max_length=300)

    @field_validator("visual_cue")
    @classmethod
    def cue_has_three_to_twelve_words(cls, value: str) -> str:
        words = re.findall(r"\b[\w'-]+\b", value)
        if not 3 <= len(words) <= 12:
            raise ValueError("visual_cue must contain 3 to 12 words")
        return value.strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_number(value: str | None) -> float | None:
    try:
        number = float((value or "").strip())
        return number if number >= 0 else None
    except ValueError:
        return None


def normalised_bbox(value: str) -> str:
    try:
        coords = json.loads(value)
        if isinstance(coords, list) and len(coords) == 4:
            return json.dumps([round(float(item), 6) for item in coords])
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    return ""


def basename(value: str) -> str:
    return Path(value.replace("\\", "/")).name.lower() if value else ""


def metadata_index(path: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        for field in ("properties.filename", "filename", "uuid", "properties.url"):
            key = basename(row.get(field, ""))
            if key:
                index[key] = row
    return index


def image_index(root: Path) -> dict[str, Path]:
    return {item.name.lower(): item for item in root.rglob("*") if item.is_file()}


def jpeg_dimensions_from_header(path: Path) -> tuple[int, int]:
    """Read JPEG dimensions from its header without decoding image pixels."""
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                   0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError("not a JPEG")
        while True:
            marker_prefix = handle.read(1)
            while marker_prefix == b"\xff":
                marker_prefix = handle.read(1)
            if not marker_prefix:
                raise ValueError("JPEG ended before a frame header")
            marker = marker_prefix[0]
            if marker in {0xD8, 0xD9}:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                raise ValueError("truncated JPEG segment")
            segment_length = int.from_bytes(length_bytes, "big")
            if segment_length < 2:
                raise ValueError("invalid JPEG segment length")
            if marker in sof_markers:
                frame = handle.read(5)
                if len(frame) != 5:
                    raise ValueError("truncated JPEG frame header")
                height = int.from_bytes(frame[1:3], "big")
                width = int.from_bytes(frame[3:5], "big")
                if not width or not height:
                    raise ValueError("invalid JPEG dimensions")
                return width, height
            handle.seek(segment_length - 2, io.SEEK_CUR)


def preview_info(path: Path | None, minimum_long_edge: int) -> tuple[str, int | None, int | None, int | None, int | None]:
    """Read preview dimensions and bytes without decoding image pixels."""
    try:
        if path is None or not path.is_file() or path.stat().st_size == 0:
            return "missing_or_zero", None, None, None, None
        file_bytes = path.stat().st_size
        width, height = jpeg_dimensions_from_header(path)
        long_edge = max(width, height)
        return ("eligible" if long_edge >= minimum_long_edge else "too_small",
                width, height, long_edge, file_bytes)
    except Exception:
        return "unreadable", None, None, None, None


def make_manifest(source_dir: Path, jpeg_dir: Path, phenology: Path, metadata: Path,
                  output: Path, min_preview_long_edge: int = 1000) -> None:
    jpgs = image_index(jpeg_dir)
    pheno = {row.get("filename", "").lower(): row for row in read_csv(phenology)}
    meta = metadata_index(metadata)
    rows: list[dict[str, str]] = []
    sources = sorted(item for item in source_dir.rglob("*") if item.is_file() and item.suffix.lower() in {".tif", ".tiff"})
    for source in sources:
        name = source.name
        p, m = pheno.get(name.lower(), {}), meta.get(name.lower(), {})
        jpeg_name = p.get("jpeg_filename") or f"{name}.jpeg"
        jpg = jpgs.get(jpeg_name.lower())
        preview_status, width, height, long_edge, file_bytes = preview_info(jpg, min_preview_long_edge)
        gsd = parse_number(m.get("gsd") or m.get("properties.resolution_in_meters"))
        category = p.get("classification") or "unknown"
        priority = category if category in {"in_season", "between_season", "out_of_season"} else "unknown"
        eligible = gsd is not None and gsd < 0.10
        selection_status = "pending" if eligible else ("metadata_review" if gsd is None else "excluded_gsd")
        if eligible and preview_status != "eligible":
            selection_status = "excluded_preview"
        rows.append({
            "image_id": source.stem, "filename": name, "source_path": str(source),
            "jpeg_path": str(jpg) if jpg else "", "gsd_m": "" if gsd is None else f"{gsd:.12g}",
            "platform": m.get("platform", "unknown"),
            "capture_date": p.get("capture_date") or m.get("acquisition_start", ""),
            "bbox": m.get("bbox") or m.get("geojson.bbox", ""),
            "dimensions": m.get("properties.dimensions", ""), "modis_category": category,
            "eligible_gsd": "true" if eligible else "false",
            "jpeg_width": "" if width is None else str(width),
            "jpeg_height": "" if height is None else str(height),
            "jpeg_long_edge": "" if long_edge is None else str(long_edge),
            "jpeg_bytes": "" if file_bytes is None else str(file_bytes),
            "preview_status": preview_status, "duplicate_of": "", "priority": priority,
            "selection_status": selection_status,
        })
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (normalised_bbox(row["bbox"]), row["capture_date"], row["platform"], row["dimensions"], row["gsd_m"])
        if key[0] and key[1]:
            groups.setdefault(key, []).append(row)
    for group in groups.values():
        if len(group) < 2:
            continue
        representative = min(group, key=lambda row: (row["preview_status"] != "eligible", row["filename"]))
        for row in group:
            if row is not representative:
                row["duplicate_of"] = representative["image_id"]
                if row["selection_status"] == "pending":
                    row["selection_status"] = "duplicate"
    write_csv(output, MANIFEST_FIELDS, rows)
    print(json.dumps({"images": len(rows), "eligible_gsd": sum(row["eligible_gsd"] == "true" for row in rows),
                      "duplicates": sum(bool(row["duplicate_of"]) for row in rows),
                      "excluded_preview": sum(row["selection_status"] == "excluded_preview" for row in rows),
                      "eligible_for_api": sum(row["selection_status"] == "pending" for row in rows),
                      "min_preview_long_edge": min_preview_long_edge}, indent=2))


def image_data(path: Path, max_side: int = 2048) -> tuple[str, str, int]:
    """Return a bounded upload JPEG from an already-eligible categorized preview."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88, optimize=True)
    upload_bytes = buffer.getvalue()
    return base64.b64encode(upload_bytes).decode("ascii"), "resized_jpeg", len(upload_bytes)


def extract_json(content: str) -> dict[str, object]:
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("model response is not a JSON object")
    return parsed


def api_review(endpoint: str, model: str, api_key: str, encoded_image: str,
               timeout: int, reasoning_effort: str | None) -> tuple[PhenologyReview, int, str, dict[str, Any]]:
    body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
            ]}]}
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort, "exclude": True}
    started = time.perf_counter()
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return PhenologyReview.model_validate(extract_json(content)), round((time.perf_counter() - started) * 1000), str(data.get("model") or model), usage


def read_attempts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def successful_ids(path: Path) -> set[str]:
    return {str(row["image_id"]) for row in read_attempts(path) if row.get("status") == "success" and row.get("image_id")}


def error_details(exc: Exception) -> tuple[int | None, str]:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return exc.code, f"HTTP {exc.code}: {body or exc.reason}"[:500]
    return None, f"{type(exc).__name__}: {exc}"[:500]


def is_transient(exc: Exception) -> bool:
    return isinstance(exc, urllib.error.HTTPError) and (exc.code == 429 or 500 <= exc.code <= 599)


def run_phenology(manifest: Path, attempts: Path, endpoint: str, model: str, api_key_env: str,
                  timeout: int, max_retries: int, max_side: int, priorities: set[str], limit: int | None,
                  reasoning_effort: str | None, workers: int) -> None:
    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {api_key_env}")
    completed = successful_ids(attempts)
    pending = [row for row in read_csv(manifest) if row["image_id"] not in completed and row.get("selection_status") == "pending" and row.get("priority") in priorities]
    if limit is not None:
        pending = pending[:limit]
    write_lock = threading.Lock()

    def append_record(record: dict[str, Any]) -> None:
        with write_lock:
            append_jsonl(attempts, record)

    def review_one(row: dict[str, str]) -> None:
        for attempt in range(1, max_retries + 2):
            submitted_image, now = "", datetime.now(timezone.utc).isoformat()
            try:
                encoded, submitted_image, upload_bytes = image_data(Path(row["jpeg_path"]), max_side)
                review, latency, returned_model, usage = api_review(endpoint, model, api_key, encoded, timeout,
                                                                      reasoning_effort)
                record: dict[str, Any] = {"status": "success", "at": now, "attempt": attempt, "image_id": row["image_id"],
                    "requested_model": model, "returned_model": returned_model, "prompt_version": PROMPT_VERSION,
                    "latency_ms": latency, "submitted_image": submitted_image, "max_side": max_side,
                    "upload_bytes": upload_bytes, "upload_base64_bytes": len(encoded),
                    "reasoning_effort": reasoning_effort or "provider_default", "review": review.model_dump(),
                    "usage": usage}
                if usage.get("cost") is not None:
                    record["cost_usd"] = usage["cost"]
                append_record(record)
                return
            except Exception as exc:
                status, detail = error_details(exc)
                append_record({"status": "error", "at": now, "attempt": attempt, "image_id": row["image_id"],
                    "requested_model": model, "prompt_version": PROMPT_VERSION, "submitted_image": submitted_image,
                    "reasoning_effort": reasoning_effort or "provider_default", "http_status": status,
                    "error": detail})
                if not is_transient(exc) or attempt > max_retries:
                    return
                time.sleep(min(2 ** (attempt - 1), 8))

    if workers == 1:
        for row in pending:
            review_one(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(review_one, pending))
    print(json.dumps({"already_complete": len(completed), "submitted": len(pending), "workers": workers}, indent=2))


def latest_by_image(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_attempts(path):
        if isinstance(row.get("image_id"), str):
            latest[row["image_id"]] = row
    return latest


def expected_state(category: str) -> str:
    return {"in_season": "leaf_on", "out_of_season": "not_leaf_on"}.get(category, "not_applicable")


def alignment(expected: str, result: dict[str, Any] | None) -> str:
    if expected == "not_applicable":
        return "not_applicable"
    if not result or result.get("status") != "success":
        return "unavailable"
    review = result.get("review", {})
    if not isinstance(review, dict) or review.get("tree_canopy_leaf_state") == "not_assessable":
        return "not_assessable"
    return "agree" if review.get("tree_canopy_leaf_state") == expected else "disagree"


def clean_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)) or "unavailable"


def report_phenology(manifest: Path, attempts: Path, output: Path, images: Path) -> None:
    if images.exists() and any(images.iterdir()):
        raise SystemExit(f"Refusing to mix results into non-empty directory: {images}")
    results = latest_by_image(attempts)
    final: list[dict[str, object]] = []
    reviewed_rows = [row for row in read_csv(manifest) if row.get("selection_status") == "pending"]
    for row in reviewed_rows:
        result = results.get(row["image_id"])
        review = result.get("review", {}) if result and result.get("status") == "success" else {}
        review = review if isinstance(review, dict) else {}
        expected = expected_state(row["modis_category"])
        final_row: dict[str, object] = {**row,
            "review_status": result.get("status", "unavailable") if result else "unavailable",
            "requested_model": result.get("requested_model", "") if result else "",
            "returned_model": result.get("returned_model", "") if result else "",
            "prompt_version": result.get("prompt_version", "") if result else "",
            "latency_ms": result.get("latency_ms", "") if result else "",
            "cost_usd": result.get("cost_usd", result.get("usage", {}).get("cost", "")) if result else "",
            "http_status": result.get("http_status", "") if result else "",
            "error": result.get("error", "") if result else "",
            "tree_canopy_leaf_state": review.get("tree_canopy_leaf_state", ""),
            "visual_cue": review.get("visual_cue", ""), "expected_tree_canopy_leaf_state": expected,
            "modis_alignment": alignment(expected, result)}
        final.append(final_row)
        source = Path(row["jpeg_path"])
        if not source.is_file() or source.stat().st_size == 0:
            raise SystemExit(f"Manifest JPEG is unreadable or absent: {source}")
        state = final_row["tree_canopy_leaf_state"] or "unavailable"
        destination = images / clean_component(row["modis_category"]) / clean_component(state)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / source.name)
    fields = list(dict.fromkeys(key for row in final for key in row))
    write_csv(output, fields, final)
    summary = {"rows": len(final), "successful": sum(row["review_status"] == "success" for row in final),
        "http_413": sum(str(row["http_status"]) == "413" for row in final),
        "in_out_comparable": sum(row["expected_tree_canopy_leaf_state"] != "not_applicable" for row in final),
        "in_out_agree": sum(row["modis_alignment"] == "agree" for row in final),
        "in_out_disagree": sum(row["modis_alignment"] == "disagree" for row in final),
        "not_assessable": sum(row["modis_alignment"] == "not_assessable" for row in final),
        "output": str(output), "images": str(images)}
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--source", type=Path, required=True)
    manifest.add_argument("--jpegs", type=Path, required=True)
    manifest.add_argument("--phenology", type=Path, required=True)
    manifest.add_argument("--metadata", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--min-preview-long-edge", type=int, default=1000)
    run = commands.add_parser("phenology-run", help="Blind visible tree-canopy phenology review")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--attempts", type=Path, required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    run.add_argument("--timeout", type=int, default=180)
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument("--max-side", type=int, default=2048)
    run.add_argument("--workers", type=int, default=1, help="Concurrent one-image API requests.")
    run.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"],
                     help="Request a supported hosted-model reasoning level.")
    run.add_argument("--priorities", nargs="+", default=["in_season", "between_season", "out_of_season"])
    run.add_argument("--limit", type=int)
    report = commands.add_parser("phenology-report", help="Join blind phenology review to MODIS and copy JPEGs")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--attempts", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--images", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "manifest":
        make_manifest(args.source, args.jpegs, args.phenology, args.metadata, args.output, args.min_preview_long_edge)
    elif args.command == "phenology-run":
        run_phenology(args.manifest, args.attempts, args.endpoint, args.model, args.api_key_env,
                      args.timeout, args.max_retries, args.max_side, set(args.priorities), args.limit,
                      args.reasoning_effort, args.workers)
    else:
        report_phenology(args.manifest, args.attempts, args.output, args.images)


if __name__ == "__main__":
    main()




