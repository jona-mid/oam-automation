"""Thin platform seam between oam_weekly and the deadtrees monorepo CLI.

All interaction with the deadtrees.earth platform lives in this module:
`DataCommands` upload/process calls and the upload-guide duplicate check.
Public API only; no forking or patching of deadtrees-cli. Heavy imports are
lazy so importing this module stays cheap and testable without the platform
stack installed.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple
import os

ENV_PATH = Path(__file__).resolve().parent / ".env"

TASK_TYPES = ["geotiff", "metadata", "cog", "thumbnail", "deadwood_v1", "treecover_v1"]
PROCESS_PRIORITY = 2
HASH_QUERY_CHUNK = 50

# OAM property_license strings -> deadtrees LicenseEnum values. Unknown values fail closed.
OAM_LICENSE_MAP = {
    "CC-BY 4.0": "CC BY",
    "CC BY-SA 4.0": "CC BY-SA",
    "CC BY-NC 4.0": "CC BY-NC",
    "CC BY-NC-SA 4.0": "CC BY-NC-SA",
    "MIT": "MIT",
}


def _platform_api() -> Tuple[Any, Any, Any]:
    """Load .env, then import the platform modules (Settings requires the env vars)."""
    from dotenv import load_dotenv

    load_dotenv(ENV_PATH)
    from deadtrees_cli.data import DataCommands
    from shared.db import use_client
    from shared.settings import settings

    return DataCommands, use_client, settings


def file_names_on_platform(filenames: List[str]) -> set:
    """Server-side duplicate check: which of these file_names the datasets table already has.

    One login and one query per HASH_QUERY_CHUNK names (the filter travels in
    the URL); names are compared lowercase like the ledger.
    """
    if not filenames:
        return set()
    DataCommands, use_client, settings = _platform_api()
    token = DataCommands()._ensure_auth()
    found = set()
    with use_client(token) as client:
        for start in range(0, len(filenames), HASH_QUERY_CHUNK):
            response = (
                client.table(settings.datasets_table)
                .select("file_name")
                .in_("file_name", filenames[start : start + HASH_QUERY_CHUNK])
                .execute()
            )
            found.update(str(row["file_name"]).strip().lower() for row in response.data if row.get("file_name"))
    return found


def file_hash(path: Path) -> str:
    """Content identifier matching the backend's algorithm in shared/hash.py."""
    from shared.hash import get_file_identifier

    return get_file_identifier(path)


def file_hashes_on_platform(hashes: List[str]) -> Any:
    """Batched content-hash check against the platform's orthos table.

    The backend processor computes get_file_identifier (shared/hash.py) for
    every processed upload and stores it in orthos.sha256. A match means the
    same file content is already on the platform. Returns {hash: dataset_id}.
    Readable with the anon SUPABASE_KEY, so no login needed.
    """
    if not hashes:
        return {}
    _, use_client, settings = _platform_api()
    found = {}
    # The filter travels in the URL; one query for hundreds of hashes fails
    # with HTTP 414, so query in chunks.
    with use_client(os.environ["SUPABASE_KEY"]) as client:
        for start in range(0, len(hashes), HASH_QUERY_CHUNK):
            response = (
                client.table(settings.orthos_table)
                .select("dataset_id,sha256")
                .in_("sha256", hashes[start : start + HASH_QUERY_CHUNK])
                .execute()
            )
            found.update(
                {row["sha256"]: row["dataset_id"] for row in response.data if row.get("sha256")}
            )
    return found


def upload_and_process(
    file_path: Path,
    authors: List[str],
    platform: str,
    license: str,
    data_access: str,
    acquisition_year: int,
    acquisition_month: int,
    acquisition_day: int,
    additional_information: Optional[str],
    citation_doi: Optional[str],
) -> int:
    """Upload one GeoTIFF and trigger processing. Returns the dataset ID."""
    DataCommands, _, _ = _platform_api()
    dc = DataCommands()  # fresh instance per file: fresh login, no stale tokens
    dataset = dc.upload(
        file_path=str(file_path),
        authors=authors,
        platform=platform,
        license=license,
        data_access=data_access,
        aquisition_year=acquisition_year,  # upstream's spelling of the kwarg
        aquisition_month=acquisition_month,
        aquisition_day=acquisition_day,
        additional_information=additional_information,
        citation_doi=citation_doi,
    )
    dataset_id = int(dataset["id"])
    dc.process(dataset_id=dataset_id, task_types=TASK_TYPES, priority=PROCESS_PRIORITY)
    return dataset_id


def start_processing(dataset_id: int) -> None:
    """Queue processing for an uploaded dataset (the second half of upload_and_process)."""
    DataCommands, _, _ = _platform_api()
    DataCommands().process(dataset_id=dataset_id, task_types=TASK_TYPES, priority=PROCESS_PRIORITY)


def find_dataset(filename: str, created_after: str) -> Optional[dict]:
    """Newest dataset with this file_name created after `created_after` (ISO time), or None.

    For recovery after an upload error: on a slow connection the client can
    time out although the platform already has the file. Returns
    {"id": ..., "processing_queued": bool}; queued means a queue row exists
    or processing already started, finished or failed.
    """
    DataCommands, use_client, settings = _platform_api()
    token = DataCommands()._ensure_auth()
    with use_client(token) as client:
        rows = (
            client.table(settings.datasets_table)
            .select("id")
            .eq("file_name", filename)
            .gte("created_at", created_after)
            .order("id", desc=True)
            .limit(1)
            .execute()
            .data
        )
        if not rows:
            return None
        dataset_id = int(rows[0]["id"])
        queued = client.table(settings.queue_table).select("id").eq("dataset_id", dataset_id).execute().data
        status = (
            client.table(settings.statuses_table)
            .select("current_status,is_ortho_done,has_error")
            .eq("dataset_id", dataset_id)
            .execute()
            .data
        )
    started = bool(status) and (
        status[0].get("current_status") != "idle" or bool(status[0].get("is_ortho_done")) or bool(status[0].get("has_error"))
    )
    return {"id": dataset_id, "processing_queued": bool(queued) or started}
