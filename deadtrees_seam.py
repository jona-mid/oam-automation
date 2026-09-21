"""Thin platform seam between oam_weekly and the deadtrees monorepo CLI.

All interaction with the deadtrees.earth platform lives in this module:
`DataCommands` upload/process calls and the upload-guide duplicate check.
Public API only; no forking or patching of deadtrees-cli. Heavy imports are
lazy so importing this module stays cheap and testable without the platform
stack installed.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

ENV_PATH = Path(__file__).resolve().parent / ".env"

TASK_TYPES = ["geotiff", "metadata", "cog", "thumbnail", "deadwood_v1", "treecover_v1"]
PROCESS_PRIORITY = 2


def _platform_api() -> Tuple[Any, Any, Any]:
    """Load .env, then import the platform modules (Settings requires the env vars)."""
    from dotenv import load_dotenv

    load_dotenv(ENV_PATH)
    from deadtrees_cli.data import DataCommands
    from shared.db import use_client
    from shared.settings import settings

    return DataCommands, use_client, settings


def file_exists_on_platform(filename: str) -> bool:
    """Server-side duplicate check: file_name in the datasets table (upload-guide pattern)."""
    DataCommands, use_client, settings = _platform_api()
    dc = DataCommands()
    token = dc._ensure_auth()
    with use_client(token) as client:
        response = (
            client.table(settings.datasets_table)
            .select("id")
            .eq("file_name", filename)
            .execute()
        )
        return len(response.data) > 0


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
