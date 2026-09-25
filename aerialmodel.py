"""aerialmodel.com source adapter: catalog, per-project screening data, login and download.

Everything up to the orthophoto is public: the marker API lists projects,
/api/cog-url/<id> reveals each project's storage path, and the ODM
stats.json under that path holds the capture time (from the photos' EXIF)
and the GSD. Only the orthophoto GeoTIFF needs a logged-in session.
"""

import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import requests

BASE_URL = "https://www.aerialmodel.com"
MARKERS_PARAMS = {"minLat": -90, "minLng": -180, "maxLat": 90, "maxLng": 180, "zoom": 1, "maxPoints": 8000}
USER_AGENT = "Mozilla/5.0 (deadtrees.earth oam-automation)"
REQUEST_DELAY = 0.5  # seconds between per-project requests
MIN_CAPTURE_YEAR = 2000  # photos without EXIF time report 1970


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def get(session: requests.Session, url: str, retries: int = 3, **kwargs) -> requests.Response:
    """GET with retries on connection errors, 429 and 5xx; other statuses are returned as is."""
    kwargs.setdefault("timeout", 60)
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, **kwargs)
        except requests.RequestException:
            if attempt == retries:
                raise
        else:
            if response.status_code != 429 and response.status_code < 500:
                return response
            if attempt == retries:
                response.raise_for_status()
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_catalog(session: requests.Session) -> list:
    """All projects on the platform (id, name, lat, lng, modelUrl, ...)."""
    response = get(session, f"{BASE_URL}/api/projects/markers", params=MARKERS_PARAMS)
    response.raise_for_status()
    data = response.json()
    if data.get("truncated"):
        raise RuntimeError(f"marker API truncated its result ({data.get('count')} of {data.get('total')})")
    return data["items"]


def storage_path_from_cog_url(url: str) -> Optional[str]:
    """'AWS/<user>/<uuid>' from the signed S3 URL the cog-url API returns."""
    match = re.search(r"amazonaws\.com/([^/?]+/[0-9a-fA-F-]{36})/", url or "")
    return f"AWS/{match.group(1)}" if match else None


def storage_path_from_page(page_html: str) -> Optional[str]:
    """Storage path from a public project page that embeds the point cloud viewer."""
    match = re.search(r"pointclouds/([^\"'\s<>]+?/[0-9a-fA-F-]{36})/entwine_pointcloud", page_html or "")
    return match.group(1) if match else None


def project_storage_path(session: requests.Session, project: dict) -> Optional[str]:
    """Storage path of a project, or None if the platform exposes none."""
    response = get(session, f"{BASE_URL}/api/cog-url/{project['id']}")
    if response.ok:
        try:
            path = storage_path_from_cog_url(response.json().get("url", ""))
        except ValueError:
            path = None
        if path:
            return path
    elif response.status_code != 404:
        response.raise_for_status()
    if not project.get("modelUrl"):
        return None
    page = get(session, f"{BASE_URL}{project['modelUrl']}")
    if page.status_code == 404:
        return None
    page.raise_for_status()
    return storage_path_from_page(page.text)


def file_url(storage_path: str, relative: str) -> str:
    return f"{BASE_URL}/API/File?/pointclouds/{storage_path}/{relative}"


def orthophoto_url(storage_path: str) -> str:
    return file_url(storage_path, "odm_orthophoto/odm_orthophoto.tif")


def fetch_stats(session: requests.Session, storage_path: str) -> Optional[dict]:
    """ODM stats.json of a project; None if the project has no ODM report."""
    response = get(session, file_url(storage_path, "odm_report/stats.json"))
    if response.status_code in (403, 404):
        return None
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        return None


def parse_stats(stats: Optional[dict]) -> Tuple[Optional[date], Optional[float]]:
    """(capture date, GSD in cm) from ODM stats.json; None where missing or implausible.

    The capture start is ODM's reading of the photos' EXIF time, formatted
    'DD/MM/YYYY at HH:MM:SS'; photos without EXIF time report 1970.
    """
    if not stats:
        return None, None
    capture = None
    raw = str((stats.get("processing_statistics") or {}).get("start_date") or "").strip()
    try:
        parsed = datetime.strptime(raw, "%d/%m/%Y at %H:%M:%S").date()
        if parsed.year >= MIN_CAPTURE_YEAR:
            capture = parsed
    except ValueError:
        capture = None
    gsd = (stats.get("odm_processing_statistics") or {}).get("average_gsd")
    try:
        gsd = float(gsd) if gsd is not None else None
    except (TypeError, ValueError):
        gsd = None
    if gsd is not None and gsd <= 0:
        gsd = None
    return capture, gsd


def login(session: requests.Session, email: str, password: str) -> None:
    """ASP.NET Identity login; raises if it does not succeed."""
    login_url = f"{BASE_URL}/Identity/Account/Login"
    page = get(session, login_url)
    page.raise_for_status()
    token = re.search(r'name="__RequestVerificationToken"[^>]*value="([^"]*)"', page.text) or re.search(
        r'value="([^"]*)"[^>]*name="__RequestVerificationToken"', page.text
    )
    if not token:
        raise RuntimeError("AerialModel login page has no anti-forgery token")
    response = session.post(
        login_url,
        data={
            "Input.Email": email,
            "Input.Password": password,
            "__RequestVerificationToken": token.group(1),
            "Input.RememberMe": "false",
        },
        allow_redirects=True,
        timeout=60,
    )
    if "/Identity/Account/Login" in response.url:
        raise RuntimeError("AerialModel login failed (check AERIALMODEL_EMAIL / AERIALMODEL_PASSWORD)")


def download(session: requests.Session, url: str, destination: Path) -> None:
    """Stream `url` to `destination` via a .part file, so a crash never leaves a truncated file."""
    partial = destination.with_name(destination.name + ".part")
    response = get(session, url, stream=True, timeout=300)
    content_type = response.headers.get("Content-Type", "")
    if (
        response.status_code in (401, 403)
        or "/Identity/Account/Login" in response.url
        or content_type.startswith("text/html")
    ):
        response.close()
        raise RuntimeError(f"not authorized for {url} (session not logged in?)")
    response.raise_for_status()
    with partial.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    partial.replace(destination)


def page_url(project: Any) -> str:
    model_url = project.get("modelUrl") if isinstance(project, dict) else project
    return f"{BASE_URL}{model_url}" if model_url else ""
