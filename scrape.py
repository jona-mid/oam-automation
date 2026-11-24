"""Scrape the OpenAerialMap catalog from HOT's STAC API into openaerial_data.csv.

The legacy api.openaerialmap.org/meta endpoint no longer lists most new
uploads, so this reads the STAC collection instead and writes the columns the
rest of the pipeline expects (uuid = download URL, uploaded_at, gsd, ...).
"""

import time

import pandas as pd
import requests

STAC_SEARCH = "https://api.imagery.hotosm.org/stac/search?collections=openaerialmap&limit=500"
STAC_ITEM = "https://api.imagery.hotosm.org/stac/collections/openaerialmap/items/{}"


def fetch_stac_items(url=STAC_SEARCH, retry_count=3, delay_between_requests=0.2):
    """All items of the collection, following the `next` links."""
    items = []
    while url:
        for attempt in range(retry_count):
            try:
                response = requests.get(url, timeout=120)
                if response.ok:
                    page = response.json()
                    break
                print(f"STAC request failed with status {response.status_code}")
            except Exception as e:
                print(f"Error fetching STAC page, attempt {attempt + 1}: {e}")
            if attempt < retry_count - 1:
                time.sleep(2)
        else:
            # A skipped page is a silent gap in the catalog; fail the run instead.
            raise Exception(f"Failed to fetch {url} after {retry_count} attempts")
        items.extend(page["features"])
        print(f"  fetched {len(items)} items", flush=True)
        url = next((link["href"] for link in page.get("links", []) if link.get("rel") == "next"), None)
        time.sleep(delay_between_requests)
    return items


def item_to_row(item):
    """Flatten one STAC item into a row, or None if it has no usable RGB image."""
    p = item.get("properties", {})
    if p.get("oam:product_type") not in (None, "visual"):
        return None  # multispectral / pseudocolor
    assets = item.get("assets", {})
    # Uploads through the new OAM uploader keep the untouched file as `original`
    # (`visual` is a lossy display copy) and often share generic names such as
    # odm_orthophoto.tif, so they are named by their item id. Items mirrored from
    # the legacy catalog only have `visual`, under the file name the ledger knows.
    if "original" in assets:
        url, filename = assets["original"]["href"], f"{item['id']}.tif"
    elif "visual" in assets:
        url, filename = assets["visual"]["href"], None
    else:
        return None
    providers = p.get("providers") or [{}]
    bbox = item.get("bbox") or [None] * 4
    return {
        "_id": item["id"],
        "uuid": url,
        "property_filename": filename,
        "title": p.get("title"),
        "provider": p.get("oam:producer_name") or providers[0].get("name"),
        "platform": p.get("oam:platform_type"),
        "gsd": p.get("gsd"),
        "property_license": p.get("license"),
        "acquisition_start": p.get("start_datetime") or p.get("datetime"),
        "acquisition_end": p.get("end_datetime") or p.get("datetime"),
        "uploaded_at": p.get("created"),
        "meta_uri": STAC_ITEM.format(item["id"]),
        "bbox": str(list(bbox)) if item.get("bbox") else None,
        "bbox_min_lon": bbox[0],
        "bbox_min_lat": bbox[1],
        "bbox_max_lon": bbox[2],
        "bbox_max_lat": bbox[3],
    }


def normalize_timestamps(df):
    """One timestamp format per column: STAC mixes '...:56Z' and '...:08.501000Z', and
    pandas' default parsing turns every value not in the first row's format into NaT."""
    for col in ("uploaded_at", "acquisition_start", "acquisition_end"):
        df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True, errors="coerce").dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
    return df


def _overlap(a, b):
    """Intersection area over the smaller bbox's area."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / smaller if smaller > 0 else 0.0


def drop_mirrored_uploads(df, min_overlap=0.5):
    """Drop new-uploader items that are also listed as a legacy item (same title,
    overlapping footprint). The two copies differ in bytes, so the content-hash
    check cannot catch them; the legacy copy is kept because the ledger knows it."""
    new = df["property_filename"].notna()
    legacy_by_title = {}
    for row in df[~new].itertuples():
        if isinstance(row.title, str):
            legacy_by_title.setdefault(row.title.strip().lower(), []).append(
                (row.bbox_min_lon, row.bbox_min_lat, row.bbox_max_lon, row.bbox_max_lat))
    def mirrored(row):
        title = row["title"].strip().lower() if isinstance(row["title"], str) else None
        box = (row["bbox_min_lon"], row["bbox_min_lat"], row["bbox_max_lon"], row["bbox_max_lat"])
        return any(_overlap(box, other) >= min_overlap for other in legacy_by_title.get(title, []))
    drop = new & df.apply(mirrored, axis=1)
    if drop.any():
        print(f"Dropped {int(drop.sum())} uploader items also listed as a legacy item")
    return df[~drop]


def main():
    items = fetch_stac_items()
    rows = [row for row in map(item_to_row, items) if row]
    df = drop_mirrored_uploads(normalize_timestamps(pd.DataFrame(rows)))
    print(f"\nFetched {len(items)} STAC items; {len(df)} RGB rows")
    print(df["platform"].value_counts().to_string())
    df.to_csv("openaerial_data.csv", index=False)
    print("Saved dataset to openaerial_data.csv")


if __name__ == "__main__":
    main()
