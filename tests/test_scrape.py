"""Tests for scrape.py."""

from unittest.mock import MagicMock, patch

import pytest

from scrape import fetch_stac_items, item_to_row


def _response(payload, ok=True):
    return MagicMock(ok=ok, status_code=200 if ok else 500, json=MagicMock(return_value=payload))


@patch("scrape.time.sleep")
@patch("scrape.requests")
def test_fetch_follows_next_links(mock_requests, _sleep):
    mock_requests.get.side_effect = [
        _response({"features": [{"id": 1}], "links": [{"rel": "next", "href": "p2"}]}),
        _response({"features": [{"id": 2}], "links": []}),
    ]
    assert [item["id"] for item in fetch_stac_items("p1")] == [1, 2]


@patch("scrape.time.sleep")
@patch("scrape.requests")
def test_fetch_fails_on_api_error(mock_requests, _sleep):
    mock_requests.get.return_value = _response({}, ok=False)
    with pytest.raises(Exception, match="Failed to fetch"):
        fetch_stac_items("p1")


PROPS = {"title": "T", "gsd": 0.05, "oam:platform_type": "uav", "license": "CC-BY-4.0",
         "created": "2026-11-01T10:00:00Z", "providers": [{"name": "P"}]}
LEGACY = "https://oin-hotosm-prod.s3.amazonaws.com/abc/0/abc1.tif"
ORIGINAL = "https://oin-hotosm-prod.s3.amazonaws.com/u-1/item-1/odm_orthophoto.tif"


@pytest.mark.parametrize("props, assets, expected", [
    # mirrored legacy item: visual asset, file name from the URL (matches the ledger)
    ({**PROPS, "start_datetime": "2026-10-01T00:00:00Z"}, {"visual": {"href": LEGACY}}, (LEGACY, None, "2026-10-01T00:00:00Z")),
    # new uploader item: the untouched original, named by item id
    ({**PROPS, "datetime": "2026-10-02T00:00:00Z", "oam:product_type": "visual"},
     {"visual": {"href": "cog.tif"}, "original": {"href": ORIGINAL}}, (ORIGINAL, "item-1.tif", "2026-10-02T00:00:00Z")),
    ({**PROPS, "oam:product_type": "multispectral"}, {"original": {"href": ORIGINAL}}, None),
    (PROPS, {"thumbnail": {"href": "t.png"}}, None),
])
def test_item_to_row(props, assets, expected):
    row = item_to_row({"id": "item-1", "bbox": [1, 2, 3, 4], "properties": props, "assets": assets})
    if expected is None:
        assert row is None
        return
    assert (row["uuid"], row["property_filename"], row["acquisition_start"]) == expected
    assert (row["uploaded_at"], row["platform"], row["property_license"], row["bbox_max_lat"]) == (
        "2026-11-01T10:00:00Z", "uav", "CC-BY-4.0", 4)


def test_timestamps_parse_after_normalizing():
    import pandas as pd
    from scrape import normalize_timestamps
    df = normalize_timestamps(pd.DataFrame({
        "uploaded_at": ["2026-11-12T11:26:56Z", "2026-11-06T19:01:08.501000Z", None],
        "acquisition_start": [None, "2026-10-01T00:00:00Z", None],
        "acquisition_end": [None, None, None],
    }))
    # downstream readers parse with pandas' default format inference
    assert pd.to_datetime(df["uploaded_at"], errors="coerce").notna().tolist() == [True, True, False]


def test_drop_mirrored_uploads_keeps_the_legacy_copy():
    import pandas as pd
    from scrape import drop_mirrored_uploads
    df = pd.DataFrame([
        # title, property_filename, bbox
        ("Kolleh Town", None, (0, 0, 10, 10)),           # legacy copy: kept
        ("kolleh town ", "a.tif", (1, 1, 10.5, 10.5)),   # same flight via the new uploader: dropped
        ("Kolleh Town", "b.tif", (50, 50, 60, 60)),      # same title, elsewhere: kept
        ("Other", "c.tif", (0, 0, 10, 10)),              # other title: kept
    ], columns=["title", "property_filename", "bbox"])
    df[["bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"]] = df["bbox"].tolist()
    assert drop_mirrored_uploads(df)["property_filename"].fillna("-").tolist() == ["-", "b.tif", "c.tif"]
