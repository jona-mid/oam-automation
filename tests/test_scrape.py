"""Tests for scrape.py."""

from unittest.mock import MagicMock, patch

import pytest

from scrape import fetch_openaerial_data, process_data


def _response(payload, ok=True):
    return MagicMock(ok=ok, status_code=200 if ok else 500, json=MagicMock(return_value=payload))


@patch("scrape.requests")
def test_fetch_follows_pages(mock_requests):
    mock_requests.get.side_effect = [
        _response({"meta": {"found": 25, "limit": 10}, "results": [{"id": 1}, {"id": 2}]}),
        _response({"results": [{"id": 3}, {"id": 4}]}),
        _response({"results": [{"id": 5}]}),
    ]
    assert [row["id"] for row in fetch_openaerial_data(max_pages=3)] == [1, 2, 3, 4, 5]


@patch("scrape.requests")
def test_fetch_fails_on_api_error(mock_requests):
    mock_requests.get.return_value = _response({}, ok=False)
    with pytest.raises(Exception, match="Failed to fetch initial data"):
        fetch_openaerial_data()


def test_process_data_flattens_nested_fields():
    [row] = process_data([{
        "id": "123",
        "bbox": [-122.5, 37.5, -122.0, 38.0],
        "footprint": "POLYGON((-122.5 37.5, -122.0 37.5, -122.0 38.0, -122.5 38.0, -122.5 37.5))",
        "properties": {"gsd": "5", "bands": "3"},
        "geojson": {"type": "Feature"},
    }])
    assert (row["bbox_min_lon"], row["bbox_min_lat"], row["bbox_max_lon"], row["bbox_max_lat"]) == (-122.5, 37.5, -122.0, 38.0)
    assert "-122.5 37.5" in row["footprint_coords"]
    assert (row["property_gsd"], row["property_bands"]) == ("5", "3")
    assert "geojson" not in row and "properties" not in row
    assert process_data([{"id": "2", "bbox": None}])[0].get("bbox_min_lon") is None
    assert process_data([]) == []
