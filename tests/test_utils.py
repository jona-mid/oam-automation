"""Tests for utils.py."""

import datetime as dt

import pandas as pd

from utils import (
    extract_author,
    extract_filename_from_url,
    is_long_campaign,
    load_download_state,
    parse_bbox_string,
    parse_iso_date,
    read_csv,
    remap_platform,
    save_download_state,
    write_csv,
)


def test_parse_bbox_string():
    assert parse_bbox_string("[1.0, 2.0, 3.0, 4.0]") == [1.0, 2.0, 3.0, 4.0]
    assert parse_bbox_string("[-122.5 37.5 -122.0 38.0]") == [-122.5, 37.5, -122.0, 38.0]
    for value in [None, "", "[1.0, 2.0, 3.0]", "[a, b, c, d]"]:
        assert parse_bbox_string(value) is None, value


def test_parse_iso_date():
    for value in ["2025-01-15", "2025-01-15T10:30:00", "2025-01-15T10:30:00Z"]:
        assert parse_iso_date(value) == dt.date(2025, 1, 15), value
    assert parse_iso_date("") is None


def test_remap_platform_and_author():
    assert [remap_platform(v) for v in ["uav", "aircraft", "satellite", pd.NA]] == ["drone", "airborne", "satellite", ""]
    assert extract_author({"contact": "John Doe, john@example.com", "provider": "Test"}) == "John Doe"
    assert extract_author({"provider": "Test Provider"}) == "Test Provider"
    assert extract_author({}) == "openaerialmap.org"


def test_is_long_campaign():
    assert is_long_campaign({"acquisition_start": "2025-01-10T08:00:00", "acquisition_end": "2025-01-12T10:00:00"}) is False
    assert is_long_campaign({"acquisition_start": "2025-01-10T08:00:00", "acquisition_end": "2025-01-20T10:00:00"}) is True
    assert is_long_campaign({}) is False


def test_csv_and_download_state_round_trip(tmp_path):
    write_csv([{"col1": "val1", "col2": "val2"}, {"col1": "val3", "col2": "val4"}], tmp_path / "a.csv")
    assert [row["col1"] for row in read_csv(tmp_path / "a.csv")] == ["val1", "val3"]
    save_download_state(tmp_path / "state.json", {"file1.tif", "file2.tif"})
    assert load_download_state(tmp_path / "state.json") == {"file1.tif", "file2.tif"}
    assert load_download_state(tmp_path / "missing.json") == set()


def test_extract_filename_from_url():
    assert extract_filename_from_url("https://example.com/path/to/file.tif") == "file.tif"
    no_ext = extract_filename_from_url("https://example.com/path/to/file", ".tif")
    assert no_ext.startswith("file_") and no_ext.endswith(".tif")
    assert extract_filename_from_url("", ".tif").endswith(".tif")
