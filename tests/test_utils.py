"""
Tests for utils.py utility functions.
"""

import pytest
import datetime as dt
import os
import tempfile
import json

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    parse_bbox_string,
    parse_bool,
    parse_iso_date,
    date_to_doy,
    normalize_doy,
    in_interval_wrap,
    in_leaf_on,
    in_leaf_on_padded,
    norm_filename,
    remap_platform,
    extract_author,
    is_long_campaign,
    read_csv,
    write_csv,
    load_download_state,
    save_download_state,
    extract_filename_from_url,
)


class TestParseBboxString:
    """Tests for parse_bbox_string function."""

    def test_parse_with_commas(self):
        result = parse_bbox_string("[1.0, 2.0, 3.0, 4.0]")
        assert result == [1.0, 2.0, 3.0, 4.0]

    def test_parse_without_commas(self):
        result = parse_bbox_string("[1.0 2.0 3.0 4.0]")
        assert result == [1.0, 2.0, 3.0, 4.0]

    def test_parse_negative_values(self):
        result = parse_bbox_string("[-122.5, 37.5, -122.0, 38.0]")
        assert result == [-122.5, 37.5, -122.0, 38.0]

    def test_parse_none(self):
        result = parse_bbox_string(None)
        assert result is None

    def test_parse_empty_string(self):
        result = parse_bbox_string("")
        assert result is None

    def test_parse_invalid_length(self):
        result = parse_bbox_string("[1.0, 2.0, 3.0]")
        assert result is None

    def test_parse_invalid_chars(self):
        result = parse_bbox_string("[a, b, c, d]")
        assert result is None


class TestParseBool:
    """Tests for parse_bool function."""

    def test_parse_true_values(self):
        assert parse_bool(True) is True
        assert parse_bool("true") is True
        assert parse_bool("1") is True
        assert parse_bool("yes") is True
        assert parse_bool("y") is True
        assert parse_bool("t") is True

    def test_parse_false_values(self):
        assert parse_bool(False) is False
        assert parse_bool("false") is False
        assert parse_bool("0") is False
        assert parse_bool("no") is False
        assert parse_bool("n") is False
        assert parse_bool("f") is False

    def test_parse_empty_string(self):
        assert parse_bool("") is False


class TestParseIsoDate:
    """Tests for parse_iso_date function."""

    def test_parse_simple_date(self):
        result = parse_iso_date("2025-01-15")
        assert result == dt.date(2025, 1, 15)

    def test_parse_with_time(self):
        result = parse_iso_date("2025-01-15T10:30:00")
        assert result == dt.date(2025, 1, 15)

    def test_parse_with_timezone(self):
        result = parse_iso_date("2025-01-15T10:30:00Z")
        assert result == dt.date(2025, 1, 15)

    def test_parse_empty_string(self):
        result = parse_iso_date("")
        assert result is None


class TestDateToDoy:
    """Tests for date_to_doy function."""

    def test_jan_1(self):
        result = date_to_doy(dt.date(2025, 1, 1))
        assert result == 1

    def test_dec_31(self):
        result = date_to_doy(dt.date(2025, 12, 31))
        assert result == 365

    def test_mid_year(self):
        result = date_to_doy(dt.date(2025, 6, 15))
        assert result == 166


class TestNormalizeDoy:
    """Tests for normalize_doy function."""

    def test_normalize_mid_range(self):
        assert normalize_doy(100, 366) == 100

    def test_normalize_wrap_above_year(self):
        assert normalize_doy(400, 366) == 34

    def test_normalize_wrap_below_one(self):
        assert normalize_doy(-5, 366) == 361

    def test_normalize_leap_year(self):
        assert normalize_doy(60, 367) == 60


class TestInIntervalWrap:
    """Tests for in_interval_wrap function."""

    def test_inside_normal_interval(self):
        assert in_interval_wrap(150, 100, 200) is True

    def test_outside_normal_interval(self):
        assert in_interval_wrap(50, 100, 200) is False

    def test_at_start(self):
        assert in_interval_wrap(100, 100, 200) is True

    def test_at_end(self):
        assert in_interval_wrap(200, 100, 200) is True

    def test_inside_wrap_interval(self):
        assert in_interval_wrap(350, 300, 50) is True

    def test_outside_wrap_interval(self):
        assert in_interval_wrap(100, 300, 50) is False


class TestInLeafOn:
    """Tests for in_leaf_on function."""

    def test_inside_window(self):
        assert in_leaf_on(150, 100, 200) is True

    def test_outside_window(self):
        assert in_leaf_on(50, 100, 200) is False

    def test_at_start(self):
        assert in_leaf_on(100, 100, 200) is True

    def test_at_end(self):
        assert in_leaf_on(200, 100, 200) is True

    def test_wrap_around_inside(self):
        assert in_leaf_on(350, 300, 50) is True

    def test_wrap_around_outside(self):
        assert in_leaf_on(100, 300, 50) is False

    def test_none_start(self):
        assert in_leaf_on(150, None, 200) is False

    def test_none_end(self):
        assert in_leaf_on(150, 100, None) is False


class TestInLeafOnPadded:
    """Tests for in_leaf_on_padded function."""

    def test_no_padding(self):
        assert in_leaf_on_padded(150, 100, 200, pad_days=0) is True

    def test_with_padding(self):
        assert in_leaf_on_padded(90, 100, 200, pad_days=15) is True

    def test_padding_outside(self):
        assert in_leaf_on_padded(50, 100, 200, pad_days=10) is False


class TestNormFilename:
    """Tests for norm_filename function."""

    def test_normalize(self):
        assert norm_filename("TestFile.TIF") == "testfile.tif"

    def test_with_spaces(self):
        assert norm_filename("  TestFile  ") == "testfile"


class TestRemapPlatform:
    """Tests for remap_platform function."""

    def test_uav_to_drone(self):
        assert remap_platform("uav") == "drone"

    def test_aircraft_to_airborne(self):
        assert remap_platform("aircraft") == "airborne"

    def test_other_unchanged(self):
        assert remap_platform("satellite") == "satellite"

    def test_none_handling(self):
        import pandas as pd

        assert remap_platform(pd.NA) == ""


class TestExtractAuthor:
    """Tests for extract_author function."""

    def test_from_contact(self):
        row = {"contact": "John Doe, john@example.com", "provider": "Test"}
        assert extract_author(row) == "John Doe"

    def test_from_provider(self):
        row = {"provider": "Test Provider"}
        assert extract_author(row) == "Test Provider"

    def test_default_openaerialmap(self):
        row = {}
        assert extract_author(row) == "openaerialmap.org"


class TestIsLongCampaign:
    """Tests for is_long_campaign function."""

    def test_short_campaign(self):
        row = {
            "acquisition_start": "2025-01-10T08:00:00",
            "acquisition_end": "2025-01-12T10:00:00",
        }
        assert is_long_campaign(row) is False

    def test_long_campaign(self):
        row = {
            "acquisition_start": "2025-01-10T08:00:00",
            "acquisition_end": "2025-01-20T10:00:00",
        }
        assert is_long_campaign(row) is True

    def test_missing_dates(self):
        row = {}
        assert is_long_campaign(row) is False


class TestReadWriteCsv:
    """Tests for CSV read/write functions."""

    def test_write_and_read_csv(self):
        data = [
            {"col1": "val1", "col2": "val2"},
            {"col1": "val3", "col2": "val4"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            write_csv(data, temp_path)
            result = read_csv(temp_path)
            assert len(result) == 2
            assert result[0]["col1"] == "val1"
        finally:
            os.remove(temp_path)


class TestDownloadState:
    """Tests for download state functions."""

    def test_save_and_load_state(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            files = {"file1.tif", "file2.tif"}
            save_download_state(temp_path, files)
            result = load_download_state(temp_path)
            assert result == files
        finally:
            os.remove(temp_path)

    def test_load_nonexistent(self):
        result = load_download_state("/nonexistent/path.json")
        assert result == set()


class TestExtractFilenameFromUrl:
    """Tests for extract_filename_from_url function."""

    def test_extract_from_path(self):
        url = "https://example.com/path/to/file.tif"
        result = extract_filename_from_url(url)
        assert result == "file.tif"

    def test_add_extension_no_ext_in_url(self):
        url = "https://example.com/path/to/file"
        result = extract_filename_from_url(url, ".tif")
        assert result.startswith("file_")
        assert result.endswith(".tif")

    def test_extract_from_path_with_ext(self):
        url = "https://example.com/path/to/file.tif"
        result = extract_filename_from_url(url)
        assert result == "file.tif"

    def test_no_path_uses_default(self):
        url = ""
        result = extract_filename_from_url(url, ".tif")
        assert ".tif" in result
