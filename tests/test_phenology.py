"""Tests for phenology.py: season classification, day-of-year parsing, derived audit columns."""

from unittest.mock import MagicMock, patch

import pandas as pd

from phenology import classify_season, parse_date_to_doy, process_bboxes_from_csv


def test_classify_season():
    cases = [
        # capture, start, end, pad, expected
        (150, 100, 200, 0, "in_season"),
        (100, 100, 200, 0, "in_season"),       # boundaries are inclusive
        (200, 100, 200, 0, "in_season"),
        (50, 100, 200, 0, "out_of_season"),
        (350, 300, 50, 0, "in_season"),        # window wraps over New Year
        (100, 300, 50, 0, "out_of_season"),
        (85, 100, 200, 20, "in_season"),       # padding
        (344, 0, 82, 30, "in_season"),         # padding reaches back into December
        (330, 0, 82, 30, "out_of_season"),
        (10, 300, 360, 30, "in_season"),       # padding reaches into January
        (30, 300, 360, 30, "out_of_season"),
        (90, 100, 80, 30, "in_season"),        # a wrapping window stays wrapping
        (150, 300, 50, 30, "out_of_season"),
        (200, 20, 320, 30, "in_season"),       # padding covers the whole year
        (None, 100, 200, 0, "unknown"),
        (150, None, 200, 0, "unknown"),
        (150, 100, None, 0, "unknown"),
    ]
    for capture, start, end, pad, expected in cases:
        assert classify_season(capture_doy=capture, pheno_start=start, pheno_end=end, pad_days=pad) == expected, (
            capture, start, end, pad)


def test_parse_date_to_doy_is_zero_indexed_like_modis():
    for value in ["2025-01-15", "2025-01-15T10:30:00", "2025-01-15T10:30:00.123456",
                  "2025-01-15T10:30:00Z", "2025-01-15T10:30:00+00:00"]:
        assert parse_date_to_doy(value) == 14, value
    assert parse_date_to_doy("2025-01-01") == 0
    assert parse_date_to_doy("2025-12-31") == 364
    assert parse_date_to_doy("2024-12-31") == 365
    for value in [None, "", "invalid-date"]:
        assert parse_date_to_doy(value) is None


def _process(tmp_path, header, row):
    csv_file = tmp_path / "in.csv"
    csv_file.write_text(f"{header},bbox_min_lon,bbox_min_lat,bbox_max_lon,bbox_max_lat,acquisition_start\n"
                        f"{row},-122.5,37.5,-122.0,38.0,2025-06-15\n")
    out_file = tmp_path / "out.csv"
    with patch("phenology.load_phenology_data", return_value=MagicMock(shape=(420, 1080, 6))), \
         patch("phenology.extract_phenology_for_bbox", return_value=(100.0, 200.0)):
        process_bboxes_from_csv(str(csv_file), str(out_file), pad_days=0)
    return pd.read_csv(out_file).iloc[0]


def test_derived_columns_from_property_filename(tmp_path):
    row = _process(tmp_path, "uuid,property_filename", "https://example.com/abc.tif,def.tif")
    assert (row["filename"], row["classification"], row["jpeg_filename"]) == ("def.tif", "in_season", "def.jpeg")


def test_derived_filename_falls_back_to_uuid(tmp_path):
    row = _process(tmp_path, "uuid", "https://oin-hotosm-temp.s3.amazonaws.com/xyz/0/abc123.tif")
    assert (row["filename"], row["jpeg_filename"]) == ("abc123.tif", "abc123.jpeg")
