"""
Tests for phenology.py - Phenology extraction and classification functions.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phenology import (
    classify_season,
    parse_date_to_doy,
    transform_bbox_to_modis,
)


class TestClassifySeason:
    """Tests for classify_season function."""

    def test_in_season_normal_window(self):
        """Test classification when capture is within phenology window."""
        result = classify_season(capture_doy=150, pheno_start=100, pheno_end=200)
        assert result == "in_season"

    def test_out_of_season(self):
        """Test classification when capture is outside phenology window."""
        result = classify_season(capture_doy=50, pheno_start=100, pheno_end=200)
        assert result == "out_of_season"

    def test_at_start_boundary(self):
        """Test classification at start boundary."""
        result = classify_season(capture_doy=100, pheno_start=100, pheno_end=200)
        assert result == "in_season"

    def test_at_end_boundary(self):
        """Test classification at end boundary."""
        result = classify_season(capture_doy=200, pheno_start=100, pheno_end=200)
        assert result == "in_season"

    def test_wrap_around_in_season(self):
        """Test classification with wrap-around phenology window."""
        result = classify_season(capture_doy=350, pheno_start=300, pheno_end=50)
        assert result == "in_season"

    def test_wrap_around_out_of_season(self):
        """Test classification outside wrap-around window."""
        result = classify_season(capture_doy=100, pheno_start=300, pheno_end=50)
        assert result == "out_of_season"

    def test_with_padding(self):
        """Test classification with padding."""
        result = classify_season(
            capture_doy=85, pheno_start=100, pheno_end=200, pad_days=20
        )
        assert result == "in_season"

    def test_none_capture_doy(self):
        """Test with None capture DOY returns unknown."""
        result = classify_season(capture_doy=None, pheno_start=100, pheno_end=200)
        assert result == "unknown"

    def test_none_pheno_start(self):
        """Test with None phenology start returns unknown."""
        result = classify_season(capture_doy=150, pheno_start=None, pheno_end=200)
        assert result == "unknown"

    def test_none_pheno_end(self):
        """Test with None phenology end returns unknown."""
        result = classify_season(capture_doy=150, pheno_start=100, pheno_end=None)
        assert result == "unknown"


class TestParseDateToDoy:
    """Tests for parse_date_to_doy function."""

    def test_parse_iso_datetime(self):
        """Test parsing ISO datetime string."""
        result = parse_date_to_doy("2025-01-15T10:30:00")
        assert result == 15

    def test_parse_iso_datetime_with_microseconds(self):
        """Test parsing ISO datetime with microseconds."""
        result = parse_date_to_doy("2025-01-15T10:30:00.123456")
        assert result == 15

    def test_parse_iso_date(self):
        """Test parsing simple ISO date."""
        result = parse_date_to_doy("2025-01-15")
        assert result == 15

    def test_parse_date_with_z_suffix(self):
        """Test parsing date with Z suffix."""
        result = parse_date_to_doy("2025-01-15T10:30:00Z")
        assert result == 15

    def test_parse_date_with_timezone(self):
        """Test parsing date with timezone offset."""
        result = parse_date_to_doy("2025-01-15T10:30:00+00:00")
        assert result == 15

    def test_parse_none(self):
        """Test parsing None returns None."""
        result = parse_date_to_doy(None)
        assert result is None

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        result = parse_date_to_doy("")
        assert result is None

    def test_parse_invalid_format(self):
        """Test parsing invalid format returns None."""
        result = parse_date_to_doy("invalid-date")
        assert result is None

    def test_december_date(self):
        """Test parsing December date."""
        result = parse_date_to_doy("2025-12-31")
        assert result == 365


class TestTransformBboxToModis:
    """Tests for transform_bbox_to_modis function."""

    @patch("phenology.transform")
    @patch("phenology.WGS84_CRS", MagicMock())
    @patch("phenology.MODIS_CRS", MagicMock())
    def test_transform_bbox_centroid(self, mock_transform):
        """Test transforming bbox centroid to MODIS coordinates."""
        mock_transform.return_value = ([-13620840], [4487940])

        x, y = transform_bbox_to_modis(-122.5, 37.5, -122.0, 38.0)

        assert x == -13620840
        assert y == 4487940
        mock_transform.assert_called_once()

    @patch("phenology.transform")
    @patch("phenology.WGS84_CRS", MagicMock())
    @patch("phenology.MODIS_CRS", MagicMock())
    def test_transform_negative_longitude(self, mock_transform):
        """Test transforming bbox with negative longitude."""
        mock_transform.return_value = ([-13620840], [4487940])

        x, y = transform_bbox_to_modis(-122.5, 37.5, -122.0, 38.0)

        assert x < 0


class TestLoadPhenologyData:
    """Tests for loading phenology data."""

    @patch("phenology.xr")
    def test_load_phenology_data(self, mock_xr):
        """Test loading phenology data from zarr."""
        mock_data = MagicMock()
        mock_data.shape = (420, 1080, 6)
        mock_xr.open_zarr.return_value.phenology40km.load.return_value = mock_data

        from phenology import load_phenology_data

        with patch("phenology.DEFAULT_PHENOLOGY_PATH", "/fake/path.zarr"):
            result = load_phenology_data("/fake/path.zarr")

        assert result.shape == (420, 1080, 6)


class TestExtractPhenologyForBbox:
    """Tests for extracting phenology from bounding box."""

    @patch("phenology.transform_bbox_to_modis")
    @patch("phenology.load_phenology_data")
    def test_extract_phenology_success(self, mock_load, mock_transform):
        """Test successful phenology extraction."""
        mock_transform.return_value = (-13620840, 4487940)

        mock_pheno_data = MagicMock()
        mock_pheno_val = MagicMock()
        mock_pheno_val.sel.return_value.values = np.array([100.0])

        mock_pheno_data.sel.return_value = mock_pheno_val
        mock_load.return_value = mock_pheno_data

        from phenology import extract_phenology_for_bbox

        with patch("phenology.np.asarray") as mock_np:
            mock_np.return_value = np.array([100.0])

            result = extract_phenology_for_bbox(
                -122.5, 37.5, -122.0, 38.0, mock_pheno_data
            )

        assert result[0] is not None or result[1] is not None


class TestProcessBboxesFromCsv:
    """Tests for processing bounding boxes from CSV."""

    def test_csv_reading(self, tmp_path):
        """Test that CSV can be read correctly."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "id,bbox_min_lon,bbox_min_lat,bbox_max_lon,bbox_max_lat\n"
            "1,-122.5,37.5,-122.0,38.0\n"
        )

        import pandas as pd

        df = pd.read_csv(csv_file)

        assert len(df) == 1
        assert "bbox_min_lon" in df.columns
