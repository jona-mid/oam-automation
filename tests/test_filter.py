"""
Tests for filter.py - Data filtering functions.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filter import filter_openaerial_data, calculate_forest_percentage, init_earthengine


class TestFilterOpendata:
    """Tests for filter_openaerial_data function."""

    def test_filter_by_gsd(self):
        """Test GSD filtering keeps only high-resolution images."""
        df = pd.DataFrame(
            [
                {"id": "1", "gsd": "0.05", "platform": "uav"},
                {"id": "2", "gsd": "0.10", "platform": "uav"},
                {"id": "3", "gsd": "0.15", "platform": "uav"},
            ]
        )

        result = filter_openaerial_data(df, max_gsd_cm=10)

        # max_gsd_cm=10 means < 0.1m (10cm/100), so only 0.05 passes
        assert len(result) == 1

    def test_filter_by_bands(self):
        """Test bands filtering keeps images with >1 band."""
        df = pd.DataFrame(
            [
                {"id": "1", "gsd": "0.05", "property_bands": "1", "platform": "uav"},
                {"id": "2", "gsd": "0.05", "property_bands": "3", "platform": "uav"},
                {"id": "3", "gsd": "0.05", "property_bands": "4", "platform": "uav"},
            ]
        )

        result = filter_openaerial_data(df, max_gsd_cm=100)

        assert len(result) == 2

    def test_filter_by_uploaded_date(self):
        """Test uploaded date filtering."""
        df = pd.DataFrame(
            [
                {
                    "id": "1",
                    "gsd": "0.05",
                    "platform": "uav",
                    "uploaded_at": "2025-01-01T00:00:00Z",
                },
                {
                    "id": "2",
                    "gsd": "0.05",
                    "platform": "uav",
                    "uploaded_at": "2025-01-15T00:00:00Z",
                },
                {
                    "id": "3",
                    "gsd": "0.05",
                    "platform": "uav",
                    "uploaded_at": "2025-02-01T00:00:00Z",
                },
            ]
        )

        result = filter_openaerial_data(
            df, max_gsd_cm=100, uploaded_after_date="2025-01-10"
        )

        assert len(result) == 2

    def test_filter_by_platform(self):
        """Test platform filtering."""
        df = pd.DataFrame(
            [
                {"id": "1", "gsd": "0.05", "platform": "uav"},
                {"id": "2", "gsd": "0.05", "platform": "aircraft"},
                {"id": "3", "gsd": "0.05", "platform": "satellite"},
            ]
        )

        result = filter_openaerial_data(
            df, max_gsd_cm=100, platform_type=["uav", "aircraft"]
        )

        assert len(result) == 2

    def test_filter_by_platform_case_insensitive(self):
        """Test platform filtering is case-insensitive."""
        df = pd.DataFrame(
            [
                {"id": "1", "gsd": "0.05", "platform": "UAV"},
                {"id": "2", "gsd": "0.05", "platform": "Aircraft"},
                {"id": "3", "gsd": "0.05", "platform": "Satellite"},
            ]
        )

        result = filter_openaerial_data(
            df, max_gsd_cm=100, platform_type=["uav", "aircraft"]
        )

        assert len(result) == 2

    @patch("filter._get_forest_mask")
    def test_filter_by_forest_percentage(self, mock_get_forest):
        """Test forest percentage filtering."""
        mock_ee = MagicMock()
        mock_reducer = MagicMock()
        mock_geometry = MagicMock()

        with patch("filter.ee", mock_ee):
            mock_ee.Image.return_value.select.return_value.eq.return_value.Or.return_value = mock_get_forest.return_value
            mock_ee.Geometry.Rectangle.return_value = mock_geometry
            mock_ee.Reducer.mean.return_value = mock_reducer

            mock_reducer.reduceRegion.return_value.getInfo.return_value = {"Map": 0.45}

            df = pd.DataFrame(
                [
                    {
                        "id": "1",
                        "gsd": "0.05",
                        "platform": "uav",
                        "bbox": "[-122.5, 37.5, -122.0, 38.0]",
                    },
                    {
                        "id": "2",
                        "gsd": "0.05",
                        "platform": "uav",
                        "bbox": "[-122.0, 37.5, -121.5, 38.0]",
                    },
                    {
                        "id": "3",
                        "gsd": "0.05",
                        "platform": "uav",
                        "bbox": "[-121.5, 37.5, -121.0, 38.0]",
                    },
                ]
            )

            df["forest_percentage_gee"] = [45.0, 20.0, 10.0]

            result = filter_openaerial_data(
                df, max_gsd_cm=100, forest_percentage_min=30, forest_percentage_max=50
            )

            assert len(result) == 1
            assert result.iloc[0]["id"] == "1"

    def test_filter_removes_duplicate_bboxes(self):
        """Test duplicate bbox removal."""
        df = pd.DataFrame(
            [
                {
                    "id": "1",
                    "gsd": "0.05",
                    "platform": "uav",
                    "bbox": "[-122.5, 37.5, -122.0, 38.0]",
                },
                {
                    "id": "2",
                    "gsd": "0.05",
                    "platform": "uav",
                    "bbox": "[-122.5, 37.5, -122.0, 38.0]",
                },
                {
                    "id": "3",
                    "gsd": "0.05",
                    "platform": "uav",
                    "bbox": "[-121.5, 37.5, -121.0, 38.0]",
                },
            ]
        )

        result = filter_openaerial_data(df, max_gsd_cm=100)

        assert len(result) == 2

    def test_filter_empty_dataframe(self):
        """Test filtering empty dataframe."""
        df = pd.DataFrame()
        result = filter_openaerial_data(df, max_gsd_cm=10)
        assert len(result) == 0


class TestCalculateForestPercentage:
    """Tests for calculate_forest_percentage function."""

    @patch("filter.ee")
    def test_calculate_forest_percentage_success(self, mock_ee):
        """Test successful forest percentage calculation."""
        mock_image = MagicMock()
        mock_image.select.return_value.eq.return_value.Or.return_value = MagicMock()

        mock_ee.Image.return_value = mock_image
        mock_ee.Geometry.Rectangle.return_value = MagicMock()
        mock_ee.Reducer.mean.return_value = MagicMock().reduceRegion.return_value.getInfo.return_value = {
            "Map": 0.45
        }

        result = calculate_forest_percentage([-122.5, 37.5, -122.0, 38.0])

        assert result is not None

    def test_calculate_forest_percentage_none_bbox(self):
        """Test with None bbox returns None."""
        result = calculate_forest_percentage(None)
        assert result is None


class TestInitEarthengine:
    """Tests for init_earthengine function."""

    @patch("filter.ee")
    def test_init_earthengine_success(self, mock_ee):
        """Test successful Earth Engine initialization."""
        mock_ee.Initialize.return_value = True

        result = init_earthengine(authenticate_if_needed=False)

        assert result is True
        mock_ee.Initialize.assert_called_once()

    @patch("filter.ee")
    def test_init_earthengine_failure(self, mock_ee):
        """Test Earth Engine initialization failure."""
        mock_ee.Initialize.side_effect = Exception("Not authenticated")

        result = init_earthengine(authenticate_if_needed=False)

        assert result is False

    @patch("filter.ee")
    def test_init_earthengine_with_auth(self, mock_ee):
        """Test Earth Engine initialization with authentication."""
        mock_ee.Initialize.side_effect = [Exception("Not authenticated"), True]
        mock_ee.Authenticate.return_value = None

        result = init_earthengine(authenticate_if_needed=True)

        assert result is True
