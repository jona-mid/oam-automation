"""
Tests for scrape.py - OpenAerialMap API scraping functions.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrape import fetch_openaerial_data, process_data


class TestFetchOpendata:
    """Tests for fetch_openaerial_data function."""

    @patch("scrape.requests")
    def test_fetch_single_page(self, mock_requests):
        """Test fetching data from a single page."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "meta": {"found": 10, "limit": 10},
            "results": [{"id": 1}, {"id": 2}],
        }
        mock_requests.get.return_value = mock_response

        result = fetch_openaerial_data(max_pages=1)

        assert len(result) == 2
        assert result[0]["id"] == 1

    @patch("scrape.requests")
    def test_fetch_multiple_pages(self, mock_requests):
        """Test fetching data from multiple pages."""
        mock_response_page1 = MagicMock()
        mock_response_page1.ok = True
        mock_response_page1.json.return_value = {
            "meta": {"found": 25, "limit": 10},
            "results": [{"id": 1}, {"id": 2}],
        }

        mock_response_page2 = MagicMock()
        mock_response_page2.ok = True
        mock_response_page2.json.return_value = {"results": [{"id": 3}, {"id": 4}]}

        mock_response_page3 = MagicMock()
        mock_response_page3.ok = True
        mock_response_page3.json.return_value = {"results": [{"id": 5}]}

        mock_requests.get.side_effect = [
            mock_response_page1,
            mock_response_page2,
            mock_response_page3,
        ]

        result = fetch_openaerial_data(max_pages=3)

        assert len(result) == 5

    @patch("scrape.requests")
    def test_fetch_api_error(self, mock_requests):
        """Test handling API errors."""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_requests.get.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            fetch_openaerial_data()

        assert "Failed to fetch initial data" in str(exc_info.value)


class TestProcessData:
    """Tests for process_data function."""

    def test_process_basic_data(self):
        """Test processing basic data without nested structures."""
        results = [
            {"id": "123", "uuid": "uuid-001", "platform": "uav"},
            {"id": "456", "uuid": "uuid-002", "platform": "aircraft"},
        ]

        processed = process_data(results)

        assert len(processed) == 2
        assert processed[0]["id"] == "123"
        assert processed[0]["platform"] == "uav"

    def test_process_with_bbox(self):
        """Test processing data with bounding box."""
        results = [
            {"id": "123", "uuid": "uuid-001", "bbox": [-122.5, 37.5, -122.0, 38.0]}
        ]

        processed = process_data(results)

        assert "bbox_min_lon" in processed[0]
        assert processed[0]["bbox_min_lon"] == -122.5
        assert processed[0]["bbox_min_lat"] == 37.5
        assert processed[0]["bbox_max_lon"] == -122.0
        assert processed[0]["bbox_max_lat"] == 38.0

    def test_process_with_footprint(self):
        """Test processing data with footprint."""
        results = [
            {
                "id": "123",
                "uuid": "uuid-001",
                "footprint": "POLYGON((-122.5 37.5, -122.0 37.5, -122.0 38.0, -122.5 38.0, -122.5 37.5))",
            }
        ]

        processed = process_data(results)

        assert "footprint_coords" in processed[0]
        assert "-122.5 37.5" in processed[0]["footprint_coords"]

    def test_process_with_properties(self):
        """Test processing data with nested properties."""
        results = [
            {"id": "123", "uuid": "uuid-001", "properties": {"gsd": "5", "bands": "3"}}
        ]

        processed = process_data(results)

        assert "property_gsd" in processed[0]
        assert processed[0]["property_gsd"] == "5"
        assert "property_bands" in processed[0]
        assert processed[0]["property_bands"] == "3"

    def test_process_removes_nested_structures(self):
        """Test that nested structures are removed after processing."""
        results = [
            {
                "id": "123",
                "uuid": "uuid-001",
                "geojson": {"type": "Feature"},
                "properties": {"key": "value"},
            }
        ]

        processed = process_data(results)

        assert "geojson" not in processed[0]
        assert "properties" not in processed[0]

    def test_process_empty_results(self):
        """Test processing empty results list."""
        processed = process_data([])
        assert processed == []

    def test_process_mixed_bbox_formats(self):
        """Test processing different bbox formats."""
        results = [{"id": "1", "bbox": [-122.5, 37.5, -122.0, 38.0]}]

        processed = process_data(results)
        assert "bbox_min_lon" in processed[0]

        results = [{"id": "2", "bbox": None}]

        processed = process_data(results)
        assert (
            "bbox_min_lon" not in processed[0]
            or processed[0].get("bbox_min_lon") is None
        )
