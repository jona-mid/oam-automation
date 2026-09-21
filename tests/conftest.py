"""
Shared fixtures and mocks for OpenAerialMap Scraper tests.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from io import StringIO
import tempfile
import os


@pytest.fixture
def sample_bbox():
    """Sample bounding box coordinates."""
    return [-122.4194, 37.7749, -122.4194 + 0.1, 37.7749 + 0.1]


@pytest.fixture
def sample_bbox_str():
    """Sample bounding box as string."""
    return "[-122.4194, 37.7749, -122.3194, 37.8749]"


@pytest.fixture
def sample_csv_data():
    """Sample OpenAerialMap CSV data."""
    return [
        {
            "id": "abc123",
            "uuid": "uuid-001",
            "platform": "uav",
            "gsd": "5",
            "property_bands": "3",
            "uploaded_at": "2025-01-15T10:00:00Z",
            "acquisition_start": "2025-01-10T08:00:00Z",
            "acquisition_end": "2025-01-10T10:00:00Z",
            "bbox": "[-122.4194, 37.7749, -122.3194, 37.8749]",
            "property_thumbnail": "https://example.com/thumbnails/img001.png",
            "provider": "Test Provider",
            "contact": "test@example.com",
        },
        {
            "id": "def456",
            "uuid": "uuid-002",
            "platform": "aircraft",
            "gsd": "8",
            "property_bands": "4",
            "uploaded_at": "2025-01-16T10:00:00Z",
            "acquisition_start": "2025-01-12T08:00:00Z",
            "acquisition_end": "2025-01-12T12:00:00Z",
            "bbox": "[-122.5194, 37.8749, -122.4194, 37.9749]",
            "property_thumbnail": "https://example.com/thumbnails/img002.png",
            "provider": "Another Provider",
            "contact": "another@test.com",
        },
        {
            "id": "ghi789",
            "uuid": "uuid-003",
            "platform": "uav",
            "gsd": "15",
            "property_bands": "1",
            "uploaded_at": "2025-01-17T10:00:00Z",
            "acquisition_start": "2025-01-14T08:00:00Z",
            "acquisition_end": "2025-01-14T10:00:00Z",
            "bbox": "[-122.6194, 37.9749, -122.5194, 38.0749]",
            "property_thumbnail": "https://example.com/thumbnails/img003.png",
            "provider": "Third Provider",
            "contact": "third@test.com",
        },
    ]


@pytest.fixture
def sample_csv_dataframe(sample_csv_data):
    """Sample CSV data as DataFrame."""
    return pd.DataFrame(sample_csv_data)


@pytest.fixture
def sample_phenology_data():
    """Sample phenology data mapping."""
    return {
        "img001.tif": (100, 200),
        "img002.tif": (150, 250),
        "img003.tif": None,
    }


@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        if sample_csv_data:
            import csv

            writer = csv.DictWriter(f, fieldnames=sample_csv_data[0].keys())
            writer.writeheader()
            writer.writerows(sample_csv_data)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def mock_requests():
    """Mock requests module."""
    with patch("scrape.requests") as mock:
        yield mock


@pytest.fixture
def mock_ee():
    """Mock Earth Engine module."""
    with patch("filter.ee") as mock:
        mock.Image.return_value.select.return_value.eq.return_value.Or.return_value = (
            MagicMock()
        )
        mock.Geometry.Rectangle.return_value = MagicMock()
        mock.Reducer.mean.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_rasterio():
    """Mock rasterio module."""
    with patch("tif_to_jpeg.rasterio") as mock:
        mock_open = MagicMock()
        mock_open.return_value.crs.is_geographic = False
        mock_open.return_value.crs = MagicMock()
        mock_open.return_value.width = 1000
        mock_open.return_value.height = 1000
        mock_open.return_value.count = 3
        mock_open.return_value.bounds = MagicMock(
            left=-122.4, bottom=37.7, right=-122.3, top=37.8
        )
        mock_open.return_value.block_windows = MagicMock(
            return_value=[(None, MagicMock())]
        )
        mock_open.return_value.__enter__ = MagicMock(return_value=mock_open)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock.open.return_value = mock_open
        yield mock


@pytest.fixture
def sample_tif_metadata():
    """Sample TIF metadata CSV data."""
    return [
        {
            "filename": "img001.tif",
            "authors": "Test Author",
            "origin": "openaerialmap.org",
            "capture_date": "2025-01-10",
            "is_long_campaign": "False",
            "platform": "drone",
            "gsd": "5",
            "forest_cover_percentage": "45.5",
            "width": "1000",
            "height": "1000",
            "pixels": "1000000",
        },
        {
            "filename": "img002.tif",
            "authors": "Another Author",
            "origin": "openaerialmap.org",
            "capture_date": "2025-01-12",
            "is_long_campaign": "False",
            "platform": "airborne",
            "gsd": "8",
            "forest_cover_percentage": "30.2",
            "width": "2000",
            "height": "2000",
            "pixels": "4000000",
        },
    ]


@pytest.fixture
def sample_jpeg_files():
    """Sample JPEG filenames."""
    return ["img001.jpeg", "img002.jpeg", "img003.jpeg"]
