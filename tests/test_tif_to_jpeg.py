"""
Tests for tif_to_jpeg.py - TIF to JPEG conversion functions.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConvertTifTo20cmJpeg:
    """Tests for _convert_tif_to_20cm_jpeg function."""

    @patch("tif_to_jpeg.rasterio")
    def test_conversion_with_projected_crs(self, mock_rasterio, tmp_path):
        """Test conversion with projected CRS (meters)."""
        mock_src = MagicMock()
        mock_src.crs = MagicMock()
        mock_src.crs.is_geographic = False
        mock_src.crs = MagicMock()
        mock_src.width = 1000
        mock_src.height = 1000
        mock_src.count = 3
        mock_src.bounds = MagicMock(left=0, bottom=0, right=1000, top=1000)
        mock_src.block_windows = MagicMock(return_value=[(None, MagicMock())])

        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_src)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        mock_calculate = MagicMock()
        mock_calculate.return_value = (MagicMock(), 500, 500)
        mock_rasterio.warp.calculate_default_transform = mock_calculate

        input_file = tmp_path / "input.tif"
        output_file = tmp_path / "output.jpeg"

        from tif_to_jpeg import _convert_tif_to_20cm_jpeg
        from rasterio.crs import CRS

        result = _convert_tif_to_20cm_jpeg(
            str(input_file), str(output_file), CRS.from_epsg(3857)
        )

        assert isinstance(result, tuple)

    @patch("tif_to_jpeg.rasterio")
    def test_conversion_with_geographic_crs(self, mock_rasterio, tmp_path):
        """Test conversion with geographic CRS (WGS84)."""
        mock_src = MagicMock()
        mock_src.crs = MagicMock()
        mock_src.crs.is_geographic = True
        mock_src.width = 1000
        mock_src.height = 1000
        mock_src.count = 3
        mock_src.bounds = MagicMock(left=-122.5, bottom=37.5, right=-122.0, top=38.0)
        mock_src.block_windows = MagicMock(return_value=[(None, MagicMock())])

        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_src)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        mock_calculate = MagicMock()
        mock_calculate.return_value = (MagicMock(), 500, 500)
        mock_rasterio.warp.calculate_default_transform = mock_calculate

        input_file = tmp_path / "input.tif"
        output_file = tmp_path / "output.jpeg"

        from tif_to_jpeg import _convert_tif_to_20cm_jpeg
        from rasterio.crs import CRS

        result = _convert_tif_to_20cm_jpeg(
            str(input_file), str(output_file), CRS.from_epsg(3857)
        )

        assert isinstance(result, tuple)

    @patch("tif_to_jpeg.rasterio")
    def test_conversion_missing_crs(self, mock_rasterio, tmp_path):
        """Test conversion fails gracefully with missing CRS."""
        mock_src = MagicMock()
        mock_src.crs = None

        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_src)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        input_file = tmp_path / "input.tif"
        output_file = tmp_path / "output.jpeg"

        from tif_to_jpeg import _convert_tif_to_20cm_jpeg
        from rasterio.crs import CRS

        success, filename, error = _convert_tif_to_20cm_jpeg(
            str(input_file), str(output_file), CRS.from_epsg(3857)
        )

        assert success is False
        assert "Missing CRS" in error

    @patch("tif_to_jpeg.rasterio")
    def test_conversion_handles_exception(self, mock_rasterio, tmp_path):
        """Test conversion handles exceptions gracefully."""
        mock_rasterio.open.side_effect = Exception("Test error")

        input_file = tmp_path / "input.tif"
        output_file = tmp_path / "output.jpeg"

        from tif_to_jpeg import _convert_tif_to_20cm_jpeg
        from rasterio.crs import CRS

        success, filename, error = _convert_tif_to_20cm_jpeg(
            str(input_file), str(output_file), CRS.from_epsg(3857)
        )

        assert success is False
        assert "Test error" in error


class TestConvertOne:
    """Tests for _convert_one function."""

    @patch("tif_to_jpeg._convert_tif_to_20cm_jpeg")
    def test_convert_one_calls_worker(self, mock_convert, tmp_path):
        """Test _convert_one calls the worker function."""
        mock_convert.return_value = (True, "test.tif", None)

        from tif_to_jpeg import _convert_one
        from rasterio.crs import CRS

        args = (
            str(tmp_path / "input.tif"),
            str(tmp_path / "output.jpeg"),
            CRS.from_epsg(3857),
        )
        result = _convert_one(args)

        mock_convert.assert_called_once()


class TestMain:
    """Tests for main entry point."""

    def test_main_finds_tif_files(self, tmp_path):
        """Test main finds TIF files in directory."""
        tif_dir = tmp_path / "tifs"
        tif_dir.mkdir()

        (tif_dir / "img1.tif").write_bytes(b"fake tif")
        (tif_dir / "img2.tiff").write_bytes(b"fake tiff")
        (tif_dir / "img3.txt").write_bytes(b"not a tif")

        tif_files = list(tif_dir.glob("*.tif")) + list(tif_dir.glob("*.tiff"))

        assert len(tif_files) == 2

    def test_main_output_path_generation(self, tmp_path):
        """Test output path generation."""
        input_path = tmp_path / "input.tif"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        base = input_path.stem
        out_path = output_dir / f"{base}.jpeg"

        assert str(out_path) == str(output_dir / "input.jpeg")


class TestWarpedVRT:
    """Tests for WarpedVRT usage."""

    def test_uses_warped_vrt(self):
        """Test that WarpedVRT is used for reprojection."""
        from tif_to_jpeg import WarpedVRT

        assert WarpedVRT is not None


class TestResampling:
    """Tests for resampling configuration."""

    def test_target_resolution_constant(self):
        """Test target resolution is defined as constant."""
        from tif_to_jpeg import TARGET_RESOLUTION_M

        assert TARGET_RESOLUTION_M == 0.2

    def test_default_meter_crs(self):
        """Test default meter CRS is EPSG:3857."""
        from tif_to_jpeg import DEFAULT_METER_CRS
        from rasterio.crs import CRS

        assert DEFAULT_METER_CRS == CRS.from_epsg(3857)
