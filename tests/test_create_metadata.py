"""
Tests for create_metadata.py - Metadata creation commands.
"""

import pytest
import os
import tempfile
import csv
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCmdTif:
    """Tests for cmd_tif function."""

    def test_cmd_tif_basic(self, tmp_path):
        """Test basic TIF metadata creation."""
        csv_file = tmp_path / "input.csv"
        tif_dir = tmp_path / "tifs"
        tif_dir.mkdir()

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["uuid", "platform", "gsd", "acquisition_start"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "uuid": "test-img-001-abc",
                    "platform": "uav",
                    "gsd": "5",
                    "acquisition_start": "2025-01-15",
                }
            )

        tif_file = tif_dir / "test-img-001.tif"
        tif_file.write_bytes(b"fake tif content")

        class Args:
            csv = str(csv_file)
            output_dir = str(tif_dir)
            metadata_output = None
            add_image_size = False

        with patch("create_metadata.utils.configure_logging"):
            with patch("create_metadata.utils.configure_utf8_stdio"):
                with patch("create_metadata.utils.read_csv") as mock_read:
                    mock_read.return_value = [
                        {
                            "uuid": "test-img-001-abc",
                            "platform": "uav",
                            "gsd": "5",
                            "acquisition_start": "2025-01-15",
                            "provider": "Test",
                            "contact": "test@test.com",
                        }
                    ]

                    from create_metadata import cmd_tif

                    cmd_tif(Args())

    def test_cmd_tif_missing_csv(self, tmp_path):
        """Test TIF command with missing CSV file."""

        class Args:
            csv = str(tmp_path / "nonexistent.csv")
            output_dir = str(tmp_path / "tifs")
            metadata_output = None
            add_image_size = False

        with patch("create_metadata.utils.configure_logging"):
            with patch("create_metadata.utils.configure_utf8_stdio"):
                from create_metadata import cmd_tif

                cmd_tif(Args())

    def test_cmd_tif_missing_output_dir(self, tmp_path):
        """Test TIF command with missing output directory."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("uuid,platform\ngs01,uav")

        class Args:
            csv = str(csv_file)
            output_dir = str(tmp_path / "nonexistent")
            metadata_output = None
            add_image_size = False

        with patch("create_metadata.utils.configure_logging"):
            with patch("create_metadata.utils.configure_utf8_stdio"):
                from create_metadata import cmd_tif

                cmd_tif(Args())


class TestCmdJpeg:
    """Tests for cmd_jpeg function."""

    def test_cmd_jpeg_basic(self, tmp_path):
        """Test basic JPEG metadata creation."""
        source_metadata = tmp_path / "source.csv"
        jpeg_dir = tmp_path / "jpegs"
        jpeg_dir.mkdir()

        with open(source_metadata, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "platform", "gsd", "capture_date"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "img001",
                    "platform": "drone",
                    "gsd": "5",
                    "capture_date": "2025-01-15",
                }
            )

        jpeg_file = jpeg_dir / "img001.jpeg"
        jpeg_file.write_bytes(b"fake jpeg content")

        class Args:
            pass

        args = Args()
        args.source_metadata = str(source_metadata)
        args.jpeg_folder = str(jpeg_dir)
        args.output_metadata = str(tmp_path / "output.csv")

        with patch("create_metadata.utils.configure_logging"):
            with patch("create_metadata.utils.configure_utf8_stdio"):
                with patch("create_metadata.utils.read_csv") as mock_read:
                    mock_read.return_value = [
                        {
                            "filename": "img001",
                            "platform": "drone",
                            "gsd": "5",
                            "capture_date": "2025-01-15",
                        }
                    ]

                    from create_metadata import cmd_jpeg

                    cmd_jpeg(args)
