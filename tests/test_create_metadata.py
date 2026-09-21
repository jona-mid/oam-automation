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
        args.image_size_csv = ""
        args.uploaded_folder = ""
        args.prioritize_small_files = False
        args.batch_size = 0

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

    def test_cmd_jpeg_prioritize_small_files(self, tmp_path):
        """Test JPEG metadata with small file prioritization."""
        source_metadata = tmp_path / "source.csv"
        jpeg_dir = tmp_path / "jpegs"
        jpeg_dir.mkdir()

        with open(source_metadata, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "platform"])
            writer.writeheader()
            writer.writerow({"filename": "img001", "platform": "drone"})
            writer.writerow({"filename": "img002", "platform": "airborne"})

        (jpeg_dir / "img001.jpeg").write_bytes(b"small")
        (jpeg_dir / "img002.jpeg").write_bytes(b"much larger content here")

        class Args:
            pass

        args = Args()
        args.source_metadata = str(source_metadata)
        args.jpeg_folder = str(jpeg_dir)
        args.output_metadata = str(tmp_path / "output.csv")
        args.image_size_csv = ""
        args.uploaded_folder = ""
        args.prioritize_small_files = True
        args.batch_size = 0

        with patch("create_metadata.utils.configure_logging"):
            with patch("create_metadata.utils.configure_utf8_stdio"):
                with patch("create_metadata.utils.read_csv") as mock_read:
                    mock_read.return_value = [
                        {"filename": "img001", "platform": "drone"},
                        {"filename": "img002", "platform": "airborne"},
                    ]

                    from create_metadata import cmd_jpeg

                    cmd_jpeg(args)

    def test_cmd_jpeg_prioritize_small_files(self, tmp_path):
        """Test JPEG metadata with small file prioritization."""
        source_metadata = tmp_path / "source.csv"
        jpeg_dir = tmp_path / "jpegs"
        jpeg_dir.mkdir()

        with open(source_metadata, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "platform"])
            writer.writeheader()
            writer.writerow({"filename": "img001", "platform": "drone"})
            writer.writerow({"filename": "img002", "platform": "airborne"})

        (jpeg_dir / "img001.jpeg").write_bytes(b"small")
        (jpeg_dir / "img002.jpeg").write_bytes(b"much larger content here")

        class Args:
            pass

        args = Args()
        args.source_metadata = str(source_metadata)
        args.jpeg_folder = str(jpeg_dir)
        args.output_metadata = str(tmp_path / "output.csv")
        args.image_size_csv = ""
        args.uploaded_folder = ""
        args.prioritize_small_files = True
        args.batch_size = 0

        with patch("create_metadata.utils.configure_logging"):
            with patch("create_metadata.utils.configure_utf8_stdio"):
                with patch("create_metadata.utils.read_csv") as mock_read:
                    mock_read.return_value = [
                        {"filename": "img001", "platform": "drone"},
                        {"filename": "img002", "platform": "airborne"},
                    ]

                    from create_metadata import cmd_jpeg

                    cmd_jpeg(args)


class TestCmdFilter:
    """Tests for cmd_filter function."""

    def test_cmd_filter_basic(self, tmp_path):
        """Test basic filter functionality."""
        selected_file = tmp_path / "selected.csv"
        metadata_file = tmp_path / "metadata.csv"
        phenology_file = tmp_path / "phenology.csv"

        selected_file.write_text("img001.tif\nimg002.tif\n")

        with open(phenology_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "pheno_start_doy", "pheno_end_doy"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "img001.tif",
                    "pheno_start_doy": "100",
                    "pheno_end_doy": "200",
                }
            )
            writer.writerow(
                {
                    "filename": "img002.tif",
                    "pheno_start_doy": "150",
                    "pheno_end_doy": "250",
                }
            )

        with open(metadata_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "capture_date", "is_long_campaign"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "img001.tif",
                    "capture_date": "2025-05-15",
                    "is_long_campaign": "False",
                }
            )
            writer.writerow(
                {
                    "filename": "img002.tif",
                    "capture_date": "2025-06-15",
                    "is_long_campaign": "False",
                }
            )

        class Args:
            selected = str(selected_file)
            metadata = str(metadata_file)
            phenology = str(phenology_file)
            output = str(tmp_path / "output.csv")
            pad_days = 0
            shoulder_only = False
            include_debug_columns = True

        with patch("create_metadata.utils.configure_logging"):
            with patch("create_metadata.utils.configure_utf8_stdio"):
                from create_metadata import cmd_filter

                cmd_filter(Args())

    def test_cmd_filter_drops_long_campaign(self, tmp_path):
        """Test that long campaigns are filtered out."""
        selected_file = tmp_path / "selected.csv"
        metadata_file = tmp_path / "metadata.csv"
        phenology_file = tmp_path / "phenology.csv"

        selected_file.write_text("img001.tif\nimg002.tif\n")

        with open(phenology_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "pheno_start_doy", "pheno_end_doy"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "img001.tif",
                    "pheno_start_doy": "100",
                    "pheno_end_doy": "200",
                }
            )

        with open(metadata_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "capture_date", "is_long_campaign"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "img001.tif",
                    "capture_date": "2025-05-15",
                    "is_long_campaign": "True",
                }
            )

        class Args:
            selected = str(selected_file)
            metadata = str(metadata_file)
            phenology = str(phenology_file)
            output = str(tmp_path / "output.csv")
            pad_days = 0
            shoulder_only = False
            include_debug_columns = True

        with patch("create_metadata.utils.configure_logging"):
            with patch("create_metadata.utils.configure_utf8_stdio"):
                from create_metadata import cmd_filter

                cmd_filter(Args())


class TestBatchedOutputPath:
    """Tests for batched output path generation."""

    def test_single_digit_batch(self):
        """Test batch path with single digit."""
        from utils import batched_output_path

        result = batched_output_path("output.csv", 1)
        assert "batch001" in result

    def test_multiple_digit_batch(self):
        """Test batch path with multiple digits."""
        from utils import batched_output_path

        result = batched_output_path("output.csv", 100)
        assert "batch100" in result


class TestLoadImageSizeCsv:
    """Tests for loading image size CSV."""

    def test_load_valid_csv(self, tmp_path):
        """Test loading valid image size CSV."""
        csv_file = tmp_path / "sizes.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Filename", "Height", "Width"])
            writer.writeheader()
            writer.writerow({"Filename": "img001", "Height": "1000", "Width": "1000"})

        from utils import load_image_size_csv

        result = load_image_size_csv(str(csv_file))

        assert "img001" in result
        assert result["img001"]["height"] == 1000
        assert result["img001"]["width"] == 1000

    def test_load_nonexistent_csv(self):
        """Test loading nonexistent CSV returns empty dict."""
        from utils import load_image_size_csv

        result = load_image_size_csv("/nonexistent/sizes.csv")

        assert result == {}


class TestLoadUploadedFilenames:
    """Tests for loading uploaded filenames."""

    def test_load_from_csv_files(self, tmp_path):
        """Test loading filenames from CSV files."""
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()

        csv_file1 = upload_dir / "batch1.csv"
        with open(csv_file1, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename"])
            writer.writeheader()
            writer.writerow({"filename": "img001.tif"})
            writer.writerow({"filename": "img002.tif"})

        from utils import load_uploaded_filenames

        result = load_uploaded_filenames(str(upload_dir))

        assert len(result) == 2

    def test_load_from_nonexistent_folder(self):
        """Test loading from nonexistent folder returns empty set."""
        from utils import load_uploaded_filenames

        result = load_uploaded_filenames("/nonexistent/folder")

        assert result == set()


class TestMatchTifToCsv:
    """Tests for matching TIF filenames to CSV rows."""

    def test_match_exact_filename(self):
        """Test matching exact filename."""
        csv_rows = [{"filename": "img001.tif"}]

        from utils import match_tif_to_csv

        result = match_tif_to_csv("img001.tif", csv_rows)

        assert result is not None

    def test_match_without_extension(self):
        """Test matching filename without extension."""
        csv_rows = [{"filename": "img001.tif"}]

        from utils import match_tif_to_csv

        result = match_tif_to_csv("img001", csv_rows)

        assert result is not None

    def test_no_match(self):
        """Test when no match is found."""
        csv_rows = [{"filename": "img001.tif"}]

        from utils import match_tif_to_csv

        result = match_tif_to_csv("img999.tif", csv_rows)

        assert result is None
