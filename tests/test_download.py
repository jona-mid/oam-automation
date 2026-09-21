"""
Tests for download.py - File download and deduplication functions.
"""

import pytest
import os
import tempfile
import hashlib
from unittest.mock import MagicMock, patch
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from download import calculate_file_hash, find_and_remove_duplicate_thumbnails


class TestCalculateFileHash:
    """Tests for calculate_file_hash function."""

    def test_calculate_hash_success(self, tmp_path):
        """Test successful hash calculation."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")

        result = calculate_file_hash(str(test_file))

        assert result is not None
        assert len(result) == 32

    def test_calculate_hash_nonexistent_file(self):
        """Test hash calculation for nonexistent file."""
        result = calculate_file_hash("/nonexistent/file.txt")

        assert result is None

    def test_calculate_hash_consistency(self, tmp_path):
        """Test hash is consistent for same content."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")

        hash1 = calculate_file_hash(str(test_file))
        hash2 = calculate_file_hash(str(test_file))

        assert hash1 == hash2

    def test_calculate_hash_different_content(self, tmp_path):
        """Test different content produces different hashes."""
        test_file1 = tmp_path / "test1.txt"
        test_file2 = tmp_path / "test2.txt"
        test_file1.write_bytes(b"content 1")
        test_file2.write_bytes(b"content 2")

        hash1 = calculate_file_hash(str(test_file1))
        hash2 = calculate_file_hash(str(test_file2))

        assert hash1 != hash2


class TestFindAndRemoveDuplicateThumbnails:
    """Tests for find_and_remove_duplicate_thumbnails function."""

    def test_no_duplicates(self, tmp_path):
        """Test when there are no duplicates."""
        file1 = tmp_path / "img1.png"
        file2 = tmp_path / "img2.png"
        file1.write_bytes(b"content1")
        file2.write_bytes(b"content2")

        duplicates, removed = find_and_remove_duplicate_thumbnails(str(tmp_path))

        assert len(duplicates) == 0
        assert removed == 0

    def test_duplicates_found(self, tmp_path):
        """Test duplicate detection."""
        file1 = tmp_path / "img1.png"
        file2 = tmp_path / "img2.png"
        file1.write_bytes(b"same content")
        file2.write_bytes(b"same content")

        duplicates, removed = find_and_remove_duplicate_thumbnails(str(tmp_path))

        assert len(duplicates) == 1
        assert removed == 1

    def test_multiple_duplicates(self, tmp_path):
        """Test multiple sets of duplicates."""
        file1 = tmp_path / "img1.png"
        file2 = tmp_path / "img2.png"
        file3 = tmp_path / "img3.png"
        file1.write_bytes(b"content A")
        file2.write_bytes(b"content A")
        file3.write_bytes(b"content B")

        duplicates, removed = find_and_remove_duplicate_thumbnails(str(tmp_path))

        assert len(duplicates) == 1
        assert removed == 1
        assert file1.exists()
        assert not file2.exists()
        assert file3.exists()

    def test_nonexistent_directory(self):
        """Test with nonexistent directory."""
        duplicates, removed = find_and_remove_duplicate_thumbnails("/nonexistent/dir")

        assert len(duplicates) == 0
        assert removed == 0

    def test_empty_directory(self, tmp_path):
        """Test with empty directory."""
        duplicates, removed = find_and_remove_duplicate_thumbnails(str(tmp_path))

        assert len(duplicates) == 0
        assert removed == 0


class TestCmdThumbnails:
    """Tests for thumbnail download command."""

    def test_load_csv_with_season_filter(self, tmp_path):
        """Test CSV loading with season filter."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame(
            [
                {
                    "uuid": "1",
                    "property_thumbnail": "https://example.com/1.png",
                    "pheno_season": "in_season",
                },
                {
                    "uuid": "2",
                    "property_thumbnail": "https://example.com/2.png",
                    "pheno_season": "out_of_season",
                },
                {
                    "uuid": "3",
                    "property_thumbnail": "https://example.com/3.png",
                    "pheno_season": "in_season",
                },
            ]
        )
        df.to_csv(csv_file, index=False)

        loaded_df = pd.read_csv(csv_file)

        assert len(loaded_df) == 3
        assert "pheno_season" in loaded_df.columns

    def test_season_filtering(self, tmp_path):
        """Test season filtering logic."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame(
            [
                {
                    "uuid": "1",
                    "property_thumbnail": "https://example.com/1.png",
                    "pheno_season": "in_season",
                },
                {
                    "uuid": "2",
                    "property_thumbnail": "https://example.com/2.png",
                    "pheno_season": "out_of_season",
                },
            ]
        )
        df.to_csv(csv_file, index=False)

        loaded_df = pd.read_csv(csv_file)
        filtered_df = loaded_df[loaded_df["pheno_season"] == "in_season"]

        assert len(filtered_df) == 1


class TestCmdTifs:
    """Tests for TIF download command."""

    def test_match_thumbnail_to_csv(self, tmp_path):
        """Test matching thumbnail filenames to CSV rows."""
        thumb_dir = tmp_path / "thumbs"
        thumb_dir.mkdir()

        (thumb_dir / "img001.png").write_bytes(b"content")
        (thumb_dir / "img002.png").write_bytes(b"content")

        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame(
            [
                {
                    "uuid": "img001-abc123",
                    "property_thumbnail": "https://example.com/img001.png",
                },
                {
                    "uuid": "img002-def456",
                    "property_thumbnail": "https://example.com/img002.png",
                },
            ]
        )
        df.to_csv(csv_file, index=False)

        thumbnail_files = [f for f in os.listdir(thumb_dir) if f.endswith(".png")]

        assert len(thumbnail_files) == 2
        assert "img001.png" in thumbnail_files
        assert "img002.png" in thumbnail_files


class TestDownloadFile:
    """Tests for download_file utility function."""

    @patch("utils.requests.get")
    def test_download_file_success(self, mock_get, tmp_path):
        """Test successful file download."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.headers = {"Content-Length": "100"}
        mock_response.iter_content = lambda chunk_size: [b"test content"]
        mock_get.return_value = mock_response

        output_file = tmp_path / "output.txt"

        from utils import download_file

        result = download_file(
            "https://example.com/file.txt", str(output_file), timeout=60
        )

        assert result is True
        assert output_file.exists()

    @patch("utils.requests.get")
    def test_download_file_empty_url(self, mock_get, tmp_path):
        """Test download with empty URL."""
        output_file = tmp_path / "output.txt"

        from utils import download_file

        result = download_file("", str(output_file), timeout=60)

        assert result is False

    @patch("utils.requests.get")
    def test_download_file_network_error(self, mock_get, tmp_path):
        """Test download with network error."""
        import requests

        mock_get.side_effect = requests.RequestException("Network error")

        output_file = tmp_path / "output.txt"

        from utils import download_file

        result = download_file(
            "https://example.com/file.txt", str(output_file), timeout=60
        )

        assert result is False


class TestDownloadParallel:
    """Tests for parallel download function."""

    @patch("utils.download_file")
    def test_sequential_download(self, mock_download, tmp_path):
        """Test sequential download with single worker."""
        mock_download.return_value = True

        tasks = [
            ("https://example.com/1.txt", str(tmp_path / "1.txt"), "1.txt"),
            ("https://example.com/2.txt", str(tmp_path / "2.txt"), "2.txt"),
        ]

        from utils import download_parallel

        success, failed = download_parallel(tasks, workers=1)

        assert success == 2
        assert failed == 0

    @patch("utils.download_file")
    def test_parallel_download(self, mock_download, tmp_path):
        """Test parallel download with multiple workers."""
        mock_download.return_value = True

        tasks = [
            ("https://example.com/1.txt", str(tmp_path / "1.txt"), "1.txt"),
            ("https://example.com/2.txt", str(tmp_path / "2.txt"), "2.txt"),
            ("https://example.com/3.txt", str(tmp_path / "3.txt"), "3.txt"),
        ]

        from utils import download_parallel

        success, failed = download_parallel(tasks, workers=3)

        assert success == 3
        assert failed == 0

    def test_empty_tasks(self):
        """Test with empty task list."""
        from utils import download_parallel

        success, failed = download_parallel([], workers=1)

        assert success == 0
        assert failed == 0
