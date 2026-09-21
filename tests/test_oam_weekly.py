"""Tests for oam_weekly pure logic: gate, diff normalization, kwargs building, ledger append."""

import csv
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import oam_weekly
import deadtrees_seam


def _write_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


REPORT_HEADER = [
    "filename",
    "modis_category",
    "tree_canopy_leaf_state",
    "review_status",
    "platform",
]
META_HEADER = [
    "filename",
    "acquisition_date",
    "platform",
    "licence",
    "authors",
    "additional_information",
]
TIF_META_HEADER = ["filename", "property_license"]
SAMPLE_ADDITIONAL = (
    "This orthophoto data is available through OpenAerialMap, provided by Contributors of "
    "Open Imagery Network. More information: https://api.openaerialmap.org/meta?_id=abc123 Accessed December 4, 2025."
)


def _write_trial_metadata(tmp_path, report_rows, meta_rows, tif_rows=None):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    _write_csv(metadata_dir / "phenology_report.csv", REPORT_HEADER, report_rows)
    _write_csv(metadata_dir / "jpeg_metadata.csv", META_HEADER, meta_rows)
    _write_csv(
        metadata_dir / "tif_metadata.csv",
        TIF_META_HEADER,
        tif_rows if tif_rows is not None else [[name, "CC-BY 4.0"] for name, *_ in report_rows],
    )


class TestLoadGateCandidates:
    def test_gate_keeps_only_in_season_leaf_on_success(self, tmp_path):
        _write_trial_metadata(
            tmp_path,
            [
                ["a.tif", "in_season", "leaf_on", "success", "drone"],
                ["b.tif", "in_season", "not_leaf_on", "success", "drone"],
                ["c.tif", "out_of_season", "leaf_on", "success", "drone"],
                ["d.tif", "in_season", "leaf_on", "unavailable", "drone"],
                ["e.tif", "in_season", "leaf_on", "success", "drone"],
            ],
            [
                ["a.tif", "2026-09-11", "drone", "CC BY", "Contributors of Open Imagery Network", SAMPLE_ADDITIONAL],
                ["b.tif", "2026-09-11", "drone", "CC BY", "Contributors of Open Imagery Network", SAMPLE_ADDITIONAL],
                ["c.tif", "2026-09-11", "drone", "CC BY", "Contributors of Open Imagery Network", SAMPLE_ADDITIONAL],
                ["e.tif", "2026-09-12", "drone", "CC BY", "Contributors of Open Imagery Network", SAMPLE_ADDITIONAL],
            ],
        )
        gate = oam_weekly.load_gate_candidates(tmp_path)
        assert sorted(gate["filename"]) == ["a.tif", "e.tif"]

    def test_join_drops_rows_without_jpeg_metadata(self, tmp_path):
        _write_trial_metadata(
            tmp_path,
            [["a.tif", "in_season", "leaf_on", "success", "drone"]],
            [],
        )
        gate = oam_weekly.load_gate_candidates(tmp_path)
        assert len(gate) == 0

    def test_overlapping_platform_column_keeps_jpeg_metadata_name(self, tmp_path):
        """Both CSVs carry `platform`; the jpeg-metadata value must survive the merge."""
        _write_trial_metadata(
            tmp_path,
            [["a.tif", "in_season", "leaf_on", "success", "drone"]],
            [["a.tif", "2026-09-11", "drone", "CC BY", "Contributors of Open Imagery Network", SAMPLE_ADDITIONAL]],
        )
        gate = oam_weekly.load_gate_candidates(tmp_path)
        kwargs = oam_weekly.build_upload_kwargs(gate.iloc[0], date(2026, 9, 21))
        assert kwargs["platform"] == "drone"

    def test_duplicated_tif_metadata_rows_do_not_multiply_candidates(self, tmp_path):
        """The upstream tif mode appends existing rows on rebuild; dedup must keep one row per file."""
        _write_trial_metadata(
            tmp_path,
            [["a.tif", "in_season", "leaf_on", "success", "drone"]],
            [["a.tif", "2026-09-11", "drone", "CC BY", "Contributors of Open Imagery Network", SAMPLE_ADDITIONAL]],
            tif_rows=[["a.tif", "CC-BY 4.0"], ["a.tif", "CC-BY 4.0"], ["a.tif", "CC-BY 4.0"]],
        )
        gate = oam_weekly.load_gate_candidates(tmp_path)
        assert len(gate) == 1


class TestLoadLedgerFilenames:
    def test_missing_file_yields_empty_set(self, tmp_path):
        assert oam_weekly.load_ledger_filenames(tmp_path / "nope.csv") == set()

    def test_filenames_normalized_lowercase(self, tmp_path):
        ledger = tmp_path / "ledger.csv"
        _write_csv(ledger, ["filename"], [["ABC.tif"], [" def.TIFF "]])
        assert oam_weekly.load_ledger_filenames(ledger) == {"abc.tif", "def.tiff"}


class TestBuildUploadKwargs:
    def _row(self, **overrides):
        base = {
            "filename": "abc.tif",
            "acquisition_date": "2026-09-11",
            "platform": "drone",
            "licence": "CC BY",
            "authors": "Contributors of Open Imagery Network",
            "additional_information": SAMPLE_ADDITIONAL,
            "property_license": "CC-BY 4.0",
        }
        base.update(overrides)
        return pd.Series(base)

    def test_happy_path(self):
        kwargs = oam_weekly.build_upload_kwargs(self._row(), date(2026, 9, 21))
        assert kwargs["authors"] == ["Contributors of Open Imagery Network"]
        assert kwargs["platform"] == "drone"
        assert kwargs["license"] == "CC BY"
        assert kwargs["data_access"] == "public"
        assert kwargs["acquisition_year"] == 2026
        assert kwargs["acquisition_month"] == 9
        assert kwargs["acquisition_day"] == 11
        assert kwargs["citation_doi"] is None

    def test_additional_information_uses_locked_template(self):
        kwargs = oam_weekly.build_upload_kwargs(self._row(), date(2026, 9, 21))
        expected = (
            "This orthophoto data is available through OpenAerialMap, provided by Contributors "
            "of Open Imagery Network. Licensed under CC-BY 4.0. More information about this "
            "dataset: https://api.openaerialmap.org/meta?_id=abc123 Accessed September 21, 2026."
        )
        assert kwargs["additional_information"] == expected

    def test_non_cc_by_license_maps_and_keeps_raw_sentence(self):
        kwargs = oam_weekly.build_upload_kwargs(
            self._row(property_license="CC BY-SA 4.0"), date(2026, 9, 21)
        )
        assert kwargs["license"] == "CC BY-SA"
        assert "Licensed under CC BY-SA 4.0." in kwargs["additional_information"]

    def test_unknown_license_fails_closed(self):
        with pytest.raises(ValueError, match="unknown OAM license"):
            oam_weekly.build_upload_kwargs(
                self._row(property_license="Public Domain"), date(2026, 9, 21)
            )

    def test_missing_property_license_fails_closed(self):
        with pytest.raises(ValueError, match="property_license"):
            oam_weekly.build_upload_kwargs(
                self._row(property_license=None), date(2026, 9, 21)
            )

    def test_missing_oam_id_fails_closed(self):
        with pytest.raises(ValueError, match="_id"):
            oam_weekly.build_upload_kwargs(
                self._row(additional_information="no url here"), date(2026, 9, 21)
            )

    def test_unknown_platform_fails_closed(self):
        with pytest.raises(ValueError, match="unknown platform"):
            oam_weekly.build_upload_kwargs(self._row(platform="balloon"), date(2026, 9, 21))

    def test_missing_acquisition_date_fails_closed(self):
        with pytest.raises(ValueError, match="acquisition_date"):
            oam_weekly.build_upload_kwargs(
                self._row(acquisition_date=None), date(2026, 9, 21)
            )


class TestFormatEnDate:
    def test_day_without_leading_zero(self):
        assert oam_weekly.format_en_date(date(2026, 9, 4)) == "September 4, 2026"


class TestAppendLedgerRow:
    def test_creates_header_then_appends_without_header(self, tmp_path):
        ledger = tmp_path / "ledger.csv"
        kwargs = {
            "license": "CC BY",
            "platform": "drone",
            "authors": ["Contributors of Open Imagery Network"],
            "acquisition_year": 2026,
            "acquisition_month": 9,
            "acquisition_day": 11,
            "data_access": "public",
            "additional_information": "info",
            "citation_doi": None,
        }
        oam_weekly.append_ledger_row(ledger, "a.tif", kwargs)
        oam_weekly.append_ledger_row(ledger, "b.tif", kwargs)
        with ledger.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == oam_weekly.LEDGER_COLUMNS
        assert rows[1][0] == "a.tif"
        assert rows[1][1] == "CC BY"
        assert rows[2][0] == "b.tif"
        assert len(rows) == 3


class TestSeamSignature:
    def test_upload_kwargs_match_seam_signature(self):
        """The wrapper spreads build_upload_kwargs into the seam; keys must match its parameters."""
        import inspect

        kwargs = oam_weekly.build_upload_kwargs(
            pd.Series(
                {
                    "filename": "abc.tif",
                    "acquisition_date": "2026-09-11",
                    "platform": "drone",
                    "licence": "CC BY",
                    "authors": "A",
                    "additional_information": SAMPLE_ADDITIONAL,
                    "property_license": "CC-BY 4.0",
                }
            ),
            date(2026, 9, 21),
        )
        params = set(inspect.signature(deadtrees_seam.upload_and_process).parameters)
        assert {"file_path", *kwargs.keys()} == params


class TestWriteStatus:
    def test_status_file_contains_counts_and_exit(self, tmp_path):
        status = tmp_path / "last_run.txt"
        counts = {"candidates": 5, "uploaded": 4, "failed": 1}
        oam_weekly.write_status(status, tmp_path / "run", False, counts, "failed")
        text = status.read_text(encoding="utf-8")
        assert "candidates: 5" in text
        assert "uploaded: 4" in text
        assert "failed: 1" in text
        assert "exit: failed" in text
        assert "dry_run: False" in text
