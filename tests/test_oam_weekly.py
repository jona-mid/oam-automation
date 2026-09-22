"""Tests for oam_weekly pure logic: gate, diff, kwargs building, ledger append, status, config."""

import csv
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import deadtrees_seam
import oam_weekly


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
    "oam_id",
    "property_license",
]
SAMPLE_OAM_ID = "abc123"
SAMPLE_LICENSE = "CC-BY 4.0"
SAMPLE_ADDITIONAL = (
    "This orthophoto data is available through OpenAerialMap, provided by Contributors of "
    "Open Imagery Network. More information: https://api.openaerialmap.org/meta?_id=abc123 Accessed December 4, 2025."
)


def _meta_row(filename, acquisition_date="2026-09-11", platform="drone"):
    return [
        filename,
        acquisition_date,
        platform,
        "CC BY",
        "Contributors of Open Imagery Network",
        SAMPLE_ADDITIONAL,
        SAMPLE_OAM_ID,
        SAMPLE_LICENSE,
    ]


def _write_trial_metadata(tmp_path, report_rows, meta_rows):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    _write_csv(metadata_dir / "phenology_report.csv", REPORT_HEADER, report_rows)
    _write_csv(metadata_dir / "jpeg_metadata.csv", META_HEADER, meta_rows)


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
                _meta_row("a.tif"),
                _meta_row("b.tif"),
                _meta_row("c.tif"),
                _meta_row("e.tif", "2026-09-12"),
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
            [_meta_row("a.tif")],
        )
        gate = oam_weekly.load_gate_candidates(tmp_path)
        kwargs = oam_weekly.build_upload_kwargs(gate.iloc[0], date(2026, 9, 21))
        assert kwargs["platform"] == "drone"


class TestNormalizeFilename:
    def test_strips_and_lowercases(self):
        assert oam_weekly.normalize_filename(" ABC.TIF ") == "abc.tif"


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
            "oam_id": SAMPLE_OAM_ID,
            "property_license": SAMPLE_LICENSE,
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
        with pytest.raises(ValueError, match="oam_id"):
            oam_weekly.build_upload_kwargs(self._row(oam_id=None), date(2026, 9, 21))

    def test_unknown_platform_fails_closed(self):
        with pytest.raises(ValueError, match="unknown platform"):
            oam_weekly.build_upload_kwargs(self._row(platform="balloon"), date(2026, 9, 21))

    def test_missing_acquisition_date_fails_closed(self):
        with pytest.raises(ValueError, match="acquisition_date"):
            oam_weekly.build_upload_kwargs(
                self._row(acquisition_date=None), date(2026, 9, 21)
            )


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
                    "oam_id": SAMPLE_OAM_ID,
                    "property_license": SAMPLE_LICENSE,
                }
            ),
            date(2026, 9, 21),
        )
        params = set(inspect.signature(deadtrees_seam.upload_and_process).parameters)
        assert {"file_path", *kwargs.keys()} == params


class TestPrepareCandidates:
    def _gate_row(self, filename):
        return pd.Series(
            {
                "filename": filename,
                "modis_category": "in_season",
                "tree_canopy_leaf_state": "leaf_on",
                "review_status": "success",
                "platform": "drone",
                "acquisition_date": "2026-09-11",
                "licence": "CC BY",
                "authors": "Contributors of Open Imagery Network",
                "additional_information": SAMPLE_ADDITIONAL,
                "oam_id": SAMPLE_OAM_ID,
                "property_license": SAMPLE_LICENSE,
            }
        )

    @pytest.fixture(autouse=True)
    def _offline_hash_leg(self, monkeypatch):
        """Keep the content-hash leg deterministic and offline by default."""
        monkeypatch.setattr(
            deadtrees_seam, "file_hash", lambda path: "hash-" + path.name.lower()
        )
        monkeypatch.setattr(deadtrees_seam, "file_hashes_on_platform", lambda hashes: {})

    def test_happy_path_builds_spec(self, tmp_path):
        (tmp_path / "tifs").mkdir()
        (tmp_path / "tifs" / "a.tif").write_bytes(b"x")
        gate = pd.DataFrame([self._gate_row("a.tif")])
        prep = oam_weekly.prepare_candidates(gate, set(), tmp_path, server_check=False)
        assert prep.candidates == 1
        assert len(prep.specs) == 1
        assert prep.rejected == 0
        assert prep.specs[0].filename == "a.tif"

    def test_ledger_excludes_filenames(self, tmp_path):
        (tmp_path / "tifs").mkdir()
        (tmp_path / "tifs" / "b.tif").write_bytes(b"x")
        gate = pd.DataFrame([self._gate_row("a.tif"), self._gate_row("b.tif")])
        prep = oam_weekly.prepare_candidates(gate, {"a.tif"}, tmp_path, server_check=False)
        assert prep.candidates == 1
        assert [spec.filename for spec in prep.specs] == ["b.tif"]

    def test_missing_tiff_rejected(self, tmp_path):
        gate = pd.DataFrame([self._gate_row("missing.tif")])
        prep = oam_weekly.prepare_candidates(gate, set(), tmp_path, server_check=False)
        assert prep.specs == []
        assert prep.rejected == 1

    def test_bad_kwargs_rejected(self, tmp_path):
        (tmp_path / "tifs").mkdir()
        (tmp_path / "tifs" / "a.tif").write_bytes(b"x")
        gate = pd.DataFrame([self._gate_row("a.tif")])
        gate.loc[0, "property_license"] = "Public Domain"
        prep = oam_weekly.prepare_candidates(gate, set(), tmp_path, server_check=False)
        assert prep.specs == []
        assert prep.rejected == 1

    def test_server_check_skips_existing(self, tmp_path, monkeypatch):
        (tmp_path / "tifs").mkdir()
        (tmp_path / "tifs" / "a.tif").write_bytes(b"x")
        monkeypatch.setattr(deadtrees_seam, "file_exists_on_platform", lambda name: True)
        gate = pd.DataFrame([self._gate_row("a.tif")])
        prep = oam_weekly.prepare_candidates(gate, set(), tmp_path, server_check=True)
        assert prep.candidates == 0
        assert prep.specs == []

    def test_server_check_failure_falls_back_for_remaining(self, tmp_path, monkeypatch):
        (tmp_path / "tifs").mkdir()
        for name in ("a.tif", "b.tif"):
            (tmp_path / "tifs" / name).write_bytes(b"x")
        calls = {"n": 0}

        def flaky(name):
            calls["n"] += 1
            if calls["n"] == 1:
                return False
            raise ConnectionError("offline")

        monkeypatch.setattr(deadtrees_seam, "file_exists_on_platform", flaky)
        gate = pd.DataFrame([self._gate_row("a.tif"), self._gate_row("b.tif")])
        prep = oam_weekly.prepare_candidates(gate, set(), tmp_path, server_check=True)
        # a.tif was confirmed absent before the failure; b.tif falls back to unchecked
        assert [spec.filename for spec in prep.specs] == ["a.tif", "b.tif"]

    def test_platform_hash_match_skips(self, tmp_path, monkeypatch):
        (tmp_path / "tifs").mkdir()
        (tmp_path / "tifs" / "a.tif").write_bytes(b"x")
        monkeypatch.setattr(
            deadtrees_seam, "file_hashes_on_platform", lambda hashes: {"hash-a.tif": 42}
        )
        gate = pd.DataFrame([self._gate_row("a.tif")])
        prep = oam_weekly.prepare_candidates(gate, set(), tmp_path, server_check=False)
        assert prep.candidates == 0
        assert prep.specs == []

    def test_intra_batch_duplicate_skipped(self, tmp_path, monkeypatch):
        (tmp_path / "tifs").mkdir()
        (tmp_path / "tifs" / "a.tif").write_bytes(b"same")
        (tmp_path / "tifs" / "b.tif").write_bytes(b"same")
        monkeypatch.setattr(deadtrees_seam, "file_hash", lambda path: "same-hash")
        gate = pd.DataFrame([self._gate_row("a.tif"), self._gate_row("b.tif")])
        prep = oam_weekly.prepare_candidates(gate, set(), tmp_path, server_check=False)
        assert [spec.filename for spec in prep.specs] == ["a.tif"]

    def test_hash_leg_failure_degrades_to_keep_all(self, tmp_path, monkeypatch):
        (tmp_path / "tifs").mkdir()
        (tmp_path / "tifs" / "a.tif").write_bytes(b"x")

        def broken(hashes):
            raise ConnectionError("supabase down")

        monkeypatch.setattr(deadtrees_seam, "file_hashes_on_platform", broken)
        gate = pd.DataFrame([self._gate_row("a.tif")])
        prep = oam_weekly.prepare_candidates(gate, set(), tmp_path, server_check=False)
        assert prep.candidates == 1
        assert len(prep.specs) == 1

    def test_hash_leg_skips_files_without_tiff(self, tmp_path, monkeypatch):
        gate = pd.DataFrame([self._gate_row("missing.tif")])
        hashes_seen = {}

        def capture(hashes):
            hashes_seen["called_with"] = hashes
            return {}

        monkeypatch.setattr(deadtrees_seam, "file_hashes_on_platform", capture)
        prep = oam_weekly.prepare_candidates(gate, set(), tmp_path, server_check=False)
        assert prep.rejected == 1
        assert hashes_seen["called_with"] == []


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


class TestWriteStatus:
    def test_status_file_contains_counts_and_exit(self, tmp_path):
        status = tmp_path / "last_run.txt"
        counts = oam_weekly.RunCounts(candidates=5, uploaded=4, failed=1)
        oam_weekly.write_status(status, tmp_path / "run", False, counts, "failed")
        text = status.read_text(encoding="utf-8")
        assert "candidates: 5" in text
        assert "uploaded: 4" in text
        assert "failed: 1" in text
        assert "exit: failed" in text
        assert "dry_run: False" in text

    def test_status_file_records_uploaded_after(self, tmp_path):
        status = tmp_path / "last_run.txt"
        counts = oam_weekly.RunCounts()
        oam_weekly.write_status(status, tmp_path / "run", True, counts, "dry-run", "2026-09-21", "2026-09-13")
        text = status.read_text(encoding="utf-8")
        assert "uploaded_after: 2026-09-21" in text
        assert "scrape_uploaded_at: 2026-09-13" in text

    def test_status_file_records_uploaded_before(self, tmp_path):
        status = tmp_path / "last_run.txt"
        counts = oam_weekly.RunCounts()
        oam_weekly.write_status(
            status, tmp_path / "run", True, counts, "dry-run", "2026-09-01", "2026-09-06", "2026-09-07"
        )
        text = status.read_text(encoding="utf-8")
        assert "uploaded_before: 2026-09-07" in text

    def test_uploaded_before_defaults_to_empty(self, tmp_path):
        status = tmp_path / "last_run.txt"
        counts = oam_weekly.RunCounts()
        oam_weekly.write_status(status, tmp_path / "run", True, counts, "dry-run")
        assert "uploaded_before: " in status.read_text(encoding="utf-8")


class TestResolveUploadedAfter:
    def test_explicit_flag_wins(self, tmp_path):
        status = tmp_path / "last_run.txt"
        status.write_text("timestamp: 2026-08-01T10:00:00\n", encoding="utf-8")
        assert oam_weekly.resolve_uploaded_after("2026-09-01", status) == "2026-09-01"

    def test_derives_from_scrape_uploaded_at(self, tmp_path):
        status = tmp_path / "last_run.txt"
        status.write_text(
            "timestamp: 2026-08-01T10:00:00\nscrape_uploaded_at: 2026-09-13\n",
            encoding="utf-8",
        )
        assert oam_weekly.resolve_uploaded_after(None, status) == "2026-09-13"

    def test_falls_back_to_timestamp_without_scrape_uploaded_at(self, tmp_path):
        status = tmp_path / "last_run.txt"
        status.write_text(
            "timestamp: 2026-08-01T10:00:00\nscrape_uploaded_at: \n", encoding="utf-8"
        )
        assert oam_weekly.resolve_uploaded_after(None, status) == "2026-08-01"

    def test_garbage_scrape_value_falls_back_to_timestamp(self, tmp_path):
        status = tmp_path / "last_run.txt"
        status.write_text(
            "timestamp: 2026-08-01T10:00:00\nscrape_uploaded_at: garbage\n", encoding="utf-8"
        )
        assert oam_weekly.resolve_uploaded_after(None, status) == "2026-08-01"

    def test_missing_status_file_yields_none(self, tmp_path):
        assert oam_weekly.resolve_uploaded_after(None, tmp_path / "missing.txt") is None


class TestScrapeUploadedAt:
    def test_computed_from_filtered_csv(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True)
        _write_csv(
            run_dir / "metadata" / "filtered.csv",
            ["uuid", "uploaded_at"],
            [["1", "2026-09-06T13:39:47.930000+00:00"], ["2", "2026-09-13T21:34:19.226000+00:00"]],
        )
        assert oam_weekly.scrape_uploaded_at(run_dir) == "2026-09-13"

    def test_missing_csv_yields_none(self, tmp_path):
        assert oam_weekly.scrape_uploaded_at(tmp_path) is None

    def test_unparseable_dates_yield_none(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True)
        _write_csv(
            run_dir / "metadata" / "filtered.csv",
            ["uuid", "uploaded_at"],
            [["1", "not-a-date"], ["2", ""]],
        )
        assert oam_weekly.scrape_uploaded_at(run_dir) is None

    def test_future_dated_upload_clamped_to_today(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True)
        _write_csv(
            run_dir / "metadata" / "filtered.csv",
            ["uuid", "uploaded_at"],
            [["1", "2031-05-01T00:00:00+00:00"]],
        )
        today = oam_weekly.datetime.now(oam_weekly.timezone.utc).date().isoformat()
        assert oam_weekly.scrape_uploaded_at(run_dir) == today


class TestAdvanceScrapeUploadedAt:
    def test_older_scrape_does_not_regress(self):
        value, warning = oam_weekly.advance_scrape_uploaded_at(
            "2026-09-20", "2026-09-06", "2026-09-06"
        )
        assert value == "2026-09-20"
        assert warning is None

    def test_missing_scrape_preserves_chain(self):
        value, warning = oam_weekly.advance_scrape_uploaded_at("2026-09-20", None, "2026-09-20")
        assert value == "2026-09-20"
        assert warning is None

    def test_newer_scrape_advances_chain(self):
        value, warning = oam_weekly.advance_scrape_uploaded_at(
            "2026-09-13", "2026-09-20", "2026-09-13"
        )
        assert value == "2026-09-20"
        assert warning is None

    def test_first_run_adopts_clamped_scrape_value(self):
        value, warning = oam_weekly.advance_scrape_uploaded_at(None, "2026-09-13", "2026-09-06")
        assert value == "2026-09-13"
        future = oam_weekly.advance_scrape_uploaded_at(None, "2031-05-01", "2026-09-06")
        assert future[0] == oam_weekly.datetime.now(oam_weekly.timezone.utc).date().isoformat()

    def test_uncovered_window_keeps_chain_and_warns(self):
        value, warning = oam_weekly.advance_scrape_uploaded_at(
            "2026-09-06", "2026-09-19", "2026-09-12"
        )
        assert value == "2026-09-06"
        assert warning is not None
        assert "2026-09-12" in warning
        assert "2026-09-06" in warning

    def test_covered_window_advances_even_from_older_after(self):
        value, warning = oam_weekly.advance_scrape_uploaded_at(
            "2026-09-06", "2026-09-19", "2026-09-01"
        )
        assert value == "2026-09-19"
        assert warning is None

    def test_future_stored_value_clamped(self):
        value, warning = oam_weekly.advance_scrape_uploaded_at("2031-05-01", "2026-09-06", "2026-09-06")
        assert value == oam_weekly.datetime.now(oam_weekly.timezone.utc).date().isoformat()

    def test_unparseable_stored_treated_as_first_run(self):
        value, warning = oam_weekly.advance_scrape_uploaded_at("garbage", "2026-09-13", None)
        assert value == "2026-09-13"
        assert warning is None


class TestReadScrapeUploadedAt:
    def test_reads_stored_value(self, tmp_path):
        status = tmp_path / "last_run.txt"
        status.write_text(
            "timestamp: 2026-09-06T10:00:00\nscrape_uploaded_at: 2026-09-13\n", encoding="utf-8"
        )
        assert oam_weekly.read_scrape_uploaded_at(status) == "2026-09-13"

    def test_missing_or_unparseable_yields_none(self, tmp_path):
        assert oam_weekly.read_scrape_uploaded_at(tmp_path / "no.txt") is None
        status = tmp_path / "last_run.txt"
        status.write_text("scrape_uploaded_at: garbage\n", encoding="utf-8")
        assert oam_weekly.read_scrape_uploaded_at(status) is None


class TestRunPipelineCommand:
    def test_without_uploaded_after(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run(command, cwd=None, check=None):
            captured["command"] = command
            return 0

        monkeypatch.setattr(oam_weekly.subprocess, "run", fake_run)
        oam_weekly.run_pipeline(tmp_path, tmp_path / "run")
        assert "--uploaded-after-date" not in captured["command"]

    def test_with_uploaded_after(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run(command, cwd=None, check=None):
            captured["command"] = command
            return 0

        monkeypatch.setattr(oam_weekly.subprocess, "run", fake_run)
        oam_weekly.run_pipeline(tmp_path, tmp_path / "run", "2026-09-14")
        assert captured["command"][-2:] == ["--uploaded-after-date", "2026-09-14"]

    def test_weekly_default_has_no_before_bound(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run(command, cwd=None, check=None):
            captured["command"] = command
            return 0

        monkeypatch.setattr(oam_weekly.subprocess, "run", fake_run)
        oam_weekly.run_pipeline(tmp_path, tmp_path / "run", "2026-09-14")
        assert "--uploaded-before-date" not in captured["command"]

    def test_with_uploaded_before(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run(command, cwd=None, check=None):
            captured["command"] = command
            return 0

        monkeypatch.setattr(oam_weekly.subprocess, "run", fake_run)
        oam_weekly.run_pipeline(tmp_path, tmp_path / "run", "2026-09-01", "2026-09-07")
        assert captured["command"][-4:] == [
            "--uploaded-after-date",
            "2026-09-01",
            "--uploaded-before-date",
            "2026-09-07",
        ]


class TestMainUploadedAfterGuard:
    def test_first_run_without_uploaded_after_fails_before_pipeline(self, tmp_path, monkeypatch):
        def no_pipeline(*args, **kwargs):
            raise AssertionError("pipeline must not run without an uploaded-after date")

        monkeypatch.setattr(oam_weekly, "run_pipeline", no_pipeline)
        status = tmp_path / "last_run.txt"
        result = oam_weekly.main(
            [
                "--output-dir",
                str(tmp_path / "run"),
                "--uploaded-csv",
                str(tmp_path / "ledger.csv"),
                "--status-file",
                str(status),
            ]
        )
        assert result == 1
        text = status.read_text(encoding="utf-8")
        assert "exit: failed" in text
        assert "uploaded_after" in text


class TestMainUploadedBeforeGuard:
    def _argv(self, tmp_path, extra):
        return [
            "--output-dir",
            str(tmp_path / "run"),
            "--uploaded-csv",
            str(tmp_path / "ledger.csv"),
            "--status-file",
            str(tmp_path / "last_run.txt"),
            *extra,
        ]

    def test_before_without_resolvable_after_fails_before_pipeline(self, tmp_path, monkeypatch):
        def no_pipeline(*args, **kwargs):
            raise AssertionError("pipeline must not run for a before-bound without an after date")

        monkeypatch.setattr(oam_weekly, "run_pipeline", no_pipeline)
        result = oam_weekly.main(self._argv(tmp_path, ["--uploaded-before", "2026-09-07"]))
        assert result == 1
        assert "exit: failed" in (tmp_path / "last_run.txt").read_text(encoding="utf-8")

    def test_before_leaving_no_common_period_fails(self, tmp_path, monkeypatch):
        def no_pipeline(*args, **kwargs):
            raise AssertionError("pipeline must not run when the window is inverted")

        monkeypatch.setattr(oam_weekly, "run_pipeline", no_pipeline)
        result = oam_weekly.main(
            self._argv(tmp_path, ["--uploaded-after", "2026-09-14", "--uploaded-before", "2026-09-07"])
        )
        assert result == 1

    def test_valid_single_day_window_runs_pipeline(self, tmp_path, monkeypatch):
        captured = {}

        def fake_run_pipeline(repo_root, run_dir, uploaded_after, uploaded_before=None):
            captured["after"] = uploaded_after
            captured["before"] = uploaded_before

        monkeypatch.setattr(oam_weekly, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(oam_weekly, "load_gate_candidates", lambda run_dir: pd.DataFrame())
        monkeypatch.setattr(
            oam_weekly,
            "prepare_candidates",
            lambda gate, ledger, run_dir, server_check: oam_weekly.Preparation(0, [], 0),
        )
        result = oam_weekly.main(
            self._argv(tmp_path, ["--uploaded-after", "2026-09-07", "--uploaded-before", "2026-09-07"])
        )
        assert result == 0
        assert captured == {"after": "2026-09-07", "before": "2026-09-07"}

    def test_adhoc_window_ahead_of_chain_keeps_chain_and_warns(self, tmp_path, monkeypatch, capsys):
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True)
        _write_csv(
            run_dir / "metadata" / "filtered.csv",
            ["uuid", "uploaded_at"],
            [["1", "2026-09-19T10:00:00+00:00"]],
        )
        status = tmp_path / "last_run.txt"
        status.write_text(
            "timestamp: 2026-09-06T10:00:00\nscrape_uploaded_at: 2026-09-06\n", encoding="utf-8"
        )
        monkeypatch.setattr(oam_weekly, "run_pipeline", lambda *a, **k: None)
        monkeypatch.setattr(oam_weekly, "load_gate_candidates", lambda run_dir: pd.DataFrame())
        monkeypatch.setattr(
            oam_weekly,
            "prepare_candidates",
            lambda gate, ledger, run_dir, server_check: oam_weekly.Preparation(0, [], 0),
        )
        result = oam_weekly.main(
            self._argv(tmp_path, ["--uploaded-after", "2026-09-12", "--uploaded-before", "2026-09-20"])
        )
        assert result == 0
        assert "scrape_uploaded_at: 2026-09-06" in status.read_text(encoding="utf-8")
        assert "were not covered" in capsys.readouterr().out

    def test_dry_run_does_not_advance_chain(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True)
        _write_csv(
            run_dir / "metadata" / "filtered.csv",
            ["uuid", "uploaded_at"],
            [["1", "2026-09-19T10:00:00+00:00"]],
        )
        (run_dir / "metadata" / "phenology_report.csv").write_text("filename\n", encoding="utf-8")
        status = tmp_path / "last_run.txt"
        status.write_text(
            "timestamp: 2026-09-06T10:00:00\nscrape_uploaded_at: 2026-09-06\n", encoding="utf-8"
        )
        monkeypatch.setattr(oam_weekly, "load_gate_candidates", lambda run_dir: pd.DataFrame())
        monkeypatch.setattr(
            oam_weekly,
            "prepare_candidates",
            lambda gate, ledger, run_dir, server_check: oam_weekly.Preparation(0, [], 0),
        )
        result = oam_weekly.main(self._argv(tmp_path, ["--dry-run", "--uploaded-after", "2026-09-12"]))
        assert result == 0
        assert "scrape_uploaded_at: 2026-09-06" in status.read_text(encoding="utf-8")

    def test_reused_window_dir_with_different_window_aborts(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True)
        (run_dir / "metadata" / "filtered.csv").write_text("uuid\n", encoding="utf-8")
        manifest = {
            "config": {
                "uploaded_after_date": "2026-09-01",
                "uploaded_before_date": "2026-09-07",
            }
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        def no_pipeline(*args, **kwargs):
            raise AssertionError("pipeline must not run into a dir holding another window")

        monkeypatch.setattr(oam_weekly, "run_pipeline", no_pipeline)
        result = oam_weekly.main(
            self._argv(tmp_path, ["--uploaded-after", "2026-09-08", "--uploaded-before", "2026-09-14"])
        )
        assert result == 1

    def test_matching_window_dir_resumes(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True)
        (run_dir / "metadata" / "filtered.csv").write_text("uuid\n", encoding="utf-8")
        manifest = {
            "config": {
                "uploaded_after_date": "2026-09-06",
                "uploaded_before_date": "None",
            }
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        captured = {}

        def fake_run_pipeline(repo_root, run_dir, uploaded_after, uploaded_before=None):
            captured["after"] = uploaded_after
            captured["before"] = uploaded_before

        monkeypatch.setattr(oam_weekly, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(oam_weekly, "load_gate_candidates", lambda run_dir: pd.DataFrame())
        monkeypatch.setattr(
            oam_weekly,
            "prepare_candidates",
            lambda gate, ledger, run_dir, server_check: oam_weekly.Preparation(0, [], 0),
        )
        result = oam_weekly.main(self._argv(tmp_path, ["--uploaded-after", "2026-09-06"]))
        assert result == 0
        assert captured == {"after": "2026-09-06", "before": None}

    def test_weekly_default_main_passes_no_before_bound(self, tmp_path, monkeypatch):
        status = tmp_path / "last_run.txt"
        status.write_text(
            "timestamp: 2026-09-06T10:00:00\nscrape_uploaded_at: 2026-09-13\n", encoding="utf-8"
        )
        captured = {}

        def fake_run_pipeline(repo_root, run_dir, uploaded_after, uploaded_before=None):
            captured["after"] = uploaded_after
            captured["before"] = uploaded_before

        monkeypatch.setattr(oam_weekly, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(oam_weekly, "load_gate_candidates", lambda run_dir: pd.DataFrame())
        monkeypatch.setattr(
            oam_weekly,
            "prepare_candidates",
            lambda gate, ledger, run_dir, server_check: oam_weekly.Preparation(0, [], 0),
        )
        result = oam_weekly.main(self._argv(tmp_path, []))
        assert result == 0
        assert captured == {"after": "2026-09-13", "before": None}

    def test_valid_window_runs_pipeline_and_records_status(self, tmp_path, monkeypatch):
        captured = {}

        def fake_run_pipeline(repo_root, run_dir, uploaded_after, uploaded_before=None):
            captured["after"] = uploaded_after
            captured["before"] = uploaded_before

        monkeypatch.setattr(oam_weekly, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(oam_weekly, "load_gate_candidates", lambda run_dir: pd.DataFrame())
        monkeypatch.setattr(
            oam_weekly,
            "prepare_candidates",
            lambda gate, ledger, run_dir, server_check: oam_weekly.Preparation(0, [], 0),
        )
        result = oam_weekly.main(
            self._argv(tmp_path, ["--uploaded-after", "2026-09-01", "--uploaded-before", "2026-09-07"])
        )
        assert result == 0
        assert captured == {"after": "2026-09-01", "before": "2026-09-07"}
        status_text = (tmp_path / "last_run.txt").read_text(encoding="utf-8")
        assert "uploaded_before: 2026-09-07" in status_text
        assert "exit: success" in status_text


class TestMainRecoveryAndManifestChecks:
    def _argv(self, tmp_path, extra):
        return [
            "--output-dir",
            str(tmp_path / "run"),
            "--uploaded-csv",
            str(tmp_path / "ledger.csv"),
            "--status-file",
            str(tmp_path / "last_run.txt"),
            *extra,
        ]

    def _seed_dir(self, tmp_path, manifest_config=None, scrape="2026-09-19T10:00:00+00:00"):
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
        _write_csv(
            run_dir / "metadata" / "filtered.csv", ["uuid", "uploaded_at"], [["1", scrape]]
        )
        if manifest_config is not None:
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"config": manifest_config}), encoding="utf-8"
            )
        return run_dir

    def _seed_chain(self, tmp_path, value="2026-09-06"):
        status = tmp_path / "last_run.txt"
        status.write_text(
            f"timestamp: 2026-09-06T10:00:00\nscrape_uploaded_at: {value}\n", encoding="utf-8"
        )
        return status

    def test_dry_run_with_explicit_flags_into_mismatched_dir_aborts(self, tmp_path, monkeypatch, capsys):
        self._seed_dir(tmp_path, {"uploaded_after_date": "2026-09-01", "uploaded_before_date": "None"})
        (tmp_path / "run" / "metadata" / "phenology_report.csv").write_text("filename\n", encoding="utf-8")

        def no_pipeline(*args, **kwargs):
            raise AssertionError("pipeline must not be touched by a dry run")

        monkeypatch.setattr(oam_weekly, "run_pipeline", no_pipeline)
        result = oam_weekly.main(self._argv(tmp_path, ["--dry-run", "--uploaded-after", "2026-09-12"]))
        assert result == 1
        assert "already holds a different window" in capsys.readouterr().out

    def test_plain_dry_run_into_mismatched_dir_proceeds(self, tmp_path, monkeypatch):
        self._seed_dir(tmp_path, {"uploaded_after_date": "2026-09-01", "uploaded_before_date": "None"})
        (tmp_path / "run" / "metadata" / "phenology_report.csv").write_text("filename\n", encoding="utf-8")
        status = self._seed_chain(tmp_path)
        monkeypatch.setattr(oam_weekly, "load_gate_candidates", lambda run_dir: pd.DataFrame())
        monkeypatch.setattr(
            oam_weekly,
            "prepare_candidates",
            lambda gate, ledger, run_dir, server_check: oam_weekly.Preparation(0, [], 0),
        )
        result = oam_weekly.main(self._argv(tmp_path, ["--dry-run"]))
        assert result == 0
        assert "scrape_uploaded_at: 2026-09-06" in status.read_text(encoding="utf-8")

    def test_fail_open_warns_when_manifest_missing(self, tmp_path, monkeypatch, capsys):
        self._seed_dir(tmp_path, manifest_config=None)
        self._seed_chain(tmp_path)
        monkeypatch.setattr(oam_weekly, "run_pipeline", lambda *a, **k: None)
        monkeypatch.setattr(oam_weekly, "load_gate_candidates", lambda run_dir: pd.DataFrame())
        monkeypatch.setattr(
            oam_weekly,
            "prepare_candidates",
            lambda gate, ledger, run_dir, server_check: oam_weekly.Preparation(0, [], 0),
        )
        result = oam_weekly.main(self._argv(tmp_path, ["--uploaded-after", "2026-09-06"]))
        assert result == 0
        assert "cannot be verified" in capsys.readouterr().out

    def test_failed_window_run_hint_includes_before_flag(self, tmp_path, monkeypatch, capsys):
        self._seed_dir(tmp_path, {"uploaded_after_date": "2026-09-06", "uploaded_before_date": "2026-09-20"})
        self._seed_chain(tmp_path)
        spec = oam_weekly.UploadSpec("a.tif", tmp_path / "a.tif", {"authors": ["x"]})
        monkeypatch.setattr(oam_weekly, "run_pipeline", lambda *a, **k: None)
        monkeypatch.setattr(oam_weekly, "load_gate_candidates", lambda run_dir: pd.DataFrame())
        monkeypatch.setattr(
            oam_weekly,
            "prepare_candidates",
            lambda gate, ledger, run_dir, server_check: oam_weekly.Preparation(1, [spec], 0),
        )

        def failing_upload(tif_path, **kwargs):
            raise RuntimeError("platform down")

        monkeypatch.setattr(oam_weekly.deadtrees_seam, "upload_and_process", failing_upload)
        result = oam_weekly.main(
            self._argv(tmp_path, ["--uploaded-after", "2026-09-06", "--uploaded-before", "2026-09-20"])
        )
        assert result == 1
        out = capsys.readouterr().out
        assert "--uploaded-after 2026-09-06 --uploaded-before 2026-09-20" in out
        assert "1 upload(s) failed" in out

    def test_aborted_run_after_scrape_prints_recovery_hint(self, tmp_path, monkeypatch, capsys):
        self._seed_dir(tmp_path, {"uploaded_after_date": "2026-09-06", "uploaded_before_date": "None"})
        self._seed_chain(tmp_path)

        def broken_gate(run_dir):
            raise RuntimeError("VLM report unreadable")

        monkeypatch.setattr(oam_weekly, "run_pipeline", lambda *a, **k: None)
        monkeypatch.setattr(oam_weekly, "load_gate_candidates", broken_gate)
        result = oam_weekly.main(self._argv(tmp_path, ["--uploaded-after", "2026-09-06"]))
        assert result == 1
        out = capsys.readouterr().out
        assert "the run aborted" in out
        assert "--uploaded-after 2026-09-06" in out


class TestParseArgs:
    def test_env_var_used_as_default(self, monkeypatch):
        monkeypatch.setenv("OAM_UPLOADED_CSV", r"H:\some\ledger.csv")
        args = oam_weekly.parse_args([])
        assert str(args.uploaded_csv) == r"H:\some\ledger.csv"

    def test_flag_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OAM_UPLOADED_CSV", r"H:\some\ledger.csv")
        args = oam_weekly.parse_args(["--uploaded-csv", "other.csv"])
        assert args.uploaded_csv == Path("other.csv")

    def test_fails_fast_without_flag_or_env(self, monkeypatch):
        monkeypatch.delenv("OAM_UPLOADED_CSV", raising=False)
        with pytest.raises(SystemExit):
            oam_weekly.parse_args([])

    def test_uploaded_after_defaults_to_none(self):
        args = oam_weekly.parse_args(["--uploaded-csv", "ledger.csv"])
        assert args.uploaded_after is None

    def test_uploaded_before_defaults_to_none(self):
        args = oam_weekly.parse_args(["--uploaded-csv", "ledger.csv"])
        assert args.uploaded_before is None

    def test_uploaded_before_parses_when_given(self):
        args = oam_weekly.parse_args(["--uploaded-csv", "ledger.csv", "--uploaded-before", "2026-09-07"])
        assert args.uploaded_before == "2026-09-07"
