"""Tests for oam_weekly: gate, upload kwargs and date checks, dedup, the chain, and main's failure handling."""

import csv
import inspect
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


SAMPLE_OAM_ID = "abc123"
SAMPLE_LICENSE = "CC-BY 4.0"
SAMPLE_ADDITIONAL = (
    "This orthophoto data is available through OpenAerialMap, provided by Contributors of "
    "Open Imagery Network. More information: https://api.openaerialmap.org/meta?_id=abc123 Accessed December 4, 2025."
)
TODAY = date(2026, 9, 21)


def _row(**overrides):
    base = {
        "filename": "abc.tif",
        "acquisition_date": "2026-09-11",
        "platform": "drone",
        "licence": "CC BY",
        "authors": "Contributors of Open Imagery Network",
        "additional_information": SAMPLE_ADDITIONAL,
        "oam_id": SAMPLE_OAM_ID,
        "property_license": SAMPLE_LICENSE,
        "modis_category": "in_season",
        "tree_canopy_leaf_state": "leaf_on",
        "review_status": "success",
    }
    base.update(overrides)
    return pd.Series(base)


def test_gate_keeps_only_reviewed_in_season_leaf_on(tmp_path):
    (tmp_path / "metadata").mkdir()
    _write_csv(
        tmp_path / "metadata" / "phenology_report.csv",
        ["filename", "modis_category", "tree_canopy_leaf_state", "review_status", "platform"],
        [
            ["a.tif", "in_season", "leaf_on", "success", "drone"],
            ["b.tif", "in_season", "not_leaf_on", "success", "drone"],
            ["c.tif", "out_of_season", "leaf_on", "success", "drone"],
            ["d.tif", "in_season", "leaf_on", "unavailable", "drone"],
            ["e.tif", "in_season", "leaf_on", "success", "drone"],  # no jpeg metadata row
        ],
    )
    meta = [
        [name, "2026-09-11", "drone", "CC BY", "A", SAMPLE_ADDITIONAL, SAMPLE_OAM_ID, SAMPLE_LICENSE]
        for name in ("a.tif", "b.tif", "c.tif", "d.tif")
    ]
    _write_csv(
        tmp_path / "metadata" / "jpeg_metadata.csv",
        ["filename", "acquisition_date", "platform", "licence", "authors",
         "additional_information", "oam_id", "property_license"],
        meta,
    )
    gate = oam_weekly.load_gate_candidates(tmp_path)
    assert list(gate["filename"]) == ["a.tif"]
    # Both CSVs carry `platform`; the merged row must still build kwargs.
    assert oam_weekly.build_upload_kwargs(gate.iloc[0], TODAY)["platform"] == "drone"


def test_upload_kwargs_match_the_seam():
    kwargs = oam_weekly.build_upload_kwargs(_row(), TODAY)
    assert kwargs["authors"] == ["Contributors of Open Imagery Network"]
    assert kwargs["license"] == "CC BY"
    assert kwargs["data_access"] == "public"
    assert (kwargs["acquisition_year"], kwargs["acquisition_month"], kwargs["acquisition_day"]) == (2026, 9, 11)
    assert kwargs["additional_information"] == (
        "This orthophoto data is available through OpenAerialMap, provided by Contributors "
        "of Open Imagery Network. Licensed under CC-BY 4.0. More information about this "
        "dataset: https://api.openaerialmap.org/meta?_id=abc123 Accessed September 21, 2026."
    )
    sa = oam_weekly.build_upload_kwargs(_row(property_license="CC BY-SA 4.0"), TODAY)
    assert sa["license"] == "CC BY-SA" and "Licensed under CC BY-SA 4.0." in sa["additional_information"]
    # main spreads these kwargs into the seam, so the keys must match its parameters.
    assert {"file_path", *kwargs} == set(inspect.signature(deadtrees_seam.upload_and_process).parameters)


def test_bad_metadata_fails_closed():
    cases = [
        ({"property_license": "Public Domain"}, "unknown OAM license"),
        ({"property_license": None}, "property_license"),
        ({"oam_id": None}, "oam_id"),
        ({"platform": "balloon"}, "unknown platform"),
        ({"acquisition_date": None}, "acquisition_date"),
        ({"acquisition_date": "2026-07-08", "title": "Lowes  New Paltz May 16, 2022"},
         "title date 2022-05-16 contradicts capture date 2026-07-08"),
        ({"acquisition_date": "2026-01-24", "title": "WattleBay_260710"}, "title date 2026-07-10"),
        ({"acquisition_date": "2016-01-01", "acquisition_start": "2016-01-01T08:00:00.000Z"}, "placeholder"),
        ({"uploaded_at": "2026-08-01T10:00:00.000Z"}, "after the OAM upload"),
    ]
    for overrides, message in cases:
        with pytest.raises(ValueError, match=message):
            oam_weekly.build_upload_kwargs(_row(**overrides), TODAY)


def test_title_dates():
    assert oam_weekly.title_dates("highland rail trail june 30 2024") == [date(2024, 6, 30)]
    assert oam_weekly.title_dates("Flight on 3rd March 2024") == [date(2024, 3, 3)]
    assert oam_weekly.title_dates("Flight March 2024") == [date(2024, 3, 15)]
    assert set(oam_weekly.title_dates("AIT Golf Course - 7/11/2015")) == {date(2015, 7, 11), date(2015, 11, 7)}
    assert oam_weekly.title_dates("Lake Helen 9 9 2021") == [date(2021, 9, 9)]
    assert date(2025, 1, 22) in oam_weekly.title_dates("3_NSD_B1_Desp_220125")
    for title in ["53646_33408sal_pembuangan", "orthomosaic_123456", "JAM-HM-SEL-AW-52-1", "Plan 2030", "", None]:
        assert oam_weekly.title_dates(title) == []


def test_title_date_check_tolerances():
    passing = [
        ("Mason Road - 1/3/2022", date(2022, 1, 3), None),
        ("Survey 2026-08-20", date(2026, 9, 11), None),              # within tolerance
        ("Incendio Chingaza Febrero 2025", date(2025, 7, 21), None),  # event before the flight
        ("Min-yr-Awel October 2025", date(2024, 9, 30), date(2025, 5, 22)),  # title date after upload
    ]
    for title, capture, uploaded in passing:
        oam_weekly.check_title_date(title, capture, uploaded)
    with pytest.raises(ValueError):
        oam_weekly.check_title_date("Hurricane Melissa 2025-10-28", date(2025, 6, 1))
    assert not oam_weekly.is_placeholder_timestamp("2026-01-01T09:13:22.000Z")
    assert oam_weekly.is_placeholder_timestamp("2013-12-31T16:00:00.000Z")  # midnight east of Greenwich


def test_tiff_processing_date_check(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    def tif(name, tag):
        path = tmp_path / name
        with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=3, dtype="uint8",
                           crs="EPSG:4326", transform=from_origin(0, 1, 0.1, 0.1)) as dst:
            dst.write(np.zeros((3, 4, 4), dtype="uint8"))
            if tag:
                dst.update_tags(TIFFTAG_DATETIME=tag)
        return path

    with pytest.raises(ValueError, match="processed on 2025-02-27"):
        oam_weekly.build_upload_kwargs(_row(acquisition_date="2026-06-19"), TODAY, tif("a.tif", "2025:02:27 10:00:00"))
    for path in (tif("b.tif", "2026:10:02 10:00:00"), tif("c.tif", None)):
        assert oam_weekly.build_upload_kwargs(_row(), TODAY, path)["acquisition_month"] == 9


class TestPrepareCandidates:
    @pytest.fixture(autouse=True)
    def _offline(self, monkeypatch):
        monkeypatch.setattr(deadtrees_seam, "file_hash", lambda path: "hash-" + path.name.lower())
        monkeypatch.setattr(deadtrees_seam, "file_hashes_on_platform", lambda hashes: {})

    def _tifs(self, tmp_path, *names, content=None):
        (tmp_path / "tifs").mkdir(exist_ok=True)
        for name in names:
            (tmp_path / "tifs" / name).write_bytes(content or name.encode())

    def test_dedup_by_ledger_platform_name_and_content_hash(self, tmp_path, monkeypatch):
        self._tifs(tmp_path, "a.tif", "b.tif", "c.tif", "d.tif", "e.tif")
        monkeypatch.setattr(deadtrees_seam, "file_names_on_platform", lambda names: {"b.tif"})
        monkeypatch.setattr(deadtrees_seam, "file_hashes_on_platform", lambda hashes: {"hash-c.tif": 42})
        monkeypatch.setattr(deadtrees_seam, "file_hash",
                            lambda path: "same" if path.name in ("d.tif", "e.tif") else "hash-" + path.name)
        gate = pd.DataFrame([_row(filename=n) for n in ("a.tif", "B.TIF", "c.tif", "d.tif", "e.tif", "z.tif")])
        prep = oam_weekly.prepare_candidates(gate, {"z.tif"}, tmp_path, server_check=True)
        # a: new; b: name on platform; c: hash on platform; e: same content as d; z: in the ledger.
        assert [spec.filename for spec in prep.specs] == ["a.tif", "d.tif"]
        assert prep.rejected == 0

    def test_missing_tiff_and_bad_metadata_are_rejected(self, tmp_path):
        self._tifs(tmp_path, "a.tif")
        gate = pd.DataFrame([_row(filename="a.tif", property_license="Public Domain"), _row(filename="missing.tif")])
        prep = oam_weekly.prepare_candidates(gate, set(), tmp_path, server_check=False)
        assert prep.specs == [] and prep.rejected == 2

    def test_platform_checks_fail_open(self, tmp_path, monkeypatch):
        self._tifs(tmp_path, "a.tif", "b.tif")

        def offline(values):
            raise ConnectionError("offline")

        monkeypatch.setattr(deadtrees_seam, "file_names_on_platform", offline)
        monkeypatch.setattr(deadtrees_seam, "file_hashes_on_platform", offline)
        gate = pd.DataFrame([_row(filename="a.tif"), _row(filename="b.tif")])
        prep = oam_weekly.prepare_candidates(gate, set(), tmp_path, server_check=True)
        assert [spec.filename for spec in prep.specs] == ["a.tif", "b.tif"]


def test_resolve_uploaded_after(tmp_path):
    status = tmp_path / "last_run.txt"
    assert oam_weekly.resolve_uploaded_after(None, status) is None
    status.write_text("timestamp: 2026-08-01T10:00:00\nscrape_uploaded_at: 2026-09-13\n", encoding="utf-8")
    assert oam_weekly.resolve_uploaded_after(None, status) == "2026-09-13"
    assert oam_weekly.resolve_uploaded_after("2026-09-01", status) == "2026-09-01"
    status.write_text("timestamp: 2026-08-01T10:00:00\nscrape_uploaded_at: garbage\n", encoding="utf-8")
    assert oam_weekly.resolve_uploaded_after(None, status) == "2026-08-01"


def test_scrape_uploaded_at_is_clamped_to_today(tmp_path):
    (tmp_path / "metadata").mkdir()
    filtered = tmp_path / "metadata" / "filtered.csv"
    _write_csv(filtered, ["uuid", "uploaded_at"],
               [["1", "2026-09-06T13:39:47+00:00"], ["2", "2026-09-13T21:34:19+00:00"], ["3", "bad"]])
    assert oam_weekly.scrape_uploaded_at(tmp_path) == "2026-09-13"
    _write_csv(filtered, ["uuid", "uploaded_at"], [["1", "2031-05-01T00:00:00+00:00"]])
    assert oam_weekly.scrape_uploaded_at(tmp_path) == oam_weekly.datetime.now(oam_weekly.timezone.utc).date().isoformat()


def test_advance_scrape_uploaded_at():
    cases = [
        ("2026-09-20", "2026-09-06", "2026-09-06", "2026-09-20", False),  # never regresses
        ("2026-09-20", None, "2026-09-20", "2026-09-20", False),          # nothing scraped
        ("2026-09-13", "2026-09-20", "2026-09-13", "2026-09-20", False),  # advances
        (None, "2026-09-13", "2026-09-06", "2026-09-13", False),          # first run
        ("garbage", "2026-09-13", None, "2026-09-13", False),             # unreadable = first run
        ("2026-09-06", "2026-09-19", "2026-09-12", "2026-09-06", True),   # gap 09-06..09-12 not covered
        ("2026-09-06", "2026-09-19", "2026-09-01", "2026-09-19", False),  # window covers the chain point
    ]
    for stored, scraped, after, expected, warns in cases:
        value, warning = oam_weekly.advance_scrape_uploaded_at(stored, scraped, after)
        assert (value, warning is not None) == (expected, warns), (stored, scraped, after)


def test_ledger_path_comes_from_flag_or_env(monkeypatch):
    monkeypatch.setenv("OAM_UPLOADED_CSV", "env.csv")
    assert oam_weekly.parse_args([]).uploaded_csv == Path("env.csv")
    assert oam_weekly.parse_args(["--uploaded-csv", "flag.csv"]).uploaded_csv == Path("flag.csv")
    monkeypatch.delenv("OAM_UPLOADED_CSV")
    with pytest.raises(SystemExit):
        oam_weekly.parse_args([])


class TestMain:
    KWARGS = {"authors": ["x"], "license": "CC BY", "platform": "drone", "data_access": "public",
              "acquisition_year": 2026, "acquisition_month": 9, "acquisition_day": 1,
              "additional_information": "x", "citation_doi": None}

    def _argv(self, tmp_path, *extra):
        return ["--output-dir", str(tmp_path / "run"), "--uploaded-csv", str(tmp_path / "ledger.csv"),
                "--status-file", str(tmp_path / "last_run.txt"), *extra]

    def _seed(self, tmp_path, after="2026-09-06", before="None", chain="2026-09-06", scrape="2026-09-19T10:00:00+00:00"):
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
        _write_csv(run_dir / "metadata" / "filtered.csv", ["uuid", "uploaded_at"], [["1", scrape]])
        if after:
            (run_dir / "run_manifest.json").write_text(json.dumps(
                {"config": {"uploaded_after_date": after, "uploaded_before_date": before}}), encoding="utf-8")
        if chain:
            (tmp_path / "last_run.txt").write_text(
                f"timestamp: 2026-09-06T10:00:00\nscrape_uploaded_at: {chain}\n", encoding="utf-8")

    def _stub(self, monkeypatch, prep=oam_weekly.Preparation(0, [], 0), captured=None):
        def fake_run_pipeline(repo_root, run_dir, uploaded_after, uploaded_before=None):
            if captured is not None:
                captured.update(after=uploaded_after, before=uploaded_before)

        monkeypatch.setattr(oam_weekly, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(oam_weekly, "load_gate_candidates", lambda run_dir: pd.DataFrame())
        monkeypatch.setattr(oam_weekly, "prepare_candidates", lambda gate, ledger, run_dir, server_check: prep)

    def _status(self, tmp_path):
        return (tmp_path / "last_run.txt").read_text(encoding="utf-8")

    @pytest.mark.parametrize("extra, manifest", [
        ([], None),                                                           # first run, no --uploaded-after
        (["--uploaded-before", "2026-09-07"], None),                          # before-bound without after
        (["--uploaded-after", "2026-09-14", "--uploaded-before", "2026-09-07"], None),  # inverted window
        (["--uploaded-after", "2026-09-08", "--uploaded-before", "2026-09-14"], ("2026-09-01", "2026-09-07")),
    ])
    def test_invalid_windows_fail_before_the_pipeline(self, tmp_path, monkeypatch, extra, manifest):
        if manifest:
            self._seed(tmp_path, *manifest, chain=None)

        def no_pipeline(*args, **kwargs):
            raise AssertionError("pipeline must not run")

        monkeypatch.setattr(oam_weekly, "run_pipeline", no_pipeline)
        assert oam_weekly.main(self._argv(tmp_path, *extra)) == 1
        assert "exit: failed" in self._status(tmp_path)

    def test_window_run_passes_bounds_and_records_status(self, tmp_path, monkeypatch):
        captured = {}
        self._stub(monkeypatch, captured=captured)
        assert oam_weekly.main(self._argv(tmp_path, "--uploaded-after", "2026-09-01", "--uploaded-before", "2026-09-07")) == 0
        assert captured == {"after": "2026-09-01", "before": "2026-09-07"}
        assert "uploaded_before: 2026-09-07" in self._status(tmp_path)
        assert "exit: success" in self._status(tmp_path)

    def test_weekly_run_resumes_from_the_chain(self, tmp_path, monkeypatch):
        (tmp_path / "last_run.txt").write_text("scrape_uploaded_at: 2026-09-13\n", encoding="utf-8")
        captured = {}
        self._stub(monkeypatch, captured=captured)
        assert oam_weekly.main(self._argv(tmp_path)) == 0
        assert captured == {"after": "2026-09-13", "before": None}

    def test_dry_run_and_adhoc_window_ahead_keep_the_chain(self, tmp_path, monkeypatch, capsys):
        self._seed(tmp_path, after=None)
        self._stub(monkeypatch)
        assert oam_weekly.main(self._argv(tmp_path, "--uploaded-after", "2026-09-12", "--uploaded-before", "2026-09-20")) == 0
        assert "scrape_uploaded_at: 2026-09-06" in self._status(tmp_path)
        assert "were not covered" in capsys.readouterr().out
        (tmp_path / "run" / "run_manifest.json").unlink(missing_ok=True)
        (tmp_path / "run" / "metadata" / "phenology_report.csv").write_text("filename\n", encoding="utf-8")
        assert oam_weekly.main(self._argv(tmp_path, "--dry-run", "--uploaded-after", "2026-09-06")) == 0
        assert "scrape_uploaded_at: 2026-09-06" in self._status(tmp_path)

    def test_failed_upload_holds_the_chain(self, tmp_path, monkeypatch, capsys):
        self._seed(tmp_path)
        self._stub(monkeypatch, oam_weekly.Preparation(1, [oam_weekly.UploadSpec("a.tif", tmp_path / "a.tif", self.KWARGS)], 0))

        def failing(tif_path, **kwargs):
            raise RuntimeError("platform down")

        monkeypatch.setattr(deadtrees_seam, "upload_and_process", failing)
        monkeypatch.setattr(deadtrees_seam, "find_dataset", lambda name, created_after: None)
        assert oam_weekly.main(self._argv(tmp_path)) == 1
        assert "scrape_uploaded_at: 2026-09-06" in self._status(tmp_path)
        assert "failed: 1" in self._status(tmp_path)
        assert not (tmp_path / "ledger.csv").exists()
        assert "next scheduled run retries it" in capsys.readouterr().out

    def test_aborted_runs_hold_or_seed_the_chain(self, tmp_path, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("VLM review failed for 3 in-season image(s)")

        monkeypatch.setattr(oam_weekly, "run_pipeline", broken)
        self._seed(tmp_path)
        assert oam_weekly.main(self._argv(tmp_path)) == 1
        assert "scrape_uploaded_at: 2026-09-06" in self._status(tmp_path)
        (tmp_path / "last_run.txt").unlink()
        self._seed(tmp_path, after="2026-09-01", chain=None)
        assert oam_weekly.main(self._argv(tmp_path, "--uploaded-after", "2026-09-01")) == 1
        # A failed first run records the window start, so the next run retries the window.
        assert "scrape_uploaded_at: 2026-09-01" in self._status(tmp_path)

    def _timed_out_upload(self, tmp_path, monkeypatch, found):
        self._seed(tmp_path)
        self._stub(monkeypatch, oam_weekly.Preparation(1, [oam_weekly.UploadSpec("a.tif", tmp_path / "a.tif", self.KWARGS)], 0))

        def timed_out(tif_path, **kwargs):
            raise TimeoutError("The read operation timed out")

        monkeypatch.setattr(deadtrees_seam, "upload_and_process", timed_out)
        monkeypatch.setattr(deadtrees_seam, "find_dataset", lambda name, created_after: found)
        processed = []
        monkeypatch.setattr(deadtrees_seam, "start_processing", processed.append)
        return processed

    @pytest.mark.parametrize("queued, started", [(True, []), (False, [42])])
    def test_upload_error_after_the_file_landed_records_it(self, tmp_path, monkeypatch, capsys, queued, started):
        processed = self._timed_out_upload(tmp_path, monkeypatch, {"id": 42, "processing_queued": queued})
        assert oam_weekly.main(self._argv(tmp_path)) == 0
        assert processed == started
        assert oam_weekly.load_ledger_filenames(tmp_path / "ledger.csv") == {"a.tif"}
        assert "uploaded: 1" in self._status(tmp_path) and "failed: 0" in self._status(tmp_path)
        assert "on the platform as dataset 42" in capsys.readouterr().out

    def test_landed_upload_whose_processing_cannot_start_fails_the_run(self, tmp_path, monkeypatch, capsys):
        self._timed_out_upload(tmp_path, monkeypatch, {"id": 42, "processing_queued": False})

        def broken(dataset_id):
            raise RuntimeError("still offline")

        monkeypatch.setattr(deadtrees_seam, "start_processing", broken)
        assert oam_weekly.main(self._argv(tmp_path)) == 1
        assert oam_weekly.load_ledger_filenames(tmp_path / "ledger.csv") == {"a.tif"}
        assert "start it by hand for dataset 42" in capsys.readouterr().out

    def test_failed_recovery_lookup_counts_as_a_failed_upload(self, tmp_path, monkeypatch, capsys):
        self._timed_out_upload(tmp_path, monkeypatch, None)

        def offline(name, created_after):
            raise ConnectionError("offline")

        monkeypatch.setattr(deadtrees_seam, "find_dataset", offline)
        assert oam_weekly.main(self._argv(tmp_path)) == 1
        assert not (tmp_path / "ledger.csv").exists()
        assert "could not check whether the failed upload landed" in capsys.readouterr().out

    def test_max_uploads_uploads_a_batch_and_holds_the_chain(self, tmp_path, monkeypatch, capsys):
        self._seed(tmp_path)
        specs = [oam_weekly.UploadSpec(f"{n}.tif", tmp_path / f"{n}.tif", self.KWARGS) for n in "abc"]
        self._stub(monkeypatch, oam_weekly.Preparation(3, specs, 0))
        uploaded = []
        monkeypatch.setattr(deadtrees_seam, "upload_and_process", lambda tif_path, **kw: uploaded.append(tif_path.name) or 1)
        assert oam_weekly.main(self._argv(tmp_path, "--max-uploads", "2")) == 0
        assert uploaded == ["a.tif", "b.tif"]
        assert "exit: partial" in self._status(tmp_path)
        assert "scrape_uploaded_at: 2026-09-06" in self._status(tmp_path)
        assert "1 candidate(s) left" in capsys.readouterr().out

    def test_rejected_candidates_do_not_fail_the_run(self, tmp_path, monkeypatch):
        self._seed(tmp_path)
        self._stub(monkeypatch, oam_weekly.Preparation(1, [], 1))
        assert oam_weekly.main(self._argv(tmp_path)) == 0
        text = self._status(tmp_path)
        assert "rejected: 1" in text and "failed: 0" in text
        assert "scrape_uploaded_at: 2026-09-19" in text

    def test_pipeline_early_stop_is_an_empty_successful_run(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        (run_dir / "metadata").mkdir(parents=True)
        _write_csv(run_dir / "metadata" / "filtered.csv", ["uuid", "uploaded_at"], [])
        (run_dir / "run_manifest.json").write_text(json.dumps({
            "config": {"uploaded_after_date": "2026-09-06", "uploaded_before_date": "None"},
            "stopped_early": "no filter-passing images in the upload window",
        }), encoding="utf-8")
        (tmp_path / "last_run.txt").write_text("scrape_uploaded_at: 2026-09-06\n", encoding="utf-8")
        monkeypatch.setattr(oam_weekly, "run_pipeline", lambda *a, **k: None)
        monkeypatch.setattr(deadtrees_seam, "file_names_on_platform", lambda names: set())
        assert oam_weekly.main(self._argv(tmp_path)) == 0
        assert "candidates: 0" in self._status(tmp_path)
        assert "scrape_uploaded_at: 2026-09-06" in self._status(tmp_path)
        # A dry run can inspect such a dir too, although it has no VLM report.
        assert oam_weekly.main(self._argv(tmp_path, "--dry-run")) == 0
