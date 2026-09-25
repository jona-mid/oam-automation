import csv
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

import aerialmodel
import aerialmodel_weekly as weekly


COG_URL = (
    "https://aerialmodel.s3.us-east-2.amazonaws.com/gfciprianihotmailcom/"
    "5497f16a-cd61-49e0-8034-0743a57ec5d2/odm_orthophoto/odm_orthophoto.pmtiles?X-Amz-Expires=3600"
)
PAGE = (
    '<script>viewer.load("/API/File?/pointclouds/AWS/anthony178outlookfr/'
    'ae23a11a-73c0-44d3-9187-b91dd41f8d5e/entwine_pointcloud/ept.json")</script>'
)


def stats(start="28/04/2026 at 13:20:12", gsd=3.57):
    return {
        "processing_statistics": {"date": "29/04/2026 at 00:41:07", "start_date": start},
        "odm_processing_statistics": {"average_gsd": gsd},
    }


class TestParsing:
    def test_storage_path_from_cog_url(self):
        assert aerialmodel.storage_path_from_cog_url(COG_URL) == (
            "AWS/gfciprianihotmailcom/5497f16a-cd61-49e0-8034-0743a57ec5d2"
        )

    def test_storage_path_from_cog_url_without_match(self):
        assert aerialmodel.storage_path_from_cog_url("https://example.com/x.pmtiles") is None

    def test_storage_path_from_page(self):
        assert aerialmodel.storage_path_from_page(PAGE) == (
            "AWS/anthony178outlookfr/ae23a11a-73c0-44d3-9187-b91dd41f8d5e"
        )
        assert aerialmodel.storage_path_from_page("<html>no viewer</html>") is None

    def test_parse_stats_reads_capture_start_not_processing_date(self):
        assert aerialmodel.parse_stats(stats()) == (date(2026, 4, 28), 3.57)

    def test_parse_stats_day_first(self):
        capture, _ = aerialmodel.parse_stats(stats(start="03/06/2026 at 13:56:03"))
        assert capture == date(2026, 6, 3)

    def test_parse_stats_rejects_1970_from_missing_exif(self):
        assert aerialmodel.parse_stats(stats(start="01/01/1970 at 00:00:00"))[0] is None

    def test_parse_stats_missing_fields(self):
        assert aerialmodel.parse_stats(None) == (None, None)
        assert aerialmodel.parse_stats({}) == (None, None)
        assert aerialmodel.parse_stats(stats(gsd="n/a"))[1] is None
        assert aerialmodel.parse_stats(stats(gsd=0))[1] is None

    def test_orthophoto_url(self):
        assert aerialmodel.orthophoto_url("AWS/u/id") == (
            "https://www.aerialmodel.com/API/File?/pointclouds/AWS/u/id/odm_orthophoto/odm_orthophoto.tif"
        )


class FakeResponse:
    def __init__(self, status=200, chunks=(b"data",), url="https://x/file.tif", content_type="image/tiff", fail_after=None):
        self.status_code = status
        self.url = url
        self.headers = {"Content-Type": content_type}
        self._chunks = chunks
        self._fail_after = fail_after

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        for number, chunk in enumerate(self._chunks):
            if self._fail_after is not None and number >= self._fail_after:
                raise ConnectionError("connection reset")
            yield chunk

    def close(self):
        pass


class FakeSession:
    def __init__(self, response):
        self.response = response

    def get(self, url, **kwargs):
        return self.response


class TestDownload:
    def test_writes_file_via_part(self, tmp_path):
        target = tmp_path / "a.tif"
        aerialmodel.download(FakeSession(FakeResponse(chunks=(b"ab", b"cd"))), "u", target)
        assert target.read_bytes() == b"abcd"
        assert not (tmp_path / "a.tif.part").exists()

    def test_interrupted_download_leaves_no_final_file(self, tmp_path):
        target = tmp_path / "a.tif"
        with pytest.raises(ConnectionError):
            aerialmodel.download(FakeSession(FakeResponse(chunks=(b"ab", b"cd"), fail_after=1)), "u", target)
        assert not target.exists()

    def test_login_page_is_an_error(self, tmp_path):
        response = FakeResponse(url="https://www.aerialmodel.com/Identity/Account/Login", content_type="text/html")
        with pytest.raises(RuntimeError, match="not authorized"):
            aerialmodel.download(FakeSession(response), "u", tmp_path / "a.tif")


def item(project_id, model=True):
    return {"id": project_id, "name": f"p{project_id}", "lat": 48.0, "lng": 8.0, "modelUrl": f"/x/p{project_id}" if model else None}


class TestWindowAndChain:
    def test_select_window_bounds_and_pending(self):
        today = date(2026, 9, 25)
        items = [item(10), item(11, model=False), item(12), item(13), item(5)]
        pending = {5: today - timedelta(days=3), 3: today - timedelta(days=40)}
        selected, next_pending, max_id = weekly.select_window(items, 10, 12, pending, today)
        # 11 has no model yet -> pending; 5 got its model -> selected; 3 expired; 13 is past before_id
        assert [project["id"] for project in selected] == [5, 12]
        assert next_pending == {11: today}
        assert max_id == 12

    def test_pending_without_model_is_kept_until_expiry(self):
        today = date(2026, 9, 25)
        items = [item(5, model=False), item(20)]
        _, next_pending, _ = weekly.select_window(items, 10, None, {5: today - timedelta(days=10)}, today)
        assert next_pending == {5: today - timedelta(days=10)}

    def test_advance_chain(self):
        assert weekly.advance_chain(100, 100, 150) == (150, None)
        assert weekly.advance_chain(100, 50, 150) == (150, None)
        assert weekly.advance_chain(None, 39083, 39500) == (39500, None)
        assert weekly.advance_chain(100, 100, None) == (100, None)
        chain, warning = weekly.advance_chain(100, 120, 150)
        assert chain == 100 and "not covered" in warning


class TestSelectionAndKwargs:
    def test_eligible_rows_gsd_bounds_and_season(self):
        screened = pd.DataFrame(
            [
                {"filename": "a", "status": "ok", "pheno_season": "in_season", "gsd_cm": "3.1"},
                {"filename": "b", "status": "ok", "pheno_season": "in_season", "gsd_cm": "0.002"},
                {"filename": "c", "status": "ok", "pheno_season": "in_season", "gsd_cm": "10"},
                {"filename": "d", "status": "ok", "pheno_season": "out_of_season", "gsd_cm": "3"},
                {"filename": "e", "status": "no_capture_date", "pheno_season": "", "gsd_cm": ""},
            ]
        )
        assert weekly.eligible_rows(screened)["filename"].tolist() == ["a"]

    def test_build_kwargs(self):
        row = pd.Series({"project_id": "44316", "capture_date": "2026-09-19", "model_url": "/ca/on/x44316"})
        kwargs = weekly.build_aerialmodel_kwargs(row, date(2026, 9, 25))
        assert kwargs["license"] == "CC BY"
        assert kwargs["data_access"] == "private"
        assert kwargs["platform"] == "drone"
        assert kwargs["authors"] == ["aerialmodel.com"]
        assert (kwargs["acquisition_year"], kwargs["acquisition_month"], kwargs["acquisition_day"]) == (2026, 9, 19)
        assert "project 44316: https://www.aerialmodel.com/ca/on/x44316" in kwargs["additional_information"]
        assert "Captured September 19, 2026. Accessed September 25, 2026." in kwargs["additional_information"]

    def test_build_kwargs_fails_closed(self):
        with pytest.raises(ValueError):
            weekly.build_aerialmodel_kwargs(pd.Series({"project_id": "1", "capture_date": ""}), date(2026, 9, 25))


REPORT_FIELDS = ["filename", "modis_category", "tree_canopy_leaf_state", "review_status"]


class TestMain:
    """main() end to end with the network, downloads, JPEG conversion and VLM stubbed."""

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        items = [item(101), item(102), item(103, model=False)]
        monkeypatch.setattr(aerialmodel, "fetch_catalog", lambda session: items)
        monkeypatch.setattr(aerialmodel, "project_storage_path", lambda session, project: f"AWS/u/{project['id']}")
        monkeypatch.setattr(aerialmodel, "fetch_stats", lambda session, path: stats(start="01/07/2026 at 10:00:00"))
        monkeypatch.setattr(aerialmodel, "REQUEST_DELAY", 0)
        monkeypatch.setattr(weekly.phenology, "load_phenology_data", lambda: None)
        monkeypatch.setattr(weekly.phenology, "extract_phenology_for_bbox", lambda *args: (120.0, 280.0))

        def fake_download(session, rows, tifs):
            for _, row in rows.iterrows():
                (tifs / row["filename"]).write_bytes(b"tif")

        def fake_vlm(output, *args):
            with (output / "metadata" / "phenology_report.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(REPORT_FIELDS)
                writer.writerow(["aerialmodel_101.tif", "in_season", "leaf_on", "success"])
                writer.writerow(["aerialmodel_102.tif", "in_season", "leaf_off", "success"])
            return {}

        monkeypatch.setattr(weekly, "download_tifs", fake_download)
        monkeypatch.setattr(weekly, "write_audit_inputs", lambda *args: None)
        monkeypatch.setattr(weekly.pipeline, "run", lambda *args, **kwargs: None)
        monkeypatch.setattr(weekly.pipeline, "run_vlm_stages", fake_vlm)
        monkeypatch.setattr(weekly.deadtrees_seam, "file_hash", lambda path: f"hash-{Path(path).name}")
        monkeypatch.setattr(weekly.deadtrees_seam, "file_hashes_on_platform", lambda hashes: {})
        uploads = []

        def fake_upload(path, **kwargs):
            uploads.append((Path(path).name, kwargs))
            return 900 + len(uploads)

        monkeypatch.setattr(weekly.deadtrees_seam, "upload_and_process", fake_upload)
        return tmp_path, uploads

    def argv(self, tmp_path, *extra):
        return [
            "--output-dir", str(tmp_path / "runs" / "run1"),
            "--uploaded-csv", str(tmp_path / "ledger.csv"),
            "--skip-server-check",
            *extra,
        ]

    def test_first_run_requires_after_id(self, env):
        tmp_path, uploads = env
        assert weekly.main(self.argv(tmp_path)) == 1
        assert uploads == []

    def test_successful_run_uploads_gate_passers_and_advances_chain(self, env):
        tmp_path, uploads = env
        assert weekly.main(self.argv(tmp_path, "--after-id", "100")) == 0
        assert [name for name, _ in uploads] == ["aerialmodel_101.tif"]
        assert uploads[0][1]["acquisition_month"] == 7
        status = (tmp_path / "runs" / "last_run.txt").read_text(encoding="utf-8")
        assert "last_project_id: 103" in status
        assert "exit: success" in status
        pending = (tmp_path / "runs" / "pending_ids.csv").read_text(encoding="utf-8")
        assert "103," in pending
        ledger = pd.read_csv(tmp_path / "ledger.csv")
        assert ledger["filename"].tolist() == ["aerialmodel_101.tif"]
        assert ledger["license"].tolist() == ["CC BY"]

    def test_rerun_skips_via_ledger(self, env):
        tmp_path, uploads = env
        weekly.main(self.argv(tmp_path, "--after-id", "100"))
        assert weekly.main(self.argv(tmp_path, "--after-id", "100")) == 0
        assert len(uploads) == 1

    def test_failed_upload_holds_chain(self, env, monkeypatch):
        tmp_path, _ = env

        def failing_upload(path, **kwargs):
            raise RuntimeError("platform down")

        monkeypatch.setattr(weekly.deadtrees_seam, "upload_and_process", failing_upload)
        assert weekly.main(self.argv(tmp_path, "--after-id", "100")) == 1
        status = (tmp_path / "runs" / "last_run.txt").read_text(encoding="utf-8")
        assert "last_project_id: 100" in status
        assert not (tmp_path / "runs" / "pending_ids.csv").exists()

    def test_screen_only_touches_no_status(self, env):
        tmp_path, uploads = env
        assert weekly.main(self.argv(tmp_path, "--after-id", "100", "--screen-only")) == 0
        assert uploads == []
        assert not (tmp_path / "runs" / "last_run.txt").exists()
        screened = pd.read_csv(tmp_path / "runs" / "run1" / "metadata" / "screened.csv")
        assert screened["project_id"].tolist() == [101, 102]
        assert set(screened["pheno_season"]) == {"in_season"}

    def test_different_window_in_same_dir_aborts(self, env):
        tmp_path, uploads = env
        weekly.main(self.argv(tmp_path, "--after-id", "100", "--screen-only"))
        assert weekly.main(self.argv(tmp_path, "--after-id", "50")) == 1
        assert uploads == []

    def test_dry_run_lists_without_uploading(self, env):
        tmp_path, uploads = env
        weekly.main(self.argv(tmp_path, "--after-id", "100", "--screen-only"))
        run_dir = tmp_path / "runs" / "run1"
        (run_dir / "tifs" / "aerialmodel_101.tif").write_bytes(b"tif")
        with (run_dir / "metadata" / "phenology_report.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([REPORT_FIELDS, ["aerialmodel_101.tif", "in_season", "leaf_on", "success"]])
        assert weekly.main(self.argv(tmp_path, "--after-id", "100", "--dry-run")) == 0
        assert uploads == []
        assert "exit: dry-run" in (tmp_path / "runs" / "last_run.txt").read_text(encoding="utf-8")
