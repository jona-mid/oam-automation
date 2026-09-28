"""Tests for download.py."""

import json

import download


def test_downloads_in_season_tifs_not_yet_fetched(tmp_path, monkeypatch):
    csv = tmp_path / "phenology.csv"
    csv.write_text(
        "uuid,pheno_season\n"
        "https://oam/a.tif,in_season\n"
        "https://oam/b.tif,out_of_season\n"
        "https://oam/c.tif,in_season\n"
        "https://oam/c.tif,in_season\n",
        encoding="utf-8",
    )
    out = tmp_path / "tifs"
    out.mkdir()
    (out / "a.tif").write_bytes(b"done")
    (out / ".download_state.json").write_text(json.dumps(["a.tif"]), encoding="utf-8")
    calls = []
    monkeypatch.setattr(download, "download_parallel", lambda tasks, **kw: calls.append(tasks) or (len(tasks), 0))

    assert download.main(["--csv", str(csv), "--season", "in_season", "--output-dir", str(out)]) == 0
    assert [name for _, _, name in calls[0]] == ["c.tif"]


def test_season_filter_without_phenology_column_fails(tmp_path):
    csv = tmp_path / "filtered.csv"
    csv.write_text("uuid\nhttps://oam/a.tif\n", encoding="utf-8")
    assert download.main(["--csv", str(csv), "--season", "in_season", "--output-dir", str(tmp_path / "t")]) == 1
