"""Tests for download.py."""

import json

import download


def test_downloads_in_season_tifs_not_yet_fetched(tmp_path, monkeypatch):
    csv = tmp_path / "phenology.csv"
    csv.write_text(
        "uuid,pheno_season,filename\n"
        "https://oam/a.tif,in_season,\n"
        "https://oam/b.tif,out_of_season,\n"
        "https://oam/c.tif,in_season,\n"
        "https://oam/c.tif,in_season,\n"
        "https://oam/item-1/odm_orthophoto.tif,in_season,item-1.tif\n"
        "https://oam/D.tif,in_season,\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("filename\nd.tif\n", encoding="utf-8")
    out = tmp_path / "tifs"
    out.mkdir()
    (out / "a.tif").write_bytes(b"done")
    (out / ".download_state.json").write_text(json.dumps(["a.tif"]), encoding="utf-8")
    calls = []
    monkeypatch.setattr(download, "download_parallel", lambda tasks, **kw: calls.append(tasks) or (len(tasks), 0))

    argv = ["--csv", str(csv), "--season", "in_season", "--output-dir", str(out),
            "--skip-csv", str(ledger), "--skip-csv", str(tmp_path / "missing.csv")]
    assert download.main(argv) == 0
    assert [name for _, _, name in calls[0]] == ["c.tif", "item-1.tif"]


def test_season_filter_without_phenology_column_fails(tmp_path):
    csv = tmp_path / "filtered.csv"
    csv.write_text("uuid\nhttps://oam/a.tif\n", encoding="utf-8")
    assert download.main(["--csv", str(csv), "--season", "in_season", "--output-dir", str(tmp_path / "t")]) == 1
