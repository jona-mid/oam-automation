"""Tests for filter.py."""

import sys

import pandas as pd
import pytest

import filter as filter_module
from filter import filter_openaerial_data


def _df(*rows):
    return pd.DataFrame([{"gsd": "0.05", "platform": "uav", **row} for row in rows])


def test_quality_filters():
    # max_gsd_cm=10 keeps only GSD < 0.1 m.
    assert list(filter_openaerial_data(_df({"id": "1"}, {"id": "2", "gsd": "0.10"}), max_gsd_cm=10)["id"]) == ["1"]
    bands = _df({"id": "1", "property_bands": "1"}, {"id": "2", "property_bands": "3"})
    assert list(filter_openaerial_data(bands, max_gsd_cm=100)["id"]) == ["2"]
    platforms = _df({"id": "1", "platform": "UAV"}, {"id": "2", "platform": "Aircraft"}, {"id": "3", "platform": "satellite"})
    assert list(filter_openaerial_data(platforms, max_gsd_cm=100, platform_type=["uav", "aircraft"])["id"]) == ["1", "2"]
    dupes = _df({"id": "1", "bbox": "[1, 2, 3, 4]"}, {"id": "2", "bbox": "[1, 2, 3, 4]"}, {"id": "3", "bbox": "[5, 6, 7, 8]"})
    assert len(filter_openaerial_data(dupes, max_gsd_cm=100)) == 2
    assert len(filter_openaerial_data(pd.DataFrame(), max_gsd_cm=10)) == 0


def test_upload_window_is_inclusive_of_both_days():
    df = _df({"id": "0", "uploaded_at": "2025-01-09T23:59:00Z"}, {"id": "1", "uploaded_at": "2025-01-10T00:00:00Z"},
             {"id": "2", "uploaded_at": "2025-01-15T14:22:00Z"}, {"id": "3", "uploaded_at": "2025-01-16T00:00:00Z"})
    result = filter_openaerial_data(df, max_gsd_cm=100, uploaded_after_date="2025-01-10", uploaded_before_date="2025-01-15")
    assert list(result["id"]) == ["1", "2"]


def test_forest_percentage_bounds():
    df = _df({"id": "1"}, {"id": "2"}, {"id": "3"})
    df["forest_percentage_gee"] = [45.0, 20.0, 60.0]
    result = filter_openaerial_data(df, max_gsd_cm=100, forest_percentage_min=30, forest_percentage_max=50)
    assert list(result["id"]) == ["1"]


def _run_main(tmp_path, monkeypatch, *extra):
    rows = [{"id": "1", "property_bands": "3", "bbox": "[1.0, 2.0, 3.0, 4.0]"},
            {"id": "2", "property_bands": "3", "bbox": "[5.0, 6.0, 7.0, 8.0]"},
            {"id": "2", "property_bands": "3", "bbox": "[5.0, 6.0, 7.0, 8.0]"}]
    _df(*rows).to_csv(tmp_path / "in.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["filter.py", "--input", str(tmp_path / "in.csv"),
                                      "--output", str(tmp_path / "out.csv"), *extra])
    filter_module.main()
    return pd.read_csv(tmp_path / "out.csv")


def test_no_forest_bounds_skips_earth_engine(tmp_path, monkeypatch):
    def no_ee(*args, **kwargs):
        raise AssertionError("init_earthengine must not run without forest bounds")

    monkeypatch.setattr(filter_module, "init_earthengine", no_ee)
    out = _run_main(tmp_path, monkeypatch)
    assert len(out) == 2 and "forest_percentage_gee" not in out.columns


def test_high_earth_engine_error_rate_aborts(tmp_path, monkeypatch):
    def failing(bbox_coords):
        filter_module._ee_errors += 1

    monkeypatch.setattr(filter_module, "init_earthengine", lambda *a, **k: True)
    monkeypatch.setattr(filter_module, "calculate_forest_percentage", failing)
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch, "--forest_percentage_min", "0", "--forest_percentage_max", "100")
