import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

import tif_to_jpeg


def write_tif(path, bands):
    data = np.full((bands, 400, 300), 120, dtype="uint8")
    if bands == 4:
        data[3] = 255  # alpha: fully opaque
    with rasterio.open(
        path, "w", driver="GTiff", width=300, height=400, count=bands, dtype="uint8",
        crs="EPSG:32632", transform=from_origin(450000, 5320000, 0.05, 0.05),
    ) as dst:
        dst.write(data)


def convert(tmp_path, name):
    return tif_to_jpeg._convert_tif_to_20cm_jpeg(
        str(tmp_path / f"{name}.tif"), str(tmp_path / f"{name}.jpeg"), tif_to_jpeg.DEFAULT_METER_CRS
    )


def test_rgba_orthophoto_converts_to_rgb_jpeg(tmp_path):
    """ODM orthophotos carry an alpha band; only the colour bands go into the JPEG."""
    write_tif(tmp_path / "rgba.tif", 4)
    ok, _, error = convert(tmp_path, "rgba")
    assert ok, error
    image = Image.open(tmp_path / "rgba.jpeg")
    assert image.mode == "RGB"
    # 15 m x 20 m at 0.2 m per pixel
    assert image.size == (75, 100)
    assert abs(float(np.asarray(image).mean()) - 120) < 5


def test_rgb_orthophoto_still_converts(tmp_path):
    write_tif(tmp_path / "rgb.tif", 3)
    ok, _, error = convert(tmp_path, "rgb")
    assert ok, error
    assert abs(float(np.asarray(Image.open(tmp_path / "rgb.jpeg")).mean()) - 120) < 5
    assert not (tmp_path / "rgb.partial.jpeg").exists()


def test_failed_conversion_leaves_no_jpeg(tmp_path, monkeypatch):
    """A resumed run skips TIFFs that have a JPEG, so a failure must not leave one behind."""
    write_tif(tmp_path / "bad.tif", 3)

    def broken_clip(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(tif_to_jpeg.np, "clip", broken_clip)
    ok, _, error = convert(tmp_path, "bad")
    assert not ok and "boom" in error
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bad.tif"]
