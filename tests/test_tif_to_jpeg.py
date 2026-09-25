import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

import tif_to_jpeg


def write_tif(path, bands):
    data = (np.random.rand(bands, 400, 300) * 255).astype("uint8")
    with rasterio.open(
        path, "w", driver="GTiff", width=300, height=400, count=bands, dtype="uint8",
        crs="EPSG:32632", transform=from_origin(450000, 5320000, 0.05, 0.05),
    ) as dst:
        dst.write(data)


def test_rgba_orthophoto_converts_to_rgb_jpeg(tmp_path):
    """ODM orthophotos carry an alpha band; only the colour bands go into the JPEG."""
    write_tif(tmp_path / "rgba.tif", 4)
    ok, _, error = tif_to_jpeg._convert_tif_to_20cm_jpeg(
        str(tmp_path / "rgba.tif"), str(tmp_path / "rgba.jpeg"), tif_to_jpeg.DEFAULT_METER_CRS
    )
    assert ok, error
    image = Image.open(tmp_path / "rgba.jpeg")
    assert image.mode == "RGB"
    # 15 m x 20 m at 0.2 m per pixel
    assert image.size == (75, 100)


def test_rgb_orthophoto_still_converts(tmp_path):
    write_tif(tmp_path / "rgb.tif", 3)
    ok, _, error = tif_to_jpeg._convert_tif_to_20cm_jpeg(
        str(tmp_path / "rgb.tif"), str(tmp_path / "rgb.jpeg"), tif_to_jpeg.DEFAULT_METER_CRS
    )
    assert ok, error
