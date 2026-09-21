# OpenAerialMap Scraper Clean Bundle

This is the supported, resumable OpenAerialMap pipeline. The historical
scraper remains unchanged in `openaerialmap_scraper`.

`pipeline.py` runs scraping, standardized metadata filtering, deterministic
MODIS phenology, thumbnail download, TIFF download, JPEG conversion, and
metadata creation. Phenology uses 30 days of padding by default. No LLM
filtering is used.

Use `--download-mode selected` for reviewed thumbnails or
`--download-mode all` for every row surviving the metadata stages.

```text
run/raw/openaerial_data.csv
run/metadata/{filtered,phenology,tif_metadata,jpeg_metadata}.csv
run/thumbnails/
run/tifs/
run/jpegs/
run/logs/
run/run_manifest.json
```

Install dependencies and authenticate Earth Engine:

```bash
pip install -r requirements.txt
python -c "import ee; ee.Authenticate()"
```

Dry run:

```bash
python pipeline.py --dry-run --output-dir D:/oam_runs/test
```

Full unattended run:

```bash
python pipeline.py --download-mode all --pad-days 30 --output-dir D:/oam_runs/run_001
```

The bundled MODIS data is under `phenology/`. Existing stage outputs are
reused. Downloads and JPEG conversion do not overwrite existing files.
