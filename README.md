# oam-automation

Patched copy of `openaerialmap_scraper_clean` for automated OAM fetching, phenology classification, and upload to deadtrees.earth.

**This repo is canonical** for the automation effort. Upstream fixes are not synced automatically; make all changes here.

## Patches applied (WP-01)

1. **v3 MODIS zarr**: Replaces buggy v2 (any-NaN interpolation) with corrected v3 (all-NaN interpolation).
2. **DOY off-by-one fix**: `parse_date_to_doy` returns `tm_yday - 1` (0-indexed) to match MODIS DOY convention (0-365).
3. **Phenology path fix**: Pipeline no longer overrides the default path, preventing path divergence.

## VLM audit integration (WP-01b)

The pipeline now runs a blind tree-canopy VLM review after the metadata stages:

```
scrape -> filter -> phenology -> thumbnails -> tifs -> jpegs -> metadata
                                                             -> audit_manifest -> phenology-run -> phenology-report
```

- Tool: `aerial_phenology_audit.py` (copied from `aerial-phenology-audit/`, unmodified).
- Model: `google/gemini-3-flash-preview` via OpenRouter (~$0.0008/image, 2-6s/image).
- Requires env var `OPENROUTER_API_KEY`. Missing key fails closed (the upload gate cannot be verified without it). Current key location on this machine: `C:\Users\jonathan\Documents\HiWi\georeferencing\georef_check_vlm\.env`. WP-03 must give the key a proper home on the scheduler host (systemd `EnvironmentFile=`).
- `--vlm-endpoint` must be the FULL chat completions URL (`https://openrouter.ai/api/v1/chat/completions`); the tool posts to it as-is.
- Only `in_season` images are reviewed (`--priorities in_season`); out-of-season images never reach upload.
- `phenology.py` derives `filename`, `classification`, and `jpeg_filename` columns so the manifest reads `phenology.csv` directly (no adapter).
- `phenology-run` is resumable: re-runs skip images with recorded successes ($0 re-run cost).
- Upload gate (WP-02, not yet built): MODIS `in_season` AND VLM `leaf_on` AND `review_status == success`.

## Upstream sources

Two upstreams, both frozen (not synced):

- `C:\Users\jonathan\Documents\HiWi\openaerialmap_scraper_clean` - scraper, pipeline, phenology base. Untouched; its vision files are intentionally excluded here.
- `C:\Users\jonathan\Documents\HiWi\aerial-phenology-audit` - source of `aerial_phenology_audit.py` (copied verbatim, unmodified).

All changes live in this repo.
