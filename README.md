# oam-automation

Patched copy of `openaerialmap_scraper_clean` for automated OAM fetching, phenology classification, and upload to deadtrees.earth.

## Patches applied (WP-01)

1. **v3 MODIS zarr**: Replaces buggy v2 (any-NaN interpolation) with corrected v3 (all-NaN interpolation).
2. **DOY off-by-one fix**: `parse_date_to_doy` returns `tm_yday - 1` (0-indexed) to match MODIS DOY convention (0-365).
3. **Phenology path fix**: Pipeline no longer overrides the default path, preventing path divergence.

## Original repo

`C:\Users\jonathan\Documents\HiWi\openaerialmap_scraper_clean` is untouched. All changes live here.
