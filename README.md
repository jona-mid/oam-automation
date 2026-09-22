# oam-automation

Weekly OpenAerialMap (OAM) harvest for deadtrees.earth: fetch new imagery, filter it, classify season with MODIS phenology, review tree-canopy state with a VLM, and upload the qualifying images as new datasets. Runs as a single command per week (`oam_weekly.py`), currently driven manually, designed for a systemd timer.

## How it works

```
scrape -> filter -> phenology -> thumbnails -> tifs -> jpegs -> metadata
       -> VLM audit (manifest -> phenology-run -> phenology-report)
       -> upload gate -> ledger/platform diff -> upload -> ledger append -> status file
```

- `pipeline.py` is the resumable pipeline: each stage is skipped if its output already exists, so an interrupted run continues where it stopped.
- The upload gate keeps only images with MODIS `in_season`, VLM `leaf_on`, and `review_status == success`.
- `deadtrees_seam.py` is the thin platform seam: it uploads via the monorepo `deadtrees-cli` with fresh credentials per file and runs the platform-side `file_name` duplicate check.
- Upload kwargs follow the canonical ledger schema. License provenance comes from the OAM record (`property_license` mapped through `OAM_LICENSE_MAP` in `deadtrees_seam.py`), not from a hardcoded default. Unknown licenses fail closed.

## Usage

`oam_weekly.py` is the entry point:

```
python oam_weekly.py [--output-dir DIR] [--dry-run] [--skip-server-check]
                     [--uploaded-after YYYY-MM-DD]
```

- `--uploaded-after YYYY-MM-DD` restricts the run to OAM uploads on or after that date, so weekly runs process only new images instead of the full catalog (~8,800 filter-passing candidates). Resolution order: explicit flag, then the date in the last run's status file, then fail fast. The first run must pass the flag explicitly.
- `--dry-run` lists gate candidates and exact upload kwargs with zero upload calls.
- `--skip-server-check` skips the platform-side duplicate check (diff leg 2).
- Each run writes a status file (`last_run.txt` next to the run dirs) with timestamp, counts, and exit status.

## Reliability

- **Dedup:** two legs, the local ledger CSV (`metadata_uploaded.csv`, canonical schema) and a server-side `file_name` check on the platform. A filename known to either is skipped.
- **Retry:** failed uploads are logged per candidate and never appended to the ledger, so the next run retries them naturally.
- **Crash recovery:** if a run dies after an upload succeeded but before its ledger append, reconcile by cross-checking platform datasets against the ledger and appending the missing row with `build_upload_kwargs` + `append_ledger_row` (exercised in practice, Sep 22, 2026).

## Notes

- **Earth Engine forest filter:** the ESA WorldCover forest-percentage filter (`forest > 0`) is included but not used by default. The VLM check covers the same concern (tree-canopy presence), and Earth Engine can cause cost. It can be re-enabled with `pipeline.py --forest-min 0 --forest-max 100`.
- **VLM audit:** `aerial_phenology_audit.py` (copied verbatim from `aerial-phenology-audit`), model `google/gemini-3-flash-preview` via OpenRouter, ~$0.0008/image, only `in_season` images are reviewed. `OPENROUTER_API_KEY` required; missing key fails closed. `phenology-run` is resumable at zero cost.
- Configuration in `.env` (OpenRouter key, platform account, endpoints, ledger path). It is not part of this repo.
