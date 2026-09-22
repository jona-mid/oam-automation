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
- `deadtrees_seam.py` does the actual upload. It uses the `deadtrees-cli` from the monorepo, logs in fresh for every file, and asks the platform whether the filename already exists before uploading.
- The uploaded metadata matches the ledger schema. The license is taken from the OAM record instead of assuming CC BY for everything. If a record has a license we don't recognize, the upload is rejected rather than guessing.

## Usage

`oam_weekly.py` is the entry point:

```
python oam_weekly.py [--output-dir DIR] [--dry-run] [--skip-server-check]
                     [--uploaded-after YYYY-MM-DD]
```

- `--uploaded-after YYYY-MM-DD` restricts the run to OAM uploads on or after that date, so weekly runs process only new images instead of the full catalog (~8,800 filter-passing candidates). Resolution order: explicit flag, then the date in the last run's status file, then fail fast. The first run must pass the flag explicitly.
- `--dry-run` lists gate candidates and exact upload kwargs with zero upload calls.
- `--skip-server-check` skips check for identical `file_name`
- Each run writes a status file (`last_run.txt` next to the run dirs) with timestamp, counts, and exit status.

## Reliability

- **Dedup:** three legs, the local ledger CSV (`metadata_uploaded.csv`, canonical schema), a server-side `file_name` check on the platform, and a content hash that matches the platform's own hash (checked against the `orthos` table). A filename known to either of the first two, or a file whose content the platform already processed, is skipped.
- **Retry:** failed uploads are logged per candidate and never appended to the ledger, so the next run retries them naturally.
- **Crash recovery:** if a run dies after an upload succeeded but before its ledger append, reconcile by cross-checking platform datasets against the ledger and appending the missing row with `build_upload_kwargs` + `append_ledger_row` (exercised in practice, Sep 22, 2026).

## Notes

- **Earth Engine forest filter:** the ESA WorldCover forest-percentage filter (`forest > 0`) is included but not used by default. The VLM check covers the same concern (tree-canopy presence), and Earth Engine can cause cost. It can be re-enabled with `pipeline.py --forest-min 0 --forest-max 100`.
- **VLM audit:** `aerial_phenology_audit.py` (copied verbatim from `aerial-phenology-audit`), model `google/gemini-3-flash-preview` via OpenRouter, ~$0.0008/image, only `in_season` images are reviewed. `OPENROUTER_API_KEY` required; missing key fails closed. `phenology-run` is resumable at zero cost.
- Configuration in `.env` (OpenRouter key, platform account, endpoints, ledger path). It is not part of this repo.
