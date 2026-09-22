# oam-automation

Weekly OpenAerialMap (OAM) harvest for deadtrees.earth: fetch new imagery, filter it, classify season with MODIS phenology, check tree-canopy leaf state with a VLM, and upload the qualifying images as new datasets. One command per week: `oam_weekly.py`.

Lightweight: no GPU. The heaviest steps are downloading the GeoTIFFs and converting them to JPEG for the VLM check. Only in-season images are downloaded; a weekly run has needed up to about 5 GB of disk.

## Setup

Python 3.10+ (developed on 3.13), Linux or Windows.

```bash
git clone <this repo> oam-automation && cd oam-automation
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install <path or git URL of the deadtrees monorepo's deadtrees-cli>
cp .env.example .env    # then fill it in
```

`deadtrees-cli` is not on PyPI. It provides `deadtrees_cli` and `shared`, which `deadtrees_seam.py` imports for login, upload, processing and the duplicate checks.

### Configuration (`.env`)

| Variable | Used for |
|---|---|
| `OPENROUTER_API_KEY` | VLM check (`google/gemini-3-flash-preview` via OpenRouter, about $0.0008 per image; only in-season images are sent) |
| `PROCESSOR_USERNAME`, `PROCESSOR_PASSWORD` | deadtrees.earth account the uploads are made with |
| `SUPABASE_URL`, `SUPABASE_KEY`, `API_ENDPOINT` | platform endpoints, read by `deadtrees-cli` |
| `OAM_UPLOADED_CSV` | upload ledger (see below) |

About the OpenRouter key: every run spends credits on it. Use a key with a credit limit. If the key is invalid or out of credits, the run fails (exit 1) and the weekly window stays put, so no images are lost. Fix the key and the next run catches up.

The ledger (`metadata_uploaded.csv`) records every uploaded file and is the first dedup check. Point `OAM_UPLOADED_CSV` at the existing ledger so earlier uploads are not re-checked, for example `/mnt/gsdata/projects/deadtrees/data_openaerialmap/metadata_uploaded.csv`. Where it lives is up to whoever runs this, but there should be exactly one.

## Deploy and schedule

### First run

The first run needs an explicit start date. After that, each run continues from where the previous one stopped.

```bash
.venv/bin/python oam_weekly.py --uploaded-after YYYY-MM-DD
```

Use the date of the last manual harvest. Starting a few days early is harmless because files already on the platform are skipped.

### Weekly timer (systemd)

`/etc/systemd/system/oam-weekly.service`:

```ini
[Unit]
Description=Weekly OpenAerialMap harvest for deadtrees.earth
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=<service user>
WorkingDirectory=/opt/oam-automation
ExecStart=/opt/oam-automation/.venv/bin/python oam_weekly.py
# Optional: prune old run dirs (about 5 GB each) after the run
# ExecStartPost=/usr/bin/find /opt/oam-automation/runs -mindepth 1 -maxdepth 1 -type d -mtime +28 -exec rm -rf {} +
```

`/etc/systemd/system/oam-weekly.timer`:

```ini
[Unit]
Description=Run the OAM harvest weekly

[Timer]
OnCalendar=Mon *-*-* 03:00
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now oam-weekly.timer
```

Run it once by hand (`sudo systemctl start oam-weekly.service`) after the first run above.

### Where things go

- `runs/<date>/` holds one run: scrape, filtered CSVs, thumbnails, TIFFs, JPEGs, VLM report, `run_manifest.json`. Symlink `runs/` to other storage if the repo disk is small.
- `runs/last_run.txt` is the status file. It holds the timestamp, the window, `scrape_uploaded_at` (where the next run starts), counts and the exit status.
- Output goes to stdout/stderr (`journalctl -u oam-weekly`).

### Monitoring

A failed run exits 1, so the service shows as failed. Hook `OnFailure=` up to a mail or chat notifier, or check `systemctl status oam-weekly` / `last_run.txt` now and then. A failed run holds the weekly window, so the next run retries it automatically. If a week keeps failing, someone has to look.

## Usage

```
python oam_weekly.py [--output-dir DIR] [--dry-run] [--skip-server-check]
                     [--uploaded-after YYYY-MM-DD] [--uploaded-before YYYY-MM-DD]
```

- No flags: the weekly run. It processes OAM uploads since `scrape_uploaded_at` in `last_run.txt` into `runs/<today>`.
- `--output-dir DIR`: run dir. Pass the same dir to resume an interrupted run, because every stage whose output exists is skipped.
- `--dry-run`: inspect an existing run dir. Lists upload candidates and their exact metadata, with no uploads and no ledger writes.
- `--skip-server-check`: skip the platform `file_name` check.
- `--uploaded-after` / `--uploaded-before`: an ad-hoc window, for example to re-harvest a past period. Both dates are inclusive. `--uploaded-before` needs an after date. Use a fresh `--output-dir` per window; a dir that holds a different window is rejected. An ad-hoc run only moves the weekly window forward if it covered the window's current start.

## How it works

```
scrape -> filter -> phenology -> thumbnails -> tifs -> jpegs -> metadata
       -> VLM audit (manifest -> phenology-run -> phenology-report)
       -> upload gate -> dedup -> upload -> ledger append -> status file
```

- **Pipeline:** `pipeline.py` runs the stages and is resumable. If a window has no filter-passing or no in-season images, it stops early and the run succeeds with 0 candidates.
- **Filter:** keeps drone/aircraft imagery under 10 cm GSD with more than one band, uploaded inside the window.
- **Phenology:** classifies each capture date against the MODIS leaf-on window at its location, widened by 30 days on each side (`pipeline.py --pad-days`). Only `in_season` images get thumbnails, TIFFs and a VLM check.
- **Upload gate:** MODIS `in_season` and VLM `leaf_on` and a successful VLM review.
- **Dedup:**
  - the local ledger;
  - a server-side `file_name` check;
  - a content hash matched against the platform's `orthos` table.
- **Upload:** `deadtrees_seam.py` uploads through `deadtrees-cli`, logs in fresh for every file and triggers processing.
- **Metadata:** matches the ledger schema. The license comes from the OAM record; an unknown license or missing authors rejects that image. Rejected images are counted as `rejected`, not `failed`, and do not hold the window.

### Failures

- **What fails a run:** a crash, a scrape that loses pages, VLM errors, or failed uploads. The run exits 1 and `scrape_uploaded_at` stays where it was, so the next scheduled run covers the same period again. Dedup skips what did get uploaded.
- **Retry now:** re-run into the same `--output-dir`. Only the unfinished stages run, and only the images whose VLM review failed are sent to OpenRouter again.
- **Crash between upload and ledger append:** the next run's server-side and content-hash checks skip the file. To fill the ledger gap, cross-check platform datasets against the ledger and append the missing row with `build_upload_kwargs` + `append_ledger_row`.
- **Skipping a period on purpose:** edit `scrape_uploaded_at` in `last_run.txt`.

## Notes

- **Earth Engine forest filter:** the ESA WorldCover forest-percentage filter is kept but off by default. The VLM check covers the same concern, and Earth Engine can cost money. `oam_weekly.py` never enables it. To try it, run `pipeline.py --forest-min 0 --forest-max 100` by hand (needs Earth Engine auth); `pipeline.py` does not load `.env`, so export `OPENROUTER_API_KEY` first. Enabling it in the weekly run means passing those flags from `run_pipeline()` in `oam_weekly.py`.
- **VLM audit:** `aerial_phenology_audit.py` is copied verbatim from `aerial-phenology-audit`.
- **MODIS phenology data:** a small zarr in `phenology/`.
- **Tests:** `pip install pytest && python -m pytest`. They need no network, credentials or `deadtrees-cli`.
