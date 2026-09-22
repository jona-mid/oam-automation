# oam-automation

Weekly OpenAerialMap harvest for deadtrees.earth: scrape new OAM uploads, filter them, check season (MODIS phenology) and tree-canopy leaf state (VLM), and upload the qualifying images via `deadtrees-cli`. Lightweight, no GPU; a run needs up to ~5 GB of disk.

## Setup

Python 3.11+. `deadtrees-cli` needs a full checkout of the deadtrees monorepo because it also installs the monorepo's `shared` package.

```bash
git clone https://github.com/Deadwood-ai/deadtrees.git ../deadtrees
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ../deadtrees/deadtrees-cli
cp .env.example .env   # fill in
```

- `OPENROUTER_API_KEY`: VLM check with `google/gemini-3-flash-preview`, ~$0.0008 per in-season image. Use a key with a credit limit.
- `OAM_UPLOADED_CSV`: the upload ledger, currently `/mnt/gsdata/projects/deadtrees/data_openaerialmap/metadata_uploaded.csv`. Keep exactly one.
- The rest are the platform account and endpoints for `deadtrees-cli`.

## Run

```bash
.venv/bin/python oam_weekly.py --uploaded-after YYYY-MM-DD   # first run only
.venv/bin/python oam_weekly.py                               # every run after that
```

The first run needs the date of the last harvest. After that, each run continues where the previous one stopped (`runs/last_run.txt`). To run it weekly, see [Deploy](#deploy).

- Each run writes a `runs/<date>/` folder (prune old ones); progress and errors go to stdout/stderr.
- A failed run exits 1.

Options: `--dry-run` lists upload candidates from an existing run folder without uploading. Re-running on the same day resumes `runs/<today>`; `--output-dir DIR` picks another folder. `--uploaded-after` / `--uploaded-before` run an ad-hoc date window (both inclusive; use a fresh `--output-dir`). `--skip-server-check` skips the platform filename check.

## How it works

```
scrape -> filter -> phenology -> thumbnails/TIFFs -> JPEGs -> VLM check -> upload filter -> duplicate checks -> upload -> ledger
```

- **Upload filter:** drone/aircraft imagery under 10 cm GSD, MODIS in season (leaf-on window ±30 days), VLM `leaf_on`. The license comes from the OAM record; images with an unknown license or incomplete metadata are rejected.
- **Duplicate checks:** the ledger, the filename on the platform, and the file hash against the platform's `orthos` table.

## Operations

- **Failed runs:** a crash, VLM/OpenRouter errors or failed downloads/uploads leave the start date in `last_run.txt` unchanged, so the next run retries automatically. To retry now, re-run into the same `--output-dir`; completed steps and files are skipped. A problem that persists (e.g. an image that never downloads) fails every run until someone looks. Images rejected for bad metadata, and TIFFs that cannot be converted, are dropped with a log line and don't block anything.
- **Skipping a period on purpose:** edit `scrape_uploaded_at` in `last_run.txt`.
- **Earth Engine forest filter:** kept but off; the weekly run never enables it (see `pipeline.py --forest-min/--forest-max`).
- **Tests:** `pip install pytest && python -m pytest` (no network or credentials needed).

## Deploy

`/etc/systemd/system/oam-weekly.service`:

```ini
[Unit]
Description=Weekly OAM harvest for deadtrees.earth
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=<service user>
WorkingDirectory=/opt/oam-automation
ExecStart=/opt/oam-automation/.venv/bin/python oam_weekly.py
```

`/etc/systemd/system/oam-weekly.timer`:

```ini
[Timer]
OnCalendar=Mon *-*-* 03:00
Persistent=true

[Install]
WantedBy=timers.target
```

After the first manual run, enable it with `sudo systemctl daemon-reload && sudo systemctl enable --now oam-weekly.timer`. Logs are in `journalctl -u oam-weekly`.
