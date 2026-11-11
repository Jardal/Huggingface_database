# Huggingface_database
Get simple catalog of datasets and gguf by its GB size to fit your hardware. You can mirror learning datasets. Not BFU friendly.

## WHY this?

Labs perform their most aggressive technical adjustments (hardening & alignment tuning) ahead of the launch of U.S. and European regulatory controls. A user who requires independent reasoning from Raw Data must reckon with the fact that, **by mid-2027, standard web interfaces will become exclusively regulator-approved terminals with a minimal degree of analytical autonomy**.

### Time Window — Primary Trigger — Affected Capability / Method of Censorship

#### Q4 2026 – Q1 2027
**US EO 14409, NY RAISE Act, EU AI Act audit**
Coding and agents: Ban on cybersecurity synthesis, blocking of autonomous script execution.
Possible Geoblocking & U.S. Ban: Pressure to remove/restrict Chinese frontier weights (Qwen, DeepSeek) and models without published training reports.

#### Q2 2027 – Q3 2027
**End of GPAI exemptions in the EU (August 2027)**
Raw Data and asymmetric analysis: Sterilization of datasets, rise of political correctness.
End of anonymous API access to code and reasoning models; introduction of strict KYC for downloading large weights.

#### Q4 2027 – Q1 2028
**EU High-Risk enforcement & margin compression**
Collapse of availability: Degradation of $20 subscriptions into compression bits; relocation of PhD-level capability into closed B2B.
Legal cleansing of the hub — what remains freely available will be primarily small, heavily lobotomized, and regulatorily approved models.

Relying on Hugging Face remaining a permanent "safe harbor" for the free download of uncensored intelligence is a mistake in the analysis of power interests. A platform facing growing shareholder and regulatory pressure will shift from an open-marketplace model to a regulated-distributor model.
If you need raw, uncensored local models (e.g., Qwen-3, DeepSeek-V3/V4, or specialized coding checkpoints), the only rational step is to build your own local mirror (cold-storage, terabyte-scale GGUF/safetensors files) immediately. Once mandatory KYC and the regulatory purges tied to the August AI Act deadlines arrive, distribution of uncorrected weights via the public API will end.

# HF Dataset & Model Tracker

A script for tracking Hugging Face models, cataloguing their associated datasets, and performing controlled mirroring of datasets to local disk. It is designed to build a **local knowledge base** (data for distillation, fine-tuning, training) **without** downloading large weights (`safetensors`, `bin`, `gguf`, etc.).

## Browsing & Editing Records

All records live in a single SQLite file (`hf_mapping.db`). For comfortable inspection and manual editing, use **[DB Browser for SQLite](https://sqlitebrowser.org/)** – a free, cross-platform GUI.

You can use it to:

- browse and filter models and datasets,
- mark datasets for download by setting `download = NULL`,
- reset a "stuck" record (`download = 2` › `NULL`),
- inspect `error` messages,
- run any of the ad-hoc queries below.

See [Example SQL queries](#example-sql-queries) for ready-to-paste snippets.

## Features

- **Scans Hugging Face** and stores models in SQLite (author, quantization, size, date, downloads).
- **Records datasets** linked to models (via `dataset:*` tags) in a separate table.
- **Estimates real GGUF sizes** via the Tree API (smallest `.gguf` file).
- **Fetches dataset metadata** – weight-free size, file count, `lastModified`.
- **Mirrors datasets** to `HF_dataset/<dataset-name>/` – skips `safetensors`, `pytorch_model.bin`, `*.pt`, `*.gguf`, `*.onnx` and similar.
- **Classifies models** by type: `base`, `lora`, `accel_head`, `alignment`, `embedding`.
- **State machine** in `datasets.download` – allows manual marking for download, crash-safe resumption, update detection.
- **Never deletes** local files that disappeared on the remote (update = fill in gaps, not sync).

## Installation

### Requirements

- Python 3.9+
- `pip`

### Steps

```bash
git clone <repository-url>
cd <repo>

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`:

```
huggingface_hub>=0.23
requests>=2.31
certifi>=2024.2.2
# Windows (recommended behind corporate proxy / TLS-inspecting antivirus):
pip-system-certs>=4.0; sys_platform == "win32"
```

### Token

Create a [Personal Access Token](https://huggingface.co/settings/tokens) (read scope is enough) and export it as an environment variable:

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN = "hf_your_token"
```

**Linux/macOS:**
```bash
export HF_TOKEN="hf_your_token"
```

> **Never** hard-code the token in the script – it will leak when you share the file.

## Configuration

Everything is at the top of the script:

```python
DB_NAME              = "hf_mapping.db"
HF_DATASET_DIR       = "HF_dataset"
MAX_UNCHANGED_STREAK = 30
```

- `DB_NAME` – path to the SQLite database
- `HF_DATASET_DIR` – where datasets are mirrored
- `MAX_UNCHANGED_STREAK` – number of consecutive unchanged models that stops incremental sync

### Excluded patterns (`IGNORE_PATTERNS`)

Files that are **never** downloaded:

```
*.safetensors, pytorch_model*.bin, model*.bin, tf_model*.h5,
*.pt, *.pth, *.ckpt, *.onnx, *.tflite, *.ot, *.msgpack,
*.gguf, *.bin.index.json, *.safetensors.index.json
```

Everything else (parquet, jsonl, csv, arrow, scripts, README, configs) is downloaded.

## Usage

```bash
python dataset.py
```

A menu appears:

```
=== HF Model/Dataset Tracker ===
1) Update models and fill in dataset metadata
2) Download datasets marked download=NULL
3) Statistics
4) Quit
Choice [1-4]:
```

### Option 1 – Model update + dataset metadata

- If the database is **empty** › downloads models from oldest (`createdAt ASC`) – full sync.
- If it **has data** › incremental update from newest changes (`lastModified DESC`), stopping after `MAX_UNCHANGED_STREAK` unchanged models.
- Then fills in metadata for every dataset that:
  - has no `size_gb`, **or**
  - is marked as downloaded (`download = 3`) – checks for remote updates.

### Option 2 – Download marked datasets

Downloads every dataset where `download IS NULL`. Order: smallest first.

For each dataset:

1. If category is `lora`, `accel_head` or `not_a_dataset` › skip and set to `1`.
2. Set `download = 2` (downloading) and **commit** – crash protection.
3. Call `snapshot_download` with `ignore_patterns` – data only.
4. On success › `download = 3` + `downloaded_at` + `local_path` + real size.
5. On error › `download = NULL` + `error` › retry next run.

### Option 3 – Statistics

Prints counts of models by type, datasets by category, and download states.

## Database schema

### `models`

| Column | Type | Description |
|---|---|---|
| `model_id` | TEXT PK | `org/name` |
| `author` | TEXT | |
| `datasets` | TEXT | comma-separated list |
| `quantization` | TEXT | `GGUF`, `GPTQ`, `AWQ`, `16-bit`, `Other` |
| `size_gb` | REAL | estimate from name |
| `real_size_gb` | REAL | smallest GGUF from Tree API |
| `last_modified` | TEXT | |
| `downloads` | INTEGER | |
| `model_type` | TEXT | `base`, `lora`, `accel_head`, `alignment`, `embedding` |

### `datasets`

| Column | Type | Description |
|---|---|---|
| `dataset_id` | INTEGER PK | |
| `name` | TEXT UNIQUE | `org/name` |
| `size_gb` | REAL | weight-free size |
| `download` | INTEGER | **state – see below** |
| `last_modified` | TEXT | |
| `num_files` | INTEGER | number of non-weight files |
| `local_path` | TEXT | |
| `downloaded_at` | TEXT | |
| `error` | TEXT | last error |
| `category` | TEXT | `dataset`, `lora`, `accel_head`, `not_a_dataset` |

### `model_datasets`

Many-to-many mapping (PK `model_id + dataset_id`, cascade delete).

## Download state machine (`datasets.download`)

| Value | Meaning | Set by |
|---|---|---|
| `NULL` | **queued for download** | **you, manually in SQL** |
| `1` | not downloaded | script upon discovery |
| `2` | currently downloading | script before download starts |
| `3` | downloaded | script after success |
| `4` | update available | script when a downloaded dataset's `lastModified` changed |

### Marking for download

```sql
-- specific dataset
UPDATE datasets SET download = NULL WHERE name = 'org/name';

-- all small datasets
UPDATE datasets SET download = NULL WHERE category = 'dataset' AND size_gb < 2;

-- one author
UPDATE datasets SET download = NULL WHERE name LIKE 'Open-Orca/%';
```

### Re-checking for updates

```sql
-- mark "downloaded" datasets to be re-checked by the script
UPDATE datasets SET download = 3 WHERE download = 4;
```

## Crash resilience

- `download = 2` is written to the DB **before** download starts.
- When entering option 2, all `2` are reset to `NULL` › safe resumption.
- `snapshot_download` supports **file-level resume** – re-downloads only missing bytes.
- Network error › `NULL` + `error`; retried next run.
- The DB is never left inconsistent – all state transitions are atomic (`COMMIT`).

## LoRA / acceleration-head classification

Filtering by keywords in the name and tags:

- **LoRA:** `lora`, `qlora`, `lycoris`, `adapter`, `peft`
- **Acceleration heads:** `mtp`, `dflash`, `medusa`, `eagle`, `speculative-head`
- **Alignment:** `reward`, `preference`, `dpo`, `orpo`, `kto`, `rlhf`
- **Embedding:** `embedding`

LoRA and acceleration-head datasets are catalogued but **skipped during download**.

---

## Example SQL queries

Paste into **DB Browser for SQLite** › *Execute SQL* tab. All queries are read-only unless you change `SELECT` to `UPDATE`/`DELETE`.

### Models

```sql
-- All models under 12 GB with "qwen" in the name
SELECT * FROM models
WHERE real_size_gb < 12
  AND real_size_gb > 4
  AND model_id LIKE '%qwen%'
-- AND quantization = 'GGUF'
-- ORDER BY last_modified DESC
-- ORDER BY author
;

-- Gemma models between 6–12 GB, modified after a given date,
-- sorted by size, datasets, model_id
SELECT model_id, author, datasets, real_size_gb
FROM models
WHERE real_size_gb < 12
  AND real_size_gb > 6
  AND model_id LIKE '%gemma%'
  AND last_modified > '2026-07-01T23:29:04.000Z'
-- AND quantization = 'GGUF'
-- ORDER BY last_modified DESC
ORDER BY real_size_gb DESC, datasets, model_id
;

-- Small (< 4 GB) Qwen 3.8 variants with specific keywords
SELECT model_id, author, datasets, real_size_gb
FROM models
WHERE real_size_gb < 4
-- AND real_size_gb > 2
  AND last_modified > '2026-07-01T23:29:04.000Z'
  AND model_id LIKE '%/qwen3.8%'
  AND (model_id LIKE '%fab%'    OR model_id LIKE '%claude%'
    OR model_id LIKE '%think%'  OR model_id LIKE '%iq%'
    OR model_id LIKE '%max%'    OR model_id LIKE '%distil%')
-- AND quantization = 'GGUF'
-- ORDER BY last_modified DESC
ORDER BY real_size_gb DESC, datasets, model_id
;

-- Newest models per author
SELECT author, COUNT(*) AS cnt, MAX(last_modified) AS latest
FROM models
GROUP BY author
ORDER BY latest DESC
LIMIT 30;

-- Only LoRA / accel-head / alignment models
SELECT model_id, author, real_size_gb, last_modified
FROM models
WHERE model_type IN ('lora', 'accel_head', 'alignment')
ORDER BY last_modified DESC;
```

### Datasets – browsing

```sql
-- Overview of all datasets with state, size, and category
SELECT dataset_id, name, category, download, size_gb, num_files,
       last_modified, downloaded_at
FROM datasets
ORDER BY size_gb DESC
;

-- Only real datasets (excludes LoRA, accel heads, missing repos)
SELECT name, size_gb, num_files, last_modified
FROM datasets
WHERE category = 'dataset'
ORDER BY size_gb DESC
;

-- Queue every real dataset under 2 GB for download
-- (UNCOMMENT to execute)
-- UPDATE datasets SET download = NULL
-- WHERE category = 'dataset' AND size_gb < 2 AND download IN (1);

-- Datasets that failed during download (with error message)
SELECT name, error, size_gb, download
FROM datasets
WHERE error IS NOT NULL
ORDER BY name
;

-- Datasets currently marked "downloading" (crashed run?)
-- The script resets them on next run, but you can do it manually:
-- UPDATE datasets SET download = NULL WHERE download = 2;
SELECT name, size_gb, downloaded_at
FROM datasets
WHERE download = 2
;

-- Datasets with an available update
SELECT name, size_gb, last_modified, downloaded_at
FROM datasets
WHERE download = 4
ORDER BY last_modified DESC
;

-- Already downloaded datasets, sorted by local size
SELECT name, size_gb, downloaded_at, local_path
FROM datasets
WHERE download = 3
ORDER BY size_gb DESC
;

-- Stats per category
SELECT category, COUNT(*) AS cnt,
       ROUND(SUM(size_gb), 2) AS total_gb,
       ROUND(AVG(size_gb), 3) AS avg_gb
FROM datasets
WHERE num_files > 0
GROUP BY category
ORDER BY total_gb DESC
;
```

### Datasets – by model / author

```sql
-- Datasets used by a specific model
SELECT d.name, d.size_gb, d.download, d.category
FROM datasets d
JOIN model_datasets md ON md.dataset_id = d.dataset_id
WHERE md.model_id = 'org/model-name'
ORDER BY d.size_gb DESC
;

-- Datasets used by models of a given author
SELECT DISTINCT d.name, d.size_gb, d.download
FROM datasets d
JOIN model_datasets md ON md.dataset_id = d.dataset_id
JOIN models m          ON m.model_id  = md.model_id
WHERE m.author = 'Qwen'
ORDER BY d.size_gb DESC
;

-- Datasets that appear in the most models (popular)
SELECT d.name, d.category, d.size_gb, COUNT(*) AS used_by
FROM datasets d
JOIN model_datasets md ON md.dataset_id = d.dataset_id
GROUP BY d.dataset_id
ORDER BY used_by DESC
LIMIT 50
;

-- Datasets matching a keyword in name
SELECT name, size_gb, num_files, download, category
FROM datasets
WHERE name LIKE '%reasoning%'
ORDER BY size_gb DESC
;

-- Datasets with data available but not yet marked (all still at state 1)
SELECT name, size_gb, num_files
FROM datasets
WHERE download = 1 AND category = 'dataset' AND num_files > 0
ORDER BY size_gb ASC
LIMIT 50
;
```

### Bulk operations (use carefully)

```sql
-- Reset every failed dataset for retry
UPDATE datasets SET download = NULL, error = NULL
WHERE download IS NULL AND error IS NOT NULL;

-- Mark every downloaded dataset for an update check
UPDATE datasets SET download = 3 WHERE download = 4;

-- Reset a single "stuck" record
UPDATE datasets SET download = NULL, error = NULL
WHERE name = 'org/name-of-dataset';
```

---

## Troubleshooting

### `SSL: CERTIFICATE_VERIFY_FAILED` (Windows)

Cause: corporate antivirus / proxy (ESET, Avast, Zscaler, Cisco Umbrella…) injects its own TLS certificate that Python does not trust.

**Solution 1 (recommended):**
```powershell
pip install pip-system-certs
```
Restart Python.

**Solution 2:** at the top of the script (before importing `huggingface_hub`):
```python
import os, certifi, ssl
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"]      = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
```

**Solution 3:** explicit session for `huggingface_hub`:
```python
from huggingface_hub import configure_http_backend
import requests, certifi

def _factory() -> requests.Session:
    s = requests.Session()
    s.verify = certifi.where()
    return s

configure_http_backend(backend_factory=_factory)
```

### `429 Too Many Requests`

The script sleeps 5–30 s depending on context. If it happens frequently, raise `time.sleep(0.15)` in `update_dataset_metadata_for_new` to `0.5`.

### `sqlite3.OperationalError: database is locked`

Close other processes holding the DB (including DB Browser for SQLite in write mode).

### The database looks inconsistent

Back up before any bigger intervention:
```bash
cp hf_mapping.db hf_mapping.db.bak
```

## Project structure

```
.
+¦¦ dataset.py            # main script
+¦¦ requirements.txt
+¦¦ hf_mapping.db         # SQLite (created on first run)
L¦¦ HF_dataset/           # mirror (created on first run)
    L¦¦ org/
        L¦¦ dataset-name/
```

## Security

- Keep the token in the `HF_TOKEN` environment variable, **never** in the code.
- `.gitignore`:
  ```
  .venv/
  __pycache__/
  *.db
  *.db-journal
  HF_dataset/
  ```
- The script only deletes rows in `model_datasets` (re-inserted immediately after); it never removes files.

## License

MIT – use, modify, share freely.
```

---

Additional useful queries I added beyond your examples:

- **Model overview:** newest-per-author, LoRA/accel-head filter.
- **Dataset browsing:** overview, real datasets only, failed ones (`error IS NOT NULL`), crashed-in-progress (`download = 2`), pending updates (`download = 4`), downloaded with local size, per-category aggregates.
- **Cross-table:** datasets of a specific model, datasets of a whole author, most popular datasets, keyword search, "ready but not marked" candidates for bulk queueing.
- **Bulk:** reset failed, mark downloaded for re-check, reset a single stuck record.

