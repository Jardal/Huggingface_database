# -*- coding: utf-8 -*-
import sqlite3
import re
import os
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
import time
import fnmatch
import requests
from datetime import datetime
from huggingface_hub import HfApi, login, snapshot_download

# --- KONFIGURACE ---
HF_TOKEN = "=================== SEM SI DEJ SVUJ HUGGINGFACE TOKEN ================================="
DB_NAME = "hf_mapping.db"
HF_DATASET_DIR = "HF_dataset"
MAX_UNCHANGED_STREAK = 30

# Vzory souborů, které se NIKDY nestahují (velké váhy a jejich příslušenství)
# Datasety obvykle obsahují parquet/jsonl/arrow - ty zůstávají.
IGNORE_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.safetensors.md5",
    "pytorch_model*.bin",
    "model*.bin",
    "tf_model*.h5",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.onnx",
    "*.tflite",
    "*.ot",
    "*.msgpack",
    "*.mlmodel",
    "*.gguf",
    "*.bin.index.json",
]

# Stavy sloupce `download`
DL_TODO     = None   # NULL - uživatel označil ke stažení
DL_NOT_DL   = 1      # nestaženo (default po nalezení)
DL_DOING    = 2      # právě stahuji (po crashi se resetuje na NULL)
DL_DONE     = 3      # úspěšně staženo
DL_UPDATE   = 4      # update k dispozici (nestahovat automaticky)


# ---------- POMOCNÉ ----------

def normalize_date(val):
    if not val:
        return ""
    s = str(val)
    return s.replace('T', ' ').replace('Z', '').split('.')[0].split('+')[0].strip()


def dir_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return round(total / (1024 ** 3), 3)


def classify_model_type(model_id, tags):
    """Rozřazení modelů pro oddělenou evidenci."""
    name = model_id.lower()
    tags_lower = [str(t).lower() for t in (tags or [])]
    blob = name + " " + " ".join(tags_lower)

    if any(k in blob for k in ('lora', 'qlora', 'lycoris', 'adapter', 'peft')):
        return 'lora'
    if any(k in blob for k in ('mtp', 'dflash', 'medusa', 'eagle', 'speculative-head')):
        return 'accel_head'
    if any(k in blob for k in ('reward', 'preference', 'dpo', 'orpo', 'kto', 'rlhf')):
        return 'alignment'
    if 'embedding' in blob:
        return 'embedding'
    return 'base'


def classify_dataset(name):
    """Rychlé rozřazení datasetu podle názvu."""
    n = name.lower()
    if any(k in n for k in ('lora', 'qlora', 'lycoris', 'adapter', 'peft')):
        return 'lora'
    if any(k in n for k in ('mtp', 'dflash', 'medusa', 'eagle')):
        return 'accel_head'
    return 'dataset'


# ---------- DB ----------

def init_db(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            author TEXT,
            datasets TEXT,
            quantization TEXT,
            size_gb REAL,
            real_size_gb REAL,
            last_modified TEXT,
            downloads INTEGER
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_datasets (
            model_id TEXT,
            dataset_id INTEGER,
            PRIMARY KEY (model_id, dataset_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        )
    ''')

    # migrace models
    for col, dtype in [
        ('real_size_gb', 'REAL'),
        ('model_type', 'TEXT'),
    ]:
        try:
            cursor.execute(f'ALTER TABLE models ADD COLUMN {col} {dtype}')
            if col == 'model_type':
                cursor.execute('UPDATE models SET model_type = "base" WHERE model_type IS NULL')
        except sqlite3.OperationalError:
            pass

    # migrace datasets
    for col, dtype in [
        ('size_gb', 'REAL'),
        ('download', 'INTEGER'),
        ('last_modified', 'TEXT'),
        ('num_files', 'INTEGER'),
        ('local_path', 'TEXT'),
        ('downloaded_at', 'TEXT'),
        ('error', 'TEXT'),
        ('category', 'TEXT'),
    ]:
        try:
            cursor.execute(f'ALTER TABLE datasets ADD COLUMN {col} {dtype}')
            # hned po přidání vyplnit default, aby se "NULL" neinterpretovalo jako "ke stažení"
            if col == 'download':
                cursor.execute('UPDATE datasets SET download = 1 WHERE download IS NULL')
            if col == 'category':
                cursor.execute('UPDATE datasets SET category = "dataset" WHERE category IS NULL')
        except sqlite3.OperationalError:
            pass

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_datasets_download ON datasets(download);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_md_dataset_id ON model_datasets(dataset_id);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_md_model_id ON model_datasets(model_id);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_real_size ON models(quantization, real_size_gb);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_last_modified ON models(last_modified);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type);')

    conn.commit()


# ---------- EXISTUJÍCÍ LOGIKA PRO MODELY ----------

def fetch_real_gguf_size_gb(model_id, session):
    try:
        url = f"https://huggingface.co/api/models/{model_id}/tree/main"
        resp = session.get(url, timeout=10)
        if resp.status_code == 429:
            time.sleep(10)
            return None
        if resp.status_code == 200:
            files = resp.json()
            gguf_files = []
            for f in files:
                path = f.get('path', '').lower()
                if path.endswith('.gguf'):
                    size_bytes = f.get('size') or (f.get('lfs', {}).get('size') if f.get('lfs') else 0)
                    if size_bytes > 0:
                        gguf_files.append(size_bytes)
            if gguf_files:
                return round(min(gguf_files) / (1024 ** 3), 2)
    except Exception:
        pass
    return None


def estimate_size_from_name(model_id):
    name = model_id.lower()
    param_match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*b', name)
    if param_match:
        try:
            params = float(param_match.group(1))
        except ValueError:
            params = 7.0
    else:
        params = 7.0

    if 'q2' in name: mult = 0.35
    elif 'q3' in name: mult = 0.50
    elif 'q4' in name or 'iq4' in name: mult = 0.60
    elif 'q5' in name or 'iq5' in name: mult = 0.70
    elif 'q6' in name: mult = 0.85
    elif 'q8' in name: mult = 1.10
    elif 'fp16' in name or 'bf16' in name: mult = 2.0
    else: mult = 0.60
    return round(params * mult, 2)


def save_model_to_db(cursor, model_id, author, datasets_str, used_datasets,
                     quant, size_gb, real_size_gb, last_modified, downloads,
                     model_type='base'):
    cursor.execute('''
        INSERT INTO models (model_id, author, datasets, quantization, size_gb,
                            real_size_gb, last_modified, downloads, model_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_id) DO UPDATE SET
        author=excluded.author,
        datasets=excluded.datasets,
        quantization=excluded.quantization,
        size_gb=excluded.size_gb,
        real_size_gb=excluded.real_size_gb,
        last_modified=excluded.last_modified,
        downloads=excluded.downloads,
        model_type=excluded.model_type
    ''', (model_id, author, datasets_str, quant, size_gb, real_size_gb,
          last_modified, downloads, model_type))

    cursor.execute('DELETE FROM model_datasets WHERE model_id = ?', (model_id,))
    for ds_name in used_datasets:
        cat = classify_dataset(ds_name)
        cursor.execute(
            'INSERT OR IGNORE INTO datasets (name, download, category) VALUES (?, ?, ?)',
            (ds_name, DL_NOT_DL, cat)
        )
        cursor.execute('SELECT dataset_id FROM datasets WHERE name = ?', (ds_name,))
        ds_row = cursor.fetchone()
        if ds_row:
            cursor.execute(
                'INSERT OR IGNORE INTO model_datasets (model_id, dataset_id) VALUES (?, ?)',
                (model_id, ds_row[0])
            )


def _common_parse(model_id, author, raw_date, downloads, tags, session, existing_real_size=None):
    last_modified = normalize_date(raw_date)
    used_datasets = [t.replace("dataset:", "") for t in tags if str(t).startswith("dataset:")]
    datasets_str = ",".join(used_datasets) if used_datasets else ""

    tags_lower = [str(t).lower() for t in tags]
    model_lower = model_id.lower()

    quant = "Other"
    if "gguf" in model_lower or any("gguf" in t for t in tags_lower):
        quant = "GGUF"
    elif "gptq" in model_lower or any("gptq" in t for t in tags_lower):
        quant = "GPTQ"
    elif "awq" in model_lower or any("awq" in t for t in tags_lower):
        quant = "AWQ"
    elif "fp16" in model_lower or "bf16" in model_lower or any("fp16" in t or "bf16" in t for t in tags_lower):
        quant = "16-bit"

    size_gb = estimate_size_from_name(model_id)
    real_size_gb = existing_real_size
    if quant == "GGUF" and real_size_gb is None:
        real_size_gb = fetch_real_gguf_size_gb(model_id, session)

    mtype = classify_model_type(model_id, tags)
    return (model_id, author, datasets_str, used_datasets, quant, size_gb,
            real_size_gb, last_modified, downloads, mtype)


def parse_model_dict(m, session):
    model_id = m.get('id')
    if not model_id:
        return None
    author = m.get('author') or (model_id.split('/')[0] if '/' in model_id else "unknown")
    raw_date = m.get('lastModified') or m.get('updatedAt') or m.get('createdAt')
    return _common_parse(model_id, author, raw_date, m.get('downloads', 0) or 0,
                         m.get('tags', []) or [], session)


def parse_model_obj(m, session, existing_real_size=None):
    model_id = getattr(m, 'id', '')
    if not model_id:
        return None
    author = getattr(m, 'author', '') or (model_id.split('/')[0] if '/' in model_id else "unknown")
    raw_date = getattr(m, 'last_modified', '') or getattr(m, 'created_at', '')
    return _common_parse(model_id, author, raw_date, getattr(m, 'downloads', 0) or 0,
                         getattr(m, 'tags', []) or [], session, existing_real_size)


# ---------- STÁVAJÍCÍ STREAMY ----------

def build_baseline_oldest_first_rest(conn, session):
    cursor = conn.cursor()
    print("[MÓD 1] Databáze je prázdná. Volám HF REST API (createdAt ASC)...")
    cursor.execute('SELECT model_id FROM models')
    existing_ids = set(row[0] for row in cursor.fetchall())
    url = "https://huggingface.co/api/models?sort=createdAt&direction=1&limit=1000&expand[]=lastModified"
    saved_count = 0
    scanned_count = 0

    while url:
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 429:
                print("\n[!] Rate limit (429). Čekám 30 s...")
                time.sleep(30)
                continue
            resp.raise_for_status()
            models_data = resp.json()
            if not models_data:
                break
            for m in models_data:
                scanned_count += 1
                model_id = m.get('id')
                if not model_id or model_id in existing_ids:
                    continue
                parsed = parse_model_dict(m, session)
                if not parsed:
                    continue
                save_model_to_db(cursor, *parsed)
                existing_ids.add(model_id)
                saved_count += 1
                if saved_count % 100 == 0:
                    conn.commit()
                    print(f"Prohledáno: {scanned_count} | Uloženo do DB: {saved_count}")
            conn.commit()
            url = resp.links['next']['url'] if 'next' in resp.links else None
        except Exception as e:
            print(f"\n[!] Chyba přenosu REST: {e}. Čekám 30 s...")
            time.sleep(30)

    conn.commit()
    print(f"\nDatový základ dokončen! Celkem uloženo {saved_count} modelů.")


def update_incremental(api, conn, session):
    cursor = conn.cursor()
    print("[MÓD 2] Inkrementální update (OOM-safe)...")
    models_stream = api.list_models(sort="lastModified", limit=None)
    new_count = updated_count = unchanged_streak = total_processed = 0

    for m in models_stream:
        total_processed += 1
        model_id = m.id
        cursor.execute('SELECT last_modified, real_size_gb FROM models WHERE model_id = ?', (model_id,))
        row = cursor.fetchone()
        stored_modified = normalize_date(row[0]) if row else None
        existing_real_size = row[1] if row else None

        parsed = parse_model_obj(m, session, existing_real_size=existing_real_size)
        if not parsed:
            continue
        current_modified = parsed[7]

        if row is not None:
            if stored_modified and stored_modified == current_modified:
                unchanged_streak += 1
                if unchanged_streak >= MAX_UNCHANGED_STREAK:
                    print(f"\n[STOP] {MAX_UNCHANGED_STREAK} nezměněných v řadě - sync hotov.")
                    break
                continue
            else:
                updated_count += 1
                unchanged_streak = 0
        else:
            new_count += 1
            unchanged_streak = 0

        save_model_to_db(cursor, *parsed)
        if (new_count + updated_count) % 50 == 0:
            conn.commit()
            print(f"Zpracováno: {total_processed} | +{new_count} | ~{updated_count}")

    conn.commit()
    print(f"\n--- Inkrementální update dokončen (+{new_count}, ~{updated_count}) ---")


# ---------- METADATA DATASETŮ ----------

def fetch_one_dataset_metadata(session, name):
    """Vrátí dict s metadaty nebo None při chybě. Respektuje rate limit."""
    try:
        resp = session.get(f"https://huggingface.co/api/datasets/{name}", timeout=15)
        if resp.status_code == 429:
            time.sleep(5)
            return {'_ratelimited': True}
        if resp.status_code == 404:
            return {'_not_found': True}
        if resp.status_code != 200:
            return None

        info = resp.json()
        new_lm = normalize_date(info.get('lastModified', ''))

        total_size = 0
        num_files = 0
        tree_resp = session.get(
            f"https://huggingface.co/api/datasets/{name}/tree/main", timeout=20
        )
        if tree_resp.status_code == 200:
            for f in tree_resp.json():
                p = f.get('path', '')
                if any(fnmatch.fnmatch(p, pat) for pat in IGNORE_PATTERNS):
                    continue
                size = f.get('size') or (f.get('lfs', {}).get('size') if f.get('lfs') else 0) or 0
                total_size += size
                num_files += 1

        return {
            'last_modified': new_lm,
            'size_gb': round(total_size / (1024 ** 3), 4),
            'num_files': num_files,
        }
    except Exception:
        return None


def update_dataset_metadata_for_new(conn, session, limit=None):
    """Doplní metadata datasetů, kde chybí size_gb (nové) nebo u stažených (update check)."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT name, last_modified, download FROM datasets
        WHERE size_gb IS NULL
           OR download = ?
        ORDER BY dataset_id
    ''', (DL_DONE,))
    rows = cursor.fetchall()
    if limit:
        rows = rows[:limit]

    print(f"[META] Zpracuji {len(rows)} datasetů (fetch metadat)...")
    for i, (name, old_lm, dl) in enumerate(rows, 1):
        md = fetch_one_dataset_metadata(session, name)
        if md is None:
            print(f"  [{i}/{len(rows)}] [!] {name} - chyba")
            continue
        if md.get('_ratelimited'):
            print(f"  [{i}/{len(rows)}] [429] {name} - pauza")
            time.sleep(10)
            continue
        if md.get('_not_found'):
            cursor.execute(
                'UPDATE datasets SET category = "not_a_dataset", download = ? WHERE name = ?',
                (DL_NOT_DL, name)
            )
            conn.commit()
            continue

        new_lm = md['last_modified']
        new_size = md['size_gb']
        num_files = md['num_files']

        new_dl = dl
        if dl == DL_DONE and old_lm and new_lm and old_lm != new_lm:
            new_dl = DL_UPDATE  # změna na remote, čeká na rozhodnutí uživatele
        elif num_files == 0:
            new_dl = DL_NOT_DL  # jen váhy / nic ke stažení

        cursor.execute('''
            UPDATE datasets
               SET size_gb = ?, last_modified = ?, num_files = ?, download = ?, error = NULL
             WHERE name = ?
        ''', (new_size, new_lm, num_files, new_dl, name))
        conn.commit()

        if i % 20 == 0:
            print(f"  [{i}/{len(rows)}] {name} -> {new_size:.3f} GB, {num_files} souborů")
        time.sleep(0.15)  # mírný rate-limit

    print("[META] Hotovo.")


# ---------- DOWNLOAD ----------

def download_pending_datasets(conn, session):
    cursor = conn.cursor()

    # po pádu resetujeme "stahuji" (2) na NULL, aby se příště zkusilo znovu
    cursor.execute('UPDATE datasets SET download = NULL WHERE download = ?', (DL_DOING,))
    conn.commit()

    cursor.execute('''
        SELECT name, category, num_files FROM datasets
        WHERE download IS NULL
        ORDER BY size_gb ASC
    ''')
    pending = cursor.fetchall()

    if not pending:
        print("Nic ke stažení (žádný dataset nemá download=NULL).")
        return

    print(f"Ke zpracování: {len(pending)} datasetů\n")
    ok = fail = skip = 0

    for idx, (name, category, num_files) in enumerate(pending, 1):
        tag = f"[{idx}/{len(pending)}]"

        if category in ('lora', 'accel_head', 'not_a_dataset'):
            print(f"{tag} [SKIP] {name} (kategorie: {category})")
            cursor.execute('UPDATE datasets SET download = ? WHERE name = ?', (DL_NOT_DL, name))
            conn.commit()
            skip += 1
            continue

        local_dir = os.path.join(HF_DATASET_DIR, name)
        os.makedirs(local_dir, exist_ok=True)

        # označit jako "stahuji" PŘED začátkem - crash detekce
        cursor.execute('UPDATE datasets SET download = ?, error = NULL WHERE name = ?',
                       (DL_DOING, name))
        conn.commit()

        try:
            print(f"{tag} Stahuji {name} ({num_files or '?'} souborů) -> {local_dir}")
            snapshot_download(
                repo_id=name,
                repo_type="dataset",
                local_dir=local_dir,
                ignore_patterns=IGNORE_PATTERNS,
                token=HF_TOKEN,
                max_workers=2,
            )
            real_size = dir_size_gb(local_dir)
            cursor.execute('''
                UPDATE datasets
                   SET download = ?, downloaded_at = ?, local_path = ?,
                       size_gb = ?, error = NULL
                 WHERE name = ?
            ''', (DL_DONE, datetime.now().isoformat(timespec='seconds'),
                  local_dir, real_size, name))
            conn.commit()
            ok += 1
            print(f"       [OK] {real_size} GB na disku")

        except KeyboardInterrupt:
            # uživatel přerušil - necháme "2", příští start resetuje na NULL
            conn.commit()
            print("\n[!] Přerušeno uživatelem.")
            raise
        except Exception as e:
            err = str(e)[:500]
            # POZOR: na NULL, aby to zkusilo příště znovu; error uložíme pro diagnostiku
            cursor.execute('UPDATE datasets SET download = NULL, error = ? WHERE name = ?',
                           (err, name))
            conn.commit()
            fail += 1
            print(f"       [FAIL] {err}")

    print(f"\n--- Download hotov: OK={ok}, FAIL={fail}, SKIP={skip} ---")
    if fail:
        print("Selhané datasety zůstaly ve stavu NULL - při dalším spuštění se zkusí znovu.")


# ---------- STATISTIKY + MENU ----------

def show_stats(conn):
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM models')
    total_models = cursor.fetchone()[0]
    cursor.execute('SELECT model_type, COUNT(*) FROM models GROUP BY model_type ORDER BY 2 DESC')
    mtypes = cursor.fetchall()

    cursor.execute('SELECT COUNT(*) FROM datasets')
    total_ds = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*), COALESCE(SUM(size_gb),0) FROM datasets WHERE num_files > 0')
    real_ds, real_gb = cursor.fetchone()

    cursor.execute('SELECT download, COUNT(*) FROM datasets GROUP BY download')
    dl_stats = cursor.fetchall()
    cursor.execute('SELECT category, COUNT(*) FROM datasets GROUP BY category ORDER BY 2 DESC')
    cat_stats = cursor.fetchall()

    labels = {None: 'ke stazeni (NULL)', 1: 'nestazeno',
              2: 'stahuji', 3: 'stazeno', 4: 'update'}

    print("\n=== STATISTIKY ===")
    print(f"Modely celkem: {total_models}")
    for t, c in mtypes:
        print(f"  - {t or '?':12}: {c}")
    print(f"Datasety celkem: {total_ds}  (s daty: {real_ds}, ~{real_gb:.2f} GB bez vah)")
    print("Kategorie:")
    for t, c in cat_stats:
        print(f"  - {t or '?':12}: {c}")
    print("Stavy download:")
    for d, c in dl_stats:
        print(f"  - {labels.get(d, '?'):18}: {c}")


def show_menu():
    print("\n=== HF Model/Dataset Tracker ===")
    print("1) Aktualizovat modely a doplnit metadata datasetu")
    print("2) Stahnout datasety oznacene jako download=NULL")
    print("3) Statistiky")
    print("4) Konec")
    return input("Volba [1-4]: ").strip()


def run_update(api, conn, session):
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM models')
    if cursor.fetchone()[0] == 0:
        build_baseline_oldest_first_rest(conn, session)
    else:
        update_incremental(api, conn, session)
    # doplnit metadata pro nove datasety (a zkontrolovat updaty u stazenych)
    update_dataset_metadata_for_new(conn, session)


# ---------- MAIN ----------

def main():
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print("HF Token úspěšně ověřen.")

    api = HfApi()
    conn = sqlite3.connect(DB_NAME)
    init_db(conn)

    session = requests.Session()
    if HF_TOKEN:
        session.headers.update({"Authorization": f"Bearer {HF_TOKEN}"})

    try:
        while True:
            choice = show_menu()
            if choice == '1':
                run_update(api, conn, session)
            elif choice == '2':
                download_pending_datasets(conn, session)
            elif choice == '3':
                show_stats(conn)
            elif choice == '4':
                break
            else:
                print("Neplatná volba.")
    except KeyboardInterrupt:
        print("\nUkončuji...")
    finally:
        session.close()
        conn.close()


if __name__ == "__main__":
    main()