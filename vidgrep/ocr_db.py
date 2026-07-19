"""Combine vidgrep scan output (.jsonl + .json pairs) into a searchable SQLite database.

Usage:
    python ocr_db.py ingest lasts4_ocr --db lasts4_ocr.db
    python ocr_db.py search "contract started" --db lasts4_ocr.db
    python ocr_db.py search "complete*" --db lasts4_ocr.db --gap 10 --min-conf 0.7

Ingest is incremental and safe to re-run while scans are still writing: each
file commits in its own transaction, malformed lines are skipped with a
warning, and files unchanged since the last ingest are not re-read.

Search groups hits into occurrence windows per video: consecutive matches whose
timestamps are within --gap seconds of each other collapse into one row with the
first and last timestamp of the window.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    source_file  TEXT PRIMARY KEY,
    pattern      TEXT,
    started_at   TEXT,
    completed_at TEXT,
    match_count  INTEGER,
    options_json TEXT
);

CREATE TABLE IF NOT EXISTS scan_runs (
    scan_id     TEXT PRIMARY KEY,
    source_file TEXT
);

CREATE TABLE IF NOT EXISTS videos (
    source_id TEXT PRIMARY KEY,
    file      TEXT,
    path      TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    match_id   TEXT PRIMARY KEY,
    scan_id    TEXT REFERENCES scan_runs(scan_id),
    source_id  TEXT REFERENCES videos(source_id),
    timestamp  REAL,
    text       TEXT,
    confidence REAL
);

CREATE INDEX IF NOT EXISTS idx_matches_video_ts ON matches(source_id, timestamp);

CREATE TABLE IF NOT EXISTS ingested_files (
    source_file     TEXT PRIMARY KEY,
    jsonl_size      INTEGER,
    jsonl_mtime_ns  INTEGER,
    sidecar_mtime_ns INTEGER
);

CREATE VIRTUAL TABLE IF NOT EXISTS matches_fts USING fts5(
    text,
    content='matches',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS matches_ai AFTER INSERT ON matches BEGIN
    INSERT INTO matches_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS matches_ad AFTER DELETE ON matches BEGIN
    INSERT INTO matches_fts(matches_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS matches_au AFTER UPDATE ON matches BEGIN
    INSERT INTO matches_fts(matches_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
    INSERT INTO matches_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""

WINDOW_QUERY = """
WITH hits AS (
    SELECT m.source_id, m.timestamp, m.text, m.confidence
    FROM matches_fts f
    JOIN matches m ON m.rowid = f.rowid
    WHERE matches_fts MATCH :query AND m.confidence >= :min_conf
),
lagged AS (
    SELECT *,
        LAG(timestamp) OVER (PARTITION BY source_id ORDER BY timestamp) AS prev_ts
    FROM hits
),
grouped AS (
    SELECT *,
        SUM(CASE WHEN prev_ts IS NULL OR timestamp - prev_ts > :gap THEN 1 ELSE 0 END)
            OVER (PARTITION BY source_id ORDER BY timestamp) AS window_num
    FROM lagged
),
best AS (
    SELECT *,
        FIRST_VALUE(text) OVER (
            PARTITION BY source_id, window_num
            ORDER BY confidence DESC, timestamp ASC
        ) AS best_text
    FROM grouped
)
SELECT
    v.file,
    v.path,
    MIN(b.timestamp)  AS first_ts,
    MAX(b.timestamp)  AS last_ts,
    COUNT(*)          AS hit_count,
    MAX(b.confidence) AS best_conf,
    b.best_text       AS sample_text
FROM best b
JOIN videos v ON v.source_id = b.source_id
GROUP BY b.source_id, b.window_num
ORDER BY v.file, first_ts
"""

MATCH_KEYS = ("match_id", "scan_id", "source_id", "file", "path", "timestamp", "text", "confidence")


def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def fmt_ts(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_rows(jsonl_path):
    """Parse match rows, skipping malformed or truncated lines instead of raising."""
    rows = []
    bad = 0
    with jsonl_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not all(k in row for k in MATCH_KEYS):
                bad += 1
                continue
            rows.append(row)
    return rows, bad


def load_sidecar(sidecar_path):
    try:
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def file_state(jsonl_path, sidecar_path):
    st = jsonl_path.stat()
    try:
        sidecar_mtime = sidecar_path.stat().st_mtime_ns
    except OSError:
        sidecar_mtime = -1
    return st.st_size, st.st_mtime_ns, sidecar_mtime


def ingest(args):
    src = Path(args.directory)
    jsonl_files = sorted(
        p for p in src.glob("*.jsonl") if not p.name.endswith(".agent.jsonl")
    )
    if not jsonl_files:
        sys.exit(f"no .jsonl files found in {src}")

    conn = connect(args.db)
    known = dict(
        (name, (size, mtime, side))
        for name, size, mtime, side in conn.execute(
            "SELECT source_file, jsonl_size, jsonl_mtime_ns, sidecar_mtime_ns"
            " FROM ingested_files"
        )
    )

    rows_added = 0
    files_done = 0
    skipped_unchanged = 0
    failed = 0
    bad_lines = 0
    missing_sidecars = []

    for jl in jsonl_files:
        sidecar = jl.with_suffix(".json")
        state = file_state(jl, sidecar)
        if known.get(jl.name) == state:
            skipped_unchanged += 1
            continue

        try:
            rows, bad = load_rows(jl)
            bad_lines += bad
            meta = load_sidecar(sidecar) if sidecar.exists() else None
            if meta is None:
                missing_sidecars.append(jl.name)

            with conn:
                if meta is not None:
                    conn.execute(
                        "INSERT OR REPLACE INTO scans VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            jl.name,
                            meta.get("pattern"),
                            meta.get("started_at"),
                            meta.get("completed_at"),
                            meta.get("match_count"),
                            json.dumps(meta.get("options", {})),
                        ),
                    )
                if rows:
                    conn.executemany(
                        "INSERT OR IGNORE INTO scan_runs VALUES (?, ?)",
                        {(r["scan_id"], jl.name) for r in rows},
                    )
                    conn.executemany(
                        "INSERT OR IGNORE INTO videos VALUES (?, ?, ?)",
                        {(r["source_id"], r["file"], r["path"]) for r in rows},
                    )
                    cur = conn.executemany(
                        "INSERT OR IGNORE INTO matches VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            (
                                r["match_id"],
                                r["scan_id"],
                                r["source_id"],
                                r["timestamp"],
                                r["text"],
                                r["confidence"],
                            )
                            for r in rows
                        ],
                    )
                    rows_added += cur.rowcount
                conn.execute(
                    "INSERT OR REPLACE INTO ingested_files VALUES (?, ?, ?, ?)", (jl.name, *state)
                )
            files_done += 1
        except Exception as exc:
            failed += 1
            print(f"warning: failed to ingest {jl.name}: {exc}", file=sys.stderr)

    total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    conn.close()

    print(
        f"ingested {files_done} scan files ({skipped_unchanged} unchanged skipped"
        f"{f', {failed} failed' if failed else ''}): "
        f"+{rows_added} new matches, {total} total across {videos} videos -> {args.db}"
    )
    if bad_lines:
        print(f"warning: skipped {bad_lines} malformed/partial jsonl line(s)", file=sys.stderr)
    if missing_sidecars:
        preview = ", ".join(missing_sidecars[:5])
        more = f" (+{len(missing_sidecars) - 5} more)" if len(missing_sidecars) > 5 else ""
        print(
            f"warning: {len(missing_sidecars)} file(s) have no .json sidecar yet"
            f" (scan still running or interrupted); matches ingested without scan"
            f" metadata: {preview}{more}",
            file=sys.stderr,
        )


def search(args):
    if args.gap < 0:
        sys.exit("--gap must be >= 0")
    if not 0.0 <= args.min_conf <= 1.0:
        sys.exit("--min-conf must be between 0 and 1")
    if not Path(args.db).exists():
        sys.exit(f"database not found: {args.db} (run ingest first)")
    conn = connect(args.db)
    try:
        rows = conn.execute(
            WINDOW_QUERY,
            {"query": args.query, "min_conf": args.min_conf, "gap": args.gap},
        ).fetchall()
    except sqlite3.OperationalError as exc:
        sys.exit(
            f"FTS query error: {exc}\n"
            'hint: quote phrases ("contract started"), use * for prefix, '
            "AND/OR/NEAR for boolean queries"
        )
    finally:
        conn.close()

    if not rows:
        print("no matches")
        return

    for file, path, first_ts, last_ts, hits, conf, sample in rows:
        print(
            f"{file}  {fmt_ts(first_ts)} - {fmt_ts(last_ts)}"
            f"  ({hits} hits, conf {conf:.2f})  {sample!r}"
        )
        if args.paths:
            print(f"    {path}")
    print(f"\n{len(rows)} occurrence window(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="load .jsonl/.json scan pairs into the database")
    p_ing.add_argument("directory", help="directory containing scan output pairs")
    p_ing.add_argument("--db", default="ocr.db", help="database file (default: ocr.db)")
    p_ing.set_defaults(func=ingest)

    p_sea = sub.add_parser("search", help="search text, grouped into occurrence windows")
    p_sea.add_argument("query", help="FTS5 query, e.g. 'contract started' or 'complete*'")
    p_sea.add_argument("--db", default="ocr.db", help="database file (default: ocr.db)")
    p_sea.add_argument(
        "--gap",
        type=float,
        default=10.0,
        help="seconds between hits before a new window starts (default: 10)",
    )
    p_sea.add_argument(
        "--min-conf", type=float, default=0.5, help="minimum OCR confidence (default: 0.5)"
    )
    p_sea.add_argument("--paths", action="store_true", help="print full video paths")
    p_sea.set_defaults(func=search)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
