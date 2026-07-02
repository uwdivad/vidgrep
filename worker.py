"""Resumable CSV-backed OCR scan worker."""
import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


JOB_COLUMNS = [
    "processed",
    "last_attempt_at",
    "completed_at",
    "output_stem",
    "jsonl_path",
    "json_path",
    "match_count",
    "exit_code",
    "error",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_processed(row: dict) -> bool:
    return str(row.get("processed", "")).casefold() == "true"


def _sanitize_stem(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", value).strip("._")
    return value[:80] or "video"


def _load_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            sys.exit(f"Error: empty CSV: {path}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if "path" not in fieldnames:
        sys.exit("Error: CSV must include a 'path' column.")

    for column in JOB_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    for row in rows:
        if not row.get("processed"):
            row["processed"] = "false"
        for column in JOB_COLUMNS:
            row.setdefault(column, "")

    return rows, fieldnames


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as fh:
        tmp_path = Path(fh.name)
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _build_scan_command(args, video_path: Path, output_stem: Path) -> list[str]:
    main_path = Path(__file__).with_name("main.py")
    cmd = [
        sys.executable,
        "-u",
        str(main_path),
        "scan",
        str(video_path),
        "--text",
        args.text,
        "--output",
        str(output_stem),
        "--skip-frames",
        str(args.skip_frames),
        "--batch-size",
        str(args.batch_size),
        "--threshold",
        str(args.threshold),
        "--lang",
        args.lang,
    ]

    if args.interval is not None:
        cmd.extend(["--interval", str(args.interval)])
    if args.region:
        cmd.extend(["--region", *(str(part) for part in args.region)])
    if args.no_gpu:
        cmd.append("--no-gpu")
    if args.stats:
        cmd.append("--stats")
    return cmd


def _run_streaming(cmd: list[str]) -> tuple[int, str]:
    tail: deque[str] = deque(maxlen=30)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line.rstrip())
        return proc.wait(), "\n".join(tail)
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise


def _read_match_count(json_path: Path) -> str:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("match_count", ""))


def _output_stem_for(args, csv_path: Path, row_index: int, video_path: Path) -> Path:
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.with_suffix("").parent / f"{csv_path.stem}_ocr"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{row_index:06d}_{_sanitize_stem(video_path.stem)}"


def _mark_attempt_started(row: dict, output_stem: Path) -> None:
    row["processed"] = "false"
    row["last_attempt_at"] = _now()
    row["completed_at"] = ""
    row["output_stem"] = str(output_stem)
    row["jsonl_path"] = str(output_stem.with_suffix(".jsonl"))
    row["json_path"] = str(output_stem.with_suffix(".json"))
    row["match_count"] = ""
    row["exit_code"] = ""
    row["error"] = ""


def _mark_success(row: dict, output_stem: Path, exit_code: int) -> None:
    row["processed"] = "true"
    row["completed_at"] = _now()
    row["exit_code"] = str(exit_code)
    row["error"] = ""
    row["match_count"] = _read_match_count(output_stem.with_suffix(".json"))


def _mark_failure(row: dict, exit_code: Optional[int], error: str) -> None:
    row["processed"] = "false"
    row["completed_at"] = _now()
    row["exit_code"] = "" if exit_code is None else str(exit_code)
    row["error"] = error.strip()[:2000]


def run_worker(args) -> None:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"Error: not found: {csv_path}")

    rows, fieldnames = _load_csv(csv_path)
    _write_csv(csv_path, rows, fieldnames)

    attempted = 0
    skipped_done = 0
    succeeded = 0
    failed = 0

    for index, row in enumerate(rows, 1):
        if _is_processed(row) and not args.force:
            skipped_done += 1
            continue
        if args.limit is not None and attempted >= args.limit:
            break

        attempted += 1
        raw_path = (row.get("path") or "").strip()
        video_path = Path(raw_path)
        output_stem = _output_stem_for(args, csv_path, index, video_path)
        _mark_attempt_started(row, output_stem)
        _write_csv(csv_path, rows, fieldnames)

        print(f"\n[{index}/{len(rows)}] {raw_path}", flush=True)
        if not raw_path:
            _mark_failure(row, None, "missing path value")
            _write_csv(csv_path, rows, fieldnames)
            failed += 1
            print("Error: missing path value", flush=True)
            continue
        if not video_path.exists():
            _mark_failure(row, None, f"not found: {video_path}")
            _write_csv(csv_path, rows, fieldnames)
            failed += 1
            print(f"Error: not found: {video_path}", flush=True)
            continue

        cmd = _build_scan_command(args, video_path, output_stem)
        print("Running: " + " ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)

        try:
            exit_code, tail = _run_streaming(cmd)
        except KeyboardInterrupt:
            _mark_failure(row, None, "interrupted")
            _write_csv(csv_path, rows, fieldnames)
            print("\nInterrupted. CSV progress was saved.", flush=True)
            raise

        if exit_code == 0:
            _mark_success(row, output_stem, exit_code)
            succeeded += 1
        else:
            _mark_failure(row, exit_code, tail or f"scan exited with {exit_code}")
            failed += 1
        _write_csv(csv_path, rows, fieldnames)

    print(
        f"\nWorker complete: attempted={attempted}, succeeded={succeeded}, "
        f"failed={failed}, skipped_done={skipped_done}",
        flush=True,
    )
    print(f"CSV updated: {csv_path}", flush=True)
