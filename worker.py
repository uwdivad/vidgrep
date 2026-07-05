"""Resumable CSV-backed OCR scan worker."""
import csv
import json
import re
import sys
import tempfile
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


def _metrics_path_for(args, output_stem: Path) -> Path:
    raw_path = getattr(args, "metrics_output", None)
    if raw_path:
        return Path(raw_path)
    return output_stem.parent / "profile.csv"


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


def _build_detector(args):
    from detector import TextDetector

    languages = [lang.strip() for lang in args.lang.split(",")]
    return TextDetector(
        args.text,
        gpu=not args.no_gpu,
        threshold=args.threshold,
        languages=languages,
    )


def _run_scan_job(args, video_path: Path, output_stem: Path, detector, options_id: str) -> int:
    from scan import scan_metadata, scan_video, write_profile_metrics

    started_at = datetime.now(timezone.utc)
    jsonl_path = output_stem.with_suffix(".jsonl")
    meta_path = output_stem.with_suffix(".json")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        def write_record(record):
            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()

        match_count, profile_row = scan_video(
            video_path=video_path,
            detector=detector,
            args=args,
            options_id=options_id,
            on_record=write_record,
        )

    metadata = scan_metadata(
        args=args,
        started_at=started_at,
        inputs=[str(video_path)],
        files_scanned=1,
        match_count=match_count,
    )
    if profile_row is not None:
        metrics_path = _metrics_path_for(args, output_stem)
        write_profile_metrics(metrics_path, [profile_row])
        metadata["profile_metrics"] = str(metrics_path)
        metadata["profile"] = profile_row
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return match_count


def run_worker(args) -> None:
    from scan import _scan_options_id

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"Error: not found: {csv_path}")

    rows, fieldnames = _load_csv(csv_path)
    _write_csv(csv_path, rows, fieldnames)

    attempted = 0
    skipped_done = 0
    succeeded = 0
    failed = 0
    detector = None
    options_id = _scan_options_id(args)

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

        try:
            if detector is None:
                detector = _build_detector(args)
            _run_scan_job(args, video_path, output_stem, detector, options_id)
        except KeyboardInterrupt:
            _mark_failure(row, None, "interrupted")
            _write_csv(csv_path, rows, fieldnames)
            print("\nInterrupted. CSV progress was saved.", flush=True)
            raise
        except Exception as exc:
            _mark_failure(row, None, str(exc))
            failed += 1
            print(f"Error: {exc}", flush=True)
        else:
            _mark_success(row, output_stem, 0)
            succeeded += 1
        finally:
            _write_csv(csv_path, rows, fieldnames)

    print(
        f"\nWorker complete: attempted={attempted}, succeeded={succeeded}, "
        f"failed={failed}, skipped_done={skipped_done}",
        flush=True,
    )
    print(f"CSV updated: {csv_path}", flush=True)
