"""Inventory video files by walking directories or local drives."""
import csv
import json
import os
import re
import string
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


VIDEO_EXTENSIONS = frozenset(
    {
        ".3g2",
        ".3gp",
        ".asf",
        ".avi",
        ".divx",
        ".f4v",
        ".flv",
        ".m2t",
        ".m2ts",
        ".m4v",
        ".mjpeg",
        ".mjpg",
        ".mkv",
        ".mod",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mts",
        ".mxf",
        ".ogm",
        ".ogv",
        ".rm",
        ".rmvb",
        ".ts",
        ".vob",
        ".webm",
        ".wmv",
    }
)

BASE_CSV_FIELDS = ["filename", "path", "video_format", "size", "date", "processed"]


@dataclass
class VideoRecord:
    filename: str
    path: str
    video_format: str
    size: int
    date: str
    parent: str
    drive: str
    processed: str = "false"


def default_stem(now: datetime) -> str:
    return now.astimezone().strftime("%Y-%m-%d")


def local_roots() -> list[Path]:
    if os.name == "nt":
        return [
            root
            for letter in string.ascii_uppercase
            if (root := Path(f"{letter}:\\")).exists() and _is_local_drive(root)
        ]
    return [Path("/")]


def _is_local_drive(root: Path) -> bool:
    """True for fixed and removable drives; excludes network/CD-ROM/RAM."""
    DRIVE_REMOVABLE, DRIVE_FIXED = 2, 3
    try:
        import ctypes
        drive_type = ctypes.windll.kernel32.GetDriveTypeW(str(root))
    except (ImportError, AttributeError, OSError):
        return True
    return drive_type in (DRIVE_REMOVABLE, DRIVE_FIXED)


def resolve_roots(raw_roots: Optional[list[str]]) -> list[Path]:
    if not raw_roots:
        return local_roots()

    roots: list[Path] = []
    for raw in raw_roots:
        path = Path(raw)
        if not path.exists():
            sys.exit(f"Error: not found: {path}")
        roots.append(path)
    return roots


def iter_video_records(
    roots: Iterable[Path],
    *,
    name_filter: Optional[str] = None,
    regex_filter: Optional[re.Pattern[str]] = None,
    regex_scope: str = "both",
) -> tuple[list[VideoRecord], list[dict]]:
    needle = name_filter.casefold() if name_filter else None
    records: list[VideoRecord] = []
    skipped: list[dict] = []

    for root in roots:
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                if current.is_file():
                    _append_record(records, current, root, needle, regex_filter, regex_scope)
                    continue
                with os.scandir(current) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                _append_record(
                                    records, Path(entry.path), root, needle, regex_filter, regex_scope
                                )
                        except OSError as exc:
                            skipped.append({"path": entry.path, "error": str(exc)})
            except OSError as exc:
                skipped.append({"path": str(current), "error": str(exc)})

    records.sort(key=lambda item: item.path.casefold())
    return records, skipped


def _append_record(
    records: list[VideoRecord],
    path: Path,
    root: Path,
    needle: Optional[str],
    regex_filter: Optional[re.Pattern[str]],
    regex_scope: str,
) -> None:
    suffix = path.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        return
    if needle and needle not in path.name.casefold():
        return
    if regex_filter and not _regex_matches(path, regex_filter, regex_scope):
        return

    try:
        stat = path.stat()
    except OSError:
        return

    modified = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
    records.append(
        VideoRecord(
            filename=path.name,
            path=str(path.resolve()),
            video_format=suffix.lstrip("."),
            size=stat.st_size,
            date=modified,
            parent=str(path.parent.resolve()),
            drive=_drive_for(path, root),
        )
    )


def _drive_for(path: Path, root: Path) -> str:
    if os.name == "nt":
        resolved_path = path.resolve()
        resolved_root = root.resolve()
        return resolved_path.drive or resolved_root.drive
    return root.anchor or str(root)


def _regex_matches(path: Path, pattern: re.Pattern[str], scope: str) -> bool:
    parent = str(path.parent)
    if scope == "filename":
        return pattern.search(path.name) is not None
    if scope == "directory":
        return pattern.search(parent) is not None
    return pattern.search(path.name) is not None or pattern.search(parent) is not None


def write_csv(path: Path, records: list[VideoRecord]) -> list[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, fieldnames = _merge_existing_csv(path, records)
    # Write to a temp file and replace atomically: this CSV carries the
    # worker's processed flags, so an interrupted rewrite must not destroy it.
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
    return rows


def _merge_existing_csv(path: Path, records: list[VideoRecord]) -> tuple[list[dict], list[str]]:
    fieldnames = list(BASE_CSV_FIELDS)
    existing_rows: list[dict] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames:
                fieldnames = list(dict.fromkeys([*reader.fieldnames, *BASE_CSV_FIELDS]))
                existing_rows = list(reader)

    rows_by_path = {_path_key(row.get("path", "")): row for row in existing_rows if row.get("path")}
    ordered_keys = [_path_key(row.get("path", "")) for row in existing_rows if row.get("path")]

    for record in records:
        row = _record_to_row(record)
        key = _path_key(record.path)
        if key in rows_by_path:
            existing_processed = rows_by_path[key].get("processed")
            rows_by_path[key].update(row)
            if existing_processed:
                rows_by_path[key]["processed"] = existing_processed
        else:
            rows_by_path[key] = row
            ordered_keys.append(key)

    rows = [rows_by_path[key] for key in ordered_keys]
    for row in rows:
        if not row.get("processed"):
            row["processed"] = "false"
        for field in fieldnames:
            row.setdefault(field, "")
    return rows, fieldnames


def _record_to_row(record: VideoRecord) -> dict:
    return {
        "filename": record.filename,
        "path": record.path,
        "video_format": record.video_format,
        "size": str(record.size),
        "date": record.date,
        "processed": record.processed,
    }


def _path_key(path: str) -> str:
    return path.casefold()


def write_json(
    path: Path,
    records: list[VideoRecord],
    *,
    started_at: datetime,
    completed_at: datetime,
    roots: list[Path],
    name_filter: Optional[str],
    regex_filter: Optional[str],
    regex_scope: str,
    skipped: list[dict],
    csv_row_count: int,
) -> None:
    payload = {
        "generated_at": completed_at.isoformat(),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "roots": [str(root.resolve()) for root in roots],
        "name_filter": name_filter,
        "regex_filter": regex_filter,
        "regex_scope": regex_scope,
        "extensions": sorted(VIDEO_EXTENSIONS),
        "file_count": len(records),
        "csv_row_count": csv_row_count,
        "total_size_bytes": sum(record.size for record in records),
        "skipped_count": len(skipped),
        "skipped": skipped[:500],
        "skipped_truncated": len(skipped) > 500,
        "files": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_inventory(args) -> None:
    started_at = datetime.now(timezone.utc)
    regex_filter = None
    if args.regex:
        try:
            regex_filter = re.compile(args.regex, re.IGNORECASE)
        except re.error as exc:
            sys.exit(f"Error: invalid --regex: {exc}")

    roots = resolve_roots(args.inputs)
    stem = args.output if args.output else default_stem(started_at)
    csv_path = Path(stem).with_suffix(".csv")
    json_path = Path(stem).with_suffix(".json")

    print(f"Scanning {len(roots)} root(s):")
    for root in roots:
        print(f"  {root}")
    if args.name:
        print(f"Filtering filenames containing: {args.name!r}")
    if args.regex:
        print(f"Filtering {args.regex_scope} with regex: {args.regex!r}")

    records, skipped = iter_video_records(
        roots,
        name_filter=args.name,
        regex_filter=regex_filter,
        regex_scope=args.regex_scope,
    )
    completed_at = datetime.now(timezone.utc)

    csv_rows = write_csv(csv_path, records)
    write_json(
        json_path,
        records,
        started_at=started_at,
        completed_at=completed_at,
        roots=roots,
        name_filter=args.name,
        regex_filter=args.regex,
        regex_scope=args.regex_scope,
        skipped=skipped,
        csv_row_count=len(csv_rows),
    )

    print(f"\nFound {len(records)} video file(s).")
    if len(csv_rows) != len(records):
        print(f"CSV now contains {len(csv_rows)} row(s) after merging existing entries.")
    if skipped:
        print(f"Skipped {len(skipped)} path(s) due to access/read errors.")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
