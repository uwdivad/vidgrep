"""scan subcommand: scan video files for text matches and write paired JSONL + JSON output."""
import csv
import hashlib
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import cv2


VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".ts", ".mts"}
)

PROFILE_COLUMNS = [
    "started_at",
    "completed_at",
    "path",
    "file",
    "width",
    "height",
    "fps",
    "duration_sec",
    "frame_count",
    "skip_frames",
    "interval",
    "batch_size",
    "threshold",
    "region",
    "lang",
    "gpu_requested",
    "total_sec",
    "sample_read_sec",
    "ocr_sec",
    "write_sec",
    "sampled_frames",
    "batch_count",
    "avg_batch_size",
    "match_count",
    "covered_frames",
    "covered_fps",
    "detect_fps",
    "realtime",
]


def _make_stem(pattern: str, now: datetime) -> str:
    safe_pattern = re.sub(r"[^\w\-]", "_", pattern)[:40]
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"scan_{safe_pattern}_{timestamp}"


def _hash_json(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_source_id(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    return _hash_json({
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    })


def _scan_options_id(args) -> str:
    return _hash_json({
        "text": args.text,
        "threshold": args.threshold,
        "skip_frames": args.skip_frames,
        "interval": args.interval,
        "region": list(args.region) if args.region else None,
        "lang": args.lang,
    })


def _make_match_id(
    *,
    source_id: str,
    scan_options_id: str,
    timestamp: float,
    text: str,
    confidence: float,
) -> str:
    return _hash_json({
        "source_id": source_id,
        "scan_options_id": scan_options_id,
        "timestamp": f"{timestamp:.3f}",
        "text": text,
        "confidence": f"{confidence:.4f}",
    })


def write_profile_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return

    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PROFILE_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PROFILE_COLUMNS})


def _metrics_path(args, stem: Path) -> Path:
    raw_path = getattr(args, "metrics_output", None)
    if raw_path:
        return Path(raw_path)
    return Path(f"{stem}.profile.csv")


def validate_video(path: Path) -> bool:
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return False
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    ok = cap.isOpened()
    cap.release()
    return ok


def find_video_files(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS)
    if path.is_file():
        return [path] if validate_video(path) else []
    raise FileNotFoundError(f"Path does not exist: {path}")


def scan_video(
    *,
    video_path: Path,
    detector,
    args,
    options_id: str,
    on_record: Callable[[dict], None],
) -> tuple[int, Optional[dict]]:
    """Scan one video with an existing detector and stream JSONL-ready records."""
    from clipper import VideoClipper

    started_at = datetime.now(timezone.utc)
    total_started = time.perf_counter()
    clipper = VideoClipper(str(video_path))
    resolved_path = str(video_path.resolve())
    scan_id = str(uuid.uuid4())
    source_id = _make_source_id(video_path)
    region = tuple(args.region) if args.region else None
    match_count = 0
    write_seconds = 0.0
    profile = {} if getattr(args, "profile", False) else None

    def write_match(m):
        nonlocal match_count, write_seconds
        record = {
            "match_id": _make_match_id(
                source_id=source_id,
                scan_options_id=options_id,
                timestamp=m["timestamp"],
                text=m["text"],
                confidence=m["confidence"],
            ),
            "scan_id": scan_id,
            "source_id": source_id,
            "file": video_path.name,
            "path": resolved_path,
            "timestamp": m["timestamp"],
            "text": m["text"],
            "confidence": m["confidence"],
        }
        write_started = time.perf_counter()
        on_record(record)
        write_seconds += time.perf_counter() - write_started
        match_count += 1

    if args.interval:
        skip_frames = max(1, round(args.interval * clipper.fps))
        print(
            f"Sampling 1 frame every {args.interval:g}s "
            f"(every {skip_frames} frames at {clipper.fps:.2f} fps)."
        )
    else:
        skip_frames = args.skip_frames

    clipper.scan_for_matches(
        detector, skip_frames=skip_frames, region=region,
        batch_size=args.batch_size, collect_stats=args.stats,
        on_match=write_match,
        profile=profile,
    )
    completed_at = datetime.now(timezone.utc)
    if profile is None:
        return match_count, None

    total_sec = time.perf_counter() - total_started
    sampled_frames = int(profile.get("sampled_frames", 0))
    covered_frames = min(sampled_frames * skip_frames, clipper.frame_count)
    return match_count, {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "path": resolved_path,
        "file": video_path.name,
        "width": clipper._w,
        "height": clipper._h,
        "fps": round(clipper.fps, 4),
        "duration_sec": round(clipper.duration, 4),
        "frame_count": clipper.frame_count,
        "skip_frames": skip_frames,
        "interval": args.interval if args.interval is not None else "",
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "region": json.dumps(list(region)) if region else "",
        "lang": args.lang,
        "gpu_requested": not args.no_gpu,
        "total_sec": round(total_sec, 6),
        "sample_read_sec": profile.get("sample_read_sec", 0.0),
        "ocr_sec": profile.get("ocr_sec", 0.0),
        "write_sec": round(write_seconds, 6),
        "sampled_frames": sampled_frames,
        "batch_count": profile.get("batch_count", 0),
        "avg_batch_size": profile.get("avg_batch_size", 0.0),
        "match_count": match_count,
        "covered_frames": covered_frames,
        "covered_fps": round(covered_frames / total_sec, 4) if total_sec > 0 else 0.0,
        "detect_fps": round(sampled_frames / total_sec, 4) if total_sec > 0 else 0.0,
        "realtime": round((covered_frames / total_sec) / clipper.fps, 4) if total_sec > 0 and clipper.fps else 0.0,
    }


def scan_metadata(*, args, started_at: datetime, inputs: list[str], files_scanned: int, match_count: int) -> dict:
    completed_at = datetime.now(timezone.utc)
    return {
        "pattern": args.text,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "files_scanned": files_scanned,
        "match_count": match_count,
        "inputs": [str(Path(i).resolve()) for i in inputs],
        "options": {
            "threshold": args.threshold,
            "skip_frames": args.skip_frames,
            "interval": args.interval,
            "region": list(args.region) if args.region else None,
            "lang": args.lang,
        },
    }


def run_scan(args) -> None:
    from detector import TextDetector

    started_at = datetime.now(timezone.utc)

    # Collect and deduplicate video files
    all_video_files: List[Path] = []
    seen: set = set()
    for raw in args.inputs:
        p = Path(raw)
        if not p.exists():
            sys.exit(f"Error: not found: {p}")
        try:
            files = find_video_files(p)
        except FileNotFoundError as exc:
            sys.exit(f"Error: {exc}")
        if not files:
            print(f"Warning: no valid video files found at '{p}'", file=sys.stderr)
        for f in files:
            resolved = f.resolve()
            if resolved not in seen:
                seen.add(resolved)
                all_video_files.append(f)

    if not all_video_files:
        sys.exit("Error: no valid video files to scan.")

    print(f"Found {len(all_video_files)} video file(s) to scan.")

    stem = args.output if args.output else _make_stem(args.text, started_at)
    # Append extensions rather than with_suffix() so a stem containing dots
    # (e.g. --output results.v2) doesn't lose part of its name.
    jsonl_path = Path(f"{stem}.jsonl")
    meta_path = Path(f"{stem}.json")
    metrics_path = _metrics_path(args, Path(stem))
    options_id = _scan_options_id(args)

    use_gpu = not args.no_gpu
    languages = [lang.strip() for lang in args.lang.split(",")]
    detector = TextDetector(args.text, gpu=use_gpu, threshold=args.threshold, languages=languages)

    match_count = 0
    profile_rows: list[dict] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for video_path in all_video_files:
            print(f"\nScanning '{video_path.name}' …")

            def write_record(record):
                jsonl_file.write(json.dumps(record) + "\n")
                jsonl_file.flush()

            try:
                video_match_count, profile_row = scan_video(
                    video_path=video_path,
                    detector=detector,
                    args=args,
                    options_id=options_id,
                    on_record=write_record,
                )
                match_count += video_match_count
                if profile_row is not None:
                    profile_rows.append(profile_row)
            except ValueError as exc:
                print(f"Warning: skipping '{video_path}': {exc}", file=sys.stderr)

    metadata = scan_metadata(
        args=args,
        started_at=started_at,
        inputs=args.inputs,
        files_scanned=len(all_video_files),
        match_count=match_count,
    )
    if profile_rows:
        write_profile_metrics(metrics_path, profile_rows)
        metadata["profile_metrics"] = str(metrics_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nWrote {match_count} match(es) across {len(all_video_files)} file(s).")
    print(f"  Matches : {jsonl_path}")
    print(f"  Metadata: {meta_path}")
    if profile_rows:
        print(f"  Profile : {metrics_path}")
