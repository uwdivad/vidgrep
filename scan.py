"""scan subcommand: scan video files for text matches and write paired JSONL + JSON output."""
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List

import cv2


VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".ts", ".mts"}
)


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
) -> int:
    """Scan one video with an existing detector and stream JSONL-ready records."""
    from clipper import VideoClipper

    clipper = VideoClipper(str(video_path))
    resolved_path = str(video_path.resolve())
    scan_id = str(uuid.uuid4())
    source_id = _make_source_id(video_path)
    region = tuple(args.region) if args.region else None
    match_count = 0

    def write_match(m):
        nonlocal match_count
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
        on_record(record)
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
    )
    return match_count


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
    jsonl_path = Path(stem).with_suffix(".jsonl")
    meta_path = Path(stem).with_suffix(".json")
    options_id = _scan_options_id(args)

    use_gpu = not args.no_gpu
    languages = [lang.strip() for lang in args.lang.split(",")]
    detector = TextDetector(args.text, gpu=use_gpu, threshold=args.threshold, languages=languages)

    match_count = 0
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for video_path in all_video_files:
            print(f"\nScanning '{video_path.name}' …")

            def write_record(record):
                jsonl_file.write(json.dumps(record) + "\n")
                jsonl_file.flush()

            try:
                match_count += scan_video(
                    video_path=video_path,
                    detector=detector,
                    args=args,
                    options_id=options_id,
                    on_record=write_record,
                )
            except ValueError as exc:
                print(f"Warning: skipping '{video_path}': {exc}", file=sys.stderr)

    metadata = scan_metadata(
        args=args,
        started_at=started_at,
        inputs=args.inputs,
        files_scanned=len(all_video_files),
        match_count=match_count,
    )
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nWrote {match_count} match(es) across {len(all_video_files)} file(s).")
    print(f"  Matches : {jsonl_path}")
    print(f"  Metadata: {meta_path}")
