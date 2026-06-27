"""scan subcommand: scan video files for text matches and write paired JSONL + JSON output."""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import cv2


VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".ts", ".mts"}
)


def _make_stem(pattern: str, now: datetime) -> str:
    safe_pattern = re.sub(r"[^\w\-]", "_", pattern)[:40]
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"scan_{safe_pattern}_{timestamp}"


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


def run_scan(args) -> None:
    from detector import TextDetector
    from clipper import VideoClipper

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

    use_gpu = not args.no_gpu
    languages = [lang.strip() for lang in args.lang.split(",")]
    detector = TextDetector(args.text, gpu=use_gpu, threshold=args.threshold, languages=languages)
    region = tuple(args.region) if args.region else None

    match_count = 0
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for video_path in all_video_files:
            print(f"\nScanning '{video_path.name}' …")
            try:
                clipper = VideoClipper(str(video_path))
            except ValueError as exc:
                print(f"Warning: skipping '{video_path}': {exc}", file=sys.stderr)
                continue

            frame_matches = clipper.scan_for_matches(
                detector, skip_frames=args.skip_frames, region=region
            )
            for m in frame_matches:
                record = {
                    "file": video_path.name,
                    "path": str(video_path.resolve()),
                    "timestamp": m["timestamp"],
                    "text": m["text"],
                    "confidence": m["confidence"],
                }
                jsonl_file.write(json.dumps(record) + "\n")
                jsonl_file.flush()
                match_count += 1

    completed_at = datetime.now(timezone.utc)
    metadata = {
        "pattern": args.text,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "files_scanned": len(all_video_files),
        "match_count": match_count,
        "inputs": [str(Path(i).resolve()) for i in args.inputs],
        "options": {
            "threshold": args.threshold,
            "skip_frames": args.skip_frames,
            "region": list(args.region) if args.region else None,
            "lang": args.lang,
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nWrote {match_count} match(es) across {len(all_video_files)} file(s).")
    print(f"  Matches : {jsonl_path}")
    print(f"  Metadata: {meta_path}")
