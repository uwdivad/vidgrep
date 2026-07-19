import argparse
import math

from vidgrep.agent import DEFAULT_MODEL


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vidgrep",
        description=(
            "Clip video segments where text or an image pattern appears. "
            "Scan-only, inventory, queue, and grouping workflows are available "
            "as subcommands (see below)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_USAGE,
    )

    p.add_argument(
        "input",
        help="Input video file, or a directory (searched recursively for videos)",
    )

    det = p.add_mutually_exclusive_group(required=True)
    det.add_argument(
        "--text", "-t", metavar="PATTERN",
        help="Text / regex to search for in each frame (case-insensitive by default)",
    )
    det.add_argument(
        "--template", "-T", metavar="IMAGE",
        help="Image file to locate via template matching",
    )

    p.add_argument(
        "--output", "-o", metavar="PATH",
        help="Output path (default: <stem>_clip[_N]<ext> next to input)",
    )
    p.add_argument(
        "--padding", "-p", type=_nonnegative_float, default=5.0, metavar="SEC",
        help="Seconds to include before and after each match interval (default: 5)",
    )
    p.add_argument(
        "--skip-frames", "-s", type=_positive_int, default=3, metavar="N",
        help="Analyse every Nth frame - higher = faster scan (default: 3)",
    )
    p.add_argument(
        "--interval", "-i", type=_positive_float, metavar="SEC",
        help=(
            "Analyse one frame every SEC seconds instead of every Nth frame. "
            "Frame-rate independent and far faster for long videos "
            "(overrides --skip-frames)."
        ),
    )
    p.add_argument(
        "--batch-size", "-b", type=_positive_int, default=8, metavar="N",
        help="Frames per OCR batch - higher = better GPU utilisation, more VRAM (default: 8)",
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Print decode/detect throughput (fps, x-realtime) after each scan",
    )
    p.add_argument(
        "--threshold", type=_probability, default=0.5, metavar="0-1",
        help="Min OCR confidence or template similarity to count as a match (default: 0.5)",
    )
    p.add_argument(
        "--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help=(
            "Only scan this pixel rectangle in each frame - speeds up OCR "
            "significantly (pick one interactively with vidgrep-region)"
        ),
    )
    p.add_argument(
        "--merge-gap", type=_nonnegative_float, default=2.0, metavar="SEC",
        help="Merge match intervals separated by less than this many seconds (default: 2)",
    )
    p.add_argument(
        "--min-duration", type=_nonnegative_float, default=0.0, metavar="SEC",
        help="Discard matched intervals shorter than this (default: 0 - keep all)",
    )
    p.add_argument(
        "--lang", default="en", metavar="CODES",
        help="Comma-separated EasyOCR language codes, e.g. en,fr (default: en)",
    )
    p.add_argument(
        "--no-gpu", action="store_true",
        help="Disable CUDA and run everything on CPU",
    )
    p.add_argument(
        "--reencode", action="store_true",
        help=(
            "Re-encode output with NVENC (h264_nvenc / hevc_nvenc / av1_nvenc). "
            "Slower but produces frame-accurate cuts.  Default: stream-copy (fast)."
        ),
    )
    p.add_argument(
        "--lossless", action="store_true",
        help=(
            "Re-encode with libx264 -crf 0 (lossless H.264).  Frame-accurate cuts, "
            "larger files.  Implies --reencode."
        ),
    )
    p.add_argument(
        "--concat", action="store_true",
        help="Concatenate all clips into a single output file",
    )

    return p


def build_scan_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vidgrep scan",
        description="Scan video files for text matches and write results to JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_SCAN_USAGE,
    )
    p.add_argument(
        "inputs", nargs="+", metavar="INPUT",
        help="Video files or directories to scan (directories searched recursively)",
    )
    p.add_argument(
        "--text", "-t", metavar="PATTERN", required=True,
        help="Text / regex to search for in each frame (case-insensitive)",
    )
    p.add_argument(
        "--output", "-o", metavar="STEM",
        help="Output file stem (default: scan_<pattern>_<timestamp> in current dir)",
    )
    p.add_argument(
        "--skip-frames", "-s", type=_positive_int, default=3, metavar="N",
        help="Analyse every Nth frame (default: 3)",
    )
    p.add_argument(
        "--interval", "-i", type=_positive_float, metavar="SEC",
        help=(
            "Analyse one frame every SEC seconds instead of every Nth frame "
            "(frame-rate independent; overrides --skip-frames)."
        ),
    )
    p.add_argument(
        "--batch-size", "-b", type=_positive_int, default=8, metavar="N",
        help="Frames per OCR batch - higher = better GPU utilisation, more VRAM (default: 8)",
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Print decode/detect throughput (fps, x-realtime) after each scan",
    )
    p.add_argument(
        "--profile", action="store_true",
        help="Record per-video profiling metrics for decode/sample, OCR, and writes",
    )
    p.add_argument(
        "--metrics-output", metavar="PATH",
        help="Metrics output path for --profile (default: <output_stem>.profile.csv)",
    )
    p.add_argument(
        "--threshold", type=_probability, default=0.5, metavar="0-1",
        help="Min OCR confidence to count as a match (default: 0.5)",
    )
    p.add_argument(
        "--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help=(
            "Only scan this pixel rectangle in each frame "
            "(pick one interactively with vidgrep-region)"
        ),
    )
    p.add_argument(
        "--lang", default="en", metavar="CODES",
        help="Comma-separated EasyOCR language codes (default: en)",
    )
    p.add_argument(
        "--no-gpu", action="store_true",
        help="Disable CUDA and run everything on CPU",
    )
    return p


def build_inventory_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vidgrep inventory",
        description="Inventory video files and write paired CSV + JSON output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_INVENTORY_USAGE,
    )
    p.add_argument(
        "inputs", nargs="*", metavar="INPUT",
        help=(
            "Optional file or directory roots to scan recursively. "
            "Default: all local drives on Windows, or / elsewhere."
        ),
    )
    p.add_argument(
        "--name", "-n", metavar="TEXT",
        help="Only include video files whose filename contains TEXT (case-insensitive)",
    )
    p.add_argument(
        "--regex", "-r", metavar="PATTERN",
        help=(
            "Only include video files whose filename and/or directory matches PATTERN "
            "(case-insensitive)"
        ),
    )
    p.add_argument(
        "--regex-scope",
        choices=("both", "filename", "directory"),
        default="both",
        help="Where --regex is matched: both, filename, or directory (default: both)",
    )
    p.add_argument(
        "--output", "-o", metavar="STEM",
        help="Output file stem (default: current date, yyyy-mm-dd)",
    )
    return p


def build_worker_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vidgrep worker",
        description="Run resumable OCR scans from an inventory CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_WORKER_USAGE,
    )
    p.add_argument("csv", help="Inventory CSV with a path column")
    p.add_argument(
        "--text", "-t", metavar="PATTERN", required=True,
        help="Text / regex to search for in each video (case-insensitive)",
    )
    p.add_argument(
        "--output-dir", metavar="DIR",
        help="Directory for per-video JSONL/JSON outputs (default: <csv_stem>_ocr)",
    )
    p.add_argument(
        "--limit", type=_positive_int, metavar="N",
        help="Process at most N eligible rows in this run",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Process rows even when processed is true",
    )
    p.add_argument(
        "--skip-frames", "-s", type=_positive_int, default=3, metavar="N",
        help="Analyse every Nth frame when --interval is not set (default: 3)",
    )
    p.add_argument(
        "--interval", "-i", type=_positive_float, metavar="SEC",
        help="Analyse one frame every SEC seconds instead of every Nth frame",
    )
    p.add_argument(
        "--batch-size", "-b", type=_positive_int, default=8, metavar="N",
        help="Frames per OCR batch (default: 8)",
    )
    p.add_argument(
        "--threshold", type=_probability, default=0.5, metavar="0-1",
        help="Min OCR confidence to count as a match (default: 0.5)",
    )
    p.add_argument(
        "--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help=(
            "Only scan this pixel rectangle in each frame "
            "(pick one interactively with vidgrep-region)"
        ),
    )
    p.add_argument(
        "--lang", default="en", metavar="CODES",
        help="Comma-separated EasyOCR language codes (default: en)",
    )
    p.add_argument(
        "--no-gpu", action="store_true",
        help="Disable CUDA and run OCR on CPU",
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Print decode/detect throughput for each scan",
    )
    p.add_argument(
        "--profile", action="store_true",
        help="Record per-video profiling metrics for decode/sample, OCR, and writes",
    )
    p.add_argument(
        "--metrics-output", metavar="PATH",
        help="Metrics output path for --profile (default: <output_dir>/profile.csv)",
    )
    return p


def build_agent_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vidgrep agent",
        description="Group scan JSONL rows into OpenAI-canonicalized text intervals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_AGENT_USAGE,
    )
    p.add_argument(
        "input",
        metavar="INPUT",
        help="Scan JSONL file, directory of JSONL files, or worker CSV with jsonl_path",
    )
    p.add_argument(
        "--search-term", "-t", metavar="TEXT", required=True,
        help="Anchor text used in the original OCR scan, e.g. uwdivad",
    )
    p.add_argument(
        "--output", "-o", metavar="STEM",
        help="Output file stem (default: derived from input)",
    )
    p.add_argument(
        "--merge-gap", type=_nonnegative_float, default=2.0, metavar="SEC",
        help="Merge adjacent rows with the same canonical label within SEC seconds (default: 2)",
    )
    p.add_argument(
        "--openai-model", default=DEFAULT_MODEL, metavar="MODEL",
        help=f"OpenAI model for OCR line canonicalization (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--env-file", default=".env", metavar="PATH",
        help="Environment file to load before reading OPENAI_API_KEY (default: .env)",
    )
    p.add_argument(
        "--no-env-override", action="store_true",
        help="Keep existing environment variables when loading --env-file",
    )
    p.add_argument(
        "--batch-size", type=_positive_int, default=100, metavar="N",
        help="Unique OCR line variants per OpenAI request (default: 100)",
    )
    p.add_argument(
        "--watch", action="store_true",
        help="Poll the input and refresh grouped outputs as new rows arrive",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Refresh OpenAI canonicalizations even when cached state exists",
    )
    p.add_argument(
        "--poll-interval", type=_positive_float, default=2.0, metavar="SEC",
        help="Seconds between --watch refreshes (default: 2)",
    )
    return p


_AGENT_USAGE = """\
Usage examples
--------------
  # Group one scan JSONL into results.agent.jsonl and results.agent.json
  vidgrep agent results.jsonl --search-term uwdivad

  # Group every JSONL under a worker output directory
  vidgrep agent cod_ocr --search-term uwdivad --merge-gap 2

  # Use a specific OpenAI model and output stem
  vidgrep agent results.jsonl --search-term uwdivad --openai-model gpt-5.4-mini --output grouped

  # Refresh grouped output while a scan is still appending rows
  vidgrep agent results.jsonl --search-term uwdivad --watch
"""


_WORKER_USAGE = """\
Usage examples
--------------
  # Process each unprocessed row in an inventory CSV
  vidgrep worker videos.csv --text "uwdivad" --region 132 476 592 388 --interval 2 --batch-size 16 --stats

  # Record per-video profiling metrics while tuning throughput
  vidgrep worker videos.csv --text "uwdivad" --batch-size 32 --profile --metrics-output metrics.csv

  # Put per-video JSONL/JSON output under a named directory
  vidgrep worker cod_videos.csv --text "GOAL" --output-dir cod_ocr

  # Resume naturally by retrying the first processed=false row
  vidgrep worker videos.csv --text "uwdivad"

  # Reprocess every row regardless of processed=true
  vidgrep worker videos.csv --text "uwdivad" --force
"""


_INVENTORY_USAGE = """\
Usage examples
--------------
  # Inventory all local drives; writes yyyy-mm-dd.csv and yyyy-mm-dd.json by default
  vidgrep inventory

  # Inventory one directory
  vidgrep inventory "D:\\Media" --output media_videos

  # Inventory all local drives but only filenames containing "match"
  vidgrep inventory --name "match" --output match_videos

  # Regex can match either filename or directory path by default
  vidgrep inventory --regex "goal|highlight" --output goal_or_highlight

  # Regex only against directory path
  vidgrep inventory --regex "2026\\\\(clips|archive)" --regex-scope directory

  # Inventory multiple roots
  vidgrep inventory "D:\\Media" "E:\\Archive" --name "goal"
"""


_SCAN_USAGE = """\
Usage examples
--------------
  # Scan a single file
  vidgrep scan match.mp4 --text "GOAL" --output results

  # Scan an entire directory recursively
  vidgrep scan /sports/videos/ --text "GOAL" --output goals

  # Mix files and directories, restrict to a region of the frame
  vidgrep scan game1.mp4 /archive/ --text "SCORE" --region 0 810 1440 270

  # Sample one frame every 2 s instead of every Nth frame
  vidgrep scan game1.mp4 --text "GOAL" --interval 2

  # Record per-video profiling metrics for batch-size tuning
  vidgrep scan game1.mp4 --text "GOAL" --batch-size 32 --profile --metrics-output metrics.csv
"""


_USAGE = """\
Subcommands
-----------
  scan       Scan videos for text matches; write JSONL + JSON results, no clipping
  inventory  Find video files across drives/directories; write CSV + JSON
  worker     Resumable OCR scan queue driven by an inventory CSV
  agent      Group scan JSONL rows into canonical text intervals via OpenAI

  Run "vidgrep <subcommand> --help" (or "vidgrep help <subcommand>") for details.
  Companion tools: vidgrep-region (pick a --region box interactively),
  vidgrep-tui (interactive scan dashboard).

Usage examples
--------------
  # Clip every segment where "GOAL" appears (+/-5 s padding)
  vidgrep match.mp4 --text "GOAL"

  # Regex search, custom padding, only scan bottom-third of frame
  vidgrep stream.mp4 --text "LIVE" --padding 10 --region 0 720 1280 360

  # Template (image) matching instead of OCR
  vidgrep gameplay.mp4 --template logo.png --threshold 0.85

  # Process every video in a directory (recursive); clips land next to each source
  vidgrep /sports/videos/ --text "GOAL"

  # Multiple hits concatenated into one file, re-encoded with NVENC
  vidgrep movie.mp4 --text "Chapter" --concat --reencode

  # Faster scan: process every 5th frame, merge gaps <= 3 s, disable GPU
  vidgrep long.mp4 --text "error" --skip-frames 5 --merge-gap 3 --no-gpu

  # Sample one frame every 2 s (frame-rate independent, much faster on long videos)
  vidgrep long.mp4 --text "error" --interval 2
"""
