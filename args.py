import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vidgrep",
        description="Clip video segments where text or an image pattern appears",
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
        "--padding", "-p", type=float, default=5.0, metavar="SEC",
        help="Seconds to include before and after each match interval (default: 5)",
    )
    p.add_argument(
        "--skip-frames", "-s", type=int, default=3, metavar="N",
        help="Analyse every Nth frame – higher = faster scan (default: 3)",
    )
    p.add_argument(
        "--interval", "-i", type=float, metavar="SEC",
        help=(
            "Analyse one frame every SEC seconds instead of every Nth frame. "
            "Frame-rate independent and far faster for long videos "
            "(overrides --skip-frames)."
        ),
    )
    p.add_argument(
        "--batch-size", "-b", type=int, default=8, metavar="N",
        help="Frames per OCR batch – higher = better GPU utilisation, more VRAM (default: 8)",
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Print decode/detect throughput (fps, x-realtime) after each scan",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5, metavar="0-1",
        help="Min OCR confidence or template similarity to count as a match (default: 0.5)",
    )
    p.add_argument(
        "--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="Only scan this rectangle in each frame (speeds up OCR significantly)",
    )
    p.add_argument(
        "--merge-gap", type=float, default=2.0, metavar="SEC",
        help="Merge match intervals separated by less than this many seconds (default: 2)",
    )
    p.add_argument(
        "--min-duration", type=float, default=0.0, metavar="SEC",
        help="Discard matched intervals shorter than this (default: 0 – keep all)",
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
        "--skip-frames", "-s", type=int, default=3, metavar="N",
        help="Analyse every Nth frame (default: 3)",
    )
    p.add_argument(
        "--interval", "-i", type=float, metavar="SEC",
        help=(
            "Analyse one frame every SEC seconds instead of every Nth frame "
            "(frame-rate independent; overrides --skip-frames)."
        ),
    )
    p.add_argument(
        "--batch-size", "-b", type=int, default=8, metavar="N",
        help="Frames per OCR batch – higher = better GPU utilisation, more VRAM (default: 8)",
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Print decode/detect throughput (fps, x-realtime) after each scan",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5, metavar="0-1",
        help="Min OCR confidence to count as a match (default: 0.5)",
    )
    p.add_argument(
        "--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="Only scan this rectangle in each frame",
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
"""


_USAGE = """\
Usage examples
--------------
  # Clip every segment where "GOAL" appears (±5 s padding)
  vidgrep match.mp4 --text "GOAL"

  # Regex search, custom padding, only scan bottom-third of frame
  vidgrep stream.mp4 --text "LIVE" --padding 10 --region 0 720 1280 360

  # Template (image) matching instead of OCR
  vidgrep gameplay.mp4 --template logo.png --threshold 0.85

  # Process every video in a directory (recursive); clips land next to each source
  vidgrep /sports/videos/ --text "GOAL"

  # Multiple hits concatenated into one file, re-encoded with NVENC
  vidgrep movie.mp4 --text "Chapter" --concat --reencode

  # Faster scan: process every 5th frame, merge gaps ≤ 3 s, disable GPU
  vidgrep long.mp4 --text "error" --skip-frames 5 --merge-gap 3 --no-gpu

  # Sample one frame every 2 s (frame-rate independent, much faster on long videos)
  vidgrep long.mp4 --text "error" --interval 2
"""
