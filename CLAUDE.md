# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this project does

`vidgrep` is a CLI tool that scans a video for text via OCR or a visual template
via image matching, then extracts padded clips around each match using FFmpeg.

```bash
vidgrep video.mp4 --text "GOAL" --padding 5
vidgrep video.mp4 --template logo.png --threshold 0.85
vidgrep scan video.mp4 --text "GOAL"
vidgrep inventory --name "GOAL"
vidgrep worker videos.csv --text "GOAL"
vidgrep-region video.mp4
vidgrep-tui
```

Direct script execution from the repo root still works for development:
`python main.py ...` and `python select_region.py ...`.

## Setup

Install CUDA-enabled PyTorch **before** the rest of the dependencies. EasyOCR
picks up the CUDA build at import time:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -e ".[tui]"  # optional Textual dashboard
```

FFmpeg and ffprobe must be available on `PATH`.

## Running

```bash
vidgrep <input> --text <pattern>       # OCR clip mode
vidgrep <input> --template <img>       # template clip mode
vidgrep scan <input> --text <pattern>  # scan-only JSONL + JSON output
vidgrep inventory [input] --name text  # video file inventory CSV + JSON
vidgrep worker <csv> --text <pattern>  # resumable OCR queue from CSV
vidgrep-tui                            # interactive scan dashboard
```

See `vidgrep --help` and `vidgrep scan --help` for all flags.

## Architecture

Flat modules with setuptools console scripts in `pyproject.toml`:

- **`main.py`** — entry point and orchestration. Dispatches the hand-rolled
  `scan` subcommand, constructs a detector and `VideoClipper`, calls
  `find_intervals`, then `extract_clips`.
- **`args.py`** — clip-mode and scan-mode argparse parser definitions.
- **`detector.py`** — `TextDetector` and `TemplateDetector`, both duck-typed on
  `detect(frame) -> bool` and `detect_batch(frames) -> list[bool]`.
- **`clipper.py`** — `VideoClipper`, frame sampling, FFmpeg readers, stats, clip
  extraction, NVENC codec detection, and concat output.
- **`scan.py`** — scan-only subcommand, recursive video discovery, streaming
  JSONL writes, and summary JSON metadata.
- **`inventory.py`** — drive/directory video inventory by extension, with CSV
  records and JSON metadata/totals/skipped path details.
- **`worker.py`** — resumable CSV-backed OCR queue runner that shells out to
  `main.py scan`, streams output, and updates `processed` status in place.
- **`select_region.py`** — interactive helper exposed as `vidgrep-region`.
- **`tui.py`** — optional Textual dashboard exposed as `vidgrep-tui`; uses
  callbacks from `VideoClipper` and detectors instead of parsing console output.

## Key behaviours to be aware of

- Stream-copy cuts snap to keyframes, usually around 2 seconds of inaccuracy.
  Use `--lossless` or `--reencode` for frame-accurate output.
- OCR model loading is lazy: startup is fast, but the first scanned frame is
  slow.
- `--skip-frames N` and `--interval SEC` are the main performance levers.
  Sampling and `--region` cropping happen inside FFmpeg via `_FFSampler` when
  available, so skipped frames and cropped-away pixels never cross the pipe.
- `--lossless` implies re-encoding and uses `libx264 -crf 0`; `--reencode`
  uses NVENC with codec auto-detection from ffprobe.
- `--output` is rejected when the input is a directory; clips are named next to
  each source file.
- `--concat` writes per-interval clips to temporary files, concatenates them
  with `ffmpeg -f concat -safe 0`, then deletes the temps.
- The TUI v1 focuses on OCR text scans. Template matching remains CLI-first.
- Inventory defaults to `yyyy-mm-dd.csv` and `yyyy-mm-dd.json` when `--output`
  is omitted. Existing CSV rows are merged by `path`; new rows append, and
  worker status columns such as `processed` are preserved.
- Inventory `--name` filters filenames by substring. Inventory `--regex` is
  case-insensitive and can match filename, directory, or both.
- Inventory CSVs include `processed=false`; the worker sets it to `true` only on
  successful scans. Failed rows remain false and are retried on resume. Reserve
  `agent` naming for a future AI layer over JSONL outputs.
