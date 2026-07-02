# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

`clip` is a CLI tool that scans a video for text (via OCR) or a visual template (image matching), then extracts padded clips around each match using FFmpeg.

```
python main.py video.mp4 --text "GOAL" --padding 5
python main.py video.mp4 --template logo.png --threshold 0.85
```

## Setup

Install CUDA-enabled PyTorch **before** the rest of the dependencies — EasyOCR picks up the CUDA build at import time:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

FFmpeg must be available on `PATH` (used for both decoding and clip extraction).

## Running

```bash
python main.py <input> --text <pattern>   # OCR mode
python main.py <input> --template <img>   # template-matching mode
```

See `python main.py --help` or the module docstring in `main.py` for all flags.

## Architecture

Three modules, no framework:

- **`main.py`** — argument parsing and orchestration only. Constructs a detector and a `VideoClipper`, calls `find_intervals`, then `extract_clips`.

- **`detector.py`** — two detector classes with a shared `detect(frame) -> bool` duck-typed interface:
  - `TextDetector`: runs EasyOCR (lazy-loaded on first call) and matches OCR results against a compiled regex.
  - `TemplateDetector`: normalised cross-correlation via OpenCV. Tries `cv2.cuda` on init and silently falls back to CPU.

- **`clipper.py`** — `VideoClipper` owns all video I/O:
  - `find_intervals()` iterates sampled frames, applies the detector, and merges nearby hit windows.
  - Frame sampling goes through `_iter_samples()`: when `skip_frames > 1` it uses `_FFSampler`, an `ffmpeg` subprocess with `select`+`crop` filters so skipped frames are dropped (and the region cropped) inside FFmpeg — only sampled crops cross the pipe. If that fails (e.g. old FFmpeg), it falls back to in-process decoding via `cv2.VideoCapture(path, cv2.CAP_FFMPEG)` or `_SWCapture` (raw BGR piped from `ffmpeg`) when hardware decoding fails on init.
  - `extract_clips()` calls FFmpeg for each interval. Default is stream-copy (fast, keyframe-accurate). `--reencode` uses NVENC (codec auto-detected from the source stream); `--lossless` uses `libx264 -crf 0`.

## Key behaviours to be aware of

- **Stream-copy cuts snap to keyframes** (~2 s inaccuracy). Use `--lossless` or `--reencode` for frame-accurate output.
- **OCR model loads lazily** on the first `detect()` call, so startup is fast but the first scanned frame is slow.
- `--skip-frames N` (default 3) / `--interval SEC` are the main performance levers. When sampling (N > 1), frame dropping and `--region` cropping happen inside the FFmpeg subprocess (`_FFSampler`), so skipped frames cost almost nothing — sparse scans are bounded by FFmpeg decode speed plus OCR of the sampled frames only.
- `--region X Y W H` crops frames before OCR (inside FFmpeg on the sampled path), which dramatically speeds up text detection on large frames.
- `--concat` writes per-interval clips to `tempfile` paths and then concatenates with `ffmpeg -f concat`, deleting the temps afterwards.
