# AGENTS.md

High-signal notes for AI agents. `CLAUDE.md` has the longer architecture overview; this file covers what's easy to get wrong.

## What this is

`vidgrep` — a CLI that scans a video for text (EasyOCR) or an image template (OpenCV cross-correlation) and extracts padded clips around each match via FFmpeg. Flat layout with `pyproject.toml` console scripts:

```bash
vidgrep video.mp4 --text "GOAL"        # clip mode
vidgrep scan video.mp4 --text "GOAL"   # scan-only: writes JSONL + JSON, no clips
vidgrep-region video.mp4               # interactive helper to pick a --region box
```

Direct script execution from the repo root still works for development:
`python main.py ...` and `python select_region.py ...`.

## Setup — install order matters

CUDA PyTorch must be installed **before** `pip install -e .`. EasyOCR binds to whatever torch build is present at import time; a CPU wheel makes OCR silently fall back to CPU.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Use the **cu128** wheel for RTX 50-series / Blackwell (`sm_120`); `cu121`/`cu118` won't drive them. FFmpeg **and** `ffprobe` must be on `PATH` (decode, clip extraction, codec detection).

## Verification — there is no test suite, linter, or typechecker

No `pytest`/`tox`/CI exists. The verification loop is:

```bash
python -m py_compile main.py args.py clipper.py detector.py scan.py select_region.py
python main.py --help && python main.py scan --help
vidgrep --help && vidgrep scan --help && vidgrep-region --help
```

`test_commands.md` is a set of manual command templates (placeholders in ALL_CAPS), not an automated suite.

## Architecture gotchas not obvious from filenames

- **Subcommand dispatch is hand-rolled**, not argparse subparsers. `main.py` peeks at `sys.argv[1] == "scan"` and calls `scan.run_scan` with a separate `build_scan_parser()`. If adding a subcommand, follow this pattern or migrate both deliberately.
- **Two modes share `VideoClipper`**: clip mode → `find_intervals()` + `extract_clips()`; scan mode → `scan_for_matches()` with an `on_match` callback that flushes each match to JSONL immediately (don't buffer the whole video).
- **Detectors are duck-typed** on `detect(frame)->bool` and `detect_batch(frames)->list[bool]`. `TextDetector` also exposes `detect_matches[_batch]` returning `{"timestamp","text","confidence"}` dicts, used only by scan mode. A new detector must implement both `detect` and `detect_batch`.
- **Frame sampling happens inside FFmpeg when possible.** Both modes read frames via `VideoClipper._iter_samples()`. For `skip_frames > 1` it uses `_FFSampler` (`clipper.py`): an `ffmpeg` subprocess with `select` + `crop` filters, so skipped frames and cropped-away pixels never cross the pipe — this is what makes sparse scans (`--interval` / large `--skip-frames`) fast. Frames from this path arrive **already cropped**; don't re-apply `region`.
- **Fallback chain if the sampler fails** (e.g. FFmpeg too old for `-fps_mode`): in-process decoding where every frame is touched — sampled frames `read()`, skipped frames `grab()`-ed — via `cv2.VideoCapture(CAP_FFMPEG)`, or `_SWCapture` (raw BGR piped from `ffmpeg -hwaccel none`) when the init probe `read()` fails. Don't assume `cv2.VideoCapture` is the only reader.
- **Sampled frames go through `detector.detect_batch`** in `batch_size` groups regardless of which reader produced them.
- **OCR model loads lazily** on the first `detect()` — startup is fast, first sampled frame is slow. A "GPU requested but CUDA unavailable" warning at load means a CPU torch wheel is installed.

## Behavioural constraints

- **`--lossless` implies `--reencode`** (wired in `main.py` as `reencode=args.reencode or args.lossless`). `--lossless` → `libx264 -crf 0`; `--reencode` → NVENC, codec auto-detected via `ffprobe` (`h264_nvenc`/`hevc_nvenc`/`av1_nvenc`, default `h264_nvenc`).
- **Stream-copy (default) snaps cuts to keyframes** (~2s inaccuracy). Use `--lossless` or `--reencode` for frame-accurate output.
- **`--output` is rejected when input is a directory** — clips are named per-file next to each source. Directory inputs are walked recursively for the extensions in `scan.py:VIDEO_EXTENSIONS`.
- **`--concat`** writes per-interval clips to tempfiles, concatenates with `ffmpeg -f concat -safe 0`, then deletes the temps.

## Files

`main.py` (entry + orchestration only) · `args.py` (both arg parsers + usage epilogs) · `detector.py` (`TextDetector`, `TemplateDetector`) · `clipper.py` (`VideoClipper`, `_FFSampler`, `_SWCapture`, all FFmpeg calls) · `scan.py` (scan subcommand, `find_video_files`, `VIDEO_EXTENSIONS`) · `select_region.py` (standalone region picker). `scrap/` is gitignored; scan-generated `*.jsonl`/`*.json` at the repo root are scratch — don't commit them.
