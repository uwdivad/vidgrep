# vidgrep test command templates

These are manual command templates, not an automated test suite. Replace
placeholders in ALL_CAPS before running. Examples use PowerShell quoting and
Windows-style paths, but forward slashes also work.

Recommended placeholders:

- `VIDEO.mp4`: a short video that contains the target text at least once
- `NO_MATCH.mp4`: a valid video that does not contain the target text
- `DIR\`: a directory with multiple supported video files, recursively
- `EMPTY_DIR\`: an existing directory with no supported video files
- `LOGO.png`: a small template image that appears in `VIDEO.mp4`
- `PATTERN`: a text string or Python regex to find with EasyOCR
- `REGION`: four integers, `X Y W H`, from `select_region.py`

## 0. Environment and parser smoke tests

```powershell
# Avoid Windows cp1252 stdout failures when argparse prints Unicode help text.
$env:PYTHONIOENCODING = "utf-8"

# Python syntax/import baseline used by this repo
python -m py_compile main.py args.py clipper.py detector.py scan.py select_region.py

# Install the console commands. Install CUDA torch first if testing OCR on GPU.
python -m pip install -e .

# CLI parser smoke tests
python main.py --help
python main.py scan --help
python select_region.py --help

# Installed CLI smoke tests, after: pip install -e .
vidgrep --help
vidgrep scan --help
vidgrep-region --help

# FFmpeg dependencies used for decode, clipping, concat, and codec detection
ffmpeg -version
ffprobe -version

# Optional GPU sanity check for OCR
python -c "import torch; print(torch.__version__); print('cuda avail:', torch.cuda.is_available()); print('cuda build:', torch.version.cuda)"
```

## 1. Clip mode: OCR

```powershell
# Minimal OCR clip extraction with default padding, skip-frames, stream copy
vidgrep "VIDEO.mp4" --text "PATTERN"

# Explicit CPU mode, useful when CUDA is unavailable or for baseline comparison
vidgrep "VIDEO.mp4" --text "PATTERN" --no-gpu

# Faster sparse scan: sample one frame every 2 seconds
vidgrep "VIDEO.mp4" --text "PATTERN" --interval 2 --stats

# Tune frame sampling, OCR batching, threshold, and language list
vidgrep "VIDEO.mp4" --text "PATTERN" --skip-frames 5 --batch-size 16 --threshold 0.6 --lang en,fr --stats

# Only scan a crop region. REGION is X Y W H.
vidgrep "VIDEO.mp4" --text "PATTERN" --region REGION

# Custom interval shaping before extraction
vidgrep "VIDEO.mp4" --text "PATTERN" --padding 10 --merge-gap 3 --min-duration 1.5

# Frame-accurate cuts. --lossless implies re-encoding with libx264 -crf 0.
vidgrep "VIDEO.mp4" --text "PATTERN" --reencode
vidgrep "VIDEO.mp4" --text "PATTERN" --lossless

# Concatenate all matched intervals into one file
vidgrep "VIDEO.mp4" --text "PATTERN" --concat --output "OUT.mp4"

# Single-file custom output path
vidgrep "VIDEO.mp4" --text "PATTERN" --output "OUT.mp4"

# No-match path should exit with "No clips written"
vidgrep "NO_MATCH.mp4" --text "PATTERN"
```

## 2. Clip mode: template image

```powershell
# Minimal template matching
vidgrep "VIDEO.mp4" --template "LOGO.png"

# Typical template threshold tuning
vidgrep "VIDEO.mp4" --template "LOGO.png" --threshold 0.85 --stats

# Region-cropped template matching
vidgrep "VIDEO.mp4" --template "LOGO.png" --region REGION

# Frame-accurate template clip extraction
vidgrep "VIDEO.mp4" --template "LOGO.png" --threshold 0.85 --reencode

# Missing template should fail cleanly
vidgrep "VIDEO.mp4" --template "DOES_NOT_EXIST.png"
```

## 3. Directory clip mode

```powershell
# Process every supported video found under DIR recursively.
# Clips are written next to each source video.
vidgrep "DIR\" --text "PATTERN"

# Directory + sparse scan + crop region
vidgrep "DIR\" --text "PATTERN" --interval 2 --region REGION --stats

# Directory + frame-accurate per-file clips
vidgrep "DIR\" --text "PATTERN" --lossless

# Template matching across a directory
vidgrep "DIR\" --template "LOGO.png" --threshold 0.85

# Expected failure: --output is rejected for directory inputs
vidgrep "DIR\" --text "PATTERN" --output "OUT.mp4"
```

## 4. Scan subcommand: JSONL + JSON only

```powershell
# Single file scan. Writes results.jsonl and results.json.
vidgrep scan "VIDEO.mp4" --text "PATTERN" --output results

# Whole directory, recursive
vidgrep scan "DIR\" --text "PATTERN" --output results_dir

# Mix files and directories; duplicate paths should only be scanned once
vidgrep scan "VIDEO.mp4" "DIR\" --text "PATTERN" --output results_mixed

# Crop region, sparse sampling, larger OCR batches, and throughput stats
vidgrep scan "VIDEO.mp4" --text "PATTERN" --region REGION --interval 2 --batch-size 16 --stats --output results_fast

# CPU-only scan
vidgrep scan "VIDEO.mp4" --text "PATTERN" --no-gpu --output results_cpu

# Default output stem should create scan_<pattern>_<timestamp>.jsonl and .json
vidgrep scan "VIDEO.mp4" --text "PATTERN"
```

## 5. Region picker helper

```powershell
# Interactive ROI selection; paste the printed X Y W H after --region
vidgrep-region "VIDEO.mp4"

# Start near a timestamp where the target appears
vidgrep-region "VIDEO.mp4" --time 12.5

# Save the selected crop to inspect or reuse as a template source
vidgrep-region "VIDEO.mp4" --time 12.5 --save-crop "crop.png"
```

## 6. Expected parser and validation failures

These should fail cleanly with argparse or an explicit `Error:` message.

```powershell
# Missing input path
vidgrep "DOES_NOT_EXIST.mp4" --text "PATTERN"

# Existing directory with no supported videos
vidgrep "EMPTY_DIR\" --text "PATTERN"

# No detector selected
vidgrep "VIDEO.mp4"

# Mutually exclusive detectors
vidgrep "VIDEO.mp4" --text "PATTERN" --template "LOGO.png"

# scan supports OCR text only, not template matching
vidgrep scan "VIDEO.mp4" --template "LOGO.png"

# scan requires at least one input
vidgrep scan --text "PATTERN"
```
