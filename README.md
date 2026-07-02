# vidgrep

Scan a video for text or a visual pattern, then extract clips around each match.

```
vidgrep match.mp4 --text "GOAL"
vidgrep gameplay.mp4 --template logo.png
```

## How it works

1. Samples every Nth frame of the video (default: every 3rd) — or one frame every N seconds with `--interval`. Sampling and `--region` cropping run inside FFmpeg, so skipped frames are nearly free
2. Runs OCR or template matching on each sampled frame
3. Merges nearby hit windows into intervals
4. Extracts a padded clip for each interval via FFmpeg

## Requirements

- Python 3.9+
- FFmpeg on `PATH`
  - **Linux:** `sudo apt install ffmpeg`
  - **Windows:** download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (`ffmpeg-release-essentials.zip`), add the `bin/` folder to your system PATH
- CUDA-capable GPU (optional but strongly recommended for OCR)

## Installation

Install CUDA-enabled PyTorch **first** — EasyOCR picks up the CUDA build at import time:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then install the rest:

```bash
pip install -e .
```

To run on CPU only, skip the PyTorch step (a CPU build will be pulled in automatically) and pass `--no-gpu` at runtime.

## Usage

```
vidgrep <input> (--text PATTERN | --template IMAGE) [options]
```

The editable install exposes two commands:

| Command | Description |
|---------|-------------|
| `vidgrep` | Scan videos and extract matching clips, or run `vidgrep scan` for JSONL metadata only |
| `vidgrep-region` | Interactive helper for choosing a `--region X Y W H` crop |

For development from the repo root, `python main.py ...` and `python select_region.py ...` still work.

### Detection modes

| Flag | Description |
|------|-------------|
| `--text PATTERN` | Match frames whose visible text contains this regex (case-insensitive) |
| `--template IMAGE` | Match frames containing this image via normalised cross-correlation |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output PATH` | `<stem>_clip[_N]<ext>` | Output file path |
| `--padding SEC` | `5.0` | Seconds to include before and after each match |
| `--skip-frames N` | `3` | Sample every Nth frame (higher = faster scan) |
| `--interval SEC` | — | Sample one frame every SEC seconds (frame-rate independent; overrides `--skip-frames`). Much faster on long videos |
| `--batch-size N` | `8` | Frames per OCR batch (higher = better GPU utilisation, more VRAM) |
| `--stats` | — | Print decode/detect throughput (fps, ×-realtime) after each scan |
| `--threshold 0-1` | `0.5` | Min OCR confidence or template similarity to count as a match |
| `--region X Y W H` | — | Crop each frame to this rectangle before scanning |
| `--merge-gap SEC` | `2.0` | Merge match windows separated by less than this |
| `--min-duration SEC` | `0.0` | Discard matched intervals shorter than this |
| `--lang CODES` | `en` | Comma-separated EasyOCR language codes (e.g. `en,fr`) |
| `--no-gpu` | — | Disable CUDA, run on CPU |
| `--reencode` | — | Re-encode with NVENC (frame-accurate cuts) |
| `--lossless` | — | Re-encode with libx264 CRF 0 (frame-accurate, larger files) |
| `--concat` | — | Concatenate all clips into a single output file |

### Output

By default clips are stream-copied (fast, but cuts snap to the nearest keyframe, ~2s inaccuracy). Use `--lossless` or `--reencode` for frame-accurate cuts.

## Examples

```bash
# Extract every moment "GOAL" appears, ±5 s
vidgrep match.mp4 --text "GOAL"

# Only scan the bottom third of the frame (faster OCR)
vidgrep stream.mp4 --text "LIVE" --region 0 720 1280 360

# Template matching with a higher similarity threshold
vidgrep gameplay.mp4 --template hud.png --threshold 0.85

# Merge all clips into one file, re-encoded for frame accuracy
vidgrep movie.mp4 --text "Chapter" --concat --lossless

# Fast scan of a long video on CPU only
vidgrep long.mp4 --text "error" --skip-frames 10 --merge-gap 5 --no-gpu

# Sample one frame every 2 s — frame-rate independent, big speedup on long videos
vidgrep long.mp4 --text "error" --interval 2
```

## Troubleshooting

### OCR is running on the CPU even though I have a GPU

**Symptom.** Scanning is slow and you see a warning like:

```
'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
```

This means PyTorch can't see your GPU (`torch.cuda.is_available()` is `False`), so EasyOCR silently falls back to the CPU. The tool also prints its own warning at model load when this happens:

```
WARNING: GPU requested but CUDA is unavailable — running OCR on CPU (much slower). …
```

**Confirm it.** With your virtualenv activated:

```bash
python -c "import torch; print(torch.__version__); print('cuda avail:', torch.cuda.is_available()); print('cuda build:', torch.version.cuda)"
```

A version ending in `+cpu` and `cuda avail: False` means the **CPU-only** PyTorch wheel got installed.

**Fix.** Reinstall the CUDA build (the **cu128** wheel — required for RTX 50-series / Blackwell GPUs; older `cu121`/`cu118` wheels don't include the `sm_120` architecture and won't drive a 5090):

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Re-run the confirm command — you want `cuda avail: True` and `cuda build: 12.8`. Order matters: install CUDA-enabled PyTorch **before** EasyOCR, since EasyOCR binds to whatever torch build is present at import time.

### Checking GPU utilisation during a scan

Run a scan with `--stats` to get an app-level throughput number (decode fps, detect fps, ×-realtime). To watch the GPU live:

- `nvidia-smi dmon -s u` — watch the `dec` column (non-zero = hardware decode active) and `sm` (compute load).
- [`nvitop`](https://github.com/XuehaiPan/nvitop) (`pip install nvitop`) — interactive per-process GPU/VRAM graphs.
- **Windows Task Manager** → Performance → GPU: switch a graph to **CUDA** and watch the **Video Decode** pane.

If GPU compute stays low while a CPU core is pegged, the GPU is starved — raise `--batch-size`, add `--region` to shrink the OCR area, or sample fewer frames with `--skip-frames` / `--interval`.
