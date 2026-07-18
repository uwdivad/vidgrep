# Making OCR video scans ~6× faster

*July 2026 · vidgrep engineering note*

vidgrep scans gameplay recordings for on-screen text (player names, killfeed
events) with EasyOCR, then extracts clips around each match with FFmpeg. The
worker pipeline felt slower than an RTX 5090 should allow, so I profiled it,
found the real bottleneck, and fixed it. The benchmark scan went from **29.6s
to 5.1s per video (5.8×)** with byte-identical match output — and the
investigation surfaced two silent correctness bugs that had been shipping on
master.

**Setup**

| | |
|---|---|
| Command | `python main.py worker videos.csv --text "uwdivad" --region 132 476 592 388 --interval 2 --batch-size 32 --stats --profile` |
| Test video | AV1, 2560×1440 @ 144 fps, 26.7 s, 3,837 frames |
| Hardware | RTX 5090 · Windows 10 · FFmpeg 8.0.1 · EasyOCR 1.7.2 (CUDA) |

## The profile: decode-bound, not OCR-bound

The obvious suspect for a slow OCR pipeline is the OCR. The profiler said
otherwise — **77% of every scan was spent waiting on FFmpeg decode**:

| Stage (warm model) | Before | After |
|---|---:|---:|
| Waiting on FFmpeg decode | 27.6 s | 4.1 s |
| OCR inference | 1.6 s | 0.6 s |
| **Total per video** | **29.6 s** | **5.1 s** |
| Scan speed vs. playback | 0.9× realtime | 5.3× realtime |
| Matches found | 8 | 8 — identical timestamps |

The pipeline already did the smart thing architecturally: frame sampling
(`select=not(mod(n,288))`) and `--region` cropping run inside FFmpeg, so
skipped frames and cropped-away pixels never cross the pipe. The problem is
that `select` drops frames *after* decode — FFmpeg was software-decoding all
3,837 AV1 1440p frames on the CPU to emit the 14 frames that `--interval 2`
actually samples.

## Change 1 — Decode on NVDEC, crop before download (27.6s → 4.1s)

`clipper.py · _FFSampler, _iter_samples`

Decode now runs on the GPU's NVDEC engine, and the region crop happens at the
decoder — so full-size frames never cross the PCIe bus, let alone the pipe:

```diff
- ffmpeg -nostdin -i video.mp4
-        -vf "select=not(mod(n\,288)),crop=592:388:132:476" ...
+ ffmpeg -nostdin -c:v av1_cuvid -crop 476x576x132x1836 -i video.mp4
+        -vf "select=not(mod(n\,288))" ...
```

Implementation notes:

- The decoder (`h264_cuvid`, `hevc_cuvid`, `av1_cuvid`, …) is chosen from the
  ffprobe codec. The generic `-hwaccel cuda` flag is deliberately **not**
  used: it fails to initialise on the H.264 archive videos under FFmpeg 8.0.1
  (`cuvidCreateDecoder → CUDA_ERROR_INVALID_VALUE`) and silently falls back
  to software — the explicit cuvid decoders work.
- Fallback chain: NVDEC sampler → software FFmpeg sampler → in-process
  decode. The sampler pre-reads frame 0 at construction, so any decoder
  rejection (unsupported profile, no NVIDIA GPU) surfaces immediately and
  falls through — machines without CUDA need no configuration.
- cuvid's crop edges must be even-aligned; odd regions are aligned outward at
  the decoder and trimmed by a cheap crop filter on the already-small frames.

## Change 2 — Actually batch EasyOCR recognition (1.6s → 0.6s)

`detector.py · TextDetector.detect_matches_batch`

`readtext_batched` defaults to `batch_size=1` — so despite `--batch-size 32`
collecting frames into batches, the recogniser was running one text crop per
GPU call. One line:

```diff
- batched = self._reader.readtext_batched(frames, detail=1)
+ batched = self._reader.readtext_batched(
+     frames, detail=1, batch_size=len(frames)
+ )
```

Warm OCR on the 14-frame benchmark batch: 1.56s → 0.53s. Output differs only
on marginal low-confidence boxes (garbled enemy names flicker between
misreadings); every match containing the search term is unchanged.

## Change 3 — Overlap decode with OCR

`clipper.py · new _Prefetcher`

A raw 592×388 frame (~690 KB) is far larger than the OS pipe buffer, so
FFmpeg stalled whenever Python was busy in OCR — decode and inference ran
back-to-back instead of concurrently. A background thread now reads sampled
frames into a bounded queue (capped at ~256 MB), keeping FFmpeg decoding
while the GPU runs inference. The short benchmark video fits in a single OCR
batch so it shows no gain there; multi-batch (i.e. real-length) videos
converge toward *max(decode, OCR)* instead of the sum.

## Isolated stage benchmarks

| Stage (measured alone) | Before | After | Speedup |
|---|---:|---:|---:|
| AV1 sampling — full benchmark video | 27 s | 4 s | **6.8×** |
| H.264 sampling — archive video, 60 s segment | 7 s | 5 s | **1.4×** |
| OCR, 14 frames — warm model | 1.56 s | 0.53 s | **2.9×** |

The dramatic decode win is codec-dependent. AV1 software decode is slow, so
NVDEC is transformative there. The H.264 archive videos already
software-decoded at ~500 fps across many cores; a single NVDEC engine tops
out near 720 fps at 1440p, so the per-video gain is modest. The next lever
for the worker queue is scanning two videos concurrently — CPU decode and
the second NVDEC engine currently sit idle.

Also measured and rejected: `-skip_frame nokey` (keyframe-only decode). The
recordings have a 0.5 s GOP, so it decodes 4× more frames than `--interval 2`
needs, gains less than NVDEC, and changes timestamp semantics.

## Bugs found during verification

Both pre-existed on master and were exposed by testing the new decode paths.

### Odd-sized regions silently returned garbage

FFmpeg's `crop` filter operates on yuv420p frames, where 4:2:0 chroma
subsampling rounds odd widths and heights *down to even*. A 591×387 region
actually produced 590×386 frames while the sampler kept reading
591×387-sized chunks from the pipe — every frame after the first was
scrambled by the accumulating offset, and scans returned zero matches with
no error. Fixed by converting to `bgr24` (packed, no subsampling) *before*
cropping; only sampled frames reach that conversion, so it's free.

| Scan with `--region 133 477 591 387` | Frames read | Matches |
|---|---|---|
| Before | 13 misaligned | 0 — silent corruption |
| After | 14 correct | 7 |

### Clip mode crashed when output was piped

The extraction progress message contained a "→" character that Windows'
cp1252 encoding can't represent. Piping or redirecting clip-mode output
(logging to a file, running under another tool) raised `UnicodeEncodeError`
and aborted *before writing the clip*. Interactive terminals masked it.
Replaced with ASCII.

## Verification

- Full test suite: 50/50 pass; `py_compile` clean on all modules.
- Scan with the worker's exact parameters: same 8 matches at identical
  timestamps as the pre-change baseline.
- Clip mode end-to-end: same 2 intervals found, both clips extracted — the
  extraction commands (stream-copy / NVENC / libx264) are untouched by these
  changes.
- Exercised full-frame (no region), even-region, and odd-region paths
  through the NVDEC sampler.
- Frame parity checked byte-for-byte: NVDEC vs software output differs only
  by NV12 colorspace rounding (mean ~1/255) — no effect on detections.

## Takeaways

- **Profile before optimising.** The received wisdom ("EasyOCR is slow, batch
  harder") pointed at the 22% slice. The 77% slice was FFmpeg decoding frames
  it was about to throw away.
- **The obvious flag isn't always the right one.** `-hwaccel cuda` looks like
  the NVDEC switch, but it broke on real files and fell back to software
  silently; the explicit cuvid decoders both worked and unlocked
  decoder-side cropping.
- **Fast paths find correctness bugs.** Verifying frame parity between the
  old and new pipelines is what exposed the odd-region corruption — a bug
  that produced clean-looking, empty results.
