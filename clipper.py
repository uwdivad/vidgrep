"""Video scanning and GPU-accelerated clip extraction via FFmpeg NVENC."""
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def _clamp_region(
    region: Tuple[int, int, int, int], w: int, h: int
) -> Tuple[int, int, int, int]:
    """Clamp a crop box to frame bounds so negative or oversized values
    can't abort FFmpeg's crop filter or silently produce a wrong numpy slice."""
    x, y, rw, rh = region
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    return x, y, rw, rh


class _SWCapture:
    """
    Frame reader that pipes raw BGR frames from an FFmpeg subprocess with
    hardware acceleration disabled.  Mirrors the cv2.VideoCapture.read()
    interface so find_intervals can use it transparently.
    """

    def __init__(self, path: str, w: int, h: int):
        self._frame_size = w * h * 3
        self._shape = (h, w, 3)
        self._buf: Optional[bytes] = None
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-hwaccel", "none",
                "-i", path,
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-loglevel", "error",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def grab(self) -> bool:
        """Advance to the next frame without decoding it into an array."""
        self._buf = self._proc.stdout.read(self._frame_size)
        return len(self._buf) == self._frame_size

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Decode the most recently grabbed frame into a BGR array."""
        if self._buf is None or len(self._buf) < self._frame_size:
            return False, None
        return True, np.frombuffer(self._buf, dtype=np.uint8).reshape(self._shape).copy()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.grab():
            return False, None
        return self.retrieve()

    def release(self):
        self._proc.stdout.close()
        self._proc.wait()


class _FFSampler:
    """
    Frame reader that lets FFmpeg do the sampling: only every Nth frame is
    decoded out of the pipe, optionally cropped to a region first.  Skipped
    frames never cross the process boundary, so the per-frame decode/pipe
    cost that dominates sparse scans (large --interval / --skip-frames)
    disappears.
    """

    def __init__(
        self,
        path: str,
        skip_frames: int,
        w: int,
        h: int,
        region: Optional[Tuple[int, int, int, int]] = None,
    ):
        vf = f"select=not(mod(n\\,{skip_frames}))"
        if region:
            x, y, rw, rh = _clamp_region(region, w, h)
            vf += f",crop={rw}:{rh}:{x}:{y}"
            out_w, out_h = rw, rh
        else:
            out_w, out_h = w, h

        self._frame_size = out_w * out_h * 3
        self._shape = (out_h, out_w, 3)
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-nostdin",
                "-i", path,
                "-vf", vf,
                "-fps_mode", "vfr",
                "-an",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-loglevel", "error",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        # Pre-read frame 0 so a rejected filter chain / unsupported flag on
        # an old FFmpeg build fails here, where callers can still fall back.
        self._pending: Optional[bytes] = self._proc.stdout.read(self._frame_size)
        if self._pending is None or len(self._pending) < self._frame_size:
            self.release()
            raise RuntimeError("FFmpeg sampler produced no frames")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        buf = self._pending
        self._pending = None
        if buf is None:
            buf = self._proc.stdout.read(self._frame_size)
        if not buf or len(buf) < self._frame_size:
            return False, None
        return True, np.frombuffer(buf, dtype=np.uint8).reshape(self._shape).copy()

    def release(self):
        self._proc.stdout.close()
        self._proc.wait()


class VideoClipper:
    def __init__(self, path: str, *, log: Optional[Callable[[str], None]] = None):
        self.path = path
        self._log = log

        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        self.fps: float = cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0:
            cap.release()
            raise ValueError(f"Cannot determine frame rate: {path}")
        self.frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration: float = self.frame_count / self.fps
        self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ret, _ = cap.read()
        cap.release()
        self._hw_ok = ret
        if not ret:
            self._emit("Hardware decoding unavailable, will use software decoder…")

        self._emit(
            f"Video: {self._w}x{self._h} @ {self.fps:.2f} fps  |  "
            f"{self.duration:.1f}s  |  {self.frame_count:,} frames"
        )

    def _emit(self, message: str) -> None:
        if self._log is not None:
            self._log(message)
        else:
            print(message, flush=True)

    def _open_reader(self):
        if self._hw_ok:
            return cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        return _SWCapture(self.path, self._w, self._h)

    def _iter_samples(
        self,
        skip_frames: int,
        region: Optional[Tuple[int, int, int, int]],
        *,
        show_progress: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ):
        """Yield (timestamp_sec, frame) for every ``skip_frames``-th frame.

        Prefers an FFmpeg-side sampler (skipped frames are dropped inside
        FFmpeg and never piped or converted); falls back to in-process
        decoding with grab() past skipped frames if that isn't available.
        """
        if region:
            region = _clamp_region(region, self._w, self._h)

        sampler = None
        if skip_frames > 1:
            try:
                sampler = _FFSampler(self.path, skip_frames, self._w, self._h, region)
            except (OSError, RuntimeError):
                self._emit("FFmpeg frame sampler unavailable — decoding every frame…")

        expected = self.frame_count // skip_frames + 1
        bar = None
        if show_progress:
            bar = tqdm(total=expected, unit="fr", dynamic_ncols=True,
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        def _advance(sampled: int) -> None:
            if bar is not None:
                bar.update(1)
            if on_progress is not None:
                on_progress(sampled, expected)

        try:
            if sampler is not None:
                idx = 0
                while True:
                    ret, frame = sampler.read()
                    if not ret:
                        break
                    yield (idx * skip_frames) / self.fps, frame
                    idx += 1
                    _advance(idx)
                return

            cap = self._open_reader()
            try:
                frame_idx = 0
                sampled = 0
                while True:
                    if frame_idx % skip_frames == 0:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if region:
                            x, y, w, h = region
                            frame = frame[y:y + h, x:x + w]
                        yield frame_idx / self.fps, frame
                        sampled += 1
                        _advance(sampled)
                    else:
                        if not cap.grab():
                            break
                    frame_idx += 1
            finally:
                cap.release()
        finally:
            if sampler is not None:
                sampler.release()
            if bar is not None:
                bar.close()

    # ------------------------------------------------------------------
    def find_intervals(
        self,
        detector,
        *,
        skip_frames: int = 3,
        region: Optional[Tuple[int, int, int, int]] = None,
        merge_gap: float = 2.0,
        min_duration: float = 0.0,
        batch_size: int = 8,
        collect_stats: bool = False,
        show_progress: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[float, float]]:
        """
        Scan the video and return (start_sec, end_sec) intervals where the
        detector fires.  Nearby intervals separated by < merge_gap seconds
        are merged into one.  Intervals shorter than min_duration are dropped.

        Sampling and region cropping happen inside FFmpeg where possible (see
        ``_iter_samples``), and sampled frames are run through the detector in
        batches of ``batch_size`` to keep the GPU fed.
        """
        intervals: List[Tuple[float, float]] = []

        interval_start: Optional[float] = None
        match_end: Optional[float] = None

        def _record(t: float, matched: bool) -> None:
            nonlocal interval_start, match_end
            if matched:
                if interval_start is None:
                    interval_start = t
                match_end = t
            elif interval_start is not None and (t - match_end) > merge_gap:
                if match_end - interval_start >= min_duration:
                    intervals.append((interval_start, match_end))
                interval_start = None
                match_end = None

        batch_frames: List[np.ndarray] = []
        batch_times: List[float] = []

        def _flush() -> None:
            if not batch_frames:
                return
            for t, matched in zip(batch_times, detector.detect_batch(batch_frames)):
                _record(t, matched)
            batch_frames.clear()
            batch_times.clear()

        started = time.perf_counter()
        sampled = 0

        for t, crop in self._iter_samples(
            skip_frames, region, show_progress=show_progress, on_progress=on_progress
        ):
            batch_frames.append(crop)
            batch_times.append(t)
            sampled += 1
            if len(batch_frames) >= batch_size:
                _flush()

        _flush()

        if interval_start is not None and match_end - interval_start >= min_duration:
            intervals.append((interval_start, match_end))

        if collect_stats:
            frames_covered = min(sampled * skip_frames, self.frame_count)
            self._print_stats(frames_covered, sampled, time.perf_counter() - started)

        return intervals

    # ------------------------------------------------------------------
    def scan_for_matches(
        self,
        detector,
        *,
        skip_frames: int = 3,
        region: Optional[Tuple[int, int, int, int]] = None,
        batch_size: int = 8,
        collect_stats: bool = False,
        on_match: Optional[Callable[[dict], None]] = None,
        show_progress: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None,
        profile: Optional[dict] = None,
    ) -> List[dict]:
        """
        Iterate frames and return one dict per matched text region per frame:
        {"timestamp": float, "text": str, "confidence": float}

        If ``on_match`` is given it is called with each match dict the moment
        the batch containing it is processed, so callers can stream results to
        disk instead of waiting for the whole video to finish.
        """
        matches: List[dict] = []

        batch_frames: List[np.ndarray] = []
        batch_times: List[float] = []
        batch_count = 0
        ocr_seconds = 0.0
        sample_seconds = 0.0
        batch_sizes: List[int] = []

        def _flush() -> None:
            nonlocal batch_count, ocr_seconds
            if not batch_frames:
                return
            batch_count += 1
            batch_sizes.append(len(batch_frames))
            ocr_started = time.perf_counter()
            found_by_frame = detector.detect_matches_batch(batch_frames)
            ocr_seconds += time.perf_counter() - ocr_started
            for t, found in zip(batch_times, found_by_frame):
                for m in found:
                    record = {
                        "timestamp": t,
                        "text": m["text"],
                        "confidence": round(m["confidence"], 4),
                    }
                    matches.append(record)
                    if on_match is not None:
                        on_match(record)
            batch_frames.clear()
            batch_times.clear()

        started = time.perf_counter()
        sampled = 0

        samples = iter(self._iter_samples(
            skip_frames, region, show_progress=show_progress, on_progress=on_progress
        ))
        while True:
            sample_started = time.perf_counter()
            try:
                t, crop = next(samples)
            except StopIteration:
                break
            sample_seconds += time.perf_counter() - sample_started
            batch_frames.append(crop)
            batch_times.append(round(t, 3))
            sampled += 1
            if len(batch_frames) >= batch_size:
                _flush()

        _flush()

        if collect_stats:
            frames_covered = min(sampled * skip_frames, self.frame_count)
            self._print_stats(frames_covered, sampled, time.perf_counter() - started)

        if profile is not None:
            profile.update({
                "sampled_frames": sampled,
                "batch_count": batch_count,
                "ocr_sec": round(ocr_seconds, 6),
                "sample_read_sec": round(sample_seconds, 6),
                "avg_batch_size": round(sum(batch_sizes) / len(batch_sizes), 2) if batch_sizes else 0.0,
            })

        return matches

    def _print_stats(self, frames: int, sampled: int, elapsed: float) -> None:
        if elapsed <= 0:
            return
        decode_fps = frames / elapsed
        ocr_fps = sampled / elapsed
        realtime = decode_fps / self.fps if self.fps else 0.0
        self._emit(
            f"\n[stats] {frames:,} frames ({sampled:,} analysed) in {elapsed:.1f}s  |  "
            f"decode {decode_fps:.1f} fps  |  detect {ocr_fps:.1f} fps  |  "
            f"{realtime:.2f}x realtime"
        )

    # ------------------------------------------------------------------
    def extract_clips(
        self,
        intervals: List[Tuple[float, float]],
        *,
        padding: float = 5.0,
        output: Optional[str] = None,
        reencode: bool = False,
        lossless: bool = False,
        concat: bool = False,
    ) -> List[str]:
        """Extract one clip per interval (start-padding … end+padding) using FFmpeg."""
        src = Path(self.path)
        # WebM cannot mux the H.264 output used by --lossless or by NVENC's
        # fallback for VP8/VP9 sources. Keep stream-copy outputs in WebM, but
        # use Matroska for re-encoded defaults so the selected codec is valid.
        ext = (
            ".mkv"
            if (reencode or lossless) and src.suffix.lower() == ".webm"
            else src.suffix
        )
        if concat and len(intervals) > 1 and output and Path(output).suffix.lower() == ".webm":
            if lossless:
                sys.exit(
                    "Error: --lossless uses H.264, which cannot be written to WebM. "
                    "Choose an .mkv or .mp4 output path."
                )
            if reencode:
                codec = self._nvenc_codec()
                if codec != "av1_nvenc":
                    sys.exit(
                        f"Error: {codec} cannot be written to WebM. "
                        "Choose an .mkv or .mp4 output path."
                    )
        clip_paths: List[str] = []

        for i, (start, end) in enumerate(intervals, 1):
            t0 = max(0.0, start - padding)
            t1 = min(self.duration, end + padding)

            if concat and len(intervals) > 1:
                # Write to a temp file; concatenate at the end
                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                out = tmp.name
                tmp.close()
            elif len(intervals) == 1:
                out = output or str(src.parent / f"{src.stem}_clip{ext}")
            else:
                if output:
                    op = Path(output)
                    out = str(op.parent / f"{op.stem}_{i}{op.suffix or ext}")
                else:
                    out = str(src.parent / f"{src.stem}_clip_{i}{ext}")

            self._emit(f"\n[{i}/{len(intervals)}] {t0:.2f}s – {t1:.2f}s  →  {out}")
            self._ffmpeg_clip(t0, t1, out, reencode, lossless)
            clip_paths.append(out)

        if concat and len(intervals) > 1:
            final = output or str(src.parent / f"{src.stem}_clips{ext}")
            self._emit(f"\nConcatenating {len(clip_paths)} clips  →  {final}")
            self._ffmpeg_concat(clip_paths, final)
            for p in clip_paths:
                Path(p).unlink(missing_ok=True)
            return [final]

        return clip_paths

    # ------------------------------------------------------------------
    def _ffmpeg_clip(self, start: float, end: float, output: str, reencode: bool, lossless: bool = False):
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.6f}",
            "-to", f"{end:.6f}",
            "-i", self.path,
        ]
        if lossless:
            if Path(output).suffix.lower() == ".webm":
                sys.exit(
                    "Error: --lossless uses H.264, which cannot be written to WebM. "
                    "Choose an .mkv or .mp4 output path."
                )
            cmd += ["-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
                    "-c:a", "copy", "-avoid_negative_ts", "make_zero"]
        elif reencode:
            codec = self._nvenc_codec()
            if Path(output).suffix.lower() == ".webm" and codec != "av1_nvenc":
                sys.exit(
                    f"Error: {codec} cannot be written to WebM. "
                    "Choose an .mkv or .mp4 output path."
                )
            cmd += ["-c:v", codec, "-c:a", "copy", "-avoid_negative_ts", "make_zero"]
        else:
            # Stream-copy is fast and bit-for-bit lossless, but cuts snap to
            # the nearest keyframe (~2s inaccuracy).  Use --lossless or
            # --reencode for frame-accurate cuts.
            cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        cmd.append(output)
        self._run(cmd)

    def _ffmpeg_concat(self, clips: List[str], output: str):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in clips:
                f.write(f"file '{Path(p).as_posix()}'\n")
            list_path = f.name
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
               "-i", list_path, "-c", "copy", output]
        self._run(cmd)
        Path(list_path).unlink(missing_ok=True)

    def _nvenc_codec(self) -> str:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", self.path],
            capture_output=True, text=True,
        )
        codec = r.stdout.strip().lower()
        return {
            "h264": "h264_nvenc",
            "hevc": "hevc_nvenc",
            "h265": "hevc_nvenc",
            "av1":  "av1_nvenc",
        }.get(codec, "h264_nvenc")

    @staticmethod
    def _run(cmd: List[str]):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Show only the tail of stderr to avoid flooding the terminal
            sys.exit(f"FFmpeg error (exit {result.returncode}):\n{result.stderr[-3000:]}")
