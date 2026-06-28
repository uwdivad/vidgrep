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


class VideoClipper:
    def __init__(self, path: str):
        self.path = path

        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        self.fps: float = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration: float = self.frame_count / self.fps
        self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ret, _ = cap.read()
        cap.release()
        self._hw_ok = ret
        if not ret:
            print("Hardware decoding unavailable, will use software decoder…", flush=True)

        print(
            f"Video: {self._w}x{self._h} @ {self.fps:.2f} fps  |  "
            f"{self.duration:.1f}s  |  {self.frame_count:,} frames"
        )

    def _open_reader(self):
        if self._hw_ok:
            return cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        return _SWCapture(self.path, self._w, self._h)

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
    ) -> List[Tuple[float, float]]:
        """
        Scan the video and return (start_sec, end_sec) intervals where the
        detector fires.  Nearby intervals separated by < merge_gap seconds
        are merged into one.  Intervals shorter than min_duration are dropped.

        Frames that are skipped (not on a ``skip_frames`` boundary) are advanced
        with ``grab()`` instead of being fully decoded, and sampled frames are
        run through the detector in batches of ``batch_size`` to keep the GPU fed.
        """
        cap = self._open_reader()
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
        frame_idx = 0

        with tqdm(total=self.frame_count, unit="fr", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as bar:
            while True:
                if frame_idx % skip_frames == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if region:
                        x, y, w, h = region
                        crop = frame[y:y + h, x:x + w]
                    else:
                        crop = frame
                    batch_frames.append(crop)
                    batch_times.append(frame_idx / self.fps)
                    sampled += 1
                    if len(batch_frames) >= batch_size:
                        _flush()
                else:
                    if not cap.grab():
                        break

                frame_idx += 1
                bar.update(1)

        _flush()
        cap.release()

        if interval_start is not None and match_end - interval_start >= min_duration:
            intervals.append((interval_start, match_end))

        if collect_stats:
            self._print_stats(frame_idx, sampled, time.perf_counter() - started)

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
    ) -> List[dict]:
        """
        Iterate frames and return one dict per matched text region per frame:
        {"timestamp": float, "text": str, "confidence": float}

        If ``on_match`` is given it is called with each match dict the moment
        the batch containing it is processed, so callers can stream results to
        disk instead of waiting for the whole video to finish.
        """
        cap = self._open_reader()
        matches: List[dict] = []

        batch_frames: List[np.ndarray] = []
        batch_times: List[float] = []

        def _flush() -> None:
            if not batch_frames:
                return
            for t, found in zip(batch_times, detector.detect_matches_batch(batch_frames)):
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
        frame_idx = 0

        with tqdm(total=self.frame_count, unit="fr", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as bar:
            while True:
                if frame_idx % skip_frames == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if region:
                        x, y, w, h = region
                        crop = frame[y:y + h, x:x + w]
                    else:
                        crop = frame
                    batch_frames.append(crop)
                    batch_times.append(round(frame_idx / self.fps, 3))
                    sampled += 1
                    if len(batch_frames) >= batch_size:
                        _flush()
                else:
                    if not cap.grab():
                        break
                frame_idx += 1
                bar.update(1)

        _flush()
        cap.release()

        if collect_stats:
            self._print_stats(frame_idx, sampled, time.perf_counter() - started)

        return matches

    def _print_stats(self, frames: int, sampled: int, elapsed: float) -> None:
        if elapsed <= 0:
            return
        decode_fps = frames / elapsed
        ocr_fps = sampled / elapsed
        realtime = decode_fps / self.fps if self.fps else 0.0
        print(
            f"\n[stats] {frames:,} frames ({sampled:,} analysed) in {elapsed:.1f}s  |  "
            f"decode {decode_fps:.1f} fps  |  detect {ocr_fps:.1f} fps  |  "
            f"{realtime:.2f}x realtime",
            flush=True,
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
        ext = src.suffix
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

            print(f"\n[{i}/{len(intervals)}] {t0:.2f}s – {t1:.2f}s  →  {out}")
            self._ffmpeg_clip(t0, t1, out, reencode, lossless)
            clip_paths.append(out)

        if concat and len(intervals) > 1:
            final = output or str(src.parent / f"{src.stem}_clips{ext}")
            print(f"\nConcatenating {len(clip_paths)} clips  →  {final}")
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
            cmd += ["-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
                    "-c:a", "copy", "-avoid_negative_ts", "make_zero"]
        elif reencode:
            codec = self._nvenc_codec()
            cmd += ["-c:v", codec, "-c:a", "copy", "-avoid_negative_ts", "make_zero"]
        else:
            # Stream-copy is fast and bit-for-bit lossless, but cuts snap to
            # the nearest keyframe (~2s inaccuracy).  Use --lossless or
            # --reencode for frame-accurate cuts.
            cmd += ["-c", "copy"]
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
