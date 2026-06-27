"""Video scanning and GPU-accelerated clip extraction via FFmpeg NVENC."""
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

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

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raw = self._proc.stdout.read(self._frame_size)
        if len(raw) < self._frame_size:
            return False, None
        return True, np.frombuffer(raw, dtype=np.uint8).reshape(self._shape).copy()

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
    ) -> List[Tuple[float, float]]:
        """
        Scan the video and return (start_sec, end_sec) intervals where the
        detector fires.  Nearby intervals separated by < merge_gap seconds
        are merged into one.  Intervals shorter than min_duration are dropped.
        """
        cap = self._open_reader()
        intervals: List[Tuple[float, float]] = []

        interval_start: Optional[float] = None
        match_end: Optional[float] = None
        frame_idx = 0

        with tqdm(total=self.frame_count, unit="fr", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as bar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % skip_frames == 0:
                    if region:
                        x, y, w, h = region
                        crop = frame[y:y + h, x:x + w]
                    else:
                        crop = frame

                    t = frame_idx / self.fps
                    matched = detector.detect(crop)

                    if matched:
                        if interval_start is None:
                            interval_start = t
                        match_end = t
                    elif interval_start is not None and (t - match_end) > merge_gap:
                        if match_end - interval_start >= min_duration:
                            intervals.append((interval_start, match_end))
                        interval_start = None
                        match_end = None

                frame_idx += 1
                bar.update(1)

        cap.release()

        if interval_start is not None and match_end - interval_start >= min_duration:
            intervals.append((interval_start, match_end))

        return intervals

    # ------------------------------------------------------------------
    def scan_for_matches(
        self,
        detector,
        *,
        skip_frames: int = 3,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[dict]:
        """
        Iterate frames and return one dict per matched text region per frame:
        {"timestamp": float, "text": str, "confidence": float}
        """
        cap = self._open_reader()
        matches: List[dict] = []
        frame_idx = 0

        with tqdm(total=self.frame_count, unit="fr", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as bar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % skip_frames == 0:
                    if region:
                        x, y, w, h = region
                        crop = frame[y:y + h, x:x + w]
                    else:
                        crop = frame
                    t = round(frame_idx / self.fps, 3)
                    for m in detector.detect_matches(crop):
                        matches.append({
                            "timestamp": t,
                            "text": m["text"],
                            "confidence": round(m["confidence"], 4),
                        })
                frame_idx += 1
                bar.update(1)

        cap.release()
        return matches

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
