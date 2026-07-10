"""Frame detectors: GPU-accelerated OCR and image template matching."""
import re
from typing import Callable, Optional

import cv2
import numpy as np


class TextDetector:
    """
    Detects frames whose visible text matches a regex pattern.
    Uses EasyOCR with CUDA for GPU-accelerated inference.
    """

    def __init__(
        self,
        pattern: str,
        *,
        gpu: bool = True,
        threshold: float = 0.5,
        languages: Optional[list] = None,
        log: Optional[Callable[[str], None]] = None,
    ):
        self._pattern = re.compile(pattern, re.IGNORECASE)
        self._threshold = threshold
        self._gpu = gpu
        self._languages = languages or ["en"]
        self._reader = None  # lazy-load on first detect() call
        self._log = log

    def _emit(self, message: str) -> None:
        if self._log is not None:
            self._log(message)
        else:
            print(message, flush=True)

    def _load(self):
        if self._reader is None:
            import easyocr
            gpu = self._gpu
            if gpu:
                import torch
                if not torch.cuda.is_available():
                    self._emit(
                        "WARNING: GPU requested but CUDA is unavailable — running OCR on "
                        "CPU (much slower). Reinstall a CUDA build of torch "
                        "(RTX 5090 needs cu128); pass --no-gpu to silence this."
                    )
                    gpu = False
            mode = "GPU" if gpu else "CPU"
            self._emit(f"Loading OCR model ({mode}) …")
            self._reader = easyocr.Reader(self._languages, gpu=gpu, verbose=False)

    @staticmethod
    def _group_lines(results, *, y_tol: float = 0.5) -> list[dict]:
        """Group EasyOCR boxes that share a horizontal line.

        EasyOCR returns one box per text region, so a single visible line
        like "uwdivad ate a whole pie" arrives as several boxes. We bucket
        boxes whose vertical centres are close (relative to box height),
        order each bucket left-to-right, and join the text with spaces.
        """
        boxes = []
        for bbox, text, conf in results:
            ys = [p[1] for p in bbox]
            xs = [p[0] for p in bbox]
            boxes.append({
                "text": text,
                "conf": float(conf),
                "x": min(xs),
                "y_center": sum(ys) / len(ys),
                "height": max(max(ys) - min(ys), 1.0),
            })

        boxes.sort(key=lambda b: b["y_center"])
        lines: list[dict] = []
        for b in boxes:
            for line in lines:
                if abs(b["y_center"] - line["y_center"]) <= y_tol * max(b["height"], line["height"]):
                    line["boxes"].append(b)
                    line["y_center"] = sum(x["y_center"] for x in line["boxes"]) / len(line["boxes"])
                    line["height"] = max(line["height"], b["height"])
                    break
            else:
                lines.append({"y_center": b["y_center"], "height": b["height"], "boxes": [b]})

        out = []
        for line in lines:
            ordered = sorted(line["boxes"], key=lambda b: b["x"])
            out.append({
                "boxes": ordered,
                "text": " ".join(b["text"] for b in ordered),
            })
        return out

    def _filter(self, results) -> list[dict]:
        matches = []
        for line in self._group_lines(results):
            if not self._pattern.search(line["text"]):
                continue
            # Confidence reflects the box(es) that actually contain the match,
            # not the surrounding words pulled in to complete the line.
            match_confs = [b["conf"] for b in line["boxes"] if self._pattern.search(b["text"])]
            conf = max(match_confs) if match_confs else max(b["conf"] for b in line["boxes"])
            if conf >= self._threshold:
                matches.append({"text": line["text"], "confidence": conf})
        return matches

    def detect_matches(self, frame: np.ndarray) -> list[dict]:
        self._load()
        return self._filter(self._reader.readtext(frame, detail=1))

    def detect_matches_batch(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """OCR a batch of equally-sized frames in one GPU call.

        Falls back to per-frame inference if the EasyOCR build lacks
        ``readtext_batched`` or the frames can't be stacked.
        """
        self._load()
        if len(frames) == 1:
            return [self.detect_matches(frames[0])]
        try:
            batched = self._reader.readtext_batched(frames, detail=1)
        except Exception:
            return [self.detect_matches(f) for f in frames]
        return [self._filter(res) for res in batched]

    def _announce(self, matches: list[dict]) -> bool:
        for m in matches:
            self._emit(f"[detected] {m['text']!r} (conf={m['confidence']:.2f})")
        return len(matches) > 0

    def detect(self, frame: np.ndarray) -> bool:
        return self._announce(self.detect_matches(frame))

    def detect_batch(self, frames: list[np.ndarray]) -> list[bool]:
        return [self._announce(m) for m in self.detect_matches_batch(frames)]


class TemplateDetector:
    """
    Detects frames containing an image template via normalised cross-correlation.
    Uses OpenCV CUDA when available, otherwise falls back to CPU.
    """

    def __init__(
        self,
        template_path: str,
        *,
        gpu: bool = True,
        threshold: float = 0.8,
        log: Optional[Callable[[str], None]] = None,
    ):
        self._threshold = threshold
        self._log = log

        tmpl = cv2.imread(template_path)
        if tmpl is None:
            raise FileNotFoundError(f"Cannot load template image: {template_path}")
        self._tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

        self._cuda = False
        if gpu:
            try:
                self._gpu_tmpl = cv2.cuda_GpuMat()
                self._gpu_tmpl.upload(self._tmpl_gray)
                self._matcher = cv2.cuda.createTemplateMatching(
                    cv2.CV_8UC1, cv2.TM_CCOEFF_NORMED
                )
                self._cuda = True
                self._emit("Template matcher: CUDA enabled")
            except Exception:
                self._emit("Template matcher: CUDA unavailable, using CPU")

    def _emit(self, message: str) -> None:
        if self._log is not None:
            self._log(message)
        else:
            print(message, flush=True)

    def detect(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = gray.shape[:2]
        template_h, template_w = self._tmpl_gray.shape[:2]
        if frame_h < template_h or frame_w < template_w:
            raise ValueError(
                "Template image "
                f"({template_w}x{template_h}) is larger than the scanned "
                f"frame/region ({frame_w}x{frame_h})"
            )
        if self._cuda:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(gray)
            result_gpu = self._matcher.match(gpu_frame, self._gpu_tmpl)
            result = result_gpu.download()
        else:
            result = cv2.matchTemplate(gray, self._tmpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val) >= self._threshold

    def detect_batch(self, frames: list[np.ndarray]) -> list[bool]:
        # Template matching is GPU-light and not meaningfully batchable here;
        # keep the uniform interface so the scan loop can call it the same way.
        return [self.detect(f) for f in frames]
