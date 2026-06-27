"""Frame detectors: GPU-accelerated OCR and image template matching."""
import re
from typing import Optional

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
    ):
        self._pattern = re.compile(pattern, re.IGNORECASE)
        self._threshold = threshold
        self._gpu = gpu
        self._languages = languages or ["en"]
        self._reader = None  # lazy-load on first detect() call

    def _load(self):
        if self._reader is None:
            import easyocr
            mode = "GPU" if self._gpu else "CPU"
            print(f"Loading OCR model ({mode}) …", flush=True)
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)

    def detect(self, frame: np.ndarray) -> bool:
        self._load()
        results = self._reader.readtext(frame, detail=1)
        matched = False
        for _, text, conf in results:
            if conf >= self._threshold and self._pattern.search(text):
                print(f"[detected] {text!r} (conf={conf:.2f})", flush=True)
                matched = True
        return matched


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
    ):
        self._threshold = threshold

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
                print("Template matcher: CUDA enabled")
            except Exception:
                print("Template matcher: CUDA unavailable, using CPU")

    def detect(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._cuda:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(gray)
            result_gpu = self._matcher.match(gpu_frame, self._gpu_tmpl)
            result = result_gpu.download()
        else:
            result = cv2.matchTemplate(gray, self._tmpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val) >= self._threshold
