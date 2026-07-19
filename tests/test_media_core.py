from pathlib import Path

import numpy as np
import pytest

from vidgrep.clipper import VideoClipper
from vidgrep.detector import TemplateDetector


@pytest.mark.parametrize("mode", ["reencode", "lossless"])
def test_reencoded_webm_defaults_to_matroska(tmp_path, mode):
    clipper = VideoClipper.__new__(VideoClipper)
    clipper.path = str(tmp_path / "source.webm")
    clipper.duration = 10.0
    clipper._emit = lambda _message: None
    calls = []
    clipper._ffmpeg_clip = lambda *args: calls.append(args)

    paths = clipper.extract_clips(
        [(1.0, 2.0)],
        reencode=mode == "reencode",
        lossless=mode == "lossless",
    )

    assert paths == [str(tmp_path / "source_clip.mkv")]
    assert Path(calls[0][2]).suffix == ".mkv"


def test_template_larger_than_scan_region_is_rejected():
    detector = TemplateDetector.__new__(TemplateDetector)
    detector._tmpl_gray = np.zeros((20, 30), dtype=np.uint8)
    detector._cuda = False

    with pytest.raises(ValueError, match="larger than the scanned frame/region"):
        detector.detect(np.zeros((10, 40, 3), dtype=np.uint8))


def test_explicit_webm_lossless_output_is_rejected(tmp_path):
    clipper = VideoClipper.__new__(VideoClipper)
    clipper.path = str(tmp_path / "source.webm")

    with pytest.raises(SystemExit, match="cannot be written to WebM"):
        clipper._ffmpeg_clip(
            0.0,
            1.0,
            str(tmp_path / "output.webm"),
            reencode=True,
            lossless=True,
        )
