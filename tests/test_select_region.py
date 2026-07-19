import sys

import numpy as np

from vidgrep import select_region


def test_save_crop_reads_before_releasing_capture(tmp_path, monkeypatch):
    saved = []

    class FakeCapture:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            values = {
                select_region.cv2.CAP_PROP_FPS: 30.0,
                select_region.cv2.CAP_PROP_FRAME_COUNT: 2,
                select_region.cv2.CAP_PROP_FRAME_WIDTH: 8,
                select_region.cv2.CAP_PROP_FRAME_HEIGHT: 6,
            }
            return values[prop]

        def set(self, _prop, _value):
            return True

        def read(self):
            if self.released:
                return False, None
            return True, np.zeros((6, 8, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    capture = FakeCapture()
    keys = iter([13, 27])
    monkeypatch.setattr(select_region.cv2, "VideoCapture", lambda *_args: capture)
    monkeypatch.setattr(select_region.cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(select_region.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(select_region.cv2, "waitKey", lambda _delay: next(keys))
    monkeypatch.setattr(select_region.cv2, "selectROI", lambda *_args, **_kwargs: (1, 1, 2, 2))
    monkeypatch.setattr(select_region.cv2, "destroyWindow", lambda *_args: None)
    monkeypatch.setattr(select_region.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(select_region.cv2, "rectangle", lambda *_args: None)
    monkeypatch.setattr(select_region.cv2, "putText", lambda *_args: None)
    monkeypatch.setattr(
        select_region.cv2,
        "imwrite",
        lambda path, image: saved.append((path, image.shape)) or True,
    )
    video_path = tmp_path / "clip.mp4"
    video_path.touch()
    crop_path = tmp_path / "crop.png"
    monkeypatch.setattr(
        sys,
        "argv",
        ["vidgrep-region", str(video_path), "--save-crop", str(crop_path)],
    )

    select_region.main()

    assert saved == [(str(crop_path), (2, 2, 3))]
    assert capture.released is True
