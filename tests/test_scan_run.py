import json
from pathlib import Path
from types import SimpleNamespace

import detector
import scan


def test_run_scan_counts_only_successful_files(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "bad.mp4").touch()
    (inputs / "good.mp4").touch()
    output_stem = tmp_path / "results"
    args = SimpleNamespace(
        inputs=[str(inputs)],
        text="goal",
        output=str(output_stem),
        threshold=0.5,
        skip_frames=3,
        interval=None,
        batch_size=8,
        stats=False,
        profile=False,
        metrics_output=None,
        region=None,
        lang="en",
        no_gpu=True,
    )

    monkeypatch.setattr(detector, "TextDetector", lambda *_args, **_kwargs: object())

    def fake_scan_video(*, video_path, detector, args, options_id, on_record):
        if video_path.name == "bad.mp4":
            raise ValueError("cannot decode")
        return 0, None

    monkeypatch.setattr(scan, "scan_video", fake_scan_video)

    scan.run_scan(args)

    metadata = json.loads(Path(f"{output_stem}.json").read_text(encoding="utf-8"))
    assert metadata["files_scanned"] == 1
