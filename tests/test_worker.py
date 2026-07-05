import csv
import json
from types import SimpleNamespace

import worker


def _args(csv_path, **overrides):
    values = {
        "csv": str(csv_path),
        "text": "uwdivad",
        "output_dir": None,
        "limit": None,
        "force": False,
        "skip_frames": 3,
        "interval": None,
        "batch_size": 8,
        "threshold": 0.5,
        "region": None,
        "lang": "en",
        "no_gpu": False,
        "stats": False,
        "profile": False,
        "metrics_output": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_rows(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_load_csv_adds_worker_columns_and_default_processed(tmp_path):
    csv_path = tmp_path / "videos.csv"
    video_path = tmp_path / "clip.mp4"
    _write_rows(csv_path, [{"path": str(video_path)}])

    rows, fieldnames = worker._load_csv(csv_path)

    assert rows[0]["processed"] == "false"
    assert "last_attempt_at" in fieldnames
    assert "error" in fieldnames


def test_run_scan_job_writes_jsonl_and_metadata(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_text("", encoding="utf-8")
    output_stem = tmp_path / "out" / "clip"
    args = _args(tmp_path / "videos.csv")

    def fake_scan_video(*, video_path, detector, args, options_id, on_record):
        on_record({
            "match_id": "match-1",
            "scan_id": "scan-1",
            "source_id": "source-1",
            "file": video_path.name,
            "path": str(video_path),
            "timestamp": 1.0,
            "text": "uwdivad enemy",
            "confidence": 0.9,
        })
        return 1, None

    monkeypatch.setattr("scan.scan_video", fake_scan_video)

    match_count = worker._run_scan_job(args, video_path, output_stem, object(), "options-1")

    assert match_count == 1
    assert json.loads(output_stem.with_suffix(".jsonl").read_text(encoding="utf-8"))["match_id"] == "match-1"
    assert json.loads(output_stem.with_suffix(".json").read_text(encoding="utf-8"))["match_count"] == 1


def test_run_worker_skips_processed_and_marks_success(tmp_path, monkeypatch):
    first = tmp_path / "done.mp4"
    second = tmp_path / "todo.mp4"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    csv_path = tmp_path / "videos.csv"
    _write_rows(
        csv_path,
        [
            {"path": str(first), "processed": "true"},
            {"path": str(second), "processed": "false"},
        ],
    )
    calls = []

    def fake_run_scan_job(args, video_path, output_stem, detector, options_id):
        calls.append((video_path, detector))
        output_stem.with_suffix(".json").write_text(json.dumps({"match_count": 3}), encoding="utf-8")
        return 3

    monkeypatch.setattr(worker, "_build_detector", lambda args: "detector")
    monkeypatch.setattr(worker, "_run_scan_job", fake_run_scan_job)

    worker.run_worker(_args(csv_path))

    rows = _read_rows(csv_path)
    assert len(calls) == 1
    assert calls[0] == (second, "detector")
    assert rows[0]["processed"] == "true"
    assert rows[1]["processed"] == "true"
    assert rows[1]["match_count"] == "3"
    assert rows[1]["error"] == ""


def test_run_worker_failed_row_does_not_stop_next_row(tmp_path, monkeypatch):
    missing = tmp_path / "missing.mp4"
    existing = tmp_path / "existing.mp4"
    existing.write_text("", encoding="utf-8")
    csv_path = tmp_path / "videos.csv"
    _write_rows(
        csv_path,
        [
            {"path": str(missing), "processed": "false"},
            {"path": str(existing), "processed": "false"},
        ],
    )

    def fake_run_scan_job(args, video_path, output_stem, detector, options_id):
        output_stem.with_suffix(".json").write_text(json.dumps({"match_count": 1}), encoding="utf-8")
        return 1

    monkeypatch.setattr(worker, "_build_detector", lambda args: "detector")
    monkeypatch.setattr(worker, "_run_scan_job", fake_run_scan_job)

    worker.run_worker(_args(csv_path))

    rows = _read_rows(csv_path)
    assert rows[0]["processed"] == "false"
    assert "not found" in rows[0]["error"]
    assert rows[1]["processed"] == "true"
    assert rows[1]["match_count"] == "1"


def test_run_worker_exception_leaves_row_unprocessed(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_text("", encoding="utf-8")
    csv_path = tmp_path / "videos.csv"
    _write_rows(csv_path, [{"path": str(video_path), "processed": "false"}])

    def fake_run_scan_job(args, video_path, output_stem, detector, options_id):
        raise RuntimeError("scan failed")

    monkeypatch.setattr(worker, "_build_detector", lambda args: "detector")
    monkeypatch.setattr(worker, "_run_scan_job", fake_run_scan_job)

    worker.run_worker(_args(csv_path))

    rows = _read_rows(csv_path)
    assert rows[0]["processed"] == "false"
    assert rows[0]["exit_code"] == ""
    assert rows[0]["error"] == "scan failed"


def test_run_worker_reuses_detector_for_multiple_rows(tmp_path, monkeypatch):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    csv_path = tmp_path / "videos.csv"
    _write_rows(
        csv_path,
        [
            {"path": str(first), "processed": "false"},
            {"path": str(second), "processed": "false"},
        ],
    )
    detector_builds = []
    detectors_used = []

    def fake_build_detector(args):
        detector_builds.append(args.text)
        return object()

    def fake_run_scan_job(args, video_path, output_stem, detector, options_id):
        detectors_used.append(detector)
        output_stem.with_suffix(".json").write_text(json.dumps({"match_count": 0}), encoding="utf-8")
        return 0

    monkeypatch.setattr(worker, "_build_detector", fake_build_detector)
    monkeypatch.setattr(worker, "_run_scan_job", fake_run_scan_job)

    worker.run_worker(_args(csv_path))

    assert detector_builds == ["uwdivad"]
    assert len(detectors_used) == 2
    assert detectors_used[0] is detectors_used[1]


def test_run_scan_job_writes_profile_metrics(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_text("", encoding="utf-8")
    output_stem = tmp_path / "out" / "clip"
    metrics_path = tmp_path / "metrics.csv"
    args = _args(tmp_path / "videos.csv", profile=True, metrics_output=str(metrics_path))

    def fake_scan_video(*, video_path, detector, args, options_id, on_record):
        return 0, {
            "started_at": "2026-07-04T00:00:00+00:00",
            "completed_at": "2026-07-04T00:00:01+00:00",
            "path": str(video_path),
            "file": video_path.name,
            "batch_size": args.batch_size,
            "match_count": 0,
            "detect_fps": 12.5,
        }

    monkeypatch.setattr("scan.scan_video", fake_scan_video)

    worker._run_scan_job(args, video_path, output_stem, object(), "options-1")

    rows = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8")))
    metadata = json.loads(output_stem.with_suffix(".json").read_text(encoding="utf-8"))
    assert rows[0]["file"] == "clip.mp4"
    assert rows[0]["detect_fps"] == "12.5"
    assert metadata["profile_metrics"] == str(metrics_path)
