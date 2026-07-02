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


def test_build_scan_command_includes_worker_options(tmp_path):
    video_path = tmp_path / "clip.mp4"
    output_stem = tmp_path / "out" / "clip"
    args = _args(
        tmp_path / "videos.csv",
        interval=2,
        region=[132, 476, 592, 388],
        no_gpu=True,
        stats=True,
        batch_size=16,
        threshold=0.7,
    )

    cmd = worker._build_scan_command(args, video_path, output_stem)

    assert cmd[:4] == [worker.sys.executable, "-u", str(worker.Path(worker.__file__).with_name("main.py")), "scan"]
    assert "--interval" in cmd
    assert cmd[cmd.index("--interval") + 1] == "2"
    assert cmd[cmd.index("--region") + 1 : cmd.index("--region") + 5] == ["132", "476", "592", "388"]
    assert "--no-gpu" in cmd
    assert "--stats" in cmd


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

    def fake_run_streaming(cmd):
        calls.append(cmd)
        output_stem = worker.Path(cmd[cmd.index("--output") + 1])
        output_stem.with_suffix(".json").write_text(json.dumps({"match_count": 3}), encoding="utf-8")
        return 0, ""

    monkeypatch.setattr(worker, "_run_streaming", fake_run_streaming)

    worker.run_worker(_args(csv_path))

    rows = _read_rows(csv_path)
    assert len(calls) == 1
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

    def fake_run_streaming(cmd):
        output_stem = worker.Path(cmd[cmd.index("--output") + 1])
        output_stem.with_suffix(".json").write_text(json.dumps({"match_count": 1}), encoding="utf-8")
        return 0, ""

    monkeypatch.setattr(worker, "_run_streaming", fake_run_streaming)

    worker.run_worker(_args(csv_path))

    rows = _read_rows(csv_path)
    assert rows[0]["processed"] == "false"
    assert "not found" in rows[0]["error"]
    assert rows[1]["processed"] == "true"
    assert rows[1]["match_count"] == "1"


def test_run_worker_nonzero_exit_leaves_row_unprocessed(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_text("", encoding="utf-8")
    csv_path = tmp_path / "videos.csv"
    _write_rows(csv_path, [{"path": str(video_path), "processed": "false"}])

    monkeypatch.setattr(worker, "_run_streaming", lambda cmd: (2, "scan failed"))

    worker.run_worker(_args(csv_path))

    rows = _read_rows(csv_path)
    assert rows[0]["processed"] == "false"
    assert rows[0]["exit_code"] == "2"
    assert rows[0]["error"] == "scan failed"

