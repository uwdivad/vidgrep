import csv
import json
import re
from datetime import datetime, timezone

from vidgrep.inventory import VideoRecord, default_stem, iter_video_records, write_csv, write_json


def _touch(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _record(path, *, processed="false", size=10):
    return VideoRecord(
        filename=path.name,
        path=str(path),
        video_format=path.suffix.lstrip("."),
        size=size,
        date="2026-07-02T12:00:00-04:00",
        parent=str(path.parent),
        drive="C:",
        processed=processed,
    )


def test_default_stem_uses_date_only():
    assert default_stem(datetime(2026, 7, 2, 12, tzinfo=timezone.utc)) == "2026-07-02"


def test_iter_video_records_filters_by_filename_substring(tmp_path):
    _touch(tmp_path / "Call Of Duty highlight.mp4")
    _touch(tmp_path / "unrelated.mkv")
    _touch(tmp_path / "Call Of Duty notes.txt")

    records, skipped = iter_video_records([tmp_path], name_filter="call of duty")

    assert skipped == []
    assert [record.filename for record in records] == ["Call Of Duty highlight.mp4"]
    assert records[0].processed == "false"
    assert records[0].video_format == "mp4"


def test_iter_video_records_regex_matches_filename_or_directory(tmp_path):
    _touch(tmp_path / "Call Of Duty clip.mp4")
    _touch(tmp_path / "CallOfDuty" / "round-01.webm")
    _touch(tmp_path / "other" / "round-02.mp4")
    pattern = re.compile(r"call\s*of\s*duty", re.IGNORECASE)

    records, _ = iter_video_records([tmp_path], regex_filter=pattern, regex_scope="both")

    assert {record.filename for record in records} == {"Call Of Duty clip.mp4", "round-01.webm"}


def test_iter_video_records_regex_scope_can_target_directory_only(tmp_path):
    _touch(tmp_path / "Call Of Duty clip.mp4")
    _touch(tmp_path / "CallOfDuty" / "round-01.webm")
    pattern = re.compile(r"call\s*of\s*duty", re.IGNORECASE)

    records, _ = iter_video_records([tmp_path], regex_filter=pattern, regex_scope="directory")

    assert [record.filename for record in records] == ["round-01.webm"]


def test_write_csv_defaults_processed_false_and_preserves_status_columns(tmp_path):
    csv_path = tmp_path / "videos.csv"
    first_path = tmp_path / "a.mp4"
    second_path = tmp_path / "b.mkv"
    write_csv(csv_path, [_record(first_path)])

    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    rows[0]["processed"] = "true"
    rows[0]["last_attempt_at"] = "2026-07-02T16:00:00+00:00"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*rows[0].keys(), "last_attempt_at"])
        writer.writeheader()
        writer.writerows(rows)

    merged = write_csv(csv_path, [_record(first_path, size=99), _record(second_path)])
    reread = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))

    assert [row["filename"] for row in merged] == ["a.mp4", "b.mkv"]
    assert reread[0]["processed"] == "true"
    assert reread[0]["last_attempt_at"] == "2026-07-02T16:00:00+00:00"
    assert reread[0]["size"] == "99"
    assert reread[1]["processed"] == "false"


def test_write_json_includes_inventory_metadata(tmp_path):
    video_path = tmp_path / "clip.mp4"
    record = _record(video_path, size=123)
    json_path = tmp_path / "videos.json"
    started = datetime(2026, 7, 2, 16, tzinfo=timezone.utc)
    completed = datetime(2026, 7, 2, 17, tzinfo=timezone.utc)

    write_json(
        json_path,
        [record],
        started_at=started,
        completed_at=completed,
        roots=[tmp_path],
        name_filter="clip",
        regex_filter="call",
        regex_scope="both",
        skipped=[{"path": "denied", "error": "access"}],
        csv_row_count=2,
    )

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["file_count"] == 1
    assert data["csv_row_count"] == 2
    assert data["total_size_bytes"] == 123
    assert data["name_filter"] == "clip"
    assert data["regex_filter"] == "call"
    assert data["skipped_count"] == 1
    assert data["files"][0]["processed"] == "false"

