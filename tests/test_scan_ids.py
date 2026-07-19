from types import SimpleNamespace

from vidgrep import scan


def _args(**overrides):
    values = {
        "text": "uwdivad",
        "threshold": 0.5,
        "skip_frames": 3,
        "interval": None,
        "region": None,
        "lang": "en",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_source_id_is_stable_for_same_file_metadata(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_text("video", encoding="utf-8")

    assert scan._make_source_id(video) == scan._make_source_id(video)


def test_match_id_is_deterministic_and_changes_with_text():
    options_id = scan._scan_options_id(_args())

    first = scan._make_match_id(
        source_id="source",
        scan_options_id=options_id,
        timestamp=1.2344,
        text="uWdiVad enemy",
        confidence=0.81234,
    )
    second = scan._make_match_id(
        source_id="source",
        scan_options_id=options_id,
        timestamp=1.2344,
        text="uWdiVad enemy",
        confidence=0.81234,
    )
    changed = scan._make_match_id(
        source_id="source",
        scan_options_id=options_id,
        timestamp=1.2344,
        text="uWdiVad teammate",
        confidence=0.81234,
    )

    assert first == second
    assert first != changed

