import json

import agent


def _row(match_id, timestamp, text, confidence=0.8, scan_id="scan-1"):
    return {
        "match_id": match_id,
        "scan_id": scan_id,
        "source_id": "source-1",
        "file": "clip.mp4",
        "path": "C:\\Videos\\clip.mp4",
        "timestamp": timestamp,
        "text": text,
        "confidence": confidence,
    }


def test_build_openai_payload_uses_structured_outputs_schema():
    payload = agent._build_openai_payload(
        agent.DEFAULT_MODEL,
        "uwdivad",
        [{"input_id": "uwdivad enemy", "text": "uWdiVad enemy"}],
    )

    assert payload["model"] == "gpt-5.4-nano"
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"]["required"] == ["items"]


def test_load_dotenv_overrides_existing_values_by_default(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join([
            "# local secrets",
            "OPENAI_API_KEY=from-file",
            "OPENAI_MODEL='quoted-model'",
            "EXISTING=from-file",
        ]),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("EXISTING", "from-env")

    loaded = agent.load_dotenv(env_path)

    assert loaded == 3
    assert agent.os.environ["OPENAI_API_KEY"] == "from-file"
    assert agent.os.environ["OPENAI_MODEL"] == "quoted-model"
    assert agent.os.environ["EXISTING"] == "from-file"


def test_load_dotenv_can_keep_existing_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("EXISTING=from-file", encoding="utf-8")
    monkeypatch.setenv("EXISTING", "from-env")

    loaded = agent.load_dotenv(env_path, override=False)

    assert loaded == 0
    assert agent.os.environ["EXISTING"] == "from-env"


def test_extract_response_text_accepts_output_text():
    response = {"output_text": json.dumps({"items": []})}

    assert agent._extract_response_text(response) == '{"items": []}'


def test_group_rows_merges_case_variants_and_averages_confidence():
    rows = [
        _row("m1", 1.0, "uWdiVad enemy", 0.7),
        _row("m2", 2.0, "uwdivad Enemy", 0.9),
    ]
    cache = {
        "uwdivad enemy": {
            "anchor_present": True,
            "canonical_label": "enemy",
            "normalized_line": "uwdivad enemy",
        }
    }

    groups = agent.group_rows(rows, cache, merge_gap=2.0)

    assert len(groups) == 1
    assert groups[0]["canonical_text"] == "enemy"
    assert groups[0]["start_timestamp"] == 1.0
    assert groups[0]["end_timestamp"] == 2.0
    assert groups[0]["match_count"] == 2
    assert groups[0]["average_confidence"] == 0.8
    assert groups[0]["match_ids"] == ["m1", "m2"]


def test_group_rows_splits_same_label_after_gap():
    rows = [
        _row("m1", 1.0, "uWdiVad enemy"),
        _row("m2", 2.0, "uwdivad enemy"),
        _row("m3", 7.0, "UWDIVAD enemy"),
    ]
    cache = {
        "uwdivad enemy": {
            "anchor_present": True,
            "canonical_label": "enemy",
            "normalized_line": "uwdivad enemy",
        }
    }

    groups = agent.group_rows(rows, cache, merge_gap=2.0)

    assert len(groups) == 2
    assert [group["match_count"] for group in groups] == [2, 1]
    assert groups[1]["start_timestamp"] == 7.0


def test_group_rows_splits_different_adjacent_labels():
    rows = [
        _row("m1", 1.0, "uWdiVad enemy"),
        _row("m2", 2.0, "uwdivad teammate"),
    ]
    cache = {
        "uwdivad enemy": {
            "anchor_present": True,
            "canonical_label": "enemy",
            "normalized_line": "uwdivad enemy",
        },
        "uwdivad teammate": {
            "anchor_present": True,
            "canonical_label": "teammate",
            "normalized_line": "uwdivad teammate",
        },
    }

    groups = agent.group_rows(rows, cache, merge_gap=2.0)

    assert [group["canonical_text"] for group in groups] == ["enemy", "teammate"]


def test_group_rows_deduplicates_match_ids():
    rows = [
        _row("m1", 1.0, "uWdiVad enemy", 0.7),
        _row("m1", 1.0, "uWdiVad enemy", 0.7),
        _row("m2", 2.0, "uwdivad enemy", 0.9),
    ]
    cache = {
        "uwdivad enemy": {
            "anchor_present": True,
            "canonical_label": "enemy",
            "normalized_line": "uwdivad enemy",
        }
    }

    groups = agent.group_rows(rows, cache, merge_gap=2.0)

    assert len(groups) == 1
    assert groups[0]["match_count"] == 2
    assert groups[0]["match_ids"] == ["m1", "m2"]
