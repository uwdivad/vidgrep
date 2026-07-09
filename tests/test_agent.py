import json
from types import SimpleNamespace

import agent


class _FakeCanonicalizer:
    def __init__(self, results_by_call):
        self.results_by_call = list(results_by_call)
        self.calls = 0

    def canonicalize(self, *, search_term, items):
        results = self.results_by_call[self.calls]
        self.calls += 1
        return results


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


def test_canonicalize_missing_ignores_unrequested_ids_and_saves_per_batch(capsys):
    rows = [_row("m1", 1.0, "uwdivad enemy"), _row("m2", 2.0, "uwdivad teammate")]
    cache = {}
    saves = []
    canonicalizer = _FakeCanonicalizer([
        [
            agent.CanonicalResult("uwdivad enemy", True, "enemy", "uwdivad enemy"),
            agent.CanonicalResult("hallucinated key", True, "bogus", "bogus"),
        ],
        [],
    ])

    completed = agent._canonicalize_missing(
        rows,
        search_term="uwdivad",
        canonicalizer=canonicalizer,
        cache=cache,
        batch_size=1,
        on_batch_done=lambda: saves.append(dict(cache)),
    )

    captured = capsys.readouterr()
    assert completed == 1
    assert "hallucinated key" not in cache
    assert cache["uwdivad enemy"]["canonical_label"] == "enemy"
    assert "uwdivad teammate" not in cache  # omitted by the model, left for retry
    assert "ignoring unrequested input_id" in captured.err
    assert "omitted 1 item(s)" in captured.err
    assert len(saves) == 2  # state persisted after every batch


def test_process_once_discards_cache_when_search_term_changes(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "results.jsonl"
    jsonl_path.write_text(json.dumps(_row("m1", 1.0, "uwdivad enemy")) + "\n", encoding="utf-8")
    state_path = tmp_path / "results.agent.state.json"

    def args_for(term):
        return SimpleNamespace(
            input=str(jsonl_path),
            search_term=term,
            output=None,
            merge_gap=2.0,
            openai_model="test-model",
            env_file=str(tmp_path / "missing.env"),
            no_env_override=False,
            batch_size=100,
            force=False,
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = []

    def fake_canonicalize_missing(rows, *, search_term, canonicalizer, cache, batch_size, on_batch_done=None):
        calls.append(search_term)
        cache["uwdivad enemy"] = {
            "anchor_present": True,
            "canonical_label": search_term,
            "normalized_line": "uwdivad enemy",
        }
        return 1

    monkeypatch.setattr(agent, "_canonicalize_missing", fake_canonicalize_missing)

    agent._process_once(args_for("first"), started_at=agent._now())
    agent._process_once(args_for("first"), started_at=agent._now())
    agent._process_once(args_for("second"), started_at=agent._now())

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert calls == ["first", "second"]  # cached on repeat, refreshed on change
    assert state["canonical_cache"]["uwdivad enemy"]["canonical_label"] == "second"


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
