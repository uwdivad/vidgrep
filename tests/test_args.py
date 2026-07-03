import pytest

from args import build_agent_parser, build_inventory_parser, build_worker_parser


def test_inventory_parser_accepts_regex_scope():
    args = build_inventory_parser().parse_args(
        ["C:\\Media", "--regex", "call\\s*of\\s*duty", "--regex-scope", "directory"]
    )

    assert args.inputs == ["C:\\Media"]
    assert args.regex == "call\\s*of\\s*duty"
    assert args.regex_scope == "directory"


def test_inventory_parser_rejects_invalid_regex_scope():
    with pytest.raises(SystemExit):
        build_inventory_parser().parse_args(["--regex", "goal", "--regex-scope", "title"])


def test_worker_parser_minimum_required_arguments():
    args = build_worker_parser().parse_args(["videos.csv", "--text", "uwdivad"])

    assert args.csv == "videos.csv"
    assert args.text == "uwdivad"
    assert args.force is False
    assert args.skip_frames == 3


def test_agent_parser_minimum_required_arguments():
    args = build_agent_parser().parse_args(["results.jsonl", "--search-term", "uwdivad"])

    assert args.input == "results.jsonl"
    assert args.search_term == "uwdivad"
    assert args.merge_gap == 2.0
    assert args.openai_model == "gpt-5.4-nano"
    assert args.env_file == ".env"
    assert args.no_env_override is False
    assert args.force is False
