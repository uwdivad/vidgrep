import pytest

from args import build_inventory_parser, build_worker_parser


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

