import pytest

from vidgrep.tui import _parse_optional_float


def test_optional_interval_requires_positive_finite_value():
    assert _parse_optional_float("") is None
    assert _parse_optional_float("2.5") == 2.5
    with pytest.raises(ValueError, match="greater than 0"):
        _parse_optional_float("-1")
    with pytest.raises(ValueError, match="greater than 0"):
        _parse_optional_float("nan")
