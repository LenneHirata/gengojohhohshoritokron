import pytest
from utils.c2n_map import C2NMap


def test_valid_c2n_map():
    valid_map = C2NMap(c2n={"a": 1, "b": 2, "c": 0, "d": 9})
    assert valid_map.c2n == {"a": 1, "b": 2, "c": 0, "d": 9}


def test_invalid_key_length():
    with pytest.raises(ValueError, match="キーは1文字である必要があります"):
        C2NMap(c2n={"ab": 1})


def test_invalid_value_range():
    with pytest.raises(ValueError, match="値は0から9の間である必要があります"):
        C2NMap(c2n={"a": 10})
