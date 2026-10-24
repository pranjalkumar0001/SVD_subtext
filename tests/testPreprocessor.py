import pytest

from src.tokenizer import tokenize


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Robot uses SENSOR. Robot moves!",
            ["robot", "uses", "sensor", "robot", "moves"],
        ),
        ("robot,sensor", ["robot", "sensor"]),
        ("", []),
        ("123 !!!", []),
        ("robot robot", ["robot", "robot"]),
    ],
)
def test_tokenize(text, expected):
    assert tokenize(text) == expected


def test_tokenize_rejects_non_string():
    with pytest.raises(TypeError):
        tokenize(None)