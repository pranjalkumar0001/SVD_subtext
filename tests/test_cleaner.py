import pytest

from src.cleaner import clean_text

@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Check out this link: https://example.com ",
            "check out this link",
        ),
        (
            "Contact me at pranjalkumar.pk.0001@gmail.com",
            "contact me at",
        ),
        (
            "Hello, World! This is a test.",
            "hello world this is a test",
        ),
        (
            "Multiple     spaces should be normalized.",
            "multiple spaces should be normalized",
        ),
        (
            "Special characters !@#$%^&*() should be removed.",
            "special characters should be removed",
        ),
        (
            "robot,sensor",
            "robot sensor"
        ),
        (
            "robot/motor",
            "robot motor"
        ),
        (
            "HTTP://example.com robot",
            "robot"
        ),
        (
            "httpology robot",
            "httpology robot"
        ),
        (
            "",
            ""
        ),
        (
            "robot@home sensor",
            "robot home sensor"
        ),
    ]
    )
def test_clean_text(text, expected):
    assert clean_text(text) == expected

def test_clean_text_rejects_non_string():
    with pytest.raises(TypeError):
        clean_text(None)
