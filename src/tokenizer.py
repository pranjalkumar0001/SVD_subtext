import re


def tokenize(text: str) -> list[str]:
    """Return lowercase English word tokens, preserving repetitions."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.findall(r"[a-z]+", text.lower())