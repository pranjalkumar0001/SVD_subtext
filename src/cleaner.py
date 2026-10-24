import re

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing unwanted characters, urls, email addresses and formatting.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove unwanted characters (keeping only alphanumeric and spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()