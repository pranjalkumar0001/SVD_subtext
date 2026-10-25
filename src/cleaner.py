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
    text = re.sub(r"\b(?:https?://|www\.)\S+", " ", text, flags=re.IGNORECASE)

    # Remove email addresses
    text = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"," ",text,flags=re.IGNORECASE)

    # Remove unwanted characters (keeping only alphanumeric and spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()