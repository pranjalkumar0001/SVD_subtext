import re
import numpy as np

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


def tokenize(text: str) -> list[str]:
    """Return lowercase English word tokens, preserving repetitions."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.findall(r"[a-z]+", text.lower())


def build_count_matrix(documents: dict[str, str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Return counts, row vocabulary, and column document names.

    Columns preserve the input dictionary's insertion order.
    """
    if not documents:
        raise ValueError("No documents provided.")

    doc_names = list(documents)
    tokenized_documents = [tokenize(clean_text(documents[name])) for name in doc_names]

    unique_words = set()
    for tokens in tokenized_documents:
        unique_words.update(tokens)

    vocabulary = sorted(unique_words)

    if not vocabulary:
        raise ValueError("No tokens found in the documents.")

    word_to_index = {word: index for index, word in enumerate(vocabulary)}

    matrix = np.zeros((len(vocabulary), len(doc_names)), dtype=np.float64)

    for doc_index, tokens in enumerate(tokenized_documents):
        for word in tokens:
            word_index = word_to_index[word]
            matrix[word_index, doc_index] += 1

    return matrix, vocabulary, doc_names


def compute_tfidf(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the TF-IDF matrix and IDF vector.

    Rows represent terms; columns represent documents.

    TF = raw count
    IDF = 1 + ln(number_of_documents / document_frequency)

    The input matrix is not modified.
    """
    counts = np.asarray(counts, dtype=np.float64)

    if counts.ndim != 2:
        raise ValueError("Counts must be a two-dimensional matrix.")

    if 0 in counts.shape:
        raise ValueError("Counts must contain at least one term and one document.")

    if not np.all(np.isfinite(counts)):
        raise ValueError("Counts must contain only finite values.")

    if np.any(counts < 0):
        raise ValueError("Counts cannot be negative.")

    number_of_documents = counts.shape[1]

    # Number of documents containing each term.
    document_frequency = np.count_nonzero(counts > 0, axis=1)

    if np.any(document_frequency == 0):
        raise ValueError("Each vocabulary term must occur in at least one document.")

    idf = 1 + np.log(number_of_documents / document_frequency)

    # Broadcast one IDF weight per row across all document columns.
    tfidf = counts * idf[:, None]

    return tfidf, idf