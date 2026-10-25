import numpy as np

from src.cleaner import clean_text
from src.tokenizer import tokenize


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