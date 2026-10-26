import numpy as np


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