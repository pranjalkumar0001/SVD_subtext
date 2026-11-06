import numpy as np


def suggest_k(Sigma: np.ndarray,energy_target: float = 0.90) -> int:
    """
    Return the smallest k that retains the requested spectral energy.

    Sigma must be a diagonal matrix of positive singular values,
    sorted in descending order, as returned by the SVD engine.

    Energy is measured relative to the singular values retained by
    the engine. This is a reconstruction heuristic, not a guarantee
    of better document similarity.
    """
    if not 0 < energy_target <= 1:
        raise ValueError("energy_target must be in (0, 1].")

    singular_values = np.diag(Sigma) # get singular values from diagonal matrix

    if singular_values.size == 0:
        raise ValueError("Cannot suggest k for a rank-zero matrix.")

    # Scaling prevents overflow when squaring large singular values.
    relative_energy = (singular_values / singular_values[0]) ** 2

    cumulative_energy = np.cumsum(relative_energy)
    cumulative_energy /= cumulative_energy[-1]

    return int(np.searchsorted(cumulative_energy, energy_target, side="left") + 1)


def document_embeddings(Sigma: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """
    Return a (k, number_of_documents) matrix of LSA coordinates.

    Each column represents one document:
        Z = Sigma_k @ Vt_k
    Sigma and Vt must be matching factors from the SVD engine.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer.")

    rank = Sigma.shape[0]

    if not 1 <= k <= rank:
        raise ValueError(f"k must be between 1 and {rank}.")

    return Sigma[:k, :k] @ Vt[:k, :]


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Return pairwise cosine similarities between document columns.

    Input shape:
        (number_of_features, number_of_documents)

    Output shape:
        (number_of_documents, number_of_documents)

    Raises ValueError for empty, complex, nonfinite, or zero vectors.
    """
    vectors = np.asarray(vectors)

    if np.iscomplexobj(vectors):
        raise ValueError("Document vectors must be real-valued.")

    vectors = np.asarray(vectors, dtype=np.float64)

    if vectors.ndim != 2 or 0 in vectors.shape:
        raise ValueError("Expected a nonempty two-dimensional matrix.")

    if not np.all(np.isfinite(vectors)):
        raise ValueError("Document vectors must contain only finite values.")

    # Scale each document before computing its length to avoid overflow.
    scales = np.max(np.abs(vectors), axis=0)

    if np.any(scales == 0):
        raise ValueError(
            "A document has a zero vector; cosine similarity is undefined."
        )

    scaled_vectors = vectors / scales
    lengths = np.sqrt(np.sum(scaled_vectors ** 2, axis=0))
    normalized_vectors = scaled_vectors / lengths

    similarities = normalized_vectors.T @ normalized_vectors

    # Floating-point rounding can produce values slightly outside [-1, 1].
    return np.clip(similarities, -1.0, 1.0)


def retained_condition_number(Sigma: np.ndarray, k: int) -> float:
    """
    Return sigma_1 / sigma_k for the retained k-dimensional subspace.

    Sigma must contain positive singular values in descending order.

    This is the condition number of the retained diagonal block,
    not the ordinary condition number of a rank-deficient
    reconstructed matrix.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer.")

    rank = Sigma.shape[0]

    if not 1 <= k <= rank:
        raise ValueError(f"k must be between 1 and {rank}.")

    return float(Sigma[0, 0] / Sigma[k - 1, k - 1])