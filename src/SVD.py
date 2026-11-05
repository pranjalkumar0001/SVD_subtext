import numpy as np


def svd_from_gram(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a compact SVD using the smaller Gram matrix.

    Parameters
    ----------
    matrix : np.ndarray Nonempty, finite, real matrix with shape (m, n).

    Returns
    -------
    U : np.ndarray, Left singular vectors, shape (m, r), orthonormal.
    s : np.ndarray, Positive singular values in descending order, shape (r, r), diagonal matrix.
    Vt : np.ndarray, Transposed right singular vectors, shape (r, n), orthonormal.

    Here r is the estimated numerical rank. Reconstruction:
        matrix ≈ (U * s) @ Vt

    Notes
    -----
    Gram-matrix formation reduces accuracy for small singular values.
    Eigenvalues below a numerical tolerance are discarded.
    The input is not modified.
    """
    raw = np.asarray(matrix)

    if np.iscomplexobj(raw):
        raise ValueError("Only real-valued matrices are supported.")

    A = np.asarray(raw, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("Input must be a two-dimensional matrix.")

    if 0 in A.shape:
        raise ValueError("Input must have at least one row and column.")

    if not np.all(np.isfinite(A)):
        raise ValueError("Input must contain only finite values.")

    m, n = A.shape
    # Scale to reduce overflow risk when forming the Gram matrix.
    scale = np.max(np.abs(A))
    
    # Compact SVD of a zero matrix has rank zero.
    if scale == 0:
        return (
            np.empty((m, 0), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty((0, n), dtype=np.float64),
        )

    B = A / scale

    # Choose the smaller symmetric eigenvalue problem.
    if m >= n:
        gram = B.T @ B
    else:
        gram = B @ B.T

    eigenvalues, eigenvectors = np.linalg.eigh(gram)

    # Sort eigenvalues and their corresponding eigenvector columns.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Numerical rank threshold at the Gram-matrix scale.
    # The factor 10 is a conservative heuristic, not an exact bound.
    eps = np.finfo(np.float64).eps
    tolerance = 10.0 * eps * max(m, n) * eigenvalues[0]

    if np.any(eigenvalues < -tolerance):
        raise np.linalg.LinAlgError(
            "Gram matrix has a significantly negative eigenvalue."
        )

    # Exclude numerical zeros and tiny negative round-off errors.
    keep = eigenvalues > tolerance
    s_scaled = np.sqrt(eigenvalues[keep])
    vectors = eigenvectors[:, keep]

    # Recover the other singular-vector factor.
    if m >= n:
        V = vectors
        U = (B @ V) / s_scaled
    else:
        U = vectors
        V = (B.T @ U) / s_scaled

    # Choose paired signs so each U column's largest entry
    # in absolute value is positive.
    columns = np.arange(U.shape[1])
    largest_rows = np.argmax(np.abs(U), axis=0)
    signs = np.where(
        U[largest_rows, columns] < 0.0,
        -1.0,
        1.0,
    )

    U = U * signs
    V = V * signs

    # Restore singular values to the original scale.
    with np.errstate(over="ignore"):
        s = s_scaled * scale

    if not np.all(np.isfinite(s)):
        raise FloatingPointError(
            "Singular values exceed the float64 numerical range."
        )

    return U, s, V.T