import numpy as np
import pytest

from src.similarity import (
    suggest_k,
    document_embeddings,
    cosine_similarity_matrix,
    retained_condition_number,
)


def test_suggest_k():
    Sigma = np.diag([5.0, 3.0, 1.0])

    # First component: 25/35 energy.
    # First two components: 34/35 energy.
    assert suggest_k(Sigma, energy_target=0.70) == 1
    assert suggest_k(Sigma, energy_target=0.90) == 2
    assert suggest_k(Sigma, energy_target=1.00) == 3


def test_suggest_k_is_scale_invariant():
    Sigma = np.diag([5.0, 3.0, 1.0])

    assert suggest_k(Sigma) == suggest_k(Sigma * 1e200)
    assert suggest_k(Sigma) == suggest_k(Sigma * 1e-200)


@pytest.mark.parametrize("target", [0.0, -0.1, 1.1, np.nan])
def test_suggest_k_rejects_invalid_target(target):
    with pytest.raises(ValueError):
        suggest_k(np.diag([3.0, 1.0]), energy_target=target)


def test_suggest_k_rejects_rank_zero():
    with pytest.raises(ValueError):
        suggest_k(np.empty((0, 0)))


def test_document_embeddings():
    Sigma = np.diag([5.0, 3.0, 1.0])
    Vt = np.array([
        [0.6, 0.8, 0.0],
        [-0.8, 0.6, 0.0],
        [0.0, 0.0, 1.0],
    ])

    embeddings = document_embeddings(Sigma, Vt, k=2)

    expected = np.array([
        [3.0, 4.0, 0.0],
        [-2.4, 1.8, 0.0],
    ])

    assert embeddings.shape == (2, 3)
    np.testing.assert_allclose(embeddings, expected)


def test_cosine_similarity_known_vectors():
    # Columns: identical direction, perpendicular, opposite direction.
    vectors = np.array([
        [1.0, 2.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    expected = np.array([
        [1.0, 1.0, 0.0, -1.0],
        [1.0, 1.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0, 1.0],
    ])

    np.testing.assert_allclose(
        cosine_similarity_matrix(vectors),
        expected,
        atol=1e-12,
    )


def test_cosine_similarity_ignores_positive_column_scaling():
    vectors = np.array([
        [1.0, 2.0, -1.0],
        [2.0, 1.0, 3.0],
    ])
    scaled = vectors * np.array([1e200, 1e-200, 7.0])

    np.testing.assert_allclose(
        cosine_similarity_matrix(scaled),
        cosine_similarity_matrix(vectors),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([1.0, 2.0]),             # Not two-dimensional.
        np.empty((2, 0)),                 # No documents.
        np.empty((0, 2)),                 # No features.
        np.array([[1.0, 0.0], [2.0, 0.0]]),  # Zero document.
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[1.0 + 1.0j]]),
    ],
)
def test_cosine_similarity_rejects_invalid_input(vectors):
    with pytest.raises(ValueError):
        cosine_similarity_matrix(vectors)


def test_retained_condition_number():
    Sigma = np.diag([12.0, 4.0, 2.0])

    assert retained_condition_number(Sigma, k=1) == pytest.approx(1.0)
    assert retained_condition_number(Sigma, k=2) == pytest.approx(3.0)
    assert retained_condition_number(Sigma, k=3) == pytest.approx(6.0)


@pytest.mark.parametrize("k", [0, -1, 4, 1.5, True])
def test_functions_reject_invalid_k(k):
    Sigma = np.diag([12.0, 4.0, 2.0])
    Vt = np.eye(3)

    with pytest.raises(ValueError):
        document_embeddings(Sigma, Vt, k)

    with pytest.raises(ValueError):
        retained_condition_number(Sigma, k)