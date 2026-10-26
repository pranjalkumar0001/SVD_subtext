import numpy as np
import pytest

from src.preprocessing import compute_tfidf


def test_tfidf_values():
    counts = np.array([
        [1, 1],  # cake: present in both documents
        [2, 0],  # coffee: present in one document
        [0, 1],  # tea: present in one document
    ])

    tfidf, idf = compute_tfidf(counts)

    expected_idf = np.array([
        1.0,
        1.6931471805599453,
        1.6931471805599453,
    ])
    expected_tfidf = np.array([
        [1.0, 1.0],
        [3.3862943611198906, 0.0],
        [0.0, 1.6931471805599453],
    ])

    np.testing.assert_allclose(idf, expected_idf)
    np.testing.assert_allclose(tfidf, expected_tfidf)


def test_single_document():
    counts = np.array([[3], [1]])

    tfidf, idf = compute_tfidf(counts)

    np.testing.assert_array_equal(idf, [1.0, 1.0])
    np.testing.assert_array_equal(tfidf, counts)


def test_empty_document_column():
    counts = np.array([
        [2, 0],
        [1, 0],
    ])

    tfidf, idf = compute_tfidf(counts)

    np.testing.assert_allclose(
        idf, [1.6931471805599453, 1.6931471805599453]
    )
    np.testing.assert_array_equal(tfidf[:, 1], [0.0, 0.0])


def test_input_is_not_modified():
    counts = np.array([
        [2.0, 0.0],
        [1.0, 1.0],
    ])
    original = counts.copy()

    compute_tfidf(counts)

    np.testing.assert_array_equal(counts, original)


@pytest.mark.parametrize(
    "counts",
    [
        np.array(3),
        np.array([1, 2, 3]),
        np.ones((2, 2, 2)),
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.array([[1.0, -1.0]]),
        np.array([[1.0, np.nan]]),
        np.array([[1.0, np.inf]]),
        np.array([[1.0, -np.inf]]),
        np.array([[1, 0], [0, 0]]),  # Absent vocabulary term
    ],
    ids=[
        "scalar",
        "one-dimensional",
        "three-dimensional",
        "no-terms",
        "no-documents",
        "negative-count",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "zero-document-frequency",
    ],
)
def test_invalid_input(counts):
    with pytest.raises(ValueError):
        compute_tfidf(counts)