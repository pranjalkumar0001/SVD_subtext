import numpy as np
import pytest

from experiments.experiment_big import precision_at_k


@pytest.fixture
def example():
    # Documents 0 and 1 belong to category A.
    # Documents 2 and 3 belong to category B.
    similarities = np.array([
        [1.0, 0.9, 0.2, 0.1],
        [0.9, 1.0, 0.1, 0.2],
        [0.2, 0.1, 1.0, 0.8],
        [0.1, 0.2, 0.8, 1.0],
    ])
    labels = ["A", "A", "B", "B"]
    return similarities, labels


def test_nearest_neighbour_has_same_category(example):
    similarities, labels = example

    scores = precision_at_k(similarities, labels, k=1)

    np.testing.assert_array_equal(scores, [1.0, 1.0, 1.0, 1.0])


def test_precision_uses_requested_number_of_neighbours(example):
    similarities, labels = example

    # Each document has exactly one relevant neighbour.
    # Among its nearest two neighbours, one is relevant.
    scores = precision_at_k(similarities, labels, k=2)

    np.testing.assert_allclose(scores, [0.5, 0.5, 0.5, 0.5])


def test_all_other_documents_can_be_requested(example):
    similarities, labels = example

    # There are three other documents and exactly one is relevant.
    scores = precision_at_k(similarities, labels, k=3)

    np.testing.assert_allclose(scores, np.full(4, 1 / 3))


def test_self_matches_are_excluded():
    similarities = np.array([
        [1.0, 0.8, 0.3],
        [0.8, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ])

    # Every category is unique. Only a document itself would be relevant.
    labels = ["A", "B", "C"]

    scores = precision_at_k(similarities, labels, k=1)

    np.testing.assert_array_equal(scores, [0.0, 0.0, 0.0])


def test_mixed_relevance_produces_per_document_scores(example):
    similarities, _ = example
    labels = ["A", "A", "B", "C"]

    scores = precision_at_k(similarities, labels, k=1)

    np.testing.assert_array_equal(scores, [1.0, 1.0, 0.0, 0.0])
    assert np.mean(scores) == pytest.approx(0.5)


def test_negative_cosines_are_ranked_by_value():
    similarities = np.array([
        [1.0, -0.1, -0.9],
        [-0.1, 1.0, -0.8],
        [-0.9, -0.8, 1.0],
    ])
    labels = ["A", "A", "B"]

    # -0.1 is more similar than -0.9; absolute values must not be used.
    scores = precision_at_k(similarities, labels, k=1)

    np.testing.assert_array_equal(scores, [1.0, 1.0, 0.0])


def test_exact_ties_follow_document_order():
    similarities = np.full((4, 4), 0.5)
    np.fill_diagonal(similarities, 1.0)
    labels = ["A", "B", "A", "B"]

    # After self-exclusion, stable sorting chooses:
    # document 0 -> 1, document 1 -> 0,
    # document 2 -> 0, document 3 -> 0.
    scores = precision_at_k(similarities, labels, k=1)

    np.testing.assert_array_equal(scores, [0.0, 0.0, 1.0, 0.0])


def test_inputs_are_not_modified(example):
    similarities, labels = example
    original_similarities = similarities.copy()
    original_labels = labels.copy()

    precision_at_k(similarities, labels, k=2)

    np.testing.assert_array_equal(
        similarities, original_similarities
    )
    assert labels == original_labels


@pytest.mark.parametrize("k", [-1, 0, 4, 5])
def test_invalid_neighbour_count(example, k):
    similarities, labels = example

    with pytest.raises(ValueError):
        precision_at_k(similarities, labels, k=k)


@pytest.mark.parametrize(
    "similarities, labels",
    [
        (np.zeros((3, 2)), ["A", "B", "C"]),
        (np.eye(3), ["A", "B"]),
        (np.array([1.0, 0.5, 0.2]), ["A", "B", "C"]),
    ],
    ids=["not-square", "label-count-mismatch", "not-two-dimensional"],
)
def test_invalid_matrix_shape(similarities, labels):
    with pytest.raises(ValueError):
        precision_at_k(similarities, labels, k=1)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_similarity_is_rejected(example, invalid_value):
    similarities, labels = example
    similarities[0, 1] = invalid_value

    with pytest.raises(ValueError):
        precision_at_k(similarities, labels, k=1)


def test_single_document_has_no_valid_neighbour():
    with pytest.raises(ValueError):
        precision_at_k(np.array([[1.0]]), ["A"], k=1)


def test_empty_collection_is_rejected():
    with pytest.raises(ValueError):
        precision_at_k(np.empty((0, 0)), [], k=1)