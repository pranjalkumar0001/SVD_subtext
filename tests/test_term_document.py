import numpy as np
import pytest

from src.preprocessing import build_count_matrix


def test_counts_and_labels():
    # Columns should preserve insertion order, not alphabetical order.
    documents = {
        "b.txt": "Coffee coffee cake.",
        "a.txt": "Cake tea.",
    }

    matrix, vocabulary, names = build_count_matrix(documents)

    assert names == ["b.txt", "a.txt"]
    assert vocabulary == ["cake", "coffee", "tea"]

    np.testing.assert_array_equal(
        matrix,
        np.array([
            [1., 1.],  # cake
            [2., 0.],  # coffee
            [0., 1.],  # tea
        ]),
    )


def test_cleaning_is_applied():
    documents = {
        "travel.txt": "Beach https://example.com sunset!",
    }

    matrix, vocabulary, names = build_count_matrix(documents)

    assert names == ["travel.txt"]
    assert vocabulary == ["beach", "sunset"]
    np.testing.assert_array_equal(matrix, [[1.], [1.]])


def test_empty_document_keeps_its_column():
    documents = {
        "empty.txt": "",
        "books.txt": "Novel novel.",
    }

    matrix, vocabulary, names = build_count_matrix(documents)

    assert names == ["empty.txt", "books.txt"]
    assert vocabulary == ["novel"]
    np.testing.assert_array_equal(matrix, [[0., 2.]])


def test_no_documents():
    with pytest.raises(ValueError):
        build_count_matrix({})


def test_no_tokens_across_documents():
    with pytest.raises(ValueError):
        build_count_matrix({
            "empty.txt": "",
            "symbols.txt": "123 !!!",
        })