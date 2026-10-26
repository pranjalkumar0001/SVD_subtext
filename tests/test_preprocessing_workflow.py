import numpy as np

from src.data_loader import load_txt_documents
from src.preprocessing import build_count_matrix, compute_tfidf


def test_complete_preprocessing_workflow(tmp_path):
    # Create files out of order to check the loader's sorting.
    (tmp_path / "b.txt").write_text(
        "Cake tea! Contact reader@example.com",
        encoding="utf-8",
    )
    (tmp_path / "a.TXT").write_text(
        "Coffee, coffee cake. https://example.com",
        encoding="utf-8-sig",
    )

    # Neither of these should become a document.
    (tmp_path / "notes.csv").write_text(
        "Ignore this content.",
        encoding="utf-8",
    )
    (tmp_path / "folder.txt").mkdir()

    # Run the actual workflow.
    documents = load_txt_documents(tmp_path)
    counts, vocabulary, document_names = build_count_matrix(documents)
    tfidf, idf = compute_tfidf(counts)

    # Verify document and term alignment.
    assert document_names == ["a.TXT", "b.txt"]
    assert vocabulary == ["cake", "coffee", "contact", "tea"]

    expected_counts = np.array([
        [1.0, 1.0],  # cake
        [2.0, 0.0],  # coffee
        [0.0, 1.0],  # contact
        [0.0, 1.0],  # tea
    ])
    np.testing.assert_array_equal(counts, expected_counts)

    # IDF = 1 + ln(N / df), with N = 2.
    expected_idf = np.array([
        1.0,
        1.6931471805599453,
        1.6931471805599453,
        1.6931471805599453,
    ])
    expected_tfidf = np.array([
        [1.0, 1.0],
        [3.3862943611198906, 0.0],
        [0.0, 1.6931471805599453],
        [0.0, 1.6931471805599453],
    ])

    np.testing.assert_allclose(idf, expected_idf)
    np.testing.assert_allclose(tfidf, expected_tfidf)