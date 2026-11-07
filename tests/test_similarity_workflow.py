import numpy as np

from src.data_loader import load_txt_documents
from src.preprocessing import build_count_matrix, compute_tfidf
from src.SVD import svd_from_gram
from src.similarity import (
    suggest_k,
    document_embeddings,
    cosine_similarity_matrix,
    retained_condition_number,
)


def test_complete_similarity_workflow(tmp_path):
    # Two identical documents and one with completely different vocabulary.
    (tmp_path / "a.txt").write_text(
        "Coffee cake coffee.",
        encoding="utf-8",
    )
    (tmp_path / "b.txt").write_text(
        "Coffee cake coffee.",
        encoding="utf-8",
    )
    (tmp_path / "c.txt").write_text(
        "Mountain river mountain.",
        encoding="utf-8",
    )

    documents = load_txt_documents(tmp_path)
    counts, vocabulary, names = build_count_matrix(documents)
    tfidf, _ = compute_tfidf(counts)

    U, Sigma, Vt = svd_from_gram(tfidf)

    k = suggest_k(Sigma, energy_target=0.90)
    embeddings = document_embeddings(Sigma, Vt, k)
    similarities = cosine_similarity_matrix(embeddings)

    assert names == ["a.txt", "b.txt", "c.txt"]
    assert vocabulary == ["cake", "coffee", "mountain", "river"]
    assert Sigma.shape == (2, 2)
    assert k == 2
    assert embeddings.shape == (2, 3)

    np.testing.assert_allclose(
        U @ Sigma @ Vt,
        tfidf,
        atol=1e-10,
    )

    expected = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    np.testing.assert_allclose(
        similarities,
        expected,
        atol=1e-10,
    )

    # Keeping all nonzero singular components preserves cosine similarity.
    baseline = cosine_similarity_matrix(tfidf)

    np.testing.assert_allclose(
        similarities,
        baseline,
        atol=1e-10,
    )

    condition = retained_condition_number(Sigma, k)
    assert np.isfinite(condition)
    assert condition >= 1.0