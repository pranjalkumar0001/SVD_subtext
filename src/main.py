import argparse
import numpy as np
from src.data_loader import load_txt_documents
from src.preprocessing import build_count_matrix, compute_tfidf
from src.SVD import svd_from_gram
from src.similarity import (
    suggest_k,
    document_embeddings,
    cosine_similarity_matrix,
    retained_condition_number)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TXT documents using TF-IDF and custom SVD.")
    parser.add_argument("folder", help="Folder containing TXT documents.")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of singular components; otherwise use the energy target.",
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=0.90,
        help="Energy target for automatic k selection (default: 0.90).",
    )
    parser.add_argument(
    "--show-vocab",
    action="store_true",
    help="Print the vocabulary with its matrix row indices."
    )
    args = parser.parse_args()

    try:
        documents = load_txt_documents(args.folder)

        if len(documents) < 2:
            raise ValueError("Provide at least two TXT documents to compare.")

        counts, vocabulary, names = build_count_matrix(documents)
        if args.show_vocab:
            print(f"\nVocabulary: {len(vocabulary)} terms")
            for row, word in enumerate(vocabulary):
                print(f"{row:4d}: {word}")
            print()

        tfidf, _ = compute_tfidf(counts)

        U, Sigma, Vt = svd_from_gram(tfidf)

        k = (
            suggest_k(Sigma, energy_target=args.energy)
            if args.k is None
            else args.k
        )

        embeddings = document_embeddings(Sigma, Vt, k)
        baseline = cosine_similarity_matrix(tfidf)
        similarities = cosine_similarity_matrix(embeddings)
        condition = retained_condition_number(Sigma, k)

    except (
        ValueError,
        OSError,
        UnicodeError,
        np.linalg.LinAlgError,
        FloatingPointError,
    ) as error:
        parser.exit(status=1, message=f"Error: {error}\n")

    singular_values = np.diag(Sigma)
    weights = (singular_values / singular_values[0]) ** 2
    retained_energy = np.sum(weights[:k]) / np.sum(weights)

    print(f"Documents: {len(names)}")
    print(f"Vocabulary: {len(vocabulary)} terms")
    print(f"Numerical rank: {Sigma.shape[0]}")
    print(f"Selected k: {k}")
    print(f"Retained spectral energy: {retained_energy:.2%}")
    print(f"Retained condition number: {condition:.4g}")

    if k == Sigma.shape[0]:
        print("All retained components are used; no further rank reduction.")

    # Print each distinct pair once.
    print("\nDocument pair | TF-IDF cosine | LSA cosine")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(
                f"{names[i]} <-> {names[j]} | "
                f"{baseline[i, j]:.4f} | "
                f"{similarities[i, j]:.4f}"
            )


if __name__ == "__main__":
    main()