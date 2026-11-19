"""Compare custom TF-IDF and Gram-matrix LSA on real text.

Run from the repository root:
    python -m experiments.experiment_big

This evaluates neighbours within one fixed collection.
It is not a held-out query retrieval experiment.
"""

import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import fetch_20newsgroups

from src.data_loader import load_txt_documents
from src.preprocessing import (
    build_count_matrix,
    clean_text,
    compute_tfidf,
    tokenize,
)
from src.SVD import svd_from_gram
from src.similarity import (
    cosine_similarity_matrix,
    document_embeddings,
    retained_condition_number,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "Data" / "20newsgroups"
DOCUMENT_DIR = DATA_DIR / "documents"
OUTPUT_DIR = ROOT / "figures" / "experiment_big"

CATEGORIES = [
    "comp.graphics",
    "rec.autos",
    "rec.sport.baseball",
    "sci.med",
    "sci.space",
]

SEED = 42
DOCUMENTS_PER_CATEGORY = 50
MIN_TOKENS = 30
NEIGHBOURS = 5
LSA_RANKS = [5, 10, 20, 50]


def prepare_documents() -> dict[str, str]:
    """Save a reproducible sample and return filename -> category labels."""
    dataset = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=False,
        data_home=str(ROOT / ".cache" / "20newsgroups"),
    )

    rng = np.random.default_rng(SEED)
    selected = []

    for category in CATEGORIES:
        category_id = dataset.target_names.index(category)
        candidates = np.flatnonzero(dataset.target == category_id)

        # Remove documents that become too short after cleaning.
        eligible = [
            int(index)
            for index in candidates
            if len(tokenize(clean_text(dataset.data[index]))) >= MIN_TOKENS
        ]

        if len(eligible) < DOCUMENTS_PER_CATEGORY:
            raise ValueError(
                f"{category}: only {len(eligible)} eligible documents."
            )

        chosen = rng.choice(
            eligible,
            size=DOCUMENTS_PER_CATEGORY,
            replace=False,
        )

        for index in sorted(chosen.tolist()):
            source_id = Path(dataset.filenames[index]).name
            filename = f"{category}__{source_id}.txt"
            selected.append(
                (filename, category, source_id, dataset.data[index])
            )

    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    expected_names = {row[0] for row in selected}

    # Prevent old samples from silently entering the experiment.
    existing_names = {
        path.name
        for path in DOCUMENT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }

    unexpected = existing_names - expected_names
    if unexpected:
        raise ValueError(
            "The document folder contains files outside this sample. "
            "Move the existing folder elsewhere before changing the "
            f"sampling configuration: {DOCUMENT_DIR}"
        )

    labels = {}

    with (DATA_DIR / "labels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "category", "source_id"])

        for filename, category, source_id, text in selected:
            (DOCUMENT_DIR / filename).write_text(text, encoding="utf-8")
            writer.writerow([filename, category, source_id])
            labels[filename] = category

    return labels


def precision_at_k(
    similarities: np.ndarray,
    labels: list[str],
    k: int = 5,
) -> np.ndarray:
    """Return one precision@k value per document, excluding itself.

    A neighbour is considered relevant when its category matches
    the query document's category.
    """
    scores = np.asarray(similarities, dtype=np.float64)
    labels_array = np.asarray(labels)
    n = len(labels_array)

    if scores.shape != (n, n):
        raise ValueError("Expected an n-by-n similarity matrix.")

    if not np.all(np.isfinite(scores)):
        raise ValueError("Similarity scores must be finite.")

    if not 1 <= k < n:
        raise ValueError("k must be between 1 and n - 1.")

    scores = scores.copy()
    np.fill_diagonal(scores, -np.inf)

    # Stable sorting resolves exact ties by document order.
    neighbours = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    relevant = labels_array[neighbours] == labels_array[:, None]

    return relevant.mean(axis=1)


def summarise_scores(
    scores: np.ndarray,
    labels: list[str],
) -> dict:
    labels_array = np.asarray(labels)

    return {
        "mean_precision_at_5": float(np.mean(scores)),
        "per_category_precision_at_5": {
            category: float(np.mean(scores[labels_array == category]))
            for category in CATEGORIES
        },
        "per_document_precision_at_5": scores.tolist(),
    }


def save_plots(singular_values: np.ndarray, results: list[dict]) -> None:
    energy = (singular_values / singular_values[0]) ** 2
    cumulative = np.cumsum(energy) / np.sum(energy)
    indices = np.arange(1, len(singular_values) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].semilogy(indices, singular_values)
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Singular value")
    axes[0].set_title("Retained singular spectrum")

    axes[1].plot(indices, cumulative)
    axes[1].axhline(0.90, color="gray", linestyle="--")
    axes[1].set_xlabel("Retained rank")
    axes[1].set_ylabel("Cumulative spectral energy")
    axes[1].set_title("Energy relative to numerical spectrum")

    for axis in axes:
        axis.grid(alpha=0.25)

    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"spectrum.{extension}", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4))
    names = [row["method"] for row in results]
    values = [row["mean_precision_at_5"] for row in results]

    bars = axis.bar(names, values)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Mean precision@5")
    axis.set_title("Same-category neighbours in the sampled collection")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)

    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"precision_at_5.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels_by_name = prepare_documents()

    documents = load_txt_documents(DOCUMENT_DIR)
    counts, vocabulary, names = build_count_matrix(documents)
    labels = [labels_by_name[name] for name in names]

    A, _ = compute_tfidf(counts)
    U, Sigma, Vt = svd_from_gram(A)

    singular_values = np.diag(Sigma)
    numerical_rank = len(singular_values)

    if numerical_rank == 0:
        raise ValueError("The collection has numerical rank zero.")

    relative_energy = (singular_values / singular_values[0]) ** 2
    total_energy = np.sum(relative_energy)

    baseline = cosine_similarity_matrix(A)
    baseline_scores = precision_at_k(baseline, labels, NEIGHBOURS)

    results = [
        {
            "method": "TF-IDF",
            "k": None,
            **summarise_scores(baseline_scores, labels),
        }
    ]

    # Include full numerical rank as a cosine-preservation check.
    ranks = sorted(
        {k for k in LSA_RANKS if k <= numerical_rank}
        | {numerical_rank}
    )

    for k in ranks:
        embeddings = document_embeddings(Sigma, Vt, k)
        similarities = cosine_similarity_matrix(embeddings)
        scores = precision_at_k(similarities, labels, NEIGHBOURS)

        results.append(
            {
                "method": f"LSA k={k}",
                "k": k,
                "retained_energy": float(
                    np.sum(relative_energy[:k]) / total_energy
                ),
                "retained_condition_number": retained_condition_number(
                    Sigma, k
                ),
                **summarise_scores(scores, labels),
            }
        )

        if k == numerical_rank:
            full_rank_cosine_error = float(
                np.max(np.abs(similarities - baseline))
            )

    output = {
        "dataset": "20 Newsgroups, by-date training subset",
        "evaluation": (
            "Within-collection category-neighbour evaluation; "
            "self-matches excluded; no held-out queries."
        ),
        "seed": SEED,
        "categories": CATEGORIES,
        "documents_per_category": DOCUMENTS_PER_CATEGORY,
        "minimum_cleaned_tokens": MIN_TOKENS,
        "removed_sections": ["headers", "footers", "quotes"],
        "neighbours": NEIGHBOURS,
        "document_names": names,
        "labels": labels,
        "matrix_shape": list(A.shape),
        "numerical_rank": numerical_rank,
        "singular_values": singular_values.tolist(),
        "relative_reconstruction_error": float(
            np.sqrt(np.sum((A - U @ Sigma @ Vt) ** 2))
            / np.sqrt(np.sum(A ** 2))
        ),
        "full_rank_max_cosine_difference": full_rank_cosine_error,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input_sha256": {
            name: hashlib.sha256(
                (DOCUMENT_DIR / name).read_bytes()
            ).hexdigest()
            for name in names
        },
        "results": results,
    }

    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    save_plots(singular_values, results)

    print(f"Documents: {len(names)}")
    print(f"Vocabulary: {len(vocabulary)}")
    print(f"Numerical rank: {numerical_rank}")
    print("\nMethod          | Mean precision@5")
    print("----------------|-----------------")

    for row in results:
        print(
            f"{row['method']:<15} | "
            f"{row['mean_precision_at_5']:.4f}"
        )

    print(
        "\nFull-rank maximum cosine difference: "
        f"{full_rank_cosine_error:.3e}"
    )
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()