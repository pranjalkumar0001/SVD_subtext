"""Experiment driver for the report (run from the repository root).

Every factorisation comes from the project's own svd_from_gram (src/SVD.py);
NumPy's SVD routine is not used anywhere in this script.
Writes results.json and figures/*.pdf.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_txt_documents
from src.preprocessing import build_count_matrix, compute_tfidf
from src.SVD import svd_from_gram
from src.similarity import (
    suggest_k,
    document_embeddings,
    cosine_similarity_matrix,
    retained_condition_number,
)

DATA = "Data/LongSamples"
FIG = Path("figures/experiment2")
FIG.mkdir(exist_ok=True)
fro = np.linalg.norm

# ---- Build the TF-IDF matrix and its SVD -------------------------------
docs = load_txt_documents(DATA)
counts, vocab, names = build_count_matrix(docs)
A, idf = compute_tfidf(counts)
m, n = A.shape
U, S, Vt = svd_from_gram(A)
s = np.diag(S)
r = len(s)
total = np.sum(s ** 2)

res = {"m": m, "n": n, "r": r, "nnz": int(np.count_nonzero(counts)),
       "singular": s.tolist()}

# ---- Verification of the factorisation (Section 2.4.2) -----------------
res["erec"] = fro(A - U @ S @ Vt) / fro(A)
res["eU"] = fro(U.T @ U - np.eye(r))
res["eV"] = fro(Vt @ Vt.T - np.eye(r))
# Independent check that does not depend on the eigensolver: the squared
# singular values must add up to the squared Frobenius norm of A.
res["frobenius_identity_err"] = float(abs(np.sum(s ** 2) - fro(A) ** 2) / fro(A) ** 2)
res["kappa_full"] = float(s[0] / s[-1])

examples = {"tall": np.array([[1., 1.], [1., 0.], [0., 1.]]),
            "rank-deficient": np.array([[1., 1.], [2., 2.], [2., 2.]]),
            "wide": np.array([[4., 11., 14.], [8., 7., -2.]])}
# Singular values worked out by hand in Chapter 1
exact = {"tall": [np.sqrt(3.0), 1.0],
         "rank-deficient": [3.0 * np.sqrt(2.0)],
         "wide": [6.0 * np.sqrt(10.0), 3.0 * np.sqrt(10.0)]}
res["examples"] = {}
for label, M in examples.items():
    Ue, Se, Ve = svd_from_gram(M)
    k = Ue.shape[1]
    res["examples"][label] = {
        "sv": np.diag(Se).tolist(),
        "sv_err": float(np.max(np.abs(np.diag(Se) - np.array(exact[label])))),
        "erec": float(fro(M - Ue @ Se @ Ve) / fro(M)),
        "eU": float(fro(Ue.T @ Ue - np.eye(k))),
        "eV": float(fro(Ve @ Ve.T - np.eye(k)))}

# ---- Effect of the retained rank (Section 2.4.3) -----------------------
sweep = []
for k in range(1, r + 1):
    Ak = U[:, :k] @ S[:k, :k] @ Vt[:k, :]
    energy = np.sum(s[:k] ** 2) / total
    sweep.append({"k": k, "energy": float(energy),
                  "rel_err": float(fro(A - Ak) / fro(A)),
                  "sqrt_1_minus_E": float(np.sqrt(max(0.0, 1 - energy))),
                  "kappa": retained_condition_number(S, k)})
res["sweep"] = sweep
res["suggest_k_90"] = suggest_k(S, 0.90)
res["suggest_k_80"] = suggest_k(S, 0.80)

# ---- Document similarity (Section 2.4.4) -------------------------------
baseline = cosine_similarity_matrix(A)
sims = {k: cosine_similarity_matrix(document_embeddings(S, Vt, k))
        for k in range(2, r + 1)}
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
res["pairs"] = [{"a": names[i], "b": names[j], "tfidf": float(baseline[i, j]),
                 **{f"k{k}": float(sims[k][i, j]) for k in sims}}
                for i, j in pairs]

# Reference grouping used only to summarise: coffee {01,02}, outdoors {03,04}, music {05}
within = [(0, 1), (2, 3)]
across = [p for p in pairs if p not in within]
def summarise(M):
    return (float(np.mean([M[i, j] for i, j in within])),
            float(np.mean([M[i, j] for i, j in across])))
res["group"] = {"tfidf": summarise(baseline),
                **{f"k{k}": summarise(sims[k]) for k in sims}}

# Leading term loadings of the first three left singular vectors
for d in range(3):
    idx = np.argsort(-np.abs(U[:, d]))[:8]
    res[f"top_terms_dir{d + 1}"] = [(vocab[i], round(float(U[i, d]), 3)) for i in idx]
res["Z2"] = document_embeddings(S, Vt, 2).tolist()
res["Z3"] = document_embeddings(S, Vt, 3).tolist()

Path("results.json").write_text(json.dumps(res, indent=1))

# ---- Figures ------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "font.family": "serif"})
short = [nm.replace(".txt", "").split("_", 1)[1] for nm in names]

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax[0].plot(range(1, r + 1), s, "o-", color="black")
ax[0].set_xlabel("index $i$"); ax[0].set_ylabel(r"$\sigma_i$")
ax[0].set_title("Singular values of the TF-IDF matrix")
ax[1].plot(range(1, r + 1), np.cumsum(s ** 2) / total, "s-", color="black")
ax[1].axhline(0.9, ls="--", color="gray")
ax[1].set_xlabel("retained rank $k$"); ax[1].set_ylabel("$E(k)$")
ax[1].set_title("Cumulative energy")
for a in ax:
    a.set_xticks(range(1, r + 1)); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "spectrum.pdf"); plt.close()

fig, ax = plt.subplots(1, 4, figsize=(8.6, 2.6), sharey=True)
panels = [(baseline, "TF-IDF"), (sims[2], "LSA $k=2$"),
          (sims[3], "LSA $k=3$"), (sims[r], f"LSA $k={r}$")]
for a, (M, t) in zip(ax, panels):
    im = a.imshow(M, vmin=-0.2, vmax=1, cmap="gray_r"); a.set_title(t, fontsize=9)
    a.set_xticks(range(n)); a.set_xticklabels(short, rotation=60, ha="right", fontsize=7)
    a.set_yticks(range(n)); a.set_yticklabels(short, fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.02)
plt.savefig(FIG / "heatmaps.pdf", bbox_inches="tight"); plt.close()

Z = document_embeddings(S, Vt, 2)
fig, a = plt.subplots(figsize=(3.6, 3.2))
for j in range(n):
    a.annotate("", xy=Z[:, j], xytext=(0, 0),
               arrowprops=dict(arrowstyle="->", color="black"))
    a.text(Z[0, j] * 1.06, Z[1, j] * 1.06, short[j], fontsize=8)
lim = 1.35 * np.max(np.abs(Z))
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.axhline(0, c="0.8"); a.axvline(0, c="0.8")
a.set_xlabel("first LSA coordinate"); a.set_ylabel("second LSA coordinate")
a.set_aspect("equal"); a.set_title("Documents in the $k=2$ LSA plane", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "plane2d.pdf"); plt.close()
print("done", res["suggest_k_90"], res["erec"])