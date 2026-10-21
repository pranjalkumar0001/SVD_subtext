# SVD Subtext

Document similarity through Singular Value Decomposition and Latent Semantic Analysis.

## Overview

This project explores how truncated SVD transforms a term-document
matrix into a lower-dimensional representation, and how that
transformation affects document similarity.

It is developed as an individual project for MAC F211:
Linear Algebra and Its Applications at BITS Pilani, Goa Campus.

## Current Status

Environment setup is completed.
SVD implementation in progress
Text pipeline,and experiments are not yet complete.

## Objectives

- Construct SVD using the eigendecomposition of a Gram matrix.
- Explain and verify low-rank approximation.
- Build term-document matrices from TXT documents.
- Compare cosine similarities before and after dimensionality reduction.
- Investigate numerical limitations and the effect of retained rank.

## Implementation Rules

The core implementation uses NumPy array operations and
`numpy.linalg.eigh` to construct the SVD.

Depending on the matrix dimensions, it uses either `A.T @ A` or
`A @ A.T` and recovers the other singular-vector factor.

`numpy.linalg.svd` is permitted only in `tests/` as a reference.
It is never used by the application pipeline.

No pretrained embeddings are used. Document representations are
computed from the text collection using truncated SVD.

## Planned Workflow

1. Read and preprocess TXT documents.
2. Construct a term-document matrix using word counts, followed by TF-IDF.
3. Compute the SVD and retain the leading k components.
4. Represent documents using Z_k = Sigma_k @ V_k.T.
5. Compare documents using cosine similarity.
6. Evaluate reconstruction error and changes in document similarity.

## Setup

Python 3.13 and Conda are used for development.

After the environment configuration is complete:

```bash
conda env create -f environment.yml
conda activate svd-project
python env_test.py
```

`environment.yml` is the primary environment specification.
`requirements.txt` provides an alternative dependency list for pip users.

## Planned Validation

- Reconstruction accuracy.
- Orthogonality of singular vectors.
- Agreement with NumPy's reference singular values.
- Tall, wide, rank-deficient, repeated-singular-value, and zero matrices.
- Reconstruction error across retained ranks.
- Document similarity before and after truncation.

## Limitations

Forming a Gram matrix can reduce the accuracy of small singular values.
This implementation is intended for mathematical study and experimentation,
not as a replacement for established numerical libraries.

LSA captures patterns of word usage. It does not understand language in
the same way as a person, and dimensionality reduction does not guarantee
better document comparisons.
