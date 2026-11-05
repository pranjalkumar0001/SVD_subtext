import numpy as np
import pytest

from src.SVD import svd_from_gram


@pytest.mark.parametrize(
    "A, expected_values",
    [
        (
            np.array([
                [4., 11., 14.],
                [8., 7., -2.],
            ]),
            [6 * np.sqrt(10), 3 * np.sqrt(10)],
        ),
        (
            np.array([
                [1., 1.],
                [1., 0.],
                [0., 1.],
            ]),
            [np.sqrt(3), 1.0],
        ),
        (
            np.array([
                [1., 1.],
                [2., 2.],
                [2., 2.],
            ]),
            [3 * np.sqrt(2)],
        ),
        (
            np.diag([2., 2., 5.]),
            [5., 2., 2.],
        ),
        (
            np.zeros((3, 2)),
            [],
        ),
    ],
    ids=["wide", "tall", "rank-deficient", "repeated", "zero"],
)
def test_svd_known_matrices(A, expected_values):
    original = A.copy()
    U, Sigma, Vt = svd_from_gram(A)

    m, n = A.shape
    r = len(expected_values)

    assert U.shape == (m, r)
    assert Sigma.shape == (r, r)
    assert Vt.shape == (r, n)

    np.testing.assert_allclose(
        Sigma,
        np.diag(expected_values),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        U @ Sigma @ Vt, A, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        U.T @ U, np.eye(r), atol=1e-12
    )
    np.testing.assert_allclose(
        Vt @ Vt.T, np.eye(r), atol=1e-12
    )
    np.testing.assert_array_equal(A, original)


@pytest.mark.parametrize("shape", [(8, 3), (3, 8), (5, 5)])
def test_svd_against_numpy(shape):
    rng = np.random.default_rng(42)
    A = rng.standard_normal(shape)

    U, Sigma, Vt = svd_from_gram(A)

    # Reference SVD is allowed in tests only.
    expected_values = np.linalg.svd(A, compute_uv=False)
    r = min(shape)

    assert U.shape == (shape[0], r)
    assert Sigma.shape == (r, r)
    assert Vt.shape == (r, shape[1])

    np.testing.assert_allclose(
        Sigma, np.diag(expected_values),
        rtol=1e-10, atol=1e-12,
    )
    np.testing.assert_allclose(
        U @ Sigma @ Vt, A, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        U.T @ U, np.eye(r), atol=1e-10
    )
    np.testing.assert_allclose(
        Vt @ Vt.T, np.eye(r), atol=1e-10
    )


@pytest.mark.parametrize(
    "A",
    [
        np.array([1., 2.]),
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[1 + 2j]]),
    ],
    ids=["not-2d", "no-rows", "no-columns", "nan", "infinity", "complex"],
)
def test_invalid_input(A):
    with pytest.raises(ValueError):
        svd_from_gram(A)