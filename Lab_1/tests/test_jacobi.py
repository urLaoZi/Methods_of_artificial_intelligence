import numpy as np
import pytest

from jacobi_method import (
    BadToleranceError,
    ConvergenceError,
    NotSymmetricMatrixError,
    NotSquareMatrixError,
    jacobi_eigen,
)


def test_1x1():
    A = np.array([[5.0]])
    vals, vecs = jacobi_eigen(A)
    assert np.allclose(vals, [5.0])
    assert np.allclose(vecs, [[1.0]])


def test_diag():
    A = np.diag([3.0, 1.0, 2.0])
    vals, _ = jacobi_eigen(A)
    assert np.allclose(vals, [1.0, 2.0, 3.0])


def test_zeros():
    A = np.zeros((3, 3))
    vals, vecs = jacobi_eigen(A)
    assert np.allclose(vals, [0, 0, 0])
    assert np.allclose(vecs.T @ vecs, np.eye(3), atol=1e-10)


def test_2x2():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    vals, vecs = jacobi_eigen(A)
    assert np.allclose(vals, [1.0, 3.0], atol=1e-10)
    for i in range(2):
        assert np.allclose(A @ vecs[:, i], vals[i] * vecs[:, i], atol=1e-10)


def test_3x3_equation():
    A = np.array([
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [1.0, 0.0, 2.0],
    ])
    vals, vecs = jacobi_eigen(A)
    assert np.allclose(A @ vecs, vecs @ np.diag(vals), atol=1e-10)
    assert np.allclose(vecs.T @ vecs, np.eye(3), atol=1e-10)


def test_random_symmetric():
    rng = np.random.default_rng(123)
    M = rng.random((6, 6))
    A = (M + M.T) / 2.0
    vals, vecs = jacobi_eigen(A, tol=1e-12)
    assert np.allclose(A @ vecs, vecs @ np.diag(vals), atol=1e-8)


def test_vs_numpy():
    rng = np.random.default_rng(7)
    M = rng.random((5, 5))
    A = (M + M.T) / 2.0
    vals, _ = jacobi_eigen(A, tol=1e-12)
    ref, _ = np.linalg.eigh(A)
    assert np.allclose(vals, ref, atol=1e-8)


def test_sorted():
    rng = np.random.default_rng(1)
    M = rng.random((4, 4))
    A = (M + M.T) / 2.0
    vals, _ = jacobi_eigen(A, tol=1e-12)
    assert np.all(np.diff(vals) >= -1e-10)


def test_not_square():
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(NotSquareMatrixError):
        jacobi_eigen(A)


def test_not_symmetric():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(NotSymmetricMatrixError):
        jacobi_eigen(A)


def test_bad_tol():
    A = np.eye(2)
    with pytest.raises(BadToleranceError):
        jacobi_eigen(A, tol=0.0)
    with pytest.raises(BadToleranceError):
        jacobi_eigen(A, tol=-1.0)


def test_bad_max_iter():
    A = np.eye(2)
    with pytest.raises(ValueError):
        jacobi_eigen(A, max_iter=0)


def test_no_convergence():
    rng = np.random.default_rng(2)
    M = rng.random((8, 8))
    A = (M + M.T) / 2.0
    with pytest.raises(ConvergenceError):
        jacobi_eigen(A, tol=1e-15, max_iter=1)


def test_nan():
    A = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(ValueError):
        jacobi_eigen(A)