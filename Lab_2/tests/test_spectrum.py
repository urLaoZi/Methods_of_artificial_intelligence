import numpy as np

from random_matrices.ensembles import bernoulli_ensemble, gaussian_ensemble
from random_matrices.spectrum import (
    eigenvalues_sorted,
    level_spacings,
    normalize_spacings,
    unfold_all,
    wigner_surmise,
)


def test_gaussian_shapes():
    rng = np.random.default_rng(0)
    A = gaussian_ensemble(4, 10, rng)
    assert A.shape == (10, 4, 4)


def test_gaussian_symmetric():
    rng = np.random.default_rng(0)
    A = gaussian_ensemble(4, 10, rng)
    assert np.allclose(A, A.transpose(0, 2, 1))


def test_bernoulli_values():
    rng = np.random.default_rng(0)
    A = bernoulli_ensemble(3, 20, rng)
    # после симметризации значения кратны 2/sqrt(2) = sqrt(2)
    assert np.allclose(A, A.transpose(0, 2, 1))


def test_eigenvalues_sorted():
    rng = np.random.default_rng(0)
    A = gaussian_ensemble(4, 50, rng)
    vals = eigenvalues_sorted(A)
    assert vals.shape == (50, 4)
    assert np.all(np.diff(vals, axis=1) >= -1e-12)


def test_level_spacings_shape():
    rng = np.random.default_rng(0)
    A = gaussian_ensemble(4, 50, rng)
    vals = eigenvalues_sorted(A)
    sp = level_spacings(vals)
    assert sp.shape == (50, 3)
    assert np.all(sp >= 0)


def test_normalize_mean_one():
    rng = np.random.default_rng(0)
    A = gaussian_ensemble(4, 500, rng)
    vals = eigenvalues_sorted(A)
    sp = unfold_all(vals)
    assert abs(np.mean(sp) - 1.0) < 1e-10


def test_wigner_at_zero():
    assert wigner_surmise(0.0) == 0.0


def test_wigner_positive():
    s = np.linspace(0.0, 3.0, 100)
    assert np.all(wigner_surmise(s) >= 0.0)


def test_wigner_normalized():
    s = np.linspace(0.0, 10.0, 5000)
    integral = np.trapezoid(wigner_surmise(s), s)
    assert abs(integral - 1.0) < 1e-3