from flint import arb

from numpy_flint_arb import np


def test_linsolve():
    A = np.arange(9, dtype=arb).reshape(3, 3)
    b = np.array([arb(1), arb(2), arb(3)])
    x = np.linalg.linsolve(A, b)
    assert np.all(np.dot(A, x) == b)
