from numpy_flint_arb import np


def test_linsolve():
    A = np.random.normal(size=(3, 3))
    b = np.random.normal(size=(3,))
    x = np.linalg.solve(A, b)
    assert np.all(np.sum(A * x, axis=-1) == b)
