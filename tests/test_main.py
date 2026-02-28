from numpy_flint_arb import np


def test_linsolve():
    A = np.random.normal(size=(2, 2))
    b = np.random.normal(size=(2,))
    x = np.linalg.solve(A, b)
    b_approx = A @ x
    assert np.all(np.contains(b_approx, b))
    assert np.all(np.overlaps(b_approx, b))
    assert np.all(~(b_approx < b))
    assert np.all(~(b_approx > b))
    assert np.all(~(b_approx == b))
    assert np.all(~(b_approx <= b))
    assert np.all(~(b_approx >= b))


def test_comparisons():
    x = np.arange(3, dtype=np.float64)
    assert np.all(x == x)
    assert np.all(x <= x)
    assert np.all(x >= x)
    y = x + 1
    assert np.all(y > x)
    assert np.all(y >= x)
