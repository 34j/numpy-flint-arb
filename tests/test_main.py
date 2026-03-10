from typing import Any

import pytest
from flint import acb, arb, arf

from numpy_flint_arb import allow_input, np


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


@pytest.mark.parametrize("input", [0.5, 1 + 1j])
@pytest.mark.parametrize("dtype", [arb, acb])
def test_allow_input(input: Any, dtype: Any) -> None:
    if dtype == arb and isinstance(input, complex):
        pytest.skip("acb does not allow float input")
    np.asarray(1, dtype=dtype)
    with pytest.raises(ExceptionGroup):
        np.asarray(input, dtype=dtype)
    with allow_input(interval=True, float=True):
        np.asarray(input, dtype=dtype)


@pytest.mark.parametrize("input", [0.5])
def test_allow_input_arf(input: Any) -> None:
    np.asarray(1, dtype=arf)
    with pytest.raises(ExceptionGroup):
        np.asarray(input, dtype=arf)
    with allow_input(float=True):
        np.asarray(input, dtype=arf)
