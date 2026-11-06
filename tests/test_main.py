from typing import Any

import pytest
from flint import acb, arb, arf, fmpq, fmpz

from numpy_flint_arb import allow_input, np


def test_linsolve():
    a = np.random.normal(size=(2, 2))
    b = np.random.normal(size=(2,))
    x = np.linalg.solve(a, b)
    b_approx = a @ x
    assert np.all(np.contains(b_approx, b))
    assert np.all(np.overlaps(b_approx, b))
    assert np.all(~(b_approx < b))
    assert np.all(~(b_approx > b))
    assert np.all(~(b_approx == b))
    assert np.all(~(b_approx <= b))
    assert np.all(~(b_approx >= b))


def test_slice():
    x = np.zeros((4, 1, 1))
    assert x[1, ...].shape == (1, 1)


def test_linspace():
    x = np.linspace(arb(1), acb(1j), 5)
    assert np.contains(x[2], (arb(1) + acb(1j)) / 2)


def test_creation():
    shape = (2, 2)
    x = np.zeros(shape)
    x = np.zeros_like(x)
    x = np.ones(shape)
    x = np.ones_like(x)


def test_dtype_functions():
    assert np.can_cast(int, fmpz)
    assert np.can_cast(float, arb)
    assert np.can_cast(complex, acb)
    assert not np.can_cast(str, arb)
    assert np.can_cast(fmpz, fmpq)
    assert np.is_dtype(fmpz, "integral")
    assert np.is_dtype(arb, "real floating")
    assert np.is_dtype(acb, "complex floating")
    assert np.is_dtype(arb, "numeric")
    assert np.finfo(arb).bits
    assert np.iinfo(fmpz).dtype == fmpz
    assert np.result_type(fmpz, arb) == arb
    info = np.__array_namespace_info__()
    assert info.default_dtypes()["real floating"] == arb
    assert info.dtypes(kind=("integral", "real floating")) == {"fmpz": fmpz, "arb": arb, "arf": arf}


def test_fft():
    fft_result = np.fft.fft(np.arange(1, stop=12))
    ifft_result = np.fft.ifft(fft_result)
    assert np.contains(ifft_result[-1], np.asarray("11.0", dtype=arb))


def test_asarray_str():
    x = np.asarray("1.0 +/- 0.1", dtype=arb)
    assert np.contains(x, np.asarray("1.0 +/- 0.05", dtype=arb))
    assert np.contains(x, np.asarray("1.0", dtype=arb))


def test_comparisons():
    x = np.arange(3, dtype=np.float64)
    same = x.copy()
    assert np.all(x == same)
    assert np.all(x <= same)
    assert np.all(x >= same)
    y = x + 1
    assert np.all(y > x)
    assert np.all(y >= x)


@pytest.mark.parametrize("value", [0.5, 1 + 1j])
@pytest.mark.parametrize("dtype", [arb, acb])
def test_allow_input(value: Any, dtype: Any) -> None:
    if dtype == arb and isinstance(value, complex):
        pytest.skip("acb does not allow float input")
    np.asarray(1, dtype=dtype)
    with pytest.raises(ExceptionGroup):
        np.asarray(value, dtype=dtype)
    with allow_input(interval=True, float=True):
        np.asarray(value, dtype=dtype)


@pytest.mark.parametrize("value", [0.5])
def test_allow_input_arf(value: Any) -> None:
    np.asarray(1, dtype=arf)
    with pytest.raises(ExceptionGroup):
        np.asarray(value, dtype=arf)
    with allow_input(float=True):
        np.asarray(value, dtype=arf)
