from typing import Any

import numpy as np
from flint import acb, arb, arf, fmpq, fmpz

dtypes = [acb, arb, arf, fmpz, fmpq]


def _fltype(x: Any) -> Any:
    el = np.asarray(x).ravel()[0]

    for t in dtypes:
        if isinstance(el, t):
            return t
    if np.issubdtype(el.dtype, np.integer):
        return fmpz
    elif np.issubdtype(el.dtype, np.floating):
        return arb
    elif np.issubdtype(el.dtype, np.complexfloating):
        return acb
    else:
        raise ValueError("Unrecognized type")


def _nptype(x: Any) -> Any:
    el = np.asarray(x).ravel()[0]

    if isinstance(el, fmpz):
        return np.int64
    elif isinstance(el, fmpq) or isinstance(el, arf) or isinstance(el, arb):
        return np.float64
    elif isinstance(el, acb):
        return np.complex128
    elif np.issubdtype(el.dtype, np.integer):
        return np.int64
    elif np.issubdtype(el.dtype, np.floating):
        return np.float64
    elif np.issubdtype(el.dtype, np.complexfloating):
        return np.complex128
    else:
        raise ValueError("Unrecognized type")


def asarray(
    obj: Any, /, *, dtype: Any = None, device: Any = None, copy: bool | None = None
) -> np.ndarray:
    if dtype is not None and dtype not in dtypes:
        raise TypeError("dtype must be flint.acb or flint.arb")
    a = np.asarray(obj)
    el = a.ravel()[0]
    if isinstance(el, dtype) and (copy is False or copy is None):
        return a
    elif copy is False:
        raise ValueError("Cannot convert to the requested dtype without copying.")
    elif dtype is None:
        dtype = _fltype(a)

    if a.dtype == np.object_:
        a = np.vectorize(lambda z: dtype(z))(a)
    elif np.issubdtype(a.dtype, np.integer):
        a = np.vectorize(lambda z: dtype(int(z)))(a)
    elif np.issubdtype(a.dtype, np.floating):
        a = np.vectorize(lambda z: dtype(float(z)))(a)
    elif np.issubdtype(a.dtype, np.complexfloating):
        a = np.vectorize(lambda z: dtype(complex(z)))(a)
    return a


print(asarray([1, 2, 3j]))
