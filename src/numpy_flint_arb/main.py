from typing import Any

import numpy as np
from flint import acb, arb, arf, fmpq, fmpz

dtypes = [acb, arb, arf, fmpz, fmpq]

# acb: complex ball
# arb: real ball
# arf: real floating-point
# fmpz: integer
# fmpq: rational number


def _fltype(x: Any) -> Any:
    el = np.asarray(x).ravel()[0]

    for t in dtypes:
        if isinstance(el, t):
            return t
    if np.isdtype(el.dtype, "integer"):
        return fmpz
    elif np.isdtype(el.dtype, "real floating"):
        return arb
    elif np.isdtype(el.dtype, "complex floating"):
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


class AttrDict[TV](dict[str, TV]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self


namespace: AttrDict[Any] = AttrDict()


# https://numpy.org/doc/stable/user/basics.subclassing.html
class flarray(np.ndarray):
    # need to patch "dtype" property to return flint dtype instead of object dtype
    _fl_dtype: Any = None

    @property
    def dtype(self) -> Any:
        return self._fl_dtype

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self._fl_dtype = getattr(obj, "_fl_dtype", None)

    def __array_namespace__(self, /, *, api_version: Any = None) -> Any:
        return namespace


def asarray(
    obj: Any, /, *, dtype: Any = None, device: Any = None, copy: bool | None = None
) -> np.ndarray:
    if dtype is not None and dtype not in dtypes:
        raise TypeError(
            f"dtype must be one of {', '.join([str(t) for t in dtypes])}, got {dtype}."
        )
    a = np.asarray(obj)
    el = a.ravel()[0]
    if dtype is not None and isinstance(el, dtype) and (copy is False or copy is None):
        return a
    elif copy is False:
        raise ValueError("Cannot convert to the requested dtype without copying.")
    elif dtype is None:
        dtype = _fltype(a)

    if a.dtype == np.object_:
        a = np.vectorize(lambda z: dtype(z))(a)
    elif np.isdtype(a.dtype, "integer"):
        a = np.vectorize(lambda z: dtype(int(z)))(a)
    elif np.isdtype(a.dtype, "floating"):
        a = np.vectorize(lambda z: dtype(float(z)))(a)
    elif np.isdtype(a.dtype, "complex floating"):
        a = np.vectorize(lambda z: dtype(complex(z)))(a)
    a = a.view(flarray)
    a._fl_dtype = dtype
    return a


namespace["asarray"] = asarray

# Constants
namespace["e"] = arb(1).exp()
namespace["pi"] = arb.pi()
namespace["nan"] = arb.nan()
namespace["newaxis"] = None
namespace["inf"] = arb("+inf")

# Creation Functions
for name in ["meshgrid", "tril", "triu"]:
    namespace[name] = getattr(np, name)

for name in [
    "arange",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "linspace",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
]:

    def _(*args: Any, _name: str = name, **kwargs: Any) -> Any:
        dtype = kwargs.pop("dtype", None)
        a = getattr(np, _name)(*args, **kwargs)
        a = asarray(a, dtype=dtype)
        return a

    namespace[name] = _


# Data Type Functions
def astype(
    x: Any, dtype: Any, /, *, copy: bool = True, device: Any = None
) -> np.ndarray:
    return asarray(x, dtype=dtype, copy=copy)


namespace["astype"] = astype


def can_cast(from_: Any, to: Any, /) -> bool:
    # numpy -> numpy fallback
    if from_ not in dtypes and to not in dtypes:
        return np.can_cast(from_, to)
    # flint -> numpy impossible
    if from_ in dtypes and to not in dtypes:
        return False
    # numpy -> flint possible in some cases
    if from_ not in dtypes and to in dtypes:
        if np.issubdtype(from_, np.integer):
            return to in [fmpz, fmpq, arb, acb]
        elif np.issubdtype(from_, np.floating):
            return to in [fmpq, arb, acb]
        elif np.issubdtype(from_, np.complexfloating):
            return to in [acb]
        else:
            return False
    # flint -> flint possible in some cases
    if (from_, to) in [
        (fmpz, fmpq),
        (fmpz, arb),
        (fmpz, acb),
        (fmpq, arb),
        (fmpq, acb),
        (arb, acb),
    ]:
        return True
    return from_ == to


namespace["can_cast"] = can_cast


def is_dtype(dtype: Any, kind: Any) -> bool:
    if dtype in dtypes:
        if kind in ["bool", "unsigned integer"]:
            return False
        elif kind in ["signed integer", "integral"]:
            return dtype == fmpz
        elif kind == "real floating":
            return dtype in [arf, arb]
        elif kind == "complex floating":
            return dtype == acb
        elif kind == "numeric":
            return dtype in dtypes
        elif kind == dtype:
            return True
        else:
            return False
    else:
        return np.isdtype(dtype, kind)


print(namespace["pi"] / namespace["arange"](10, dtype=arb))
