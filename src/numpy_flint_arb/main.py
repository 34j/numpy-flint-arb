from collections.abc import Callable
from typing import Any

import numpy as np
from flint import acb, acb_mat, arb, arb_mat, arf, ctx, fmpq, fmpq_mat, fmpz, fmpz_mat

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
    if np.isdtype(el.dtype, "integral") or np.isdtype(el.dtype, "bool"):
        return fmpz
    elif np.isdtype(el.dtype, "real floating"):
        return arb
    elif np.isdtype(el.dtype, "complex floating"):
        return acb
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
    elif np.isdtype(a.dtype, "integral") or np.isdtype(a.dtype, "bool"):
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


namespace["is_dtype"] = is_dtype


def finfo(type: Any, /) -> Any:
    return AttrDict(
        {
            "bits": ctx.prec(),
            "eps": None,
            "max": None,
            "min": None,
            "smallest_normal": None,
            "dtype": arf,
        }
    )


namespace["finfo"] = finfo


def iinfo(type: Any, /) -> Any:
    return AttrDict(
        {
            "bits": None,
            "max": None,
            "min": None,
            "dtype": fmpz,
        }
    )


namespace["iinfo"] = iinfo


def result_type(*arrays_and_dtypes: Any) -> Any:
    types = []
    for x in arrays_and_dtypes:
        if x in dtypes or np.issubdtype(x, np.number):
            types.append(x)
        else:
            types.append(_fltype(x))
    if acb in types:
        return acb
    elif arb in types:
        return arb
    elif arf in types:
        return arf
    elif fmpq in types:
        return fmpq
    else:
        return fmpz


namespace["result_type"] = result_type

# Elementwise Functions
namespace["abs"] = np.abs
namespace["acos"] = np.vectorize(lambda x: x.acos())
namespace["acosh"] = np.vectorize(lambda x: x.acosh())
namespace["add"] = np.add
namespace["asin"] = np.vectorize(lambda x: x.asin())
namespace["asinh"] = np.vectorize(lambda x: x.asinh())
namespace["atan"] = np.vectorize(lambda x: x.atan())
namespace["atan2"] = np.vectorize(lambda y, x: y.atan2(x))
namespace["atanh"] = np.vectorize(lambda x: x.atanh())
# no bitwise operations
namespace["ceil"] = np.vectorize(lambda x: x.ceil())
namespace["clip"] = np.clip
namespace["conj"] = np.vectorize(lambda x: acb.conjugate(x))
namespace["copysign"] = np.copysign
namespace["cos"] = np.vectorize(lambda x: x.cos())
namespace["cosh"] = np.vectorize(lambda x: x.cosh())
namespace["divide"] = np.divide
namespace["equal"] = np.equal
namespace["exp"] = np.vectorize(lambda x: x.exp())
namespace["expm1"] = np.vectorize(lambda x: x.expm1())
namespace["floor"] = np.vectorize(lambda x: x.floor())
namespace["floor_divide"] = np.floor_divide
namespace["greater"] = np.greater
namespace["greater_equal"] = np.greater_equal
namespace["hypot"] = np.vectorize(lambda x1, x2: abs(x1 + x2 * 1j))
namespace["imag"] = np.vectorize(
    lambda x: x.imag if hasattr(x, "imag") else acb.imag(x)
)
namespace["isfinite"] = np.vectorize(
    lambda x: x.is_finite() if hasattr(x, "is_finite") else np.isfinite(x)
)
# namespace["isinf"] = None
namespace["isnan"] = np.vectorize(lambda x: x.is_nan())
namespace["less"] = np.less
namespace["less_equal"] = np.less_equal
namespace["log"] = np.vectorize(lambda x: x.log())
namespace["log1p"] = np.vectorize(lambda x: x.log1p())
namespace["log2"] = np.vectorize(lambda x: x.log() / arb(2).log())
namespace["log10"] = np.vectorize(lambda x: x.log() / arb(10).log())
namespace["logaddexp"] = np.vectorize(lambda x1, x2: (x1.exp() + x2.exp()).log())
# no logical operations
namespace["maximum"] = np.vectorize(lambda x1, x2: x1.max(x2))
namespace["minimum"] = np.vectorize(lambda x1, x2: x1.min(x2))
namespace["multiply"] = np.multiply
namespace["negative"] = np.negative
# no nextafter
namespace["nextafter"] = np.vectorize(lambda x1, x2: x1)
namespace["not_equal"] = np.not_equal
namespace["positive"] = np.positive
namespace["pow"] = np.pow
namespace["real"] = np.real
namespace["reciprocal"] = np.reciprocal
namespace["remainder"] = np.remainder
# namespace["round"] = None
namespace["sign"] = np.vectorize(lambda x: x.sgn())
namespace["signbit"] = np.vectorize(lambda x: x.sgn())
namespace["sin"] = np.vectorize(lambda x: x.sin())
namespace["sinh"] = np.vectorize(lambda x: x.sinh())
namespace["square"] = np.vectorize(lambda x: x**2)
namespace["sqrt"] = np.vectorize(lambda x: x.sqrt())
namespace["subtract"] = np.subtract
namespace["tan"] = np.vectorize(lambda x: x.tan())
namespace["tanh"] = np.vectorize(lambda x: x.tanh())
namespace["trunc"] = np.vectorize(lambda x: x.floor() if x >= 0 else x.ceil())


def __array_namespace_info__() -> Any:
    info = np.__array_namespace_info__()

    def default_dtypes(*, device: Any = None) -> Any:
        return {
            "real floating": arb,
            "complex floating": acb,
            "integral": fmpz,
            "indexing": info.default_dtypes(device=device)["indexing"],
        }

    info["default_dtypes"] = default_dtypes

    def dtypes(*, device: Any = None, kind: Any = None) -> list[Any]:
        if kind in ["bool", "unsigned integer"]:
            return []
        elif kind in ["signed integer", "integral"]:
            return [fmpz]
        elif kind == "real floating":
            return [arf, arb]
        elif kind == "complex floating":
            return [acb]
        elif kind == "numeric":
            return [fmpz, fmpq, arf, arb, acb]
        elif isinstance(kind, tuple):
            return list(set().union(*(dtypes(kind=k) for k in kind)))
        else:
            return []

    info["dtypes"] = dtypes
    return info


namespace["__array_namespace_info__"] = __array_namespace_info__

# Linear Algebra Functions
for name in ["matmul", "matrix_transpose", "tensordot", "vecdot"]:
    namespace[name] = getattr(np, name)

# Manipulation Functions
for name in [
    "broadcast_arrays",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "moveaxis",
    "permute_dims",
    "repeat",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "tile",
    "unstack",
]:
    namespace[name] = getattr(np, name)

# Searching Functions
for name in ["argmax", "argmin", "count_nonzero", "nonzero", "searchsorted", "where"]:
    namespace[name] = getattr(np, name)

# Set Functions
for name in ["unique_all", "unique_counts", "unique_inverse", "unique_values"]:
    namespace[name] = getattr(np, name)

# Sorting Functions
for name in ["argsort", "sort"]:
    namespace[name] = getattr(np, name)

# Statistical Functions
for name in [
    "cumulative_prod",
    "cumulative_sum",
    "max",
    "mean",
    "min",
    "prod",
    "std",
    "sum",
    "var",
]:
    namespace[name] = getattr(np, name)

# Utility Functions
for name in ["all", "any", "diff"]:
    namespace[name] = getattr(np, name)

__array_api_version__ = "2024.12"
namespace["__array_api_version__"] = __array_api_version__

linalg: AttrDict[Any] = AttrDict()


def tomat(a: Any, /) -> Any:
    if a.dtype == acb:
        mattype = acb_mat
    elif a.dtype == arb:
        mattype = arb_mat
    elif a.dtype == fmpq:
        mattype = fmpq_mat
    elif a.dtype == fmpz:
        mattype = fmpz_mat
    else:
        raise TypeError("Unsupported dtype for matrix conversion.")
    ashape = a.shape
    a = np.reshape(a, (-1, a.shape[-2], a.shape[-1]))
    a = np.asarray([mattype(el.tolist()) for el in a])
    a = np.reshape(a, ashape[:-2])
    a = a.view(flarray)
    a._fl_dtype = a.dtype
    return a


def frommat(a: Any, /) -> Any:
    ashape = a.shape
    a = np.reshape(a, (-1,))
    a = np.asarray([asarray(el.table(), dtype=a.dtype.__element_type__) for el in a])
    a = np.reshape(a, ashape + a[0].shape[1:])
    a = a.view(flarray)
    a._fl_dtype = a.dtype
    return a


def linalg_wrapper(f: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(a: Any, /, **kwargs: Any) -> Any:
        pass


linalg["cholesky"] = np.linalg.cholesky
