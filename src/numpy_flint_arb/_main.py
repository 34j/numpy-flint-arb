from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Protocol

import numpy as np
from array_api.latest import Array, ArrayNamespaceFull
from flint import acb, acb_mat, arb, arb_mat, arf, ctx, fmpq, fmpq_mat, fmpz, fmpz_mat

dtypes = [acb, arb, arf, fmpz, fmpq]

_ALLOW_FLOAT_INPUT = False
_ALLOW_NONINTERVAL_INPUT = False


@contextmanager
def allow_input(*, interval: bool = False, float: bool = False) -> Any:
    global _ALLOW_FLOAT_INPUT, _ALLOW_NONINTERVAL_INPUT
    old_allow_float_input = _ALLOW_FLOAT_INPUT
    old_allow_noninterval_input = _ALLOW_NONINTERVAL_INPUT
    _ALLOW_FLOAT_INPUT = float
    _ALLOW_NONINTERVAL_INPUT = interval
    try:
        yield
    finally:
        _ALLOW_FLOAT_INPUT = old_allow_float_input
        _ALLOW_NONINTERVAL_INPUT = old_allow_noninterval_input


class NotAllowedError(ValueError):
    pass


class FloatInputNotAllowedError(NotAllowedError):
    def __init__(self) -> None:
        super().__init__("Float input is not allowed. Use allow_input context manager to allow it.")


class NonIntervalInputNotAllowedError(NotAllowedError):
    def __init__(self) -> None:
        super().__init__(
            "Non-interval input is not allowed. Use allow_input context manager to allow it."
        )


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


class ArrayNamespaceFullFlintArb[TArray: Array, TDtype, TDevice](
    ArrayNamespaceFull[TArray, TDtype, TDevice], Protocol
):
    def contains(self, x: TArray, y: TArray) -> TArray:
        """Returns nonzero iff y is contained in x."""
        ...

    def contains_integer(self, x: TArray) -> TArray:
        """
        Returns nonzero iff the complex interval
        represented by x contains an integer.
        """
        ...

    def overlaps(self, x: TArray, y: TArray) -> TArray:
        """Returns nonzero iff x and y have some point in common."""
        ...


namespace: ArrayNamespaceFullFlintArb["flarray", Any, Any] = AttrDict()  # type: ignore


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

    def __contains__(self, value: object) -> "flarray":
        raise NotImplementedError(
            "Since 'in' operator tries to convert the return value to bool, "
            "use np.contains(x, y) instead of y in x."
        )
        # return np.vectorize(lambda self, value: self in value)(self, value)


def asarray(
    obj: Any, /, *, dtype: Any = None, device: Any = None, copy: bool | None = None
) -> np.ndarray:
    if dtype is not None and dtype not in dtypes:
        raise TypeError(f"dtype must be one of {', '.join([str(t) for t in dtypes])}, got {dtype}.")
    a = np.asarray(obj)
    el = a.ravel()[0]
    if dtype is not None and isinstance(el, dtype) and (copy is False or copy is None):
        return a
    elif copy is False:
        raise ValueError("Cannot convert to the requested dtype without copying.")
    elif dtype is None:
        dtype = _fltype(a)

    dtype_a = a.dtype
    if dtype_a == np.object_:
        if isinstance(el, arf) and dtype != arf:
            raise NonIntervalInputNotAllowedError()
        a = np.vectorize(lambda z: dtype(z))(a)
    elif np.isdtype(dtype_a, "integral") or np.isdtype(dtype_a, "bool"):
        a = np.vectorize(lambda z: dtype(int(z)))(a)
    elif np.isdtype(dtype_a, "real floating") or np.isdtype(dtype_a, "complex floating"):
        errors: list[NotAllowedError] = []
        if not _ALLOW_FLOAT_INPUT:
            errors.append(FloatInputNotAllowedError())
        if not _ALLOW_NONINTERVAL_INPUT and dtype in [arb, acb]:
            errors.append(NonIntervalInputNotAllowedError())
        if errors:
            raise ExceptionGroup("Input is not allowed.", errors)
        a = np.vectorize(
            lambda z: dtype((float if np.isdtype(dtype_a, "real floating") else complex)((z)))
        )(a)
    a = a.view(flarray)
    a._fl_dtype = dtype
    return a


namespace["asarray"] = asarray

# Specific functions
namespace["contains"] = np.vectorize(lambda x, y: x.contains(y))
namespace["contains_integer"] = np.vectorize(lambda x: x.contains_integer())
namespace["overlaps"] = np.vectorize(lambda x, y: x.overlaps(y))

# Constants
namespace["e"] = arb(1).exp()
namespace["pi"] = arb.pi()
namespace["nan"] = arb.nan()
namespace["newaxis"] = None
namespace["inf"] = arb("+inf")

# Creation Functions
# Simply call numpy functions
for name in ["meshgrid", "tril", "triu"]:
    namespace[name] = getattr(np, name)

# Need to asarray after creation
# The values numpy returns are integers
# and exact (no rounding issues) except for linspace
for name in [
    "arange",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
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


def linspace(
    start: Any, stop: Any, /, num: int = 50, dtype: Any = None, device: Any = None
) -> np.ndarray:
    if start not in dtypes:
        raise TypeError("start must be a flint type.")
    if stop not in dtypes:
        raise TypeError("stop must be a flint type.")
    dtype = result_type(start, stop, dtype)
    diff = (stop - start) / (num - 1) if num > 1 else 0
    diff = dtype(diff)
    return start + diff * namespace["arange"](num, dtype=dtype)


namespace["linspace"] = linspace


# Data Type Functions
def astype(x: Any, dtype: Any, /, *, copy: bool = True, device: Any = None) -> np.ndarray:
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
    return AttrDict({
        "bits": ctx.prec,
        "eps": None,
        "max": None,
        "min": None,
        "smallest_normal": None,
        "dtype": arf,
    })


namespace["finfo"] = finfo


def iinfo(type: Any, /) -> Any:
    return AttrDict({
        "bits": None,
        "max": None,
        "min": None,
        "dtype": fmpz,
    })


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
# Use np.vectorize to wrap flint methods
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
namespace["conj"] = lambda x: np.vectorize(lambda x: acb.conjugate(x))(x) if x.dtype == acb else x
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
namespace["imag"] = np.vectorize(lambda x: x.imag if hasattr(x, "imag") else acb.imag(x))
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
    new_info: AttrDict[Any] = AttrDict({
        "capabilities": info.capabilities,
        "default_device": info.default_device,
        "devices": info.devices,
    })

    def default_dtypes(*, device: Any = None) -> Any:
        return {
            "real floating": arb,
            "complex floating": acb,
            "integral": fmpz,
            "indexing": info.default_dtypes(device=device)["indexing"],
        }

    new_info["default_dtypes"] = default_dtypes

    def dtypes(*, device: Any = None, kind: Any = None) -> dict[str, Any]:
        if kind in ["bool"]:
            return {"bool": np.bool}
        elif kind in ["unsigned integer"]:
            return {}
        elif kind in ["signed integer", "integral"]:
            return {"fmpz": fmpz}
        elif kind == "real floating":
            return {"arb": arb, "arf": arf}
        elif kind == "complex floating":
            return {"acb": acb}
        elif kind == "numeric":
            return {"fmpz": fmpz, "fmpq": fmpq, "arb": arb, "acb": acb, "arf": arf}
        elif isinstance(kind, tuple):
            res = {}
            for k in kind:
                res.update(dtypes(device=device, kind=k))
            return res
        else:
            return {}

    new_info["dtypes"] = dtypes
    return new_info


namespace["__array_namespace_info__"] = __array_namespace_info__

# Linear Algebra Functions
# Simply call numpy functions
for name in ["matmul", "matrix_transpose", "tensordot", "vecdot"]:
    namespace[name] = getattr(np, name)

# Manipulation Functions
# Simply call numpy functions
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
# Simply call numpy functions
for name in ["argmax", "argmin", "count_nonzero", "nonzero", "searchsorted", "where"]:
    namespace[name] = getattr(np, name)

# Set Functions
# Simply call numpy functions
for name in ["unique_all", "unique_counts", "unique_inverse", "unique_values"]:
    namespace[name] = getattr(np, name)

# Sorting Functions
# Simply call numpy functions
for name in ["argsort", "sort"]:
    namespace[name] = getattr(np, name)

# Statistical Functions
# Simply call numpy functions
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
# Simply call numpy functions
for name in ["all", "any", "diff"]:
    namespace[name] = getattr(np, name)

# Data Types
for t in ["bool", "bool_"]:
    namespace[t] = np.bool
for t in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]:
    namespace[t] = fmpz
for t in ["float32", "float64"]:
    namespace[t] = arb
for t in ["complex64", "complex128"]:
    namespace[t] = acb

# Must be a ModuleType
namespace["__name__"] = "numpy_flint_arb.np"

__array_api_version__ = "2024.12"
namespace["__array_api_version__"] = __array_api_version__

# Linear Algebra Functions
linalg: AttrDict[Any] = AttrDict()
namespace["linalg"] = linalg


def tomat(a: Any, /) -> Any:
    """
    Convert array of shape (..., m, n) to
    array of flint matrices of shape (m, n) of shape (..., ).

    Parameters
    ----------
    a : Any
        The input array of shape (..., m, n).

    Returns
    -------
    Any
        The output array of flint matrices of shape (..., ).

    """
    dtype = a.dtype
    if dtype == acb:
        mattype = acb_mat
    elif dtype == arb:
        mattype = arb_mat
    elif dtype == fmpq:
        mattype = fmpq_mat
    elif dtype == fmpz:
        mattype = fmpz_mat
    else:
        raise TypeError("Unsupported dtype for matrix conversion.")
    ashape = a.shape
    a = np.reshape(a, (-1, a.shape[-2], a.shape[-1]))
    a = np.asarray([mattype(el.tolist()) for el in a])
    a = np.reshape(a, ashape[:-2])
    a = a.view(flarray)
    a._fl_dtype = dtype
    return a


def frommat(a: Any, /) -> Any:
    """
    Convert array of flint matrices of shape (m, n) of shape (..., )
    to array of shape (..., m, n).

    Parameters
    ----------
    a : Any
        The input array of flint matrices of shape (..., ).

    Returns
    -------
    Any
        The output array of shape (..., m, n).

    """
    ashape = a.shape
    dtype = a.dtype
    a = np.reshape(a, (-1,))
    a = np.asarray([el.table() for el in a])
    a = np.reshape(a, ashape + a.shape[1:])
    a = a.view(flarray)
    a._fl_dtype = dtype
    return a


def vectorize_mat(f_mat: Callable[..., Any], /, *, n_args: int = 1) -> Callable[..., Any]:
    """
    Return a function to call a function for flint matrices
    along with last 2 axes.

    Parameters
    ----------
    f_mat : Callable[..., Any]
        The function to be called for flint matrices.
    n_args : int, optional
        The number of arguments to be converted to flint matrices,
        by default 1.

    Returns
    -------
    Callable[..., Any]
        The wrapped function.

    """

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        args_ = list(args)
        for i in range(n_args):
            args_[i] = tomat(args_[i])
        res = np.vectorize(lambda x: f_mat(x, *args_[n_args:], **kwargs))(*args_[:n_args])
        if isinstance(res.ravel()[0], (acb_mat, arb_mat, fmpz_mat, fmpq_mat)):
            res = frommat(res)
        return res

    return wrapped


# linalg["cholesky"] = None
# linalg["cross"] = None
linalg["det"] = vectorize_mat(lambda x: x.det())
linalg["diagonal"] = np.linalg.diagonal
linalg["eigh"] = vectorize_mat(lambda x: x.eig(right=True))
linalg["eigvalsh"] = vectorize_mat(lambda x: x.eig())
linalg["inv"] = vectorize_mat(lambda x: x.inv())
linalg["matmul"] = vectorize_mat(lambda x, y: x * y)
linalg["matrix_norm"] = np.linalg.matrix_norm
linalg["matrix_power"] = vectorize_mat(lambda x, n: x**n)
# linalg["matrix_rank"] = None
linalg["matrix_transpose"] = np.linalg.matrix_transpose
# linalg["outer"] = None
# linalg["pinv"] = None
# linalg["qr"] = None
# linalg["slogdet"] = None


def solve(a: Any, b: Any, /) -> Any:
    expand_b = b.ndim < a.ndim
    if expand_b:
        b = b[..., :, None]
    a_mat = tomat(a)
    b_mat = tomat(b)
    x_mat = np.vectorize(lambda A, B: A.solve(B))(a_mat, b_mat)
    x = frommat(x_mat)
    if expand_b:
        x = x[..., :, 0]
    return x


linalg["solve"] = solve
# linalg["svd"] = None
# linalg["svdvals"] = None
linalg["tensordot"] = np.linalg.tensordot
linalg["trace"] = np.linalg.trace
linalg["vecdot"] = np.linalg.vecdot
linalg["vector_norm"] = np.linalg.vector_norm

# Random Functions
# Simply call asarray after generating with numpy
random: AttrDict[Any] = AttrDict()
namespace["random"] = random


def _uniform(*args: Any, **kwargs: Any) -> Any:
    with allow_input(float=True, interval=True):
        return asarray(np.random.uniform(*args, **kwargs), dtype=kwargs.get("dtype", arb))


def _normal(*args: Any, **kwargs: Any) -> Any:
    with allow_input(float=True, interval=True):
        return asarray(np.random.normal(*args, **kwargs), dtype=kwargs.get("dtype", arb))


random["uniform"] = _uniform
random["normal"] = _normal

# Special Functions (scipy.special)
special: AttrDict[Any] = AttrDict()
namespace["special"] = special

special["airy"] = np.vectorize(lambda x: x.airy())
for name in ["jv", "jn"]:
    special[name] = lambda v, x: np.vectorize(lambda x: x.bessel_j(v))(x)
for name in ["yv", "yn"]:
    special[name] = lambda v, x: np.vectorize(lambda x: x.bessel_y(v))(x)
for name in ["iv"]:  # somewhat "in" is not in scipy.special
    special[name] = lambda v, x: np.vectorize(lambda x: x.bessel_i(v))(x)
for name in ["kv", "kn"]:
    special[name] = lambda v, x: np.vectorize(lambda x: x.bessel_k(v))(x)
special["hankel1"] = lambda v, x: np.vectorize(
    lambda x: (
        acb(2)
        / acb(1j)
        / acb.pi()
        * acb.exp(-1j * acb.pi() * v / 2)
        * acb.bessel_k(x * acb.exp(-1j * acb.pi() / 2), v)
    )
)(x)
special["hankel2"] = lambda v, x: np.vectorize(
    lambda x: (
        acb(-2)
        / acb(1j)
        / acb.pi()
        * acb.exp(1j * acb.pi() * v / 2)
        * acb.bessel_k(x * acb.exp(1j * acb.pi() / 2), v)
    )
)(x)

special["gamma"] = np.vectorize(lambda x: x.gamma())
special["gammaln"] = np.vectorize(lambda x: x.lgamma())
special["loggamma"] = np.vectorize(lambda x: x.lgamma())
special["beta"] = np.vectorize(lambda x, y: x.beta(y))
special["rgamma"] = np.vectorize(lambda x: x.rgamma())
special["digamma"] = np.vectorize(lambda x: x.digamma())

special["erf"] = np.vectorize(lambda x: x.erf())
special["erfc"] = np.vectorize(lambda x: x.erfc())
special["erfi"] = np.vectorize(lambda x: x.erfi())
special["fresnel"] = lambda x: (
    np.vectorize(lambda x: x.fresnel_s())(x),
    np.vectorize(lambda x: x.fresnel_c())(x),
)

special["legendre_p"] = lambda n, x: np.vectorize(lambda x: x.legendre_p(n))(x)
special["lqn"] = lambda n, x: np.vectorize(lambda x: x.legendre_q(n))(x)
