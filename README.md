# numpy-flint-arb

<p align="center">
  <a href="https://github.com/34j/numpy-flint-arb/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/numpy-flint-arb/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://numpy-flint-arb.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/numpy-flint-arb.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/numpy-flint-arb">
    <img src="https://img.shields.io/codecov/c/github/34j/numpy-flint-arb.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/numpy-flint-arb/">
    <img src="https://img.shields.io/pypi/v/numpy-flint-arb.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/numpy-flint-arb.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/numpy-flint-arb.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://numpy-flint-arb.readthedocs.io" target="_blank">https://numpy-flint-arb.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/numpy-flint-arb" target="_blank">https://github.com/34j/numpy-flint-arb </a>

---

Arbitrary precision ball arithmetic (interval arithmetic) dtype in NumPy

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install numpy-flint-arb
```

## Usage

Import `numpy_flint_arb.np` instead of `numpy`:

```python
from numpy_flint_arb import np

A = np.random.normal(size=(2, 2))
b = np.random.normal(size=(2,))
x = np.linalg.solve(A, b)
b_approx = A @ x
assert np.all(np.contains(b_approx, b))
```

### `asarray()` and Input Check

To avoid mixing ordinary floats like `float` or `np.float`, `flarray` for `arb`, `acb` only accepts integers, `arb` or `acb` and `flarray` for `arf` only accepts integers and `arf, arb`.

To relax this, `allow_input()` may be used:

```python
import pytest
from numpy_flint_arb import allow_input
from flint import arb, arf

# arb array
with pytest.raises(Exception):
    np.asarray(0.5, dtype=arb)
with pytest.raises(Exception):
    with allow_input(float=True):
        np.asarray(0.5, dtype=arb)
with allow_input(interval=True, float=True):
    np.asarray(0.5, dtype=arb)

# arf array
with pytest.raises(Exception):
    np.asarray(0.5, dtype=arf)
with allow_input(float=True):
    np.asarray(0.5, dtype=arf)
with allow_input(interval=True, float=True):
    np.asarray(0.5, dtype=arf)
```

Note that `allow_input()` does not affect for `arf()`, `arb()`, `acb()` constructors, but only for `np.asarray()` and `flarray()`.

#### `str` input

One can input `str` to `asarray()`.
If `dtype` is not specified, it will be automatically detected as `arb`. However, specifying `dtype` explicitly is recommended.

#### `asarray(dtype=acb)`

`asarray()` does not support separated input for real and imaginary parts.

Do the following instead, as `acb(1j)` is exact.

```python
from flint import arb, acb
from numpy_flint_arb import np

with pytest.raises(Exception):
    # python-flint does not support single argument str input for acb
    np.asarray("[0.5 +/- 0.001] + [0.5 +/- 0.001]j", dtype=acb)
with pytest.raises(Exception):
    # This is inexact and raises an error without allow_input()
    np.asarray(0.5 + 0.5j, dtype=acb)
with pytest.raises(Exception):
    # Mixing complex and arb is not supported by python-flint
    np.asarray("0.5 +/- 0.001", dtype=arb) + 1j * np.asarray("0.5 +/- 0.001", dtype=arb)
```

```python
>>> # This is possible but not recommended
>>> np.asarray("0.5 +/- 0.001", dtype=arb) + 1j * np.asarray("0.5 +/- 0.001", dtype=acb)
flarray([0.50 +/- 1.01e-3] + [0.50 +/- 1.01e-3]j,
        dtype=<class 'flint.types.arb.arb'>)
>>> # Recommended
>>> np.asarray("0.5 +/- 0.001", dtype=arb) + acb(1j) * np.asarray("0.5 +/- 0.001", dtype=arb)
flarray([0.50 +/- 1.01e-3] + [0.50 +/- 1.01e-3]j,
        dtype=<class 'flint.types.arb.arb'>)
```

## Randomness

Since `python-flint` does not support random number generation, the `random` module just uses `np.random`.
Therefore, the return values may not be random up to the precision of `arb`, `acb`.

## What it does

- This package adds a `flarray` which [subclasses `ndarray`](https://numpy.org/doc/stable/user/basics.subclassing.html) in order to
  - Override `__array_namespace__` to `numpy_flint_arb.np`
  - Override `dtype` to return newly added `_fl_dtype` private attribute, since the actual internal dtype `object` cannot be overridden.
  - Override `__array_finalize__` as recommended by the NumPy docs to return `flarray` with proper `_fl_dtype` instead of `ndarray` after Numpy operations.
- Partially supports `linalg` and `(scipy.)special` functions.
- Adds `tomat()` and `frommat()` to treat `flarray` as array of `arb_mat` / `acb_mat`, so that we can perform matrix operations like `np.linalg.solve` on `flarray`.
- Does not perform any parallelization to avoid complexity and to fully utilize the great `python-flint` library
  - Using `arb_series` and `acb_series` may be faster for additions but this is too hacky.
  - Defining custom `dtype` is way too complicated
  - Writing C extension would be theoretically also possible but is still too complicated.
- Does not support `in` operator since it tries to convert the return value to bool. Use newly added `np.contains(x, y)` and `np.overlaps(x, y)` instead.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/34j"><img src="https://avatars.githubusercontent.com/u/55338215?v=4?s=80" width="80px;" alt="34j"/><br /><sub><b>34j</b></sub></a><br /><a href="https://github.com/34j/numpy-flint-arb/commits?author=34j" title="Code">💻</a> <a href="#ideas-34j" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/34j/numpy-flint-arb/commits?author=34j" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
