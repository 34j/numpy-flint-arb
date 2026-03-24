import re
from pathlib import Path

from flint import acb, acb_mat, arb, arb_mat, arf, fmpq, fmpq_mat, fmpz, fmpz_mat


def _public_attributes() -> set[str]:
    types = [acb, acb_mat, arb, arb_mat, arf, fmpq, fmpq_mat, fmpz, fmpz_mat]
    attrs: set[str] = set()
    for t in types:
        attrs.update(name for name in dir(t) if not name.startswith("_"))
    return attrs


def _main_identifiers() -> set[str]:
    main_path = Path(__file__).parents[1] / "src" / "numpy_flint_arb" / "_main.py"
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", main_path.read_text(encoding="utf-8")))


def test_missing_flint_attributes_in_main() -> None:
    all_attrs = _public_attributes()
    main_names = _main_identifiers()
    missing = sorted(all_attrs - main_names)
    print(missing)
