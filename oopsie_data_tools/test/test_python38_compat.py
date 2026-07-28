"""Guard the Python 3.8 floor against runtime-only annotation breakage.

Ruff's ``target-version`` only tunes autofixes; it does not reject newer syntax, and it
cannot see runtime-only breakage at all. tyro (used by the ``examples/``) resolves dataclass
field annotations at runtime via ``typing.get_type_hints``, which raises ``TypeError`` on 3.8
for a PEP 604 union (``str | None``) *even under* ``from __future__ import annotations``.

The check is static rather than an import test because the examples import ``droid`` and
``openpi_client``, neither of which is on PyPI, so CI cannot import them.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_DIR = _REPO_ROOT / "oopsie_data_tools"
_EXAMPLES_DIR = _REPO_ROOT / "examples"


def _decorator_name(node):
    """Last dotted component of a decorator, without ast.unparse (absent on 3.8)."""
    while isinstance(node, ast.Call):
        node = node.func
    while isinstance(node, ast.Attribute):
        node = node.attr if isinstance(node.attr, str) else node.value
        return node if isinstance(node, str) else ""
    return node.id if isinstance(node, ast.Name) else ""


def _scan(root: Path):
    """Return ``(offenders, dataclasses_inspected)`` for every .py file under *root*."""
    offenders = []
    inspected = 0
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any("dataclass" in _decorator_name(d) for d in node.decorator_list):
                continue
            inspected += 1
            for stmt in node.body:
                if not isinstance(stmt, ast.AnnAssign):
                    continue
                if isinstance(stmt.annotation, ast.BinOp) and isinstance(
                    stmt.annotation.op, ast.BitOr
                ):
                    field = stmt.target.id if isinstance(stmt.target, ast.Name) else "?"
                    rel = path.relative_to(_REPO_ROOT)
                    offenders.append(f"{rel}:{stmt.lineno}  {node.name}.{field}")
    return offenders, inspected


@pytest.mark.parametrize(
    "root",
    [
        pytest.param(_PACKAGE_DIR, id="package"),
        pytest.param(_EXAMPLES_DIR, id="examples"),
    ],
)
def test_no_pep604_in_dataclass_fields(root: Path) -> None:
    if not root.is_dir():
        # examples/ is not shipped in the wheel; nothing to scan from an installed package.
        pytest.skip(f"{root.name}/ not present")

    offenders, inspected = _scan(root)

    # Without this the test passes trivially if the scan ever stops finding dataclasses —
    # exactly how the original version of this check silently reported success.
    assert inspected > 0, f"no dataclasses found under {root}; the scan is not working"
    assert not offenders, (
        "PEP 604 union in a dataclass field. tyro resolves these via typing.get_type_hints, "
        "which raises TypeError on the Python 3.8 floor. Use Optional[X] instead:\n  "
        + "\n  ".join(offenders)
    )
