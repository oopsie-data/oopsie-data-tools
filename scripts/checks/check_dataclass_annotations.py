"""Flag PEP 604 (`X | Y`) in dataclass field annotations.

tyro resolves these at runtime via typing.get_type_hints, which raises TypeError on
Python 3.8 even under `from __future__ import annotations`. Static, because the example
scripts import droid/openpi_client, which are not on PyPI.
"""
import ast
import sys
from pathlib import Path


def _decorator_name(node):
    """Last dotted component of a decorator, without ast.unparse (absent on 3.8)."""
    while isinstance(node, ast.Call):
        node = node.func
    while isinstance(node, ast.Attribute):
        node = node.attr if isinstance(node.attr, str) else node.value
        return node if isinstance(node, str) else ""
    return node.id if isinstance(node, ast.Name) else ""


bad, checked = [], 0
for path in sorted(Path("examples").rglob("*.py")) + sorted(Path("oopsie_data_tools").rglob("*.py")):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any("dataclass" in _decorator_name(d) for d in node.decorator_list):
            continue
        checked += 1
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.annotation, ast.BinOp):
                if isinstance(stmt.annotation.op, ast.BitOr):
                    name = stmt.target.id if isinstance(stmt.target, ast.Name) else "?"
                    bad.append(f"  {path}:{stmt.lineno}  {node.name}.{name}")

print(f"inspected {checked} dataclasses")
print("\n".join(bad) if bad else "  no PEP 604 in any dataclass field")
sys.exit(1 if bad else 0)
