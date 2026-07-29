"""Human-readable dump of an HDF5 file: groups, datasets, shapes, dtypes, attributes.

Backs ``oopsie-data inspect``. This is a debugging aid for looking at a recorded episode,
not a validator — it never rejects anything and makes no assumptions about the schema, so
it is equally useful on a file that fails ``oopsie-data validate``.

Output goes to stdout via ``print`` rather than the logger: it is the command's result,
not a progress report.
"""

from __future__ import annotations

import datetime as _dt
import math
from pprint import pformat
from typing import Any

import h5py
import numpy as np

from oopsie_data_tools.utils.h5 import decode_h5_scalar

_MAX_STR = 200  # truncation width for string scalars
_MAX_ELEMS = 32  # array elements shown before eliding
_MAX_LIST = 16  # list entries shown before eliding


def _human_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    if n < 1024:
        return f"{n} B"
    units = ["KiB", "MiB", "GiB", "TiB", "PiB"]
    f = float(n)
    for u in units:
        f /= 1024.0
        if f < 1024.0:
            return f"{f:.2f} {u}"
    return f"{f:.2f} EiB"


def _truncate(s: str) -> str:
    return s[:_MAX_STR] + "…" if len(s) > _MAX_STR else s


def _fmt_scalar(v: Any) -> str:
    if isinstance(v, (np.generic,)):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, bytes):
        return repr(_truncate(decode_h5_scalar(v)))
    if isinstance(v, str):
        return repr(_truncate(v))
    if isinstance(v, (_dt.datetime, _dt.date)):
        return v.isoformat()
    return repr(v)


def _fmt_array(a: np.ndarray, *, max_elems: int = _MAX_ELEMS) -> str:
    # Keep output stable and short for big arrays.
    if a.size == 0:
        return f"array(shape={a.shape}, dtype={a.dtype}, empty)"

    flat = a.ravel()
    head = flat[: min(flat.size, max_elems)]
    suffix = f", … (+{flat.size - max_elems} more)" if flat.size > max_elems else ""

    try:
        content = np.array2string(head, threshold=max_elems, edgeitems=math.inf)
    except Exception:
        content = repr(head)

    return f"array(shape={a.shape}, dtype={a.dtype}, head={content}{suffix})"


def _fmt_attr_value(v: Any) -> str:
    # h5py may return scalars, bytes, numpy arrays, or lists.
    if isinstance(v, np.ndarray):
        return _fmt_array(v)
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return "[]"
        if len(v) <= _MAX_LIST:
            return pformat([_fmt_scalar(x) for x in v])
        head = [_fmt_scalar(x) for x in v[:_MAX_LIST]]
        return pformat(head)[:-1] + f", … (+{len(v) - _MAX_LIST} more)]"
    return _fmt_scalar(v)


# ── one walk, two renderers ───────────────────────────────────────────────────
#
# The tree is built once and rendered twice, so the printed dump and ``--json`` cannot
# disagree about what is in the file. Attribute values are carried raw: the printed form
# elides them for a terminal, the JSON form keeps them whole for a program.


def _raw_attrs(obj: h5py.Group | h5py.Dataset) -> dict:
    out = {}
    for k in sorted(obj.attrs.keys()):
        try:
            out[k] = obj.attrs[k]
        except Exception as e:  # a corrupt attr must not abort the whole dump
            out[k] = f"<error reading attr: {e}>"
    return out


def _structure(obj: h5py.Group | h5py.Dataset) -> dict:
    if isinstance(obj, h5py.Group):
        node: dict = {"type": "group", "attrs": _raw_attrs(obj), "children": {}}
        for k in sorted(obj.keys()):
            # Show links explicitly rather than following them (common in some layouts).
            link = obj.get(k, getlink=True)
            if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                node["children"][k] = {"type": "link", "target": str(link)}
            else:
                node["children"][k] = _structure(obj[k])
        return node

    if isinstance(obj, h5py.Dataset):
        # An h5py.Empty dataset has a dtype but no shape, and the schema uses those as
        # placeholders for undeclared action keys — so "empty" has to be reportable.
        return {
            "type": "dataset",
            "shape": None if obj.shape is None else list(obj.shape),
            "dtype": str(obj.dtype),
            "empty": obj.shape is None,
            "chunks": obj.chunks,
            "compression": obj.compression,
            "fillvalue": obj.fillvalue,
            "nbytes": None if obj.size is None else int(obj.size) * obj.dtype.itemsize,
            "attrs": _raw_attrs(obj),
        }

    return {"type": "unknown", "repr": str(type(obj))}


def _print_node(name: str, node: dict, *, indent: int) -> None:
    pad = " " * indent
    if node["type"] == "link":
        print(f"{pad}[link] {name} -> {node['target']}")
        return

    if node["type"] == "group":
        print(f"{pad}[group] {name}")
    elif node["type"] == "dataset":
        parts = [f"shape={tuple(node['shape']) if node['shape'] is not None else None}",
                 f"dtype={node['dtype']}"]
        if node["chunks"] is not None:
            parts.append(f"chunks={node['chunks']}")
        if node["compression"] is not None:
            parts.append(f"compression={node['compression']!r}")
        if node["fillvalue"] is not None:
            parts.append(f"fill={_fmt_scalar(node['fillvalue'])}")
        if node["nbytes"] is not None:
            parts.append(f"approx_nbytes={_human_bytes(node['nbytes'])}")
        print(f"{pad}[dataset] {name} ({', '.join(parts)})")
    else:
        print(f"{pad}[unknown] {name}: {node['repr']}")

    if node["attrs"]:
        print(f"{pad}  attrs:")
        for k, v in node["attrs"].items():
            print(f"{pad}    - {k!r}: {_fmt_attr_value(v)}")

    for key, child in node.get("children", {}).items():
        child_name = f"{name.rstrip('/')}/{key}" if name != "/" else f"/{key}"
        _print_node(child_name, child, indent=indent + 2)


def inspect_h5(path: str) -> None:
    """Print the full structure of the HDF5 file at *path* to stdout."""
    with h5py.File(path, "r") as f:
        print(f"HDF5: {path}")
        _print_node("/", _structure(f), indent=0)


def _jsonable(v: Any) -> Any:
    """An attribute value as something ``json.dumps`` accepts.

    Attrs are the interesting half of an episode — the profile, the annotations, the
    instruction — so these are converted rather than stringified. Arrays keep their full
    contents here: unlike the printed dump this output is read by a program, and eliding
    is the caller's business.
    """
    if isinstance(v, bytes):
        return decode_h5_scalar(v)
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return _jsonable(v.item())
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        # JSON has no NaN/Infinity literal; emit the repr rather than a document that
        # only Python's json module can read back.
        return repr(v)
    return v


def _jsonable_node(node: dict) -> dict:
    """One tree node with its attrs converted, dropping the print-only dataset detail."""
    if node["type"] == "link":
        return node
    if node["type"] == "group":
        return {
            "type": "group",
            "attrs": {k: _jsonable(v) for k, v in node["attrs"].items()},
            "children": {k: _jsonable_node(c) for k, c in node["children"].items()},
        }
    if node["type"] == "dataset":
        return {
            "type": "dataset",
            "shape": node["shape"],
            "dtype": node["dtype"],
            "empty": node["empty"],
            "attrs": {k: _jsonable(v) for k, v in node["attrs"].items()},
        }
    return node


def inspect_h5_structure(path: str) -> dict:
    """The same structure ``inspect_h5`` prints, as a JSON-serializable dict.

    Backs ``oopsie-data inspect --json``. Like the printed dump it assumes nothing about
    the schema, so it works on a file ``oopsie-data validate`` rejects.
    """
    with h5py.File(path, "r") as f:
        return {"path": path, "root": _jsonable_node(_structure(f))}
