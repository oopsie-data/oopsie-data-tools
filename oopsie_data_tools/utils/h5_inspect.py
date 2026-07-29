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


def _print_attrs(obj: h5py.Group | h5py.Dataset, *, indent: int) -> None:
    if len(obj.attrs) == 0:
        return
    print(" " * indent + "attrs:")
    for k in sorted(obj.attrs.keys()):
        try:
            v = obj.attrs[k]
        except Exception as e:
            print(" " * (indent + 2) + f"- {k!r}: <error reading attr: {e}>")
            continue
        print(" " * (indent + 2) + f"- {k!r}: {_fmt_attr_value(v)}")


def _describe_dataset(ds: h5py.Dataset) -> str:
    parts = [f"shape={ds.shape}", f"dtype={ds.dtype}"]

    try:
        if ds.chunks is not None:
            parts.append(f"chunks={ds.chunks}")
    except Exception:
        pass

    try:
        comp = ds.compression
        if comp is not None:
            parts.append(f"compression={comp!r}")
    except Exception:
        pass

    try:
        fill = ds.fillvalue
        if fill is not None:
            parts.append(f"fill={_fmt_scalar(fill)}")
    except Exception:
        pass

    try:
        nbytes = int(ds.size) * int(ds.dtype.itemsize)
        parts.append(f"approx_nbytes={_human_bytes(nbytes)}")
    except Exception:
        pass

    return ", ".join(parts)


def _walk(name: str, obj: h5py.Group | h5py.Dataset, *, indent: int) -> None:
    if isinstance(obj, h5py.Group):
        title = "/" if name == "" else name
        print(" " * indent + f"[group] {title}")
        _print_attrs(obj, indent=indent + 2)

        for k in sorted(obj.keys()):
            # Show links explicitly rather than following them (common in some layouts).
            try:
                link = obj.get(k, getlink=True)
                if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                    print(" " * (indent + 2) + f"[link] {title.rstrip('/')}/{k} -> {link}")
                    continue
            except Exception:
                pass

            child_name = (title.rstrip("/") + "/" + k) if title != "/" else ("/" + k)
            _walk(child_name, obj[k], indent=indent + 2)

    elif isinstance(obj, h5py.Dataset):
        print(" " * indent + f"[dataset] {name} ({_describe_dataset(obj)})")
        _print_attrs(obj, indent=indent + 2)
    else:
        print(" " * indent + f"[unknown] {name}: {type(obj)}")


def inspect_h5(path: str) -> None:
    """Print the full structure of the HDF5 file at *path* to stdout."""
    with h5py.File(path, "r") as f:
        print(f"HDF5: {path}")
        _walk("", f, indent=0)


# ── structured form ───────────────────────────────────────────────────────────


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


def _attrs_dict(obj: h5py.Group | h5py.Dataset) -> dict:
    out = {}
    for k in sorted(obj.attrs.keys()):
        try:
            out[k] = _jsonable(obj.attrs[k])
        except Exception as e:
            out[k] = f"<error reading attr: {e}>"
    return out


def _structure(obj: h5py.Group | h5py.Dataset) -> dict:
    if isinstance(obj, h5py.Group):
        node: dict = {"type": "group", "attrs": _attrs_dict(obj), "children": {}}
        for k in sorted(obj.keys()):
            try:
                link = obj.get(k, getlink=True)
                if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                    node["children"][k] = {"type": "link", "target": str(link)}
                    continue
            except Exception:
                pass
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
            "attrs": _attrs_dict(obj),
        }

    return {"type": "unknown", "repr": str(type(obj))}


def inspect_h5_structure(path: str) -> dict:
    """The same structure ``inspect_h5`` prints, as a JSON-serializable dict.

    Backs ``oopsie-data inspect --json``. Like the printed dump it assumes nothing about
    the schema, so it works on a file ``oopsie-data validate`` rejects.
    """
    with h5py.File(path, "r") as f:
        return {"path": path, "root": _structure(f)}
