"""Small HDF5 helpers shared by the package and the scripts.

Deliberately dependency-light — numpy only, no h5py or cv2 — so conversion and inspection
scripts can use it without pulling in the validation stack.

Both helpers replaced a family of near-identical local copies: four separate
bytes-to-string decoders across the scripts, and three hand-rolled episode globs that
disagreed about whether ``.hdf5`` counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

#: Extensions an episode file may carry. ``validate`` accepts both.
EPISODE_SUFFIXES = (".h5", ".hdf5")


def decode_h5_scalar(value: Any) -> str:
    """Decode an HDF5 attribute or dataset scalar to a plain ``str``.

    Handles the shapes h5py hands back depending on how a value was written: raw ``bytes``,
    ``str``, numpy scalars, and 0-d or single-element arrays. Anything else is stringified,
    so this never raises.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, np.generic):
        return decode_h5_scalar(value.item())
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_h5_scalar(value.item())
        if value.size == 0:
            return ""
        return decode_h5_scalar(value.reshape(-1)[0])
    return str(value)


def find_episode_files(
    root: Path | str, suffixes: Iterable[str] = EPISODE_SUFFIXES
) -> list[Path]:
    """Every episode file under ``root``, recursively, in a stable order.

    Args:
        root: Directory to search.
        suffixes: Extensions to accept. Pass a narrower set where downstream code only
            handles one — the annotation server's path guard accepts ``.h5`` alone.
    """
    root = Path(root)
    wanted = {s.lower() for s in suffixes}
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in wanted
    )
