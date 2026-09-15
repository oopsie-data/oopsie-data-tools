"""Video references stored in an episode HDF5.

An episode stores each video as a string dataset holding the MP4's path relative to the
directory of the episode file: one per camera under ``observations/video_paths``, and one per
video-format additional sensor under ``additional_data``. Relative paths keep a session
directory valid after it is moved or uploaded.

Everything that writes, reads or follows these references goes through this module.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import h5py

from oopsie_data_tools.utils.h5 import decode_h5_scalar

logger = logging.getLogger(__name__)

#: Groups whose scalar string datasets are video references. Numeric datasets under
#: ``additional_data`` are sensor arrays, not references.
VIDEO_PATH_GROUPS = ("observations/video_paths", "additional_data")


def relative_to_episode(path: str | os.PathLike[str], episode_dir: str | os.PathLike[str]) -> str:
    """The stored form of ``path``: relative to ``episode_dir``, with ``/`` separators.

    ``path`` may be absolute or relative to ``episode_dir``; ``~`` is expanded. Both sides are
    resolved before the relpath is taken, so a symlinked directory cannot produce a path
    that climbs out of the dataset and back in through the link target.
    """
    base = Path(episode_dir).resolve()
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = base / target
    return Path(os.path.relpath(target.resolve(), start=base)).as_posix()


def resolve_from_episode(stored: str, episode_dir: str | os.PathLike[str]) -> Path:
    """The absolute path a stored reference points to. Absolute references are kept as such."""
    path = Path(stored)
    return path.resolve() if path.is_absolute() else (Path(episode_dir) / path).resolve()


def write_video_path(
    group: h5py.Group,
    name: str,
    path: str | os.PathLike[str],
    episode_dir: str | os.PathLike[str],
) -> None:
    """Store ``path`` as the reference ``group[name]``, replacing any existing one."""
    if name in group:
        del group[name]
    group.create_dataset(
        name,
        data=relative_to_episode(path, episode_dir),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def is_video_reference(ds: Any) -> bool:
    """A scalar string dataset, which is how a video reference is stored."""
    return (
        isinstance(ds, h5py.Dataset)
        and ds.shape == ()
        and h5py.check_string_dtype(ds.dtype) is not None
    )


def read_video_paths(h5_path: Path) -> dict[str, str]:
    """Every video reference in the episode, as ``{dataset path: stored string}``.

    Keys are full dataset paths such as ``observations/video_paths/front`` or
    ``additional_data/tactile``. An unreadable file yields ``{}`` with a warning; rejecting
    it is the validator's job.
    """
    paths: dict[str, str] = {}
    try:
        with h5py.File(h5_path, "r") as f:
            for group_name in VIDEO_PATH_GROUPS:
                group = f.get(group_name)
                if not isinstance(group, h5py.Group):
                    continue
                for name, ds in group.items():
                    if is_video_reference(ds):
                        paths[f"{group_name}/{name}"] = decode_h5_scalar(ds[()]).strip()
    except Exception as exc:
        logger.warning("  ! Could not read video paths from %s: %s", h5_path.name, exc)
    return paths
