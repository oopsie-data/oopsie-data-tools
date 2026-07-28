"""Shared utilities for dataset-to-HDF5 conversion scripts.

A converter writes ``oopsiedata_format_v1`` directly instead of going through
:class:`~oopsie_data_tools.annotation_tool.episode_recorder.EpisodeRecorder`, so none of the
recording-time checks run. These helpers exist so the parts that are easy to get silently
wrong — the annotation layout and the image bounds — come from the same definitions the
validator uses. See ``skill/reference/conversion.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import h5py
import numpy as np

from oopsie_data_tools.annotation_tool.annotation_schema import write_annotation_attrs
from oopsie_data_tools.utils.h5 import decode_h5_scalar
from oopsie_data_tools.utils.robot_profile.robot_profile import (
    RobotProfile,
    robot_profile_to_json,
)
from oopsie_data_tools.utils.validation.episode_loader import OOPSIE_DATA_SCHEMA_V1
from oopsie_data_tools.utils.validation.episode_validator import (
    MAX_IMAGE_SIZE,
    MIN_IMAGE_SIZE,
)

#: Kept as an alias so existing converters keep working. Imported from the validator rather
#: than restated: a local copy drifted to 1080 once already, which silently downscaled every
#: video for a limit that was never 1080.
SCHEMA_VERSION = OOPSIE_DATA_SCHEMA_V1
MAX_DIM = MAX_IMAGE_SIZE


def _resize_frames(frames: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    """Resize ``(T, H, W, 3)`` frames so that ``max(H, W) <= max_dim``.

    Raises:
        ValueError: If the result would fall below the validator's minimum side, which
            would produce an episode rejected at upload rather than here.
    """
    h, w = frames.shape[1:3]
    if w <= max_dim and h <= max_dim:
        new_h, new_w = h, w
    else:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

    if min(new_w, new_h) < MIN_IMAGE_SIZE:
        raise ValueError(
            f"Frames are {w}x{h}; at most {max_dim}px on the long side that becomes "
            f"{new_w}x{new_h}, whose short side is under the {MIN_IMAGE_SIZE}px minimum "
            "the validator enforces. Crop or re-render the source video instead."
        )

    if (new_w, new_h) == (w, h):
        return frames
    return np.stack(
        [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
    )


def _parse_fps(control_freq: Any, default_fps: float = 15.0) -> float:
    """Coerce a profile's ``control_freq`` to a usable FPS, falling back on nonsense."""
    try:
        parsed = float(control_freq)
    except (TypeError, ValueError):
        return default_fps
    return parsed if parsed > 0 else default_fps


def _decode_text(value: Any) -> str:
    """Decode bytes, numpy scalars/arrays, or arbitrary values to a plain str."""
    # Thin alias: decode_h5_scalar handles every case this did, plus None and str.
    return decode_h5_scalar(value)


def write_root_attrs(
    file_handle: h5py.File,
    *,
    episode_id: str,
    language_instruction: str,
    lab_id: str,
    operator_name: str,
    robot_profile: RobotProfile,
    timestamp: float | None = None,
) -> None:
    """Write the six required root attributes, plus the optional ``timestamp``.

    All six are checked by the loader before anything else is read, and four of them are
    additionally checked for emptiness. ``lab_id`` must be the real one from registration —
    ``"your_lab_id"`` is rejected by name.
    """
    str_dtype = h5py.string_dtype(encoding="utf-8")
    file_handle.attrs["schema"] = SCHEMA_VERSION
    file_handle.attrs["episode_id"] = episode_id
    file_handle.attrs["language_instruction"] = language_instruction
    file_handle.attrs["lab_id"] = lab_id
    file_handle.attrs["operator_name"] = operator_name
    file_handle.attrs.create(
        "robot_profile", robot_profile_to_json(robot_profile), dtype=str_dtype
    )
    if timestamp is not None:
        file_handle.attrs["timestamp"] = float(timestamp)


def write_episode_annotations(
    file_handle: h5py.File,
    *,
    annotator_name: str,
    success: float,
    failure_description: str = "",
    failure_category: Sequence[str] | None = None,
    severity: str = "",
    additional_notes: str = "",
    timestamp: str = "",
) -> None:
    """Write one annotator's annotation into ``episode_annotations/<annotator_name>/``.

    The per-annotator subgroup is not optional: the loader iterates the subgroups of
    ``episode_annotations`` and reads attributes off each one, so attributes written on the
    parent group are invisible and the episode fails as unannotated. Identity fields
    (``lab_id``, ``operator_name``) are root attributes — see :func:`write_root_attrs` —
    not annotation fields.

    ``success`` is stored as the exact float given, so qualified successes survive; the
    failure taxonomy is only required when it is below 0.5.
    """
    if not 0.0 <= float(success) <= 1.0:
        raise ValueError(f"success must be in [0.0, 1.0], got {success!r}")

    categories = list(failure_category or [])
    if float(success) < 0.5:
        filled = [bool(categories), bool(str(failure_description).strip()), bool(str(severity).strip())]
        if any(filled) and not all(filled):
            raise ValueError(
                "For a failure, failure_category, failure_description and severity must be "
                "all filled or all empty; got "
                f"category={categories!r}, description={failure_description!r}, "
                f"severity={severity!r}."
            )

    group = file_handle.require_group("episode_annotations").require_group(annotator_name)
    write_annotation_attrs(
        group,
        {
            # write_annotation_attrs speaks the annotation tool's dict; the numeric success
            # it derives from this is overwritten below with the exact value.
            "binary_success": "success" if float(success) >= 0.5 else "failure",
            "timestamp": timestamp,
            "failure_description": failure_description,
            "failure_category": categories,
            "severity": severity,
            "additional_notes": additional_notes,
        },
    )
    group.attrs["success"] = float(success)


def write_actions(
    file_handle: h5py.File,
    actions: dict[str, np.ndarray],
    action_space: Sequence[str],
) -> None:
    """Write ``/actions``: real arrays for ``action_space``, ``h5py.Empty`` for the rest.

    Every canonical key must exist as a dataset. A key in ``action_space`` stored as Empty is
    rejected, and a non-empty dataset the profile does not declare is rejected too, so the
    split has to follow the profile exactly.
    """
    from oopsie_data_tools.annotation_tool.episode_recorder import VALID_ACTION_KEYS

    declared = set(action_space)
    unknown = declared - set(VALID_ACTION_KEYS)
    if unknown:
        raise ValueError(f"action_space contains unrecognized keys: {sorted(unknown)}")
    missing = declared - set(actions)
    if missing:
        raise ValueError(f"No action array supplied for declared keys: {sorted(missing)}")

    group = file_handle.create_group("actions")
    for key in sorted(VALID_ACTION_KEYS):
        if key in declared:
            group.create_dataset(key, data=np.asarray(actions[key]), dtype=np.float64)
        else:
            group.create_dataset(key, data=h5py.Empty(dtype=np.float64))


def write_video_paths(
    file_handle: h5py.File,
    video_paths: dict[str, str],
    h5_path: Path | str,
) -> None:
    """Write ``/observations/video_paths`` as paths relative to the episode file."""
    import os

    str_dtype = h5py.string_dtype(encoding="utf-8")
    episode_dir = Path(h5_path).resolve().parent
    observations = file_handle.require_group("observations")
    group = observations.require_group("video_paths")
    for cam, raw in video_paths.items():
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = episode_dir / target
        rel = os.path.relpath(target.resolve(), start=episode_dir)
        group.create_dataset(cam, data=rel.replace(os.sep, "/"), dtype=str_dtype)
