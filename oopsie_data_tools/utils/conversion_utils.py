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

from oopsie_data_tools.annotation_tool.annotation_schema import (
    OUTCOME_SUCCESS,
    OUTCOMES,
    SUCCESS_THRESHOLD,
    success_to_outcome,
    write_annotation_attrs,
)
from oopsie_data_tools.utils.h5 import decode_h5_scalar
from oopsie_data_tools.utils.robot_profile.robot_profile import (
    RobotProfile,
    robot_profile_to_json,
)
from oopsie_data_tools.utils.robot_profile.rotation_utils import (
    ActionQuatConversion,
    RotOption,
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


def resize_frames(frames: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
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


#: Kept so converters written against the older private name keep importing.
_resize_frames = resize_frames


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
    outcome: str | None = None,
    episode_description: str = "",
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

    ``success`` is stored as the exact float given, so qualified successes survive; it is
    authoritative for magnitude, and ``outcome`` for which of the four branches this is.
    Omit ``outcome`` and it is derived from the float, which can only ever yield the coarse
    ``success``/``failure`` — pass it explicitly to record a qualified success.

    Every taxonomy field is optional: a partial annotation is valid.
    """
    if not 0.0 <= float(success) <= 1.0:
        raise ValueError(f"success must be in [0.0, 1.0], got {success!r}")

    resolved_outcome = (outcome or "").strip().lower() or success_to_outcome(success)
    if resolved_outcome not in OUTCOME_SUCCESS:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    # A float and a slug that disagree would make every downstream reader pick a different
    # answer depending on which one it happens to consult.
    if (resolved_outcome == "failure") != (float(success) < SUCCESS_THRESHOLD):
        raise ValueError(
            f"outcome={resolved_outcome!r} disagrees with success={success!r} "
            f"(threshold {SUCCESS_THRESHOLD})"
        )

    group = file_handle.require_group("episode_annotations").require_group(annotator_name)
    write_annotation_attrs(
        group,
        {
            "outcome": resolved_outcome,
            "timestamp": timestamp,
            "episode_description": episode_description,
            "failure_category": list(failure_category or []),
            "severity": severity,
            "additional_notes": additional_notes,
        },
    )
    # write_annotation_attrs derives a 1.0/0.0 success from the outcome; restore the exact
    # value so a fractional success survives.
    group.attrs["success"] = float(success)


def to_quaternion_poses(
    poses: np.ndarray,
    source_format: str,
    *,
    is_biarm: bool = False,
) -> np.ndarray:
    """Convert ``(T, D)`` cartesian poses to ``(T, 7)`` — or ``(T, 14)`` when biarm.

    ``source_format`` is a profile orientation string (``"euler_xyz"``, ``"rotvec"``,
    ``"quat"``, …). The recorder applies this conversion in ``record_step``; a converter
    writing HDF5 directly bypasses it, so it has to do the same thing here or write a pose
    the validator rejects on width.
    """
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected (T, D) poses, got shape {array.shape}")

    convert = ActionQuatConversion(RotOption.from_string(source_format), is_biarm=is_biarm)
    converted = np.stack([convert.convert_position(row) for row in array]).astype(np.float64)

    expected = 14 if is_biarm else 7
    if converted.shape[-1] != expected:
        raise ValueError(
            f"{source_format!r} poses of width {array.shape[-1]} converted to width "
            f"{converted.shape[-1]}, but a {'biarm' if is_biarm else 'single-arm'} pose "
            f"must be {expected}. Check is_biarm and the source layout."
        )
    return converted


def write_robot_states(
    file_handle: h5py.File,
    robot_states: dict[str, np.ndarray],
    robot_state_keys: Sequence[str],
) -> None:
    """Write ``/observations/robot_states``, one ``(T, D)`` float64 dataset per key.

    The keys must equal ``profile.robot_state_keys`` exactly: a declared key that is absent
    and an undeclared key that is present are both rejected by the validator, so neither is
    allowed to slip through here.
    """
    declared = set(robot_state_keys)
    supplied = set(robot_states)
    if declared != supplied:
        missing = sorted(declared - supplied)
        extra = sorted(supplied - declared)
        raise ValueError(
            "robot_states must match profile.robot_state_keys exactly "
            f"(missing: {missing}, undeclared: {extra})"
        )

    observations = file_handle.require_group("observations")
    group = observations.require_group("robot_states")
    for key in robot_state_keys:
        array = np.asarray(robot_states[key], dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        group.create_dataset(key, data=array, dtype=np.float64)


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
