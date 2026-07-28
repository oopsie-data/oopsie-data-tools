"""Load episode HDF5 files into the schema-agnostic EpisodeData representation.

This module owns all file I/O:
  - HDF5 readability checks
  - Schema detection
  - Video file existence and MP4 metadata extraction

It does NOT perform semantic validation — that is episode_validator's job.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import h5py
import numpy as np

from oopsie_data_tools.utils.robot_profile.robot_profile import robot_profile_from_json
from oopsie_data_tools.utils.validation.episode_data import EpisodeData, VideoInfo
from oopsie_data_tools.utils.validation.errors import EpisodeValidationError

OOPSIE_DATA_SCHEMA_V1 = "oopsiedata_format_v1"

_OOPSIE_V1_REQUIRED_ROOT_ATTRS = (
    "schema",
    "episode_id",
    "language_instruction",
    "lab_id",
    "operator_name",
    "robot_profile",
)


# ── HDF5 scalar helpers ────────────────────────────────────────────────────────


def _decode_h5_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, np.generic):
        return _decode_h5_scalar(value.item())
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_h5_scalar(value.item())
    return str(value)


def _read_string_dataset(ds: h5py.Dataset) -> str:
    return _decode_h5_scalar(ds[()]).strip()


# ── Video loading ──────────────────────────────────────────────────────────────


def load_video_info(mp4_path: str) -> VideoInfo:
    """Open an MP4 and extract frame metadata. Raises EpisodeValidationError on any failure."""
    if not os.path.exists(mp4_path):
        raise EpisodeValidationError(f"Video file does not exist: {mp4_path}")
    if not os.path.isfile(mp4_path):
        raise EpisodeValidationError(f"Video path is not a file: {mp4_path}")

    cap = cv2.VideoCapture(mp4_path)
    try:
        if not cap.isOpened():
            raise EpisodeValidationError(f"Could not open video: {mp4_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not (width > 0 and height > 0):
            raise EpisodeValidationError(f"Invalid dimensions ({width}x{height}): {mp4_path}")
        if not (fps > 0):
            raise EpisodeValidationError(f"Invalid FPS ({fps}): {mp4_path}")
        if not (frame_count > 0):
            raise EpisodeValidationError(f"Invalid frame count ({frame_count}): {mp4_path}")
        return VideoInfo(frame_count=frame_count, fps=fps, width=width, height=height)
    finally:
        cap.release()


def _resolve_video_path(rel: str, h5_dir: str, label: str) -> str:
    if not rel:
        raise EpisodeValidationError(f"Empty video path for {label}")
    return os.path.normpath(os.path.join(h5_dir, rel))


# ── Schema-specific loaders ────────────────────────────────────────────────────


def _load_oopsie_v1(f: h5py.File, h5_dir: str) -> EpisodeData:
    for attr in _OOPSIE_V1_REQUIRED_ROOT_ATTRS:
        if attr not in f.attrs:
            raise EpisodeValidationError(f"Missing root attr: {attr}")

    try:
        profile = robot_profile_from_json(_decode_h5_scalar(f.attrs["robot_profile"]))
    except ValueError as e:
        raise EpisodeValidationError(f"Invalid robot_profile JSON: {e}") from e

    if "observations" not in f:
        raise EpisodeValidationError("Missing group: observations")
    if "robot_states" not in f["observations"]:
        raise EpisodeValidationError("Missing group: observations/robot_states")
    if "actions" not in f:
        raise EpisodeValidationError("Missing group: actions")

    rs = f["observations/robot_states"]
    for key in profile.robot_state_keys:
        if key not in rs:
            raise EpisodeValidationError(f"Missing observations/robot_states/{key}")
    observations = {k: rs[k][()] for k in rs.keys()}

    action_group = f["actions"]
    for key in profile.action_space:
        if key not in action_group:
            raise EpisodeValidationError(f"Missing actions/{key}")
        ds = action_group[key]
        if ds.shape is None:
            raise EpisodeValidationError(
                f"actions/{key} is in profile.action_space but stored as h5py.Empty"
            )
    # Load only non-empty action datasets.
    actions: dict[str, np.ndarray] = {
        k: action_group[k][()] for k in action_group.keys() if action_group[k].shape is not None
    }

    if "video_paths" not in f["observations"]:
        raise EpisodeValidationError("Missing group: observations/video_paths")
    vp_group = f["observations/video_paths"]
    for cam in profile.camera_names:
        if cam not in vp_group:
            raise EpisodeValidationError(f"Missing observations/video_paths/{cam}")
    videos: dict[str, VideoInfo] = {}
    for cam in vp_group.keys():
        rel = _read_string_dataset(vp_group[cam])
        abs_path = _resolve_video_path(rel, h5_dir, f"camera {cam}")
        videos[cam] = load_video_info(abs_path)

    annotations = _load_annotations_oopsie_v1(f)

    trajectory_length = next(iter(observations.values())).shape[0]

    return EpisodeData(
        robot_profile=profile,
        language_instruction=_decode_h5_scalar(f.attrs["language_instruction"]),
        episode_id=_decode_h5_scalar(f.attrs["episode_id"]),
        lab_id=_decode_h5_scalar(f.attrs["lab_id"]),
        operator_name=_decode_h5_scalar(f.attrs["operator_name"]),
        trajectory_length=trajectory_length,
        control_freq=float(profile.control_freq),
        observations=observations,
        actions=actions,
        videos=videos,
        annotations=annotations,
    )


def _load_annotations_oopsie_v1(f: h5py.File) -> dict[str, dict[str, Any]] | None:
    if "episode_annotations" not in f:
        return None
    ea = f["episode_annotations"]
    # Everything in this layer reports failure as EpisodeValidationError. A file storing
    # episode_annotations as a dataset used to reach .keys() and raise AttributeError,
    # which escapes that contract and surfaces as an "unexpected error" with no
    # indication of what is actually wrong with the file.
    if not isinstance(ea, h5py.Group):
        raise EpisodeValidationError(
            "episode_annotations must be a group of per-annotator subgroups, got "
            f"{type(ea).__name__}"
        )
    annotations: dict[str, dict[str, Any]] = {}
    for annotator in ea.keys():
        subgroup = ea[annotator]
        if not isinstance(subgroup, h5py.Group):
            raise EpisodeValidationError(
                f"episode_annotations/{annotator} must be a group, got {type(subgroup).__name__}"
            )
        annotations[annotator] = {k: subgroup.attrs[k] for k in subgroup.attrs}
    return annotations


def load_episode_from_h5(h5_path: str) -> EpisodeData:
    """Load an episode HDF5 file into a schema-agnostic EpisodeData.

    Raises EpisodeValidationError with a descriptive message if the file is unreadable,
    structurally invalid, or references missing/corrupt video files.
    """
    resolved = os.path.abspath(os.path.normpath(h5_path))
    if not os.path.exists(resolved):
        raise EpisodeValidationError(f"H5 file does not exist: {resolved}")
    if not os.path.isfile(resolved):
        raise EpisodeValidationError(f"H5 path is not a file: {resolved}")

    try:
        f = h5py.File(resolved, "r")
    except Exception as e:
        raise EpisodeValidationError(f"H5 file is not readable: {resolved}. Error: {e}") from e

    h5_dir = os.path.dirname(resolved)
    try:
        schema = _decode_h5_scalar(f.attrs.get("schema", ""))
        if schema == OOPSIE_DATA_SCHEMA_V1:
            return _load_oopsie_v1(f, h5_dir)
        else:
            raise EpisodeValidationError(
                f"Unsupported or missing schema: '{schema}' in file: {resolved}"
            )
    finally:
        f.close()
