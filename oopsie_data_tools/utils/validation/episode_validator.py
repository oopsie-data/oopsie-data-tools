"""Semantic validation of EpisodeData.

All checks here operate on in-memory data (numpy arrays, VideoInfo structs)
with no file I/O.  This makes the same validation callable from:
  - The HDF5 validation pipeline (after episode_loader produces EpisodeData)
  - EpisodeRecorder pre-save (build EpisodeData from in-memory buffers first)
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from oopsie_data_tools.utils.validation.annotation_completeness import failure_trio_flags
from oopsie_data_tools.utils.validation.episode_data import EpisodeData
from oopsie_data_tools.utils.validation.errors import EpisodeValidationError

MAX_IMAGE_SIZE = 1280
MIN_IMAGE_SIZE = 180
# Bounds on episode *duration*, not step count: trajectory_length / control_freq. They were
# named MIN/MAX_EPISODE_LENGTH, so the error reported the raw step count against them and
# read as nonsense ("trajectory_length 20 out of range [1, 300]" for a 0.67 s episode).
MIN_EPISODE_DURATION_S = 1
MAX_EPISODE_DURATION_S = 300


def validate_episode(data: EpisodeData, strict_annotation_check: bool = False) -> None:
    """Run all semantic checks on a loaded EpisodeData.

    Raises EpisodeValidationError with a descriptive message on the first failure.
    """
    _validate_metadata(data)
    _validate_profile_consistency(data)
    _validate_trajectory_lengths(data)
    _validate_video_specs(data)
    if strict_annotation_check:
        if not data.annotations:
            raise EpisodeValidationError("Annotations dict is empty, must be provided for upload")
        _validate_annotations(data)


# ── Individual checks ──────────────────────────────────────────────────────────


def _validate_metadata(data: EpisodeData) -> None:
    if not data.language_instruction:
        raise EpisodeValidationError("language_instruction is empty")
    if not data.episode_id:
        raise EpisodeValidationError("episode_id is empty")
    if not data.lab_id:
        raise EpisodeValidationError("lab_id is empty")
    if data.lab_id == "your_lab_id":
        raise EpisodeValidationError("lab_id has not been changed from the placeholder value")
    if not data.operator_name:
        raise EpisodeValidationError("operator_name is empty")
    if not (data.control_freq > 0):
        raise EpisodeValidationError("control_freq must be > 0")
    duration_s = data.trajectory_length / data.control_freq
    if not (MIN_EPISODE_DURATION_S <= duration_s <= MAX_EPISODE_DURATION_S):
        raise EpisodeValidationError(
            f"episode duration {duration_s:.2f}s out of range "
            f"[{MIN_EPISODE_DURATION_S}, {MAX_EPISODE_DURATION_S}]s "
            f"({data.trajectory_length} steps at {data.control_freq:g} Hz)"
        )


def _validate_profile_consistency(data: EpisodeData) -> None:
    """Check that observations, actions, and videos match the embedded robot profile."""
    profile = data.robot_profile

    for key in profile.robot_state_keys:
        if key not in data.observations:
            raise EpisodeValidationError(
                f"Missing observations key required by profile: {key}. Got {list(data.observations.keys())}, required by profile.robot_state_keys={profile.robot_state_keys}"
            )

    for key in profile.action_space:
        if key not in data.actions:
            raise EpisodeValidationError(
                f"Missing actions key required by profile.action_space: {key}. Got {list(data.actions.keys())}, required by profile.action_space={profile.action_space}"
            )

    for cam in profile.camera_names:
        if cam not in data.videos:
            raise EpisodeValidationError(
                f"Missing video for camera required by profile: {cam}. Got {list(data.videos.keys())}, required by profile.camera_names={profile.camera_names}"
            )

    jp_obs = data.observations.get("joint_position")
    if jp_obs is not None and jp_obs.ndim >= 2:
        if len(profile.robot_state_joint_names) != jp_obs.shape[-1]:
            raise EpisodeValidationError(
                "robot_state_joint_names count does not match observations/joint_position DOF: "
                f"the robot profile lists {len(profile.robot_state_joint_names)} joint name(s) in "
                f"robot_state_joint_names, but the recorded observations/joint_position has "
                f"{jp_obs.shape[-1]} DOF (last axis). Fix robot_state_joint_names in the robot "
                "profile (or the recorded joint_position) so the two counts match."
            )

    if profile.action_joint_names:
        for key in ("joint_position", "joint_velocity"):
            arr = data.actions.get(key)
            if arr is not None and arr.ndim >= 2:
                if len(profile.action_joint_names) != arr.shape[-1]:
                    raise EpisodeValidationError(
                        f"action_joint_names count does not match actions/{key} DOF: "
                        f"the robot profile lists {len(profile.action_joint_names)} joint name(s) in "
                        f"action_joint_names, but the recorded actions/{key} has {arr.shape[-1]} DOF "
                        "(last axis). Fix action_joint_names in the robot profile (or the recorded "
                        "actions) so the two counts match."
                    )


def _validate_trajectory_lengths(data: EpisodeData) -> None:
    """All observation and action arrays must share the same trajectory length."""
    lengths: dict[str, int] = {}

    for key, arr in data.observations.items():
        if arr.ndim > 0:
            lengths[f"observations/{key}"] = arr.shape[0]

    for key, arr in data.actions.items():
        if arr.ndim > 0:
            lengths[f"actions/{key}"] = arr.shape[0]

    if not lengths:
        raise EpisodeValidationError("No trajectory data found in observations or actions")

    unique = set(lengths.values())
    if len(unique) != 1:
        raise EpisodeValidationError(f"Inconsistent trajectory lengths: {lengths}")

    actual_T = unique.pop()
    if actual_T != data.trajectory_length:
        raise EpisodeValidationError(
            f"trajectory_length field ({data.trajectory_length}) does not match "
            f"array shapes ({actual_T})"
        )


def _validate_video_specs(data: EpisodeData) -> None:
    """Check per-camera resolution, frame count alignment, and duration alignment."""
    if not data.videos:
        raise EpisodeValidationError("No video entries found")

    T = data.trajectory_length
    frame_tolerance = max(5, int(0.1 * T))
    expected_duration = T / data.control_freq

    frame_counts: dict[str, int] = {}
    for cam, info in data.videos.items():
        if not (info.width >= MIN_IMAGE_SIZE and info.height >= MIN_IMAGE_SIZE):
            raise EpisodeValidationError(
                f"Video too small for camera {cam}: {info.width}x{info.height} "
                f"(min {MIN_IMAGE_SIZE}px)"
            )
        if not (info.width <= MAX_IMAGE_SIZE and info.height <= MAX_IMAGE_SIZE):
            raise EpisodeValidationError(
                f"Video too large for camera {cam}: {info.width}x{info.height} "
                f"(max {MAX_IMAGE_SIZE}px)"
            )
        if not (abs(info.frame_count - T) <= frame_tolerance):
            raise EpisodeValidationError(
                f"Frame count / trajectory mismatch for camera {cam}: "
                f"frames={info.frame_count}, trajectory={T}"
            )
        duration = info.frame_count / info.fps
        if not (abs(duration - expected_duration) <= 0.5):
            raise EpisodeValidationError(
                f"Video duration / control_freq mismatch for camera {cam}: "
                f"duration={duration:.2f}s, expected={expected_duration:.2f}s"
            )
        frame_counts[cam] = info.frame_count

    if len(frame_counts) > 1:
        counts = list(frame_counts.values())
        if not (max(counts) - min(counts) <= 1):
            raise EpisodeValidationError(
                f"Inconsistent frame counts across cameras: {frame_counts}"
            )


def _annotation_attr_scalar_str(val: Any) -> str:
    """Normalize HDF5 attr scalars (bytes, numpy, str) to a trimmed string."""
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").strip()
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, np.generic):
        return _annotation_attr_scalar_str(val.item())
    if isinstance(val, np.ndarray) and val.shape == ():
        return _annotation_attr_scalar_str(val.item())
    return str(val).strip()


def _failure_trio_fill_flags(attrs: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Read the failure trio out of HDF5 annotation attrs and report what is filled.

    Extraction is specific to the stored layout — category and severity live inside the
    ``taxonomy`` JSON blob — but the "is this filled?" rule is shared with the annotation
    server via :mod:`annotation_completeness`, so the two cannot drift apart.
    """
    tax: dict[str, Any] = {}
    tax_raw = _annotation_attr_scalar_str(attrs.get("taxonomy", ""))
    if tax_raw:
        try:
            parsed = json.loads(tax_raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            tax = parsed

    return failure_trio_flags(
        failure_category=tax.get("failure_category"),
        failure_description=attrs.get("failure_description", ""),
        severity=tax.get("severity"),
    )


def _validate_annotations(data: EpisodeData) -> None:
    """Every annotator subgroup must have a numeric success score in [0.0, 1.0].

    The failure taxonomy trio (category/description/severity) is all-or-nothing, but
    only for *failures* — a qualified success may record severity/notes on its own (#29).
    """
    if not data.annotations:
        raise EpisodeValidationError("annotations dict is empty")

    for annotator, attrs in data.annotations.items():
        if "success" not in attrs:
            raise EpisodeValidationError(
                f"episode_annotations/{annotator} is missing 'success' — "
                "episode has not been fully annotated yet"
            )
        try:
            success = float(attrs["success"])
        except (TypeError, ValueError) as e:
            raise EpisodeValidationError(
                f"episode_annotations/{annotator}/success is not numeric: {attrs['success']!r}"
            ) from e
        if np.isnan(success):
            raise EpisodeValidationError(
                f"episode_annotations/{annotator}/success is NaN — "
                "episode has not been fully annotated yet"
            )
        if not (0.0 <= success <= 1.0):
            raise EpisodeValidationError(
                f"episode_annotations/{annotator}/success out of range [0.0, 1.0]: {success}"
            )

        # The failure taxonomy trio is all-or-nothing, but only for failures;
        # successes (including qualified successes) are exempt.
        if success < 0.5:
            cat_ok, desc_ok, sev_ok = _failure_trio_fill_flags(attrs)
            filled = int(cat_ok) + int(desc_ok) + int(sev_ok)
            if filled not in (0, 3):
                raise EpisodeValidationError(
                    f"episode_annotations/{annotator}: failure_category (taxonomy), "
                    f"failure_description, and severity must be all filled or all empty "
                    f"(found {filled} of 3 filled). Either complete all three or leave all three empty."
                )
