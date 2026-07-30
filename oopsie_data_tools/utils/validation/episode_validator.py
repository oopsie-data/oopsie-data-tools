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

from oopsie_data_tools.annotation_tool.annotation_schema import (
    OUTCOME_SUCCESS,
    OUTCOMES,
    SUCCESS_THRESHOLD,
    read_annotation_attrs,
    validate_annotation_vocabulary,
)
from oopsie_data_tools.utils.validation.array_validation import (
    require_finite_real_array,
    validate_cartesian_quaternions,
    validate_gripper_binary_trajectory,
)
from oopsie_data_tools.utils.validation.episode_data import EpisodeData
from oopsie_data_tools.utils.validation.errors import EpisodeValidationError

MAX_IMAGE_SIZE = 1280
MIN_IMAGE_SIZE = 180
# Bounds on episode *duration*, not step count: trajectory_length / control_freq.
MIN_EPISODE_DURATION_S = 1
MAX_EPISODE_DURATION_S = 600


def validate_episode(data: EpisodeData, strict_annotation_check: bool = False) -> None:
    """Run all semantic checks on a loaded EpisodeData.

    Raises EpisodeValidationError with a descriptive message on the first failure.
    """
    _validate_metadata(data)
    _validate_profile_consistency(data)
    _validate_trajectory_lengths(data)
    _validate_trajectory_values(data)
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
    if not np.isfinite(data.control_freq):
        raise EpisodeValidationError(
            f"control_freq must be finite, got {data.control_freq!r}"
        )
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

    # The reverse direction. The profile is the episode's documentation, so anything present
    # but undeclared has no joint names, no units and no DOF to check against, and cannot be
    # interpreted by anyone downstream. EpisodeRecorder cannot produce this — it writes
    # exactly profile.robot_state_keys — so this only ever catches hand-built or externally
    # converted files.
    _reject_undeclared(
        "observations/robot_states", data.observations, profile.robot_state_keys,
        "profile.robot_state_keys",
    )
    _reject_undeclared("actions", data.actions, profile.action_space, "profile.action_space")

    _validate_cartesian_arm_count(data)

    jp_obs = data.observations.get("joint_position")
    if jp_obs is not None:
        dof = _recorded_dof(jp_obs)
        if len(profile.robot_state_joint_names) != dof:
            raise EpisodeValidationError(
                "robot_state_joint_names count does not match observations/joint_position DOF: "
                f"the robot profile lists {len(profile.robot_state_joint_names)} joint name(s) in "
                f"robot_state_joint_names, but the recorded observations/joint_position has "
                f"{dof} DOF (last axis). Fix robot_state_joint_names in the robot "
                "profile (or the recorded joint_position) so the two counts match."
            )

    if profile.action_joint_names:
        for key in ("joint_position", "joint_velocity"):
            arr = data.actions.get(key)
            if arr is not None:
                dof = _recorded_dof(arr)
                if len(profile.action_joint_names) != dof:
                    raise EpisodeValidationError(
                        f"action_joint_names count does not match actions/{key} DOF: "
                        f"the robot profile lists {len(profile.action_joint_names)} joint name(s) in "
                        f"action_joint_names, but the recorded actions/{key} has {dof} DOF "
                        "(last axis). Fix action_joint_names in the robot profile (or the recorded "
                        "actions) so the two counts match."
                    )


def _recorded_dof(arr) -> int:
    """DOF per timestep: the last axis, or 1 for a flat ``(T,)`` trajectory."""
    return arr.shape[-1] if arr.ndim >= 2 else 1


def _reject_undeclared(
    group: str, present: dict[str, Any], declared: list[str], declared_by: str
) -> None:
    """Every dataset in ``group`` must be declared by the profile."""
    undeclared = sorted(set(present) - set(declared))
    if undeclared:
        raise EpisodeValidationError(
            f"{group} contains {len(undeclared)} key(s) the robot profile does not declare: "
            f"{undeclared}. The profile is what documents an episode, so undeclared data has "
            f"no joint names, units or expected DOF, and nothing downstream can interpret it. "
            f"Add the key(s) to {declared_by}, or stop recording them. "
            f"{declared_by}={list(declared)}"
        )


def _validate_cartesian_arm_count(data: EpisodeData) -> None:
    """A cartesian pose must carry one arm's worth of DOF per arm the profile declares.

    ``[x, y, z, qx, qy, qz, qw]`` is 7 values, so a bimanual robot records 14. Joint counts
    cannot be constrained this way — two arms need not have the same DOF, and a 7+6 pair is
    legitimate — but the end-effector pose is fixed by its representation.
    """
    expected = 14 if data.robot_profile.is_biarm else 7
    arms = "biarm" if data.robot_profile.is_biarm else "single-arm"

    for group, arrays in (("observations", data.observations), ("actions", data.actions)):
        arr = arrays.get("cartesian_position")
        if arr is None or arr.ndim < 2:
            continue
        if arr.shape[-1] != expected:
            raise EpisodeValidationError(
                f"{group}/cartesian_position has {arr.shape[-1]} DOF, but the profile "
                f"declares a {arms} robot (is_biarm={data.robot_profile.is_biarm}), which "
                f"means {expected} — [x, y, z, qx, qy, qz, qw] per arm. Either the pose is "
                f"missing an arm or is_biarm is wrong."
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


def _validate_trajectory_values(data: EpisodeData) -> None:
    """Reject unreadable numeric values before applying field-specific semantics."""
    groups = (
        ("observations/robot_states", data.observations),
        ("actions", data.actions),
    )
    for group, arrays in groups:
        for key, value in arrays.items():
            label = f"{group}/{key}"
            try:
                require_finite_real_array(value, label)
                if key == "cartesian_position":
                    validate_cartesian_quaternions(value, label)
                if key == "gripper_binary":
                    validate_gripper_binary_trajectory(
                        value,
                        is_biarm=data.robot_profile.is_biarm,
                        label=label,
                    )
            except ValueError as e:
                raise EpisodeValidationError(str(e)) from e


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


def _validate_annotations(data: EpisodeData) -> None:
    """Every annotator subgroup must have a numeric success score in [0.0, 1.0].

    Beyond that, the taxonomy fields are all optional: a partial annotation is valid, and
    a failure with no taxonomy at all is valid. What is checked is only that what *is*
    stored is readable and self-consistent — a malformed ``taxonomy`` blob, or an
    ``outcome`` that contradicts the ``success`` float, would make different readers reach
    different conclusions about the same episode.

    v1 files carry no ``outcome``, so they skip that check and stay valid unchanged.
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
        if not np.isfinite(success):
            raise EpisodeValidationError(
                f"episode_annotations/{annotator}/success must be finite: {success}"
            )
        if not (0.0 <= success <= 1.0):
            raise EpisodeValidationError(
                f"episode_annotations/{annotator}/success out of range [0.0, 1.0]: {success}"
            )

        taxonomy_raw = _annotation_attr_scalar_str(attrs.get("taxonomy", ""))
        if taxonomy_raw:
            try:
                parsed = json.loads(taxonomy_raw)
            except json.JSONDecodeError as e:
                raise EpisodeValidationError(
                    f"episode_annotations/{annotator}/taxonomy is not valid JSON: "
                    f"{taxonomy_raw!r}"
                ) from e
            if not isinstance(parsed, dict):
                raise EpisodeValidationError(
                    f"episode_annotations/{annotator}/taxonomy must be a JSON object, "
                    f"got {type(parsed).__name__}"
                )

            outcome = _annotation_attr_scalar_str(parsed.get("outcome", "")).lower()
            if outcome:
                if outcome not in OUTCOME_SUCCESS:
                    raise EpisodeValidationError(
                        f"episode_annotations/{annotator}/taxonomy has unrecognized "
                        f"outcome {outcome!r}; expected one of {OUTCOMES}"
                    )
                if (outcome == "failure") != (success < SUCCESS_THRESHOLD):
                    raise EpisodeValidationError(
                        f"episode_annotations/{annotator}: outcome {outcome!r} disagrees "
                        f"with success {success} (threshold {SUCCESS_THRESHOLD})"
                    )

            # Normalize known v1 prose to v2 slugs before applying the shared vocabulary checks.
            normalized = read_annotation_attrs(attrs)
            vocabulary_error = validate_annotation_vocabulary(
                normalized, require_outcome=False
            )
            if vocabulary_error:
                raise EpisodeValidationError(
                    f"episode_annotations/{annotator}/taxonomy {vocabulary_error}"
                )
