"""Toggles and thresholds for the ``oopsie-data qa`` checks.

:data:`DEFAULT_CHECKS` is the full set. ``oopsie-data qa --config FILE`` overrides any value
from a YAML file with the same ``<category>: <check>: <parameter>`` nesting.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import yaml

# Every check, grouped by category (the YAML's top-level check sections), with its default
# parameters. The YAML overrides any value; each value is coerced to its default's type.
DEFAULT_CHECKS: dict[str, dict[str, dict]] = {
    "general": {
        "duplicate_pairs": {"enabled": False, "atol": 1.0e-9, "rtol": 0.0},
        "length_mismatch": {"enabled": True},
        "action_copies_state": {"enabled": True, "atol": 1.0e-6, "min_fraction": 0.9},
        "scale_consistency": {"enabled": False, "max_ratio": 5.0, "min_ratio": 0.2},
        "tracking_residual": {"enabled": False, "max_normalized_rms": 0.5},
        "delta_detection": {"enabled": False, "ratio_threshold": 0.3},
        "gripper_binary_present": {"enabled": True},
        "gripper_range": {"enabled": True, "atol": 0.05},
        "gripper_command_range": {"enabled": True, "max_fraction": 0.25, "min_margin": 0.05},
    },
    "joint_position": {
        "joint_names": {"enabled": True},
        "dof": {"enabled": True},
        "radians": {"enabled": True, "max_abs_rad": 6.5},
        "step_jump": {"enabled": True, "max_joint_speed_rad_s": 6.28},
        "gripper_in_joints": {"enabled": True},
        "normalized_range": {"enabled": True},
        "delta_detection": {"enabled": True, "ratio_threshold": 0.3},
        "tracking": {"enabled": True, "max_median_error_rad": 0.2},
    },
    "joint_velocity": {
        "dof": {"enabled": True},
        "speed_limit": {"enabled": True, "max_joint_speed_rad_s": 3.14},
        "normalized_range": {"enabled": True},
        "integration": {
            "enabled": True,
            "min_ratio": 0.25,
            "max_ratio": 8.0,
            "min_correlation": 0.3,
            "min_motion_rad": 0.2,
        },
        "looks_like_position": {"enabled": True, "max_position_match": 0.2},
        "smoothness": {"enabled": True, "max_sign_flip_rate": 0.3},
    },
    "cartesian_position": {
        "biarm_ordering": {"enabled": True, "max_swap_ratio": 0.5},
        "delta_detection": {"enabled": True, "ratio_threshold": 0.3},
        "pose_tracking": {
            "enabled": True,
            "max_median_position_error_m": 0.10,
            "max_median_orientation_error_deg": 20.0,
        },
        "position_units": {
            "enabled": True,
            "max_abs_position_m": 3.0,
            "max_action_state_ratio": 10.0,
            "typical_reach_min_m": 0.2,
            "typical_reach_max_m": 1.5,
        },
        "quaternion_shape": {"enabled": True, "norm_atol": 1.0e-3},
        "quaternion_sign_flip": {"enabled": True},
        "rotation_jump": {"enabled": True, "max_angular_speed_deg_s": 360.0},
        "normalized_range": {"enabled": True},
        "quaternion_scalar_order": {"enabled": False},
    },
    "cartesian_velocity": {
        "shape": {"enabled": True},
        "speed_limit": {"enabled": True, "max_linear_m_s": 2.0, "max_angular_rad_s": 6.28},
        "normalized_range": {"enabled": True},
        "integration": {
            "enabled": True,
            "min_ratio": 0.25,
            "max_ratio": 8.0,
            "min_correlation": 0.3,
            "min_translation_m": 0.05,
            "min_rotation_deg": 15.0,
        },
        "linear_angular_order": {"enabled": True, "min_improvement": 0.2},
        "looks_like_position": {"enabled": True, "max_position_match": 0.2},
        "smoothness": {"enabled": True, "max_sign_flip_rate": 0.3},
    },
    "video": {
        "resolution": {"enabled": True, "min_short_side": 360},
        "decoding": {"enabled": True, "max_frame_count_diff": 1},
        "brightness": {
            "enabled": True,
            "dark_median": 30.0,
            "bright_median": 225.0,
            "dark_frame": 15.0,
            "bright_frame": 240.0,
            "extreme_frame_fraction": 0.1,
        },
        "speed": {
            "enabled": True,
            "fps_rel_tol": 0.05,
            "duration_tol_s": 0.5,
            "duration_rel_tol": 0.05,
            "frame_interval_cv": 0.1,
        },
        "heavy_artifacts_and_desync": {
            "enabled": True,
            "moving_fraction_of_p95": 0.1,
            "static_diff": 0.05,
            "duplicate_diff": 0.01,
            "frozen_s": 1.0,
            "choppy_fraction": 0.2,
            "jump_diff": 40.0,
            "jump_ratio": 20.0,
            "glitch_diff": 20.0,
            "glitch_ratio": 4.0,
        },
        "periodic_skips": {
            "enabled": True,
            "jump_ratio": 3.0,
            "min_jumps": 6,
            "min_period_score": 0.4,
            "min_coverage": 0.6,
            "min_joint_jump_rad": 0.015,
            "min_ee_jump_cm": 1.0,
            "min_video_jump": 2.0,
            "min_confirmed_fraction": 0.4,
            "command_explains_ratio": 3.0,
        },
    },
}

# One line per check, written into every finding of the report.
CHECK_DESCRIPTIONS = {
    ("general", "duplicate_pairs"): "Two different action types hold the same values (a likely copy-paste bug)",
    ("general", "length_mismatch"): "An action and its matching robot state have different numbers of steps",
    ("general", "action_copies_state"): "Actions are exact copies of the recorded state, not real commands",
    ("general", "scale_consistency"): "Gripper position actions are on a very different scale from the state",
    ("general", "tracking_residual"): "The commanded gripper position doesn't match where the gripper ends up",
    ("general", "delta_detection"): "Gripper actions were saved as step-to-step changes instead of absolute positions",
    ("general", "gripper_binary_present"): "No gripper_binary action is recorded",
    ("general", "gripper_range"): "Binary gripper actions fall outside [0, 1] (e.g. -1/+1)",
    ("general", "gripper_command_range"): "Gripper position commands go far beyond the range the gripper state covers",
    ("joint_position", "joint_names"): "Action/state joint names are missing, don't match, or are in an odd order",
    ("joint_position", "dof"): "Number of joint values doesn't match the number of joint names",
    ("joint_position", "radians"): "Joint values aren't in radians (degrees or a -100..100 range)",
    ("joint_position", "step_jump"): "A joint moves unrealistically far between two steps",
    ("joint_position", "gripper_in_joints"): "A gripper value is mixed into the joint vector",
    ("joint_position", "normalized_range"): "Joint actions peak at exactly +-1 (normalized, not radians)",
    ("joint_position", "delta_detection"): "Joint actions were saved as step-to-step changes instead of absolute positions",
    ("joint_position", "tracking"): "The commanded joint positions don't match where the joints actually end up",
    ("joint_velocity", "dof"): "Number of joint velocities doesn't match the number of joint names",
    ("joint_velocity", "speed_limit"): "A joint is commanded faster than a real arm can move",
    ("joint_velocity", "normalized_range"): "Velocities peak at exactly +-1 (normalized, not real units)",
    ("joint_velocity", "integration"): "Velocity x step time doesn't match how far the joints actually moved",
    ("joint_velocity", "looks_like_position"): "The velocity is really a joint position saved under the wrong key",
    ("joint_velocity", "smoothness"): "Velocities flip direction from step to step",
    ("cartesian_velocity", "shape"): "Velocity isn't 6 values per arm (3 linear + 3 angular)",
    ("cartesian_velocity", "speed_limit"): "The end effector is commanded faster than a real arm can move",
    ("cartesian_velocity", "normalized_range"): "Velocities peak at exactly +-1 (normalized, not real units)",
    ("cartesian_velocity", "integration"): "Velocity x step time doesn't match how the end effector actually moved and turned",
    ("cartesian_velocity", "linear_angular_order"): "Linear and angular halves look swapped",
    ("cartesian_velocity", "looks_like_position"): "The velocity is really a position saved under the wrong key",
    ("cartesian_velocity", "smoothness"): "Velocities flip direction from step to step",
    ("cartesian_position", "quaternion_shape"): "Pose isn't [x, y, z, qx, qy, qz, qw] per arm with a unit-length quaternion",
    ("cartesian_position", "position_units"): "Positions aren't in metres (cm/mm or a far-away frame)",
    ("cartesian_position", "quaternion_sign_flip"): "The quaternion suddenly flips sign (q -> -q)",
    ("cartesian_position", "rotation_jump"): "The commanded orientation turns unrealistically fast between two steps",
    ("cartesian_position", "normalized_range"): "xyz actions peak at exactly +-1 (normalized, not metres)",
    ("cartesian_position", "pose_tracking"): "The commanded position or orientation doesn't match where the robot ends up",
    ("cartesian_position", "delta_detection"): "xyz actions were saved as step-to-step changes instead of absolute positions",
    ("cartesian_position", "biarm_ordering"): "Left and right arms look swapped in the actions",
    ("cartesian_position", "quaternion_scalar_order"): "The quaternion's scalar (w) component looks like it was written first instead of last (rough heuristic, often unreliable)",
    ("video", "resolution"): "A camera video's shorter side is below the minimum resolution",
    ("video", "decoding"): "A camera video is missing, fails to decode, or decodes to the wrong frame count",
    ("video", "brightness"): "A camera video is too dark or too bright",
    ("video", "speed"): "Video timing doesn't match control_freq or the trajectory length, or cameras disagree",
    ("video", "heavy_artifacts_and_desync"): "Frozen or duplicated frames while the robot moves, image jumps, or corrupted frames",
    ("video", "periodic_skips"): "The recording pauses at a regular period: robot position and video jump at each gap",
}


def _reject_unknown(where: str, given, known) -> None:
    # A typo'd name would otherwise silently fall back to the default (e.g. leave a check on).
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise ValueError(f"QA config: unknown {where}: {unknown}")


def build_checks(raw: dict) -> dict:
    """Merge ``raw`` over :data:`DEFAULT_CHECKS`, coercing each value to its default's type."""
    _reject_unknown("categories", raw, DEFAULT_CHECKS)
    checks: dict[str, dict] = {}
    for category, defaults in DEFAULT_CHECKS.items():
        raw_category = raw.get(category) or {}
        _reject_unknown(f"check(s) under {category!r}", raw_category, defaults)
        checks[category] = {}
        for check, params in defaults.items():
            raw_check = raw_category.get(check) or {}
            _reject_unknown(f"parameter(s) under {category}.{check}", raw_check, params)
            checks[category][check] = {
                name: type(default)(raw_check.get(name, default)) for name, default in params.items()
            }
    return checks


def load_qa_config(path: Optional[Path] = None) -> dict:
    """The check config: defaults, overridden by the YAML at ``path`` if given."""
    if path is None:
        return build_checks({})
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"QA config {path}: expected a mapping at the top level")
    return build_checks(raw)


def disable_category(config: dict, category: str) -> dict:
    """A copy of ``config`` with every check in ``category`` switched off."""
    out = copy.deepcopy(config)
    for check in out[category].values():
        check["enabled"] = False
    return out
