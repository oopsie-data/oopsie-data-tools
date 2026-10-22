"""Toggles and thresholds for the ``oopsie-data qa`` checks.

:data:`DEFAULT_CHECKS` is the set the Oopsie Data team runs on submissions. ``oopsie-data qa
--config FILE`` overrides any value from a YAML file with the same
``<category>: <check>: <parameter>`` nesting. A category also takes ``enabled: false`` to
switch off all its checks, and ``video`` takes the settings in :data:`CATEGORY_SETTINGS`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import yaml

# Every check, grouped by category, with its default parameters. A YAML override's values are
# coerced to the default's type.
DEFAULT_CHECKS: dict[str, dict[str, dict]] = {
    "general": {
        "duplicate_pairs": {"enabled": True, "atol": 1e-09, "rtol": 0.0},
        "length_mismatch": {"enabled": True},
        "action_chunk_shape": {"enabled": True},
        "constant_action": {"enabled": True, "atol": 1e-09},
        "command_rate": {"enabled": True, "atol": 1e-09, "min_holds": 10, "min_fraction": 0.8},
        "required_state_keys": {"enabled": True},
        "placeholder_metadata": {
            "enabled": True,
            "placeholder_values": ["unknown", "none", "null", "n/a", "na", "tbd", "todo", "test", "default", "placeholder", "example", "xxx", "-", "?", "robot", "gripper", "policy"],
            "template_prefixes": ["your_", "example_", "my_"],
            "operator_placeholder_values": ["operator", "pending_operator", "test_operator", "user", "admin", "human", "teleoperator"],
            "operator_non_person_words": ["policy", "eval", "autoeval", "orchestrator", "script", "bot", "lab", "pipeline", "system", "oopsie"],
        },
        "action_copies_state": {"enabled": True, "atol": 1e-06, "min_fraction": 0.9},
        "scale_consistency": {"enabled": False, "max_ratio": 5.0, "min_ratio": 0.2},
        "tracking_residual": {"enabled": False, "max_normalized_rms": 0.5},
        "delta_detection": {"enabled": False, "ratio_threshold": 0.3},
        "gripper_binary_present": {"enabled": True},
        "gripper_range": {"enabled": True, "atol": 0.05},
        "gripper_command_range": {"enabled": True, "max_fraction": 0.25, "min_margin": 0.05},
    },
    "joint_position": {
        "delta_detection": {"enabled": True, "ratio_threshold": 0.3},
        "tracking": {"enabled": True, "max_median_error_rad": 0.2},
        "joint_names": {"enabled": True},
        "dof": {"enabled": True},
        "radians": {"enabled": True, "max_abs_rad": 6.5},
        "step_jump": {"enabled": True, "max_joint_speed_rad_s": 20.0},
        "gripper_in_joints": {"enabled": True},
        "normalized_range": {"enabled": True},
    },
    "joint_velocity": {
        "dof": {"enabled": True},
        "tracking": {
            "enabled": True,
            "min_ratio": 0.25,
            "max_ratio": 8.0,
            "min_correlation": 0.3,
            "min_motion_rad": 0.2,
        },
        "looks_like_position": {"enabled": True, "max_position_match": 0.2},
        "normalized_range": {"enabled": True},
        "speed_limit": {"enabled": True, "max_joint_speed_rad_s": 3.14},
        "smoothness": {"enabled": True, "max_sign_flip_rate": 0.3},
    },
    "cartesian_position": {
        "biarm_ordering": {"enabled": True, "max_swap_ratio": 0.5},
        "delta_detection": {"enabled": True, "ratio_threshold": 0.3},
        "pose_tracking": {
            "enabled": True,
            "max_median_position_error_m": 0.1,
            "max_median_orientation_error_deg": 20.0,
        },
        "position_units": {
            "enabled": True,
            "max_abs_position_m": 3.0,
            "max_action_state_ratio": 10.0,
            "typical_reach_min_m": 0.2,
            "typical_reach_max_m": 1.5,
        },
        "quaternion_shape": {"enabled": True, "norm_atol": 0.001},
        "quaternion_sign_flip": {"enabled": False},
        "rotation_jump": {"enabled": True, "max_angular_speed_deg_s": 360.0},
        "normalized_range": {"enabled": True},
        "quaternion_scalar_order": {"enabled": False},
    },
    "cartesian_velocity": {
        "shape": {"enabled": True},
        "speed_limit": {"enabled": True, "max_linear_m_s": 2.0, "max_angular_rad_s": 6.28},
        "normalized_range": {"enabled": True},
        "tracking": {
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
        "decoding": {"enabled": True, "max_frame_count_diff": 1},
        "corrupted_frames": {"enabled": True, "glitch_diff": 20.0, "glitch_ratio": 4.0},
        "state_jump": {"enabled": True, "jump_ratio": 10.0, "min_fraction_of_p95": 1.0},
        "video_jump": {"enabled": True, "jump_diff": 40.0, "jump_ratio": 20.0, "state_jump_ratio": 5.0},
        "recording_gap": {"enabled": True, "video_jump_ratio": 5.0},
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
        "choppy": {"enabled": True, "duplicate_diff": 0.01, "choppy_fraction": 0.2},
        "frozen": {"enabled": True, "static_diff": 0.05, "frozen_s": 1.0},
        "resolution": {"enabled": False, "min_short_side": 360},
        "brightness": {
            "enabled": False,
            "dark_median": 30.0,
            "bright_median": 225.0,
            "dark_frame": 15.0,
            "bright_frame": 240.0,
            "extreme_frame_fraction": 0.1,
        },
        "fps": {"enabled": True, "fps_rel_tol": 0.05},
        "duration": {"enabled": True, "duration_tol_s": 0.5, "duration_rel_tol": 0.05},
        "camera_duration_mismatch": {"enabled": True, "duration_tol_s": 0.5},
    },
}

# Category-level settings that are not checks, returned as ``config["<category>_settings"]``.
CATEGORY_SETTINGS: dict[str, dict] = {
    "video": {"ignore_cameras": [], "moving_fraction_of_p95": 0.1},
}

CONFIDENCE_LEVELS = ("high_confidence", "lower_confidence")
SUGGESTED_ACTIONS = ("fix_and_resubmit", "confirm_and_discuss", "informational")

# Per check, as (confidence, suggested action), written into every finding of the report.
#   confidence: "high_confidence" - a flagged episode almost certainly has the issue;
#     "lower_confidence" - a heuristic that can flag fine episodes or miss real issues.
#   suggested action: "fix_and_resubmit" - correct the data and re-record or reconvert;
#     "confirm_and_discuss" - check whether the issue is real and tell the Oopsie Data team;
#     "informational" - not acted on by the team.
FEEDBACK: dict[str, tuple[str, str]] = {
    "general.duplicate_pairs": ("high_confidence", "fix_and_resubmit"),
    "general.length_mismatch": ("high_confidence", "fix_and_resubmit"),
    "general.action_chunk_shape": ("high_confidence", "fix_and_resubmit"),
    "general.constant_action": ("high_confidence", "confirm_and_discuss"),
    "general.command_rate": ("lower_confidence", "confirm_and_discuss"),
    "general.required_state_keys": ("high_confidence", "fix_and_resubmit"),
    "general.placeholder_metadata": ("high_confidence", "fix_and_resubmit"),
    "general.action_copies_state": ("lower_confidence", "confirm_and_discuss"),
    "general.scale_consistency": ("lower_confidence", "informational"),
    "general.tracking_residual": ("lower_confidence", "informational"),
    "general.delta_detection": ("lower_confidence", "informational"),
    "general.gripper_binary_present": ("high_confidence", "fix_and_resubmit"),
    "general.gripper_range": ("high_confidence", "fix_and_resubmit"),
    "general.gripper_command_range": ("lower_confidence", "confirm_and_discuss"),
    "joint_position.delta_detection": ("high_confidence", "fix_and_resubmit"),
    "joint_position.tracking": ("lower_confidence", "confirm_and_discuss"),
    "joint_position.joint_names": ("lower_confidence", "confirm_and_discuss"),
    "joint_position.dof": ("high_confidence", "fix_and_resubmit"),
    "joint_position.radians": ("high_confidence", "fix_and_resubmit"),
    "joint_position.step_jump": ("lower_confidence", "confirm_and_discuss"),
    "joint_position.gripper_in_joints": ("high_confidence", "fix_and_resubmit"),
    "joint_position.normalized_range": ("lower_confidence", "confirm_and_discuss"),
    "joint_velocity.dof": ("high_confidence", "fix_and_resubmit"),
    "joint_velocity.tracking": ("lower_confidence", "confirm_and_discuss"),
    "joint_velocity.looks_like_position": ("high_confidence", "fix_and_resubmit"),
    "joint_velocity.normalized_range": ("lower_confidence", "confirm_and_discuss"),
    "joint_velocity.speed_limit": ("lower_confidence", "confirm_and_discuss"),
    "joint_velocity.smoothness": ("lower_confidence", "informational"),
    "cartesian_position.biarm_ordering": ("lower_confidence", "confirm_and_discuss"),
    "cartesian_position.delta_detection": ("high_confidence", "fix_and_resubmit"),
    "cartesian_position.pose_tracking": ("lower_confidence", "confirm_and_discuss"),
    "cartesian_position.position_units": ("high_confidence", "fix_and_resubmit"),
    "cartesian_position.quaternion_shape": ("high_confidence", "fix_and_resubmit"),
    "cartesian_position.quaternion_sign_flip": ("lower_confidence", "informational"),
    "cartesian_position.rotation_jump": ("lower_confidence", "confirm_and_discuss"),
    "cartesian_position.normalized_range": ("lower_confidence", "confirm_and_discuss"),
    "cartesian_position.quaternion_scalar_order": ("lower_confidence", "informational"),
    "cartesian_velocity.shape": ("high_confidence", "fix_and_resubmit"),
    "cartesian_velocity.speed_limit": ("lower_confidence", "confirm_and_discuss"),
    "cartesian_velocity.normalized_range": ("lower_confidence", "confirm_and_discuss"),
    "cartesian_velocity.tracking": ("lower_confidence", "confirm_and_discuss"),
    "cartesian_velocity.linear_angular_order": ("lower_confidence", "confirm_and_discuss"),
    "cartesian_velocity.looks_like_position": ("high_confidence", "fix_and_resubmit"),
    "cartesian_velocity.smoothness": ("lower_confidence", "informational"),
    "video.decoding": ("high_confidence", "fix_and_resubmit"),
    "video.corrupted_frames": ("lower_confidence", "confirm_and_discuss"),
    "video.state_jump": ("lower_confidence", "confirm_and_discuss"),
    "video.video_jump": ("lower_confidence", "confirm_and_discuss"),
    "video.recording_gap": ("lower_confidence", "confirm_and_discuss"),
    "video.periodic_skips": ("lower_confidence", "confirm_and_discuss"),
    "video.choppy": ("lower_confidence", "confirm_and_discuss"),
    "video.frozen": ("lower_confidence", "confirm_and_discuss"),
    "video.resolution": ("lower_confidence", "informational"),
    "video.brightness": ("lower_confidence", "informational"),
    "video.fps": ("high_confidence", "confirm_and_discuss"),
    "video.duration": ("high_confidence", "confirm_and_discuss"),
    "video.camera_duration_mismatch": ("high_confidence", "confirm_and_discuss"),
}

# One line per check, written into every finding of the report.
CHECK_DESCRIPTIONS = {
    ("general", "duplicate_pairs"): "Two different action types hold the same values (a likely copy-paste bug)",
    ("general", "length_mismatch"): "An action and its matching robot state have different numbers of steps",
    ("general", "action_chunk_shape"): "An arm action step holds a policy's whole action chunk instead of the one action executed",
    ("general", "constant_action"): "An arm action never changes over the whole episode (stuck policy or placeholder actions)",
    ("general", "command_rate"): "Each arm action is held for the same N steps: commands arrive at control_freq / N, not the declared control_freq",
    ("general", "required_state_keys"): "A state the action space needs is missing (gripper_position always; joint/cartesian position for joint/cartesian actions)",
    ("general", "placeholder_metadata"): "Operator, policy, robot or gripper name is blank or a template placeholder",
    ("general", "action_copies_state"): "Actions are exact copies of the recorded state, not real commands",
    ("general", "scale_consistency"): "Gripper position actions are on a very different scale from the state",
    ("general", "tracking_residual"): "The commanded gripper position doesn't match where the gripper ends up",
    ("general", "delta_detection"): "Gripper actions were saved as step-to-step changes instead of absolute positions",
    ("general", "gripper_binary_present"): "No gripper_binary action is recorded",
    ("general", "gripper_range"): "Binary gripper actions fall outside [0, 1] (e.g. -1/+1)",
    ("general", "gripper_command_range"): "Gripper position commands go far beyond the range the gripper state covers",
    ("joint_position", "delta_detection"): "Joint actions were saved as step-to-step changes instead of absolute positions",
    ("joint_position", "tracking"): "The commanded joint positions don't match where the joints actually end up",
    ("joint_position", "joint_names"): "Action/state joint names are missing, don't match, or are in an odd order",
    ("joint_position", "dof"): "Number of joint values doesn't match the number of joint names",
    ("joint_position", "radians"): "Joint values aren't in radians (degrees or a -100..100 range)",
    ("joint_position", "step_jump"): "A joint moves unrealistically far between two steps",
    ("joint_position", "gripper_in_joints"): "A gripper value is mixed into the joint vector",
    ("joint_position", "normalized_range"): "Joint actions peak at exactly +-1 (normalized, not radians)",
    ("joint_velocity", "dof"): "Number of joint velocities doesn't match the number of joint names",
    ("joint_velocity", "tracking"): "Velocity x step time doesn't match how far the joints actually moved",
    ("joint_velocity", "looks_like_position"): "The velocity is really a joint position saved under the wrong key",
    ("joint_velocity", "normalized_range"): "Velocities peak at exactly +-1 (normalized, not real units)",
    ("joint_velocity", "speed_limit"): "A joint is commanded faster than a real arm can move",
    ("joint_velocity", "smoothness"): "Velocities flip direction from step to step",
    ("cartesian_position", "biarm_ordering"): "Left and right arms look swapped in the actions",
    ("cartesian_position", "delta_detection"): "xyz actions were saved as step-to-step changes instead of absolute positions",
    ("cartesian_position", "pose_tracking"): "The commanded position or orientation doesn't match where the robot ends up",
    ("cartesian_position", "position_units"): "Positions aren't in metres (cm/mm or a far-away frame)",
    ("cartesian_position", "quaternion_shape"): "Pose isn't [x, y, z, qx, qy, qz, qw] per arm with a unit-length quaternion",
    ("cartesian_position", "quaternion_sign_flip"): "The quaternion suddenly flips sign (q -> -q)",
    ("cartesian_position", "rotation_jump"): "The commanded orientation turns unrealistically fast between two steps",
    ("cartesian_position", "normalized_range"): "xyz actions peak at exactly +-1 (normalized, not metres)",
    ("cartesian_position", "quaternion_scalar_order"): "The quaternion's scalar (w) component looks like it was written first instead of last (rough heuristic, often unreliable)",
    ("cartesian_velocity", "shape"): "Velocity isn't 6 values per arm (3 linear + 3 angular)",
    ("cartesian_velocity", "speed_limit"): "The end effector is commanded faster than a real arm can move",
    ("cartesian_velocity", "normalized_range"): "Velocities peak at exactly +-1 (normalized, not real units)",
    ("cartesian_velocity", "tracking"): "Velocity x step time doesn't match how the end effector actually moved and turned",
    ("cartesian_velocity", "linear_angular_order"): "Linear and angular halves look swapped",
    ("cartesian_velocity", "looks_like_position"): "The velocity is really a position saved under the wrong key",
    ("cartesian_velocity", "smoothness"): "Velocities flip direction from step to step",
    ("video", "decoding"): "A camera video is missing, fails to decode, or decodes to the wrong frame count",
    ("video", "corrupted_frames"): "A single frame is unlike both its neighbours (decoding or recording glitch)",
    ("video", "state_jump"): "Robot state jumps in one step while the video doesn't: a state glitch, or state and video out of step",
    ("video", "video_jump"): "The image changes suddenly while the robot state changes smoothly (cut, dropped frames, or video out of step)",
    ("video", "recording_gap"): "Robot state and video both jump at the same moment: time looks missing from the recording",
    ("video", "periodic_skips"): "The recording skips time at a regular period (robot state jumps throughout the motion)",
    ("video", "choppy"): "The video repeats frames while the robot moves",
    ("video", "frozen"): "The video stops changing for a while as the robot moves",
    ("video", "resolution"): "A camera video's shorter side is below the minimum resolution",
    ("video", "brightness"): "A camera video is too dark or too bright",
    ("video", "fps"): "The video's fps header doesn't match control_freq (playback speed; frames may still align)",
    ("video", "duration"): "The video is longer or shorter than the trajectory: frame count doesn't match the number of steps",
    ("video", "camera_duration_mismatch"): "The cameras of one episode have different durations",
}


def _reject_unknown(where: str, given, known) -> None:
    # A typo'd name would otherwise silently fall back to the default (e.g. leave a check on).
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise ValueError(f"QA config: unknown {where}: {unknown}")


def build_checks(raw: dict) -> dict:
    """Merge ``raw`` over :data:`DEFAULT_CHECKS` and :data:`CATEGORY_SETTINGS`."""
    _reject_unknown("categories", raw, DEFAULT_CHECKS)
    config: dict[str, dict] = {}
    for category, defaults in DEFAULT_CHECKS.items():
        raw_category = dict(raw.get(category) or {})
        category_enabled = bool(raw_category.pop("enabled", True))
        if category in CATEGORY_SETTINGS:
            config[f"{category}_settings"] = {
                name: type(default)(raw_category.pop(name, default) or default)
                for name, default in CATEGORY_SETTINGS[category].items()
            }
        _reject_unknown(f"check(s) under {category!r}", raw_category, defaults)
        config[category] = {}
        for check, params in defaults.items():
            raw_check = raw_category.get(check) or {}
            _reject_unknown(f"parameter(s) under {category}.{check}", raw_check, params)
            config[category][check] = {
                name: type(default)(raw_check.get(name, default)) for name, default in params.items()
            }
            if not category_enabled:
                config[category][check]["enabled"] = False
    return config


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
