"""Load robot / lab profiles for annotation and episode recording."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from oopsie_data_tools.utils.robot_profile.rotation_utils import RotOption

ACTION_SPACE_SET_1 = {
    "joint_position",
    "joint_velocity",
    "cartesian_position",
    "cartesian_velocity",
}

ACTION_SPACE_SET_2 = {
    "gripper_position",
    "gripper_velocity",
    "gripper_binary",
}

ACTION_SPACE_SET_3 = {
    "base_velocity",
    "base_position",
}

REQUIRED_KEYS = frozenset(
    {
        "policy_name",
        "robot_name",
        "gripper_name",
        "control_freq",
        "is_biarm",
        "uses_mobile_base",
        "camera_names",
        "robot_state_keys",
        # Read unconditionally below, so list it here to get the friendly "missing keys"
        # error instead of a bare KeyError.
        "robot_state_joint_names",
        "action_space",
    }
)

REQUIRED_ROBOT_STATE_KEYS = frozenset(
    {
        "gripper_position",
    }
)

ARM_ACTION_REQUIRED_ROBOT_STATE_KEY = {
    "joint_position": "joint_position",
    "joint_velocity": "joint_position",
    "cartesian_position": "cartesian_position",
    "cartesian_velocity": "cartesian_position",
}


@dataclasses.dataclass(frozen=True)
class RobotProfile:
    """Robot / dataset identity and recording semantics (paths, joints, cameras).

    Options specific to :class:`WebRolloutAnnotator` (browser server port, blocking
    until annotation, resuming a session directory) are *not* part of this profile;
    pass those separately when constructing the annotator, e.g. via
    :meth:`WebRolloutAnnotator.from_robot_profile`.
    """

    # typing.List/Dict/Optional rather than the PEP 585/604 spellings used elsewhere in
    # this package. Those are fine in annotations that are never evaluated, but a dataclass
    # is exactly the thing something later introspects — tyro, pydantic, or a plain
    # get_type_hints — and resolving them raises TypeError on the declared Python 3.8 floor.
    # The tyro examples hit precisely this. Revisit when the floor moves past 3.9.
    policy_name: str
    robot_name: str
    is_biarm: bool
    uses_mobile_base: bool
    gripper_name: str
    control_freq: int
    camera_names: List[str]
    robot_state_keys: List[str]
    robot_state_joint_names: List[str]
    action_space: List[str]
    action_joint_names: Optional[List[str]] = None
    orientation_representation: Optional[str] = None
    robot_state_orientation_representation: Optional[str] = None
    controller: Optional[str] = None
    gains: Optional[Dict[str, Any]] = None
    intrinsic_calibration_matrix: Optional[Dict[str, Any]] = None
    extrinsic_calibration_matrix: Optional[Dict[str, Any]] = None

    def get_rot_option(self) -> RotOption | None:
        if self.orientation_representation is None:
            return None
        return RotOption.from_string(self.orientation_representation)

    def get_robot_state_rot_option(self) -> RotOption | None:
        if self.robot_state_orientation_representation is None:
            return None
        return RotOption.from_string(self.robot_state_orientation_representation)


def robot_profile_to_json(profile: RobotProfile) -> str:
    """Serialize ``RobotProfile`` to a JSON string (for HDF5 file attributes)."""
    return json.dumps(dataclasses.asdict(profile), ensure_ascii=False)


def load_robot_profile(path: Path | str) -> RobotProfile:
    """Parse a robot profile YAML file into ``Robotprofile``."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Robot profile file not found: {p}")
    raw = yaml.safe_load(p.read_text())
    return robot_profile_from_raw(raw)


def robot_profile_from_json(payload: str) -> RobotProfile:
    """Parse a robot profile JSON string into ``RobotProfile``."""
    try:
        raw = json.loads(payload)
    except Exception as e:
        raise ValueError(f"Invalid robot_profile JSON: {e}") from e
    return robot_profile_from_raw(raw)


def is_valid_action_space(action_space):
    s = set(action_space)

    arm_count = len(s & ACTION_SPACE_SET_1)
    gripper_count = len(s & ACTION_SPACE_SET_2)
    base_count = len(s & ACTION_SPACE_SET_3)

    return (
        arm_count >= 1 and
        gripper_count >= 1 and
        base_count <= 1 and
        len(s) == arm_count + gripper_count + base_count  # no extras
    )

def robot_profile_from_raw(raw: Any) -> RobotProfile:
    """Validate and parse raw mapping data into ``RobotProfile``."""
    if not isinstance(raw, dict):
        raise ValueError(f"Robot profile must be a mapping, got {type(raw).__name__}")

    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Robot profile missing keys: {missing}")

    action_space = list(raw["action_space"])
    if not is_valid_action_space(action_space):
        raise ValueError(
            f"Invalid action_space: {action_space!r}. "
            "Expected at least 1 arm action from "
            f"{sorted(ACTION_SPACE_SET_1)}, "
            "at least 1 gripper action from "
            f"{sorted(ACTION_SPACE_SET_2)}, "
            "at most 1 base action from "
            f"{sorted(ACTION_SPACE_SET_3)}, "
            "and no other keys."
        )

    required_robot_state_keys = set(REQUIRED_ROBOT_STATE_KEYS)
    required_robot_state_keys.update(
        ARM_ACTION_REQUIRED_ROBOT_STATE_KEY[action]
        for action in action_space
        if action in ARM_ACTION_REQUIRED_ROBOT_STATE_KEY
    )
    robot_state_keys = list(raw["robot_state_keys"])
    missing_robot_state_keys = sorted(required_robot_state_keys - set(robot_state_keys))
    if missing_robot_state_keys:
        raise ValueError(
            "Robot profile missing robot state keys required by its action_space: "
            f"{missing_robot_state_keys}"
        )

    robot_state_joint_names = _optional_str_list(
        raw.get("robot_state_joint_names")
    )
    if "joint_position" in robot_state_keys and not robot_state_joint_names:
        raise ValueError(
            "robot_state_joint_names is required when joint_position is included "
            "in robot_state_keys"
        )

    action_joint_names = _optional_str_list(raw.get("action_joint_names"))
    if (
        any(k in {"joint_position", "joint_velocity"} for k in action_space)
        and not action_joint_names
    ):
        raise ValueError(
            "action_joint_names is required for joint_position and joint_velocity "
            "action spaces"
        )

    # is_valid_action_space already guarantees at least one gripper action, so only the
    # mobile-base rule is left to check: it allows zero base actions in general, but a
    # profile declaring a mobile base has to drive it.
    if bool(raw["uses_mobile_base"]) and set(action_space).isdisjoint(ACTION_SPACE_SET_3):
        raise ValueError(
            f"Invalid action_space {action_space!r} for mobile base: must include at least one of "
            f"{sorted(ACTION_SPACE_SET_3)}"
        )

    return RobotProfile(
        policy_name=raw["policy_name"],
        robot_name=raw["robot_name"],
        # Both are in REQUIRED_KEYS, so there is no default to fall back on; bool() only
        # normalizes a blank YAML value (``is_biarm:``) into False.
        is_biarm=bool(raw["is_biarm"]),
        uses_mobile_base=bool(raw["uses_mobile_base"]),
        gripper_name=raw["gripper_name"],
        control_freq=raw["control_freq"],
        camera_names=list(raw["camera_names"]),
        # Observation Related
        robot_state_keys=robot_state_keys,
        robot_state_joint_names=robot_state_joint_names or [],
        # Action Related
        action_space=action_space,
        action_joint_names=action_joint_names,
        orientation_representation=raw.get("orientation_representation", None),
        # Optional Keys
        robot_state_orientation_representation=raw.get("robot_state_orientation_representation", None),
        controller=raw.get("controller"),
        gains=raw.get("gains"),
        intrinsic_calibration_matrix=_calibration_matrix(raw, "intrinsic"),
        extrinsic_calibration_matrix=_calibration_matrix(raw, "extrinsic"),
    )


def _calibration_matrix(raw: dict[str, Any], kind: str) -> dict[str, Any] | None:
    """Read a calibration matrix under either spelling.

    Profiles written from the bundled template use the spaced key
    (``intrinsic calibration matrix``); the underscored key is canonical.
    """
    return raw.get(f"{kind}_calibration_matrix", raw.get(f"{kind} calibration matrix"))


def _optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(x) for x in value]
    raise ValueError(f"Expected a list or null, got {type(value).__name__}")
