"""Load robot / lab profiles for annotation and episode recording."""

from __future__ import annotations

import dataclasses
import json
import re
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
        "action_space",
    }
)

REQUIRED_ROBOT_STATE_KEYS = frozenset(
    {
        "gripper_position",
    }
)

# The state has to observe whatever space the action controls: velocity commands still
# need the corresponding *position* state. Applied as a union over the action space, so a
# profile mixing joint and cartesian actions needs both state keys. Base actions are not
# covered — nothing yet requires base_position state for a base action.
ARM_ACTION_REQUIRED_ROBOT_STATE_KEY = {
    "joint_position": "joint_position",
    "joint_velocity": "joint_position",
    "cartesian_position": "cartesian_position",
    "cartesian_velocity": "cartesian_position",
}

# How an additional sensor's per-step data is stored: "array" as a (T, ...) numeric dataset
# under additional_data/<key>, "video" as an MP4 whose relative path is stored there instead.
ADDITIONAL_DATA_FORMAT_ARRAY = "array"
ADDITIONAL_DATA_FORMAT_VIDEO = "video"
ADDITIONAL_DATA_FORMATS = (ADDITIONAL_DATA_FORMAT_ARRAY, ADDITIONAL_DATA_FORMAT_VIDEO)

_ADDITIONAL_DATA_FIELDS = frozenset({"sensor", "sensor_info", "format"})
# Keys become HDF5 dataset names and MP4 filename suffixes.
_ADDITIONAL_DATA_KEY_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


@dataclasses.dataclass(frozen=True)
class AdditionalDataSource:
    """One additional sensor stream declared by a robot profile.

    ``sensor`` names the device recording the data; ``sensor_info`` is free-form (a string
    or a JSON-serializable mapping) for units, frames, mounting, native rate and the like.
    """

    sensor: str
    sensor_info: str | Dict[str, Any] | None = None
    format: str = ADDITIONAL_DATA_FORMAT_ARRAY

    @property
    def is_video(self) -> bool:
        return self.format == ADDITIONAL_DATA_FORMAT_VIDEO


@dataclasses.dataclass(frozen=True)
class RobotProfile:
    """Robot / dataset identity and recording semantics (paths, joints, cameras).

    Options specific to :class:`WebRolloutAnnotator` (browser server port, blocking
    until annotation, resuming a session directory) are *not* part of this profile;
    pass those separately when constructing the annotator.
    """
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
    additional_data: Dict[str, AdditionalDataSource] = dataclasses.field(default_factory=dict)

    def additional_array_keys(self) -> list[str]:
        return [k for k, src in self.additional_data.items() if not src.is_video]

    def additional_video_keys(self) -> list[str]:
        return [k for k, src in self.additional_data.items() if src.is_video]

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

    if bool(raw["uses_mobile_base"]) and set(action_space).isdisjoint(ACTION_SPACE_SET_3):
        raise ValueError(
            f"Invalid action_space {action_space!r} for mobile base: must include at least one of "
            f"{sorted(ACTION_SPACE_SET_3)}"
        )

    camera_names = list(raw["camera_names"])
    additional_data = _additional_data(raw.get("additional_data"), camera_names)

    return RobotProfile(
        policy_name=raw["policy_name"],
        robot_name=raw["robot_name"],
        # Both are in REQUIRED_KEYS, so there is no default to fall back on; bool() only
        # normalizes a blank YAML value (``is_biarm:``) into False.
        is_biarm=bool(raw["is_biarm"]),
        uses_mobile_base=bool(raw["uses_mobile_base"]),
        gripper_name=raw["gripper_name"],
        control_freq=raw["control_freq"],
        camera_names=camera_names,
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
        additional_data=additional_data,
    )


def _additional_data(value: Any, camera_names: list[str]) -> dict[str, AdditionalDataSource]:
    """Parse the ``additional_data`` mapping of sensor key → sensor metadata."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"additional_data must be a mapping of data key to sensor metadata, "
            f"got {type(value).__name__}"
        )

    # A video sensor's MP4 is named <episode>_<key>.mp4, the same pattern as a camera's, and
    # case-insensitive filesystems (macOS, Windows) treat names differing in case as one file.
    taken = {name.casefold(): f"camera {name!r}" for name in camera_names}
    sources: dict[str, AdditionalDataSource] = {}
    for key, entry in value.items():
        label = f"additional_data[{key!r}]"
        if not isinstance(key, str) or not _ADDITIONAL_DATA_KEY_PATTERN.fullmatch(key):
            raise ValueError(
                f"{label}: keys may only contain letters, digits, '_' and '-'"
            )
        if key.casefold() in taken:
            raise ValueError(
                f"{label}: key collides with {taken[key.casefold()]} (names are compared "
                "ignoring case, since their files would collide on macOS and Windows)"
            )
        taken[key.casefold()] = f"additional_data key {key!r}"
        if not isinstance(entry, dict):
            raise ValueError(
                f"{label} must be a mapping with at least 'sensor', got {type(entry).__name__}"
            )
        unknown = sorted(set(entry) - _ADDITIONAL_DATA_FIELDS)
        if unknown:
            raise ValueError(
                f"{label} has unknown field(s) {unknown}; "
                f"allowed: {sorted(_ADDITIONAL_DATA_FIELDS)}"
            )

        sensor = entry.get("sensor")
        if not isinstance(sensor, str) or not sensor.strip():
            raise ValueError(f"{label}.sensor must be a non-empty string naming the sensor")

        sensor_info = entry.get("sensor_info")
        if sensor_info is not None:
            if not isinstance(sensor_info, (str, dict)):
                raise ValueError(
                    f"{label}.sensor_info must be a string or a mapping, "
                    f"got {type(sensor_info).__name__}"
                )
            try:
                json.dumps(sensor_info)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{label}.sensor_info is not JSON-serializable: {e}") from e

        fmt = entry.get("format") or ADDITIONAL_DATA_FORMAT_ARRAY
        if fmt not in ADDITIONAL_DATA_FORMATS:
            raise ValueError(
                f"{label}.format must be one of {list(ADDITIONAL_DATA_FORMATS)}, got {fmt!r}"
            )

        sources[key] = AdditionalDataSource(
            sensor=sensor, sensor_info=sensor_info, format=fmt
        )
    return sources


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
