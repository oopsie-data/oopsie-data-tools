"""Tests for robot profile YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from oopsie_data_tools.utils.paths import PROFILES_DIR_NAME
from oopsie_data_tools.utils.robot_profile.robot_profile import (
    RobotProfile,
    load_robot_profile,
)

# The shipped example profiles, addressed directly rather than through the cwd-sensitive
# profile lookup chain, which $OOPSIE_ROBOT_PROFILES_DIR or a local ./robot_profiles moves.
BUNDLED_PROFILES = Path(__file__).resolve().parents[2] / "configs" / PROFILES_DIR_NAME

VALID_PROFILE = {
    "policy_name": "test_policy",
    "robot_name": "test_robot",
    "is_biarm": False,
    "uses_mobile_base": False,
    "gripper_name": "test_gripper",
    "control_freq": 10,
    "camera_names": ["cam0"],
    "robot_state_keys": ["cartesian_position", "gripper_position"],
    "action_space": ["cartesian_position", "gripper_position"],
}

JOINT_STATE = {
    "robot_state_keys": ["joint_position", "gripper_position"],
    "robot_state_joint_names": ["j0", "j1"],
}


@pytest.fixture
def write_profile(tmp_path):
    """Write a profile dict to YAML and return its path."""

    def _write(**overrides) -> Path:
        path = tmp_path / "profile.yaml"
        path.write_text(yaml.dump({**VALID_PROFILE, **overrides}))
        return path

    return _write


def test_load_openpi_example_yaml() -> None:
    assert BUNDLED_PROFILES.is_dir(), "these tests run from a checkout"
    profile = load_robot_profile(BUNDLED_PROFILES / "openpi_example_robot_profile.yaml")
    assert isinstance(profile, RobotProfile)
    assert "joint_velocity" in profile.action_space
    assert len(profile.action_joint_names or []) == 7


def test_load_minimal_valid_profile(write_profile) -> None:
    profile = load_robot_profile(write_profile())
    assert isinstance(profile, RobotProfile)
    assert profile.is_biarm is False


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_robot_profile("/nonexistent/path/profile.yaml")


def test_non_mapping_yaml_raises(tmp_path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_robot_profile(path)


@pytest.mark.parametrize(
    "key",
    [
        "policy_name",
        "robot_name",
        "gripper_name",
        "control_freq",
        "camera_names",
        "robot_state_keys",
        "action_space",
    ],
)
def test_every_required_key_is_required(tmp_path, key: str) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.dump({k: v for k, v in VALID_PROFILE.items() if k != key}))
    with pytest.raises(ValueError):
        load_robot_profile(path)


# robot_state_keys must document what the action space controls: a profile that promises
# cartesian actions but records no cartesian state produces episodes the validator rejects.
@pytest.mark.parametrize(
    "overrides, match",
    [
        pytest.param({"robot_state_keys": ["gripper_position"]}, "cartesian_position",
                     id="cartesian action, no cartesian state"),
        pytest.param({"robot_state_keys": ["cartesian_position"]}, "gripper_position",
                     id="no gripper state"),
        pytest.param({"robot_state_keys": []}, "missing robot state keys",
                     id="empty robot_state_keys"),
        pytest.param({"robot_state_keys": ["gripper_position"],
                      "action_space": ["joint_velocity", "gripper_position"],
                      "action_joint_names": ["j0", "j1"]}, "joint_position",
                     id="joint action, no joint state"),
    ],
)
def test_robot_state_keys_must_cover_the_action_space(write_profile, overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        load_robot_profile(write_profile(**overrides))


@pytest.mark.parametrize("joint_names", [None, []], ids=["absent", "empty"])
def test_joint_position_state_without_joint_names(write_profile, joint_names) -> None:
    overrides = {
        "robot_state_keys": ["joint_position", "cartesian_position", "gripper_position"]
    }
    if joint_names is not None:
        overrides["robot_state_joint_names"] = joint_names
    with pytest.raises(ValueError, match="robot_state_joint_names is required"):
        load_robot_profile(write_profile(**overrides))


@pytest.mark.parametrize(
    "action_space",
    [
        pytest.param(["invalid_key", "gripper_position"], id="unrecognized key"),
        pytest.param(["gripper_position"], id="no arm action at all"),
        pytest.param(["cartesian_position"], id="no gripper action"),
    ],
)
def test_action_space_must_be_a_usable_control_mode(write_profile, action_space) -> None:
    with pytest.raises(ValueError):
        load_robot_profile(write_profile(action_space=action_space))


@pytest.mark.parametrize("joint_action", ["joint_position", "joint_velocity"])
def test_joint_actions_require_action_joint_names(write_profile, joint_action) -> None:
    with pytest.raises(ValueError):
        load_robot_profile(
            write_profile(**JOINT_STATE, action_space=[joint_action, "gripper_position"])
        )


def test_mobile_base_without_base_action_key(write_profile) -> None:
    with pytest.raises(ValueError):
        load_robot_profile(write_profile(uses_mobile_base=True))
