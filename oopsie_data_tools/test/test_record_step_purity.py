"""``record_step`` must not modify the dicts the caller passes in.

It used to: ``_check_step_data`` wrote the quaternion-converted pose back into the caller's
``action``/``robot_state``, and ``record_step`` then read those same dicts to fill its
buffer. The mutation was therefore load-bearing — the conversion reached the recording
buffer through it — so "just copy the dict" would have silently recorded raw euler values
labelled as quaternions. These tests pin down both halves: the caller's data is untouched,
*and* the conversion still lands in the episode.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from oopsie_data_tools.annotation_tool.episode_recorder import EpisodeRecorder
from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile

EULER = [0.3, -0.5, 1.1]
POSITION = [0.1, -0.2, 0.3]


def _cartesian_profile(**overrides) -> RobotProfile:
    defaults = dict(
        policy_name="p",
        robot_name="r",
        is_biarm=False,
        uses_mobile_base=False,
        gripper_name="g",
        control_freq=10,
        camera_names=["left"],
        robot_state_keys=["joint_position", "gripper_position", "cartesian_position"],
        robot_state_joint_names=["j1", "j2"],
        action_space=["cartesian_position", "gripper_position"],
        orientation_representation="euler_xyz",
        robot_state_orientation_representation="euler_xyz",
    )
    defaults.update(overrides)
    return RobotProfile(**defaults)


@pytest.fixture
def recorder():
    with tempfile.TemporaryDirectory() as tmp:
        rec = EpisodeRecorder(
            robot_profile=_cartesian_profile(),
            data_root_dir=Path(tmp),
            operator_name="op",
        )
        rec.reset_episode_recorder()
        yield rec


def _observation() -> dict:
    return {
        "robot_state": {
            "joint_position": np.zeros(2, dtype=np.float32),
            "gripper_position": np.zeros(1, dtype=np.float32),
            "cartesian_position": np.array(POSITION + EULER, dtype=np.float32),
        },
        "image_observation": {"left": np.zeros((64, 64, 3), dtype=np.uint8)},
    }


def _action() -> dict:
    return {
        "cartesian_position": np.array(POSITION + EULER, dtype=np.float32),
        "gripper_position": np.zeros(1, dtype=np.float32),
    }


def test_caller_dicts_are_not_mutated(recorder):
    observation, action = _observation(), _action()
    observation_before = copy.deepcopy(observation)
    action_before = copy.deepcopy(action)

    recorder.record_step(observation=observation, action=action)

    np.testing.assert_array_equal(
        action["cartesian_position"],
        action_before["cartesian_position"],
        err_msg="record_step rewrote the caller's action dict",
    )
    np.testing.assert_array_equal(
        observation["robot_state"]["cartesian_position"],
        observation_before["robot_state"]["cartesian_position"],
        err_msg="record_step rewrote the caller's robot_state dict",
    )
    assert action["cartesian_position"].shape == (6,), "still euler, not converted in place"


def test_conversion_still_reaches_the_buffer(recorder):
    """The half that a naive de-mutation would have broken, silently."""
    recorder.record_step(observation=_observation(), action=_action())

    buffered_action = recorder.timesteps[0]["action_dict"]["cartesian_position"]
    buffered_state = recorder.timesteps[0]["robot_state"]["cartesian_position"]

    assert buffered_action.shape == (7,), "euler must have been converted to a quaternion"
    assert buffered_state.shape == (7,)
    np.testing.assert_allclose(buffered_action[:3], POSITION, rtol=1e-6)
    np.testing.assert_allclose(
        R.from_quat(buffered_action[3:]).as_matrix(),
        R.from_euler("xyz", EULER).as_matrix(),
        atol=1e-6,
    )


def test_the_same_action_dict_can_be_reused_across_steps(recorder):
    """A caller reusing one preallocated dict must not accumulate conversions."""
    observation, action = _observation(), _action()

    for _ in range(3):
        recorder.record_step(observation=observation, action=action)

    assert recorder.num_steps == 3
    for step in recorder.timesteps:
        np.testing.assert_allclose(
            step["action_dict"]["cartesian_position"],
            recorder.timesteps[0]["action_dict"]["cartesian_position"],
            err_msg="repeated conversion of an already-converted pose",
        )
