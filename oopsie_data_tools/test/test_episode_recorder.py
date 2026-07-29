"""Tests for EpisodeRecorder: buffering, validation, and HDF5 output."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from oopsie_data_tools.annotation_tool.episode_recorder import EpisodeRecorder, write_mp4
from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile


def _profile(**overrides) -> RobotProfile:
    defaults = dict(
        policy_name="test_policy",
        robot_name="test_robot",
        is_biarm=False,
        uses_mobile_base=False,
        gripper_name="test_gripper",
        control_freq=10,
        camera_names=["left", "wrist"],
        robot_state_keys=["joint_position", "gripper_position"],
        robot_state_joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
        action_space=["joint_velocity", "gripper_position"],
        action_joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
    )
    defaults.update(overrides)
    return RobotProfile(**defaults)


PROFILE = _profile()


def _obs() -> dict:
    return {
        "robot_state": {
            "joint_position": np.zeros(7, dtype=np.float32),
            "gripper_position": np.zeros(1, dtype=np.float32),
        },
        "image_observation": {
            cam: np.zeros((64, 64, 3), dtype=np.uint8) for cam in PROFILE.camera_names
        },
    }


def _action() -> dict:
    sizes = {"joint_velocity": 7, "gripper_position": 1}
    return {k: np.zeros(sizes.get(k, 1), dtype=np.float32) for k in PROFILE.action_space}


def _save_data(recorder: EpisodeRecorder) -> dict:
    return {
        "language_instruction": "pick up the cup",
        "metadata": {"episode_id": recorder.save_fname, "operator_name": "tester"},
        "video_paths": {cam: f"/tmp/{cam}.mp4" for cam in PROFILE.camera_names},
    }


@pytest.fixture
def recorder(tmp_path) -> EpisodeRecorder:
    rec = EpisodeRecorder(
        robot_profile=PROFILE, data_root_dir=str(tmp_path), operator_name="test_operator"
    )
    rec.reset_episode_recorder()
    return rec


def test_construction_prepares_an_empty_session(recorder):
    assert recorder.session_dir.is_dir()
    assert recorder.num_steps == 0


def test_steps_accumulate_and_reset_clears_them(recorder):
    for _ in range(5):
        recorder.record_step(_obs(), _action())
    assert recorder.num_steps == 5
    recorder.reset_episode_recorder()
    assert recorder.num_steps == 0


# Each case mutates one thing about an otherwise valid step. The recorder is the only
# gate between a robot script and a recorded episode, so every one of these has to be
# refused at record time rather than surfacing as a validation failure at upload.
def _no_robot_state(obs, action):
    del obs["robot_state"]


def _no_image_observation(obs, action):
    del obs["image_observation"]


def _missing_camera(obs, action):
    del obs["image_observation"]["left"]


def _missing_robot_state_key(obs, action):
    del obs["robot_state"]["joint_position"]


def _undeclared_robot_state_key(obs, action):
    obs["robot_state"] = {"unexpected_key": np.zeros(7)}


def _empty_action(obs, action):
    action.clear()


def _unrecognized_action_key(obs, action):
    action["bad_key"] = np.zeros(1)


def _action_missing_a_profile_key(obs, action):
    action.pop("gripper_position")


def _action_beyond_the_profile(obs, action):
    action["cartesian_position"] = np.zeros(7, dtype=np.float32)


def _valid_keys_but_not_the_profile_s(obs, action):
    action.clear()
    action["cartesian_position"] = np.zeros(7, dtype=np.float32)
    action["gripper_position"] = np.zeros(1, dtype=np.float32)


def _all_action_values_none(obs, action):
    for k in action:
        action[k] = None


def _one_action_value_none(obs, action):
    action[next(iter(action))] = None


@pytest.mark.parametrize(
    "mutate",
    [
        _no_robot_state,
        _no_image_observation,
        _missing_camera,
        _missing_robot_state_key,
        _undeclared_robot_state_key,
        _empty_action,
        _unrecognized_action_key,
        _action_missing_a_profile_key,
        _action_beyond_the_profile,
        _valid_keys_but_not_the_profile_s,
        _all_action_values_none,
        _one_action_value_none,
    ],
    ids=lambda f: f.__name__.strip("_"),
)
def test_record_step_rejects_a_malformed_step(recorder, mutate):
    obs, action = _obs(), _action()
    mutate(obs, action)
    with pytest.raises(ValueError):
        recorder.record_step(obs, action)


@pytest.mark.parametrize("bad", ["not a dict", np.zeros(8)], ids=["observation", "action"])
def test_record_step_rejects_a_non_dict(recorder, bad):
    if isinstance(bad, str):
        with pytest.raises(ValueError):
            recorder.record_step(bad, _action())
    else:
        with pytest.raises(ValueError):
            recorder.record_step(_obs(), bad)


def test_save_writes_the_episode_it_buffered(recorder):
    for _ in range(4):
        recorder.record_step(_obs(), _action())
    h5_path = recorder.save(_save_data(recorder))

    assert isinstance(h5_path, Path)
    assert h5_path.suffix == ".h5"
    assert h5_path.exists()
    assert h5_path.parent == recorder.session_dir
    with h5py.File(h5_path, "r") as f:
        assert f.attrs["language_instruction"] == "pick up the cup"
        assert f.attrs["schema"] == "oopsiedata_format_v1"
        assert "actions" in f
        assert "joint_position" in f["observations/robot_states"]
        assert f["observations/robot_states/joint_position"].shape[0] == 4


def test_save_raises_without_steps(recorder):
    with pytest.raises(ValueError):
        recorder.save(_save_data(recorder))


@pytest.mark.parametrize(
    "shape",
    [(64, 64, 3), (4, 64, 64, 4), (0, 64, 64, 3)],
    ids=["no time dimension", "RGBA not RGB", "zero frames"],
)
def test_write_mp4_rejects_a_bad_frame_array(tmp_path, shape):
    with pytest.raises(ValueError):
        write_mp4(tmp_path / "out.mp4", np.zeros(shape, dtype=np.uint8), fps=10.0)
