"""End-to-end: a bulk inference loop records episodes that validate.

This is the only test that exercises record -> HDF5 -> MP4 -> validate as one flow. It
lived in ``mock_bulk_inference.py``, which pytest never collected because of the filename,
so the one integration check in the repo never ran in CI.

It also no longer sleeps between episodes. That sleep was hiding a real defect: episode
names are second-resolution, so a fast loop produced two episodes with the same name and
the second overwrote the first. ``test_back_to_back_episodes_do_not_collide`` pins the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from oopsie_data_tools.annotation_tool.episode_recorder import EpisodeRecorder
from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile
from oopsie_data_tools.utils.validation.validation_utils import validate_session_dir

IMG_H, IMG_W = 224, 224

PROFILE = RobotProfile(
    policy_name="mock_pi0",
    robot_name="franka_research_3",
    is_biarm=False,
    uses_mobile_base=False,
    gripper_name="robotiq_2f_85",
    control_freq=10,
    camera_names=["front", "wrist"],
    robot_state_keys=["joint_position", "gripper_position"],
    robot_state_joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
    action_space=["joint_velocity", "gripper_position"],
    action_joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
)

EPISODES = [
    {"instruction": "pick up the red block", "n_steps": 25, "success": 1.0},
    {"instruction": "place the cup on the tray", "n_steps": 30, "success": 0.0},
    {"instruction": "open the drawer", "n_steps": 22, "success": 1.0},
]


def _observation(step: int) -> dict:
    rng = np.random.default_rng(step)
    return {
        "robot_state": {
            "joint_position": rng.standard_normal(7).astype(np.float32),
            "gripper_position": np.array([rng.uniform(-1, 1)], dtype=np.float32),
        },
        "image_observation": {
            cam: rng.integers(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
            for cam in PROFILE.camera_names
        },
    }


def _action() -> dict:
    return {
        "joint_velocity": (np.random.randn(7) * 0.05).astype(np.float32),
        "gripper_position": np.clip(np.random.randn(1).astype(np.float32), -1.0, 1.0),
    }


@pytest.fixture
def recorded_session(tmp_path):
    """Record every episode back to back, with no pause between them."""
    recorder = EpisodeRecorder(
        robot_profile=PROFILE, data_root_dir=tmp_path, operator_name="mock_operator"
    )
    for episode in EPISODES:
        recorder.reset_episode_recorder()
        for step in range(episode["n_steps"]):
            recorder.record_step(_observation(step), _action())
        recorder.finish_rollout(
            instruction=episode["instruction"], success=episode["success"]
        )
    return recorder.session_dir


def test_a_recorded_session_validates(recorded_session):
    """What the recorder writes must be what the validator accepts."""
    assert validate_session_dir(str(recorded_session), strict_annotation_check=True) == 0


def test_every_episode_is_written(recorded_session):
    assert len(list(recorded_session.glob("*.h5"))) == len(EPISODES)


def test_back_to_back_episodes_do_not_collide(recorded_session):
    """The defect the old sleep(1.1) was working around.

    These episodes are recorded within the same second, so second-resolution names used to
    repeat and each episode overwrote the previous one's HDF5 and videos.
    """
    episodes = sorted(p.stem for p in recorded_session.glob("*.h5"))

    assert len(set(episodes)) == len(EPISODES), f"names collided: {episodes}"
    expected_videos = len(EPISODES) * len(PROFILE.camera_names)
    assert len(list(recorded_session.glob("*.mp4"))) == expected_videos


def test_each_episode_keeps_its_own_instruction(recorded_session):
    """A weaker collision would leave the right number of files with the wrong contents."""
    import h5py

    instructions = set()
    for h5_path in recorded_session.glob("*.h5"):
        with h5py.File(h5_path, "r") as f:
            instructions.add(f.attrs["language_instruction"])

    assert instructions == {e["instruction"] for e in EPISODES}
