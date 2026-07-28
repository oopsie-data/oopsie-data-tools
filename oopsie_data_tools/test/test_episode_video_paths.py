"""How an episode records where its videos live.

Paths are stored relative to the episode file so a session directory can be moved or
uploaded from anywhere. That only holds if both sides of the relpath use the same base:
measuring a resolved video against an unresolved directory used to store
``../../../private/tmp/<session>/x.mp4`` for a video sitting right beside the episode,
because /tmp is a symlink to /private/tmp on macOS. Such a path still resolves in place,
so nothing failed — until the directory moved.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from oopsie_data_tools.annotation_tool.episode_recorder import EpisodeRecorder
from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile
from oopsie_data_tools.utils.validation.validation_utils import validate_h5_file

CAMERAS = ["left", "wrist"]


def _profile() -> RobotProfile:
    return RobotProfile(
        policy_name="p",
        robot_name="r",
        is_biarm=False,
        uses_mobile_base=False,
        gripper_name="g",
        control_freq=10,
        camera_names=CAMERAS,
        robot_state_keys=["joint_position", "gripper_position"],
        robot_state_joint_names=["j1", "j2"],
        action_space=["joint_position", "gripper_position"],
        action_joint_names=["j1", "j2"],
    )


@pytest.fixture
def episode(tmp_path) -> Path:
    """A recorded episode, returned as the path to its .h5 file."""
    profile = _profile()
    recorder = EpisodeRecorder(
        robot_profile=profile, data_root_dir=tmp_path, operator_name="op"
    )
    for _ in range(30):
        recorder.record_step(
            observation={
                "robot_state": {
                    "joint_position": np.zeros(2, dtype=np.float32),
                    "gripper_position": np.zeros(1, dtype=np.float32),
                },
                "image_observation": {
                    cam: np.zeros((240, 320, 3), dtype=np.uint8) for cam in CAMERAS
                },
            },
            action={
                "joint_position": np.zeros(2, dtype=np.float32),
                "gripper_position": np.zeros(1, dtype=np.float32),
            },
        )
    recorder.finish_rollout(instruction="pick up the block", success=1.0)
    return next(tmp_path.rglob("*.h5"))


def _stored_paths(h5_path: Path) -> dict[str, str]:
    with h5py.File(h5_path, "r") as f:
        group = f["observations/video_paths"]
        return {cam: group[cam][()].decode("utf-8") for cam in group}


def test_videos_next_to_the_episode_are_stored_as_bare_filenames(episode):
    stored = _stored_paths(episode)

    assert set(stored) == set(CAMERAS)
    for cam, rel in stored.items():
        assert "/" not in rel and ".." not in rel, (
            f"{cam} stored as {rel!r}; a video beside the episode needs no path at all"
        )
        assert (episode.parent / rel).is_file()


def test_exactly_one_video_per_camera_is_written(episode):
    """Two writers used to run over the same frames; only one should now."""
    mp4s = sorted(p.name for p in episode.parent.glob("*.mp4"))

    assert len(mp4s) == len(CAMERAS), f"expected one mp4 per camera, got {mp4s}"


def test_the_session_directory_can_be_moved(episode, tmp_path):
    """The reason paths are relative in the first place."""
    destination = tmp_path / "relocated"
    shutil.move(str(episode.parent), str(destination))
    moved = next(destination.glob("*.h5"))

    assert validate_h5_file(str(moved), strict_annotation_check=True)
