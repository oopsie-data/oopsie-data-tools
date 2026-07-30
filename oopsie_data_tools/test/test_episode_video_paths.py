"""How an episode records where its videos live.

Paths are stored relative to the episode file so a session directory can be moved or
uploaded from anywhere. That only holds if both sides of the relpath use the same base:
measuring a resolved video against an unresolved directory used to store
``../../../private/tmp/<session>/x.mp4`` for a video sitting right beside the episode,
because /tmp is a symlink to /private/tmp on macOS. Such a path still resolves in place,
so nothing failed — until the directory moved.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from oopsie_data_tools.annotation_tool.episode_recorder import EpisodeRecorder
from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile
from oopsie_data_tools.utils.validation.errors import EpisodeValidationError
from oopsie_data_tools.utils.validation.validation_utils import (
    collect_validation_results,
    validate_h5_file,
    validate_session_dir,
)

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


def _record_episode(data_root: Path) -> Path:
    """Record a short episode into ``data_root``; returns the path to its .h5 file."""
    profile = _profile()
    recorder = EpisodeRecorder(
        robot_profile=profile, data_root_dir=data_root, operator_name="op"
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
    return next(Path(data_root).rglob("*.h5"))


@pytest.fixture
def episode(tmp_path) -> Path:
    return _record_episode(tmp_path)


@pytest.fixture
def symlinked_episode(tmp_path) -> Path:
    """An episode recorded through a symlinked data root.

    This is what makes the surrounding tests mean anything off macOS. There, /tmp is a
    symlink to /private/tmp, so pytest's tmp_path exposed the resolved/unresolved mismatch
    for free; on Linux tmp_path is a real directory and every assertion here passes even
    against the unfixed code. Putting the symlink in explicitly makes the regression
    reproducible on any platform — which matters, because CI only runs Linux.

    The link has to be *shallower than its target* to reproduce the fault. With both at
    the same depth the stray ".." segments still resolve back to the right place and the
    stored path comes out clean anyway; only when they overshoot the link's own parent —
    exactly the shape of /tmp -> /private/tmp — does the broken path survive into the file.
    """
    real = tmp_path / "deep" / "deeper" / "real_root"
    real.mkdir(parents=True)
    link = tmp_path / "linked_root"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError) as e:  # Windows without developer mode
        pytest.skip(f"cannot create symlinks on this platform: {e}")
    return _record_episode(link)


def _stored_paths(h5_path: Path) -> dict[str, str]:
    with h5py.File(h5_path, "r") as f:
        group = f["observations/video_paths"]
        return {cam: group[cam][()].decode("utf-8") for cam in group}


def _replace_stored_path(h5_path: Path, camera: str, path: str) -> None:
    with h5py.File(h5_path, "r+") as f:
        f[f"observations/video_paths/{camera}"][()] = path


def test_stored_paths_are_relative_and_point_at_the_videos(episode):
    """Deliberately loose: any relative layout is fine as long as it resolves.

    Pinning the exact string would freeze today's flat "<stem>_<cam>.mp4" layout and break
    if videos ever move into a subdirectory, which is not what this is protecting.
    """
    stored = _stored_paths(episode)

    assert set(stored) == set(CAMERAS)
    for cam, rel in stored.items():
        assert not Path(rel).is_absolute(), f"{cam} stored an absolute path: {rel!r}"
        assert (episode.parent / rel).is_file(), f"{cam} path does not resolve: {rel!r}"


def test_stored_paths_stay_inside_the_data_root(episode, tmp_path):
    """The actual regression: a path must not leave the dataset and climb back in.

    A video beside its episode was stored as "../../../private/tmp/<session>/x.mp4". That
    resolves correctly in place, so nothing failed — it is only wrong because it encodes
    the absolute location of the machine that recorded it. Subdirectories and sibling
    directories inside the data root remain perfectly acceptable.
    """
    root = tmp_path.resolve()

    for cam, rel in _stored_paths(episode).items():
        resolved = (episode.parent / rel).resolve()
        assert root in resolved.parents, (
            f"{cam} stored {rel!r}, which escapes the data root {root} before returning. "
            "Both sides of the relpath must use the same resolved base."
        )


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


def test_a_symlinked_data_root_still_stores_portable_paths(symlinked_episode, tmp_path):
    """The regression itself, reproducible on any platform.

    Recording through a symlinked directory used to store a path that walked up to the
    filesystem root and back down the resolved side of the link.
    """
    for cam, rel in _stored_paths(symlinked_episode).items():
        assert not Path(rel).is_absolute(), f"{cam}: {rel!r}"
        assert not rel.startswith("../"), (
            f"{cam} stored {rel!r}: the relpath was measured against an unresolved base, "
            "so it escapes the dataset and walks back in through the symlink target."
        )
        assert (symlinked_episode.parent / rel).is_file()


def test_a_symlinked_session_can_also_be_moved(symlinked_episode, tmp_path):
    destination = tmp_path / "relocated_from_symlink"
    shutil.move(str(symlinked_episode.parent), str(destination))
    moved = next(destination.glob("*.h5"))

    assert validate_h5_file(str(moved), strict_annotation_check=True)


def test_validator_rejects_a_relative_video_path_outside_the_submitted_directory(
    episode, tmp_path
):
    submitted = episode.parent
    original = submitted / _stored_paths(episode)["left"]
    outside = tmp_path / "outside.mp4"
    shutil.copy2(original, outside)
    rel = os.path.relpath(outside, submitted)
    _replace_stored_path(episode, "left", rel)

    with pytest.raises(EpisodeValidationError, match="escapes the submitted directory"):
        validate_h5_file(str(episode), strict_annotation_check=True)

    results = collect_validation_results(str(submitted))
    failed = next(result for result in results if result["path"] == str(episode))
    assert failed["passed"] is False
    assert failed["error_type"] == "validation"
    assert "escapes the submitted directory" in failed["error"]


def test_validator_rejects_a_symlink_to_a_video_outside_the_submitted_directory(
    episode, tmp_path
):
    submitted = episode.parent
    original = submitted / _stored_paths(episode)["left"]
    outside = tmp_path / "outside.mp4"
    shutil.copy2(original, outside)
    link = submitted / "linked-outside.mp4"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"cannot create symlinks on this platform: {e}")
    _replace_stored_path(episode, "left", link.name)

    with pytest.raises(EpisodeValidationError, match="escapes the submitted directory"):
        validate_h5_file(str(episode), strict_annotation_check=True)


def test_directory_validation_allows_a_parent_relative_video_inside_the_submission(tmp_path):
    submitted = tmp_path / "submission"
    nested = submitted / "episodes"
    nested.mkdir(parents=True)
    episode = _record_episode(nested)
    original = episode.parent / _stored_paths(episode)["left"]
    shared = submitted / "shared-left.mp4"
    shutil.move(str(original), shared)
    rel = os.path.relpath(shared, episode.parent)
    _replace_stored_path(episode, "left", rel)

    assert validate_session_dir(str(submitted), strict_annotation_check=True) == 0
