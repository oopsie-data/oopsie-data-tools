"""``to_quaternion_poses`` and ``write_robot_states`` — the two writers a converter needs
that are not just "put this array here".

Both exist because a converter bypasses ``EpisodeRecorder`` and therefore every check it
runs: the pose conversion that ``record_step`` would have applied, and the exact-match rule
between ``robot_state_keys`` and what is on disk.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from oopsie_data_tools.utils.conversion_utils import to_quaternion_poses, write_robot_states


def test_euler_poses_become_scalar_last_quaternions():
    poses = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])

    converted = to_quaternion_poses(poses, "euler_xyz")

    assert converted.shape == (1, 7)
    np.testing.assert_allclose(converted[0, :3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        converted[0, 3:], Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat()
    )
    assert np.isclose(np.linalg.norm(converted[0, 3:]), 1.0)


def test_a_biarm_pose_is_converted_per_arm():
    poses = np.tile(np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]]), (1, 2))

    converted = to_quaternion_poses(poses, "euler_xyz", is_biarm=True)

    assert converted.shape == (1, 14)
    np.testing.assert_allclose(converted[0, :7], converted[0, 7:])


def test_a_width_that_does_not_match_is_biarm_is_rejected():
    """``is_biarm`` splits the pose in half, so single-arm data under it would convert
    three position values and no rotation. That has to fail loudly, not produce a pose."""
    with pytest.raises(ValueError, match="orientation expects"):
        to_quaternion_poses(np.zeros((2, 6)), "euler_xyz", is_biarm=True)


def test_only_2d_pose_arrays_are_accepted():
    with pytest.raises(ValueError, match=r"Expected \(T, D\) poses"):
        to_quaternion_poses(np.zeros(6), "euler_xyz")


def test_robot_states_must_match_the_profile_exactly(tmp_path):
    with h5py.File(tmp_path / "ep.h5", "w") as f:
        with pytest.raises(ValueError, match=r"missing: \['gripper_position'\]"):
            write_robot_states(
                f, {"joint_position": np.zeros((4, 7))}, ["joint_position", "gripper_position"]
            )
        with pytest.raises(ValueError, match=r"undeclared: \['cartesian_position'\]"):
            write_robot_states(
                f,
                {"gripper_position": np.zeros((4, 1)), "cartesian_position": np.zeros((4, 7))},
                ["gripper_position"],
            )


def test_a_1d_state_is_stored_as_a_column(tmp_path):
    """Every trajectory array is (T, D); a bare (T,) gripper would fail the shape checks."""
    path = tmp_path / "ep.h5"
    with h5py.File(path, "w") as f:
        write_robot_states(f, {"gripper_position": np.arange(4.0)}, ["gripper_position"])

    with h5py.File(path, "r") as f:
        stored = f["observations/robot_states/gripper_position"]
        assert stored.shape == (4, 1)
        assert stored.dtype == np.float64
