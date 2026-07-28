"""Tests for orientation-representation conversion.

This module had no coverage at all, which is how the ``matrix`` branch shipped broken:
it handed scipy a flat 9-element slice without reshaping, so every profile declaring
``orientation_representation: matrix`` raised at the first recorded step.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from oopsie_data_tools.utils.robot_profile.rotation_utils import ActionQuatConversion, RotOption

POSITION = np.array([0.1, -0.2, 0.3])
# An arbitrary rotation, expressed once and then re-encoded per representation below.
REFERENCE = R.from_euler("xyz", [0.3, -0.5, 1.1])


def _assert_same_rotation(quat: np.ndarray, expected: R) -> None:
    """Compare as matrices: q and -q are the same rotation, and Rotation.approx_equal
    does not exist on the scipy that ships with our Python 3.8 floor."""
    np.testing.assert_allclose(R.from_quat(quat).as_matrix(), expected.as_matrix(), atol=1e-8)


def _encode(option: RotOption) -> np.ndarray:
    if option == RotOption.QUAT:
        return REFERENCE.as_quat()
    if option == RotOption.MATRIX:
        return REFERENCE.as_matrix().reshape(9)
    if option == RotOption.ROT6D:
        # First two columns of the rotation matrix, flattened row-wise as the converter reads it.
        return REFERENCE.as_matrix()[:, :2].T.reshape(6)
    if option == RotOption.ROTVEC:
        return REFERENCE.as_rotvec()
    return REFERENCE.as_euler(option.name)


@pytest.mark.parametrize(
    "option",
    [
        RotOption.QUAT,
        RotOption.MATRIX,
        RotOption.ROT6D,
        RotOption.ROTVEC,
        RotOption.xyz,
        RotOption.zyx,
        RotOption.XYZ,
    ],
)
def test_every_representation_round_trips_to_the_same_rotation(option):
    action = np.concatenate([POSITION, _encode(option)])

    result = ActionQuatConversion(option).convert_position(action)

    assert result.shape == (7,), "position + scalar-last quaternion"
    np.testing.assert_allclose(result[:3], POSITION)
    _assert_same_rotation(result[3:], REFERENCE)


def test_matrix_is_accepted_flat():
    """The regression: a flat 3x3 in the action vector must not reach scipy unreshaped."""
    action = np.concatenate([POSITION, np.eye(3).reshape(9)])

    result = ActionQuatConversion(RotOption.MATRIX).convert_position(action)

    # 1e-9 rather than exact: the value comes out of scipy's matrix-to-quaternion routine,
    # and pinning it to machine epsilon would make this brittle across scipy/BLAS builds
    # for no gain — the point is that it converts at all, and to the identity rotation.
    np.testing.assert_allclose(result, [*POSITION, 0.0, 0.0, 0.0, 1.0], atol=1e-9)


def test_biarm_converts_each_arm_independently():
    left = R.from_euler("xyz", [0.2, 0.0, 0.0])
    right = R.from_euler("xyz", [0.0, 0.7, 0.0])
    action = np.concatenate(
        [POSITION, left.as_euler("xyz"), -POSITION, right.as_euler("xyz")]
    )

    result = ActionQuatConversion(RotOption.xyz, is_biarm=True).convert_position(action)

    assert result.shape == (14,)
    np.testing.assert_allclose(result[:3], POSITION)
    np.testing.assert_allclose(result[7:10], -POSITION)
    _assert_same_rotation(result[3:7], left)
    _assert_same_rotation(result[10:14], right)


def test_wrong_orientation_width_names_the_profile_field():
    """A profile/policy mismatch should say so, not surface a scipy internal error."""
    action = np.concatenate([POSITION, np.zeros(3)])  # euler-sized, but declared as quat

    with pytest.raises(ValueError, match="QUAT orientation expects 4 value"):
        ActionQuatConversion(RotOption.QUAT).convert_position(action)


def test_biarm_rejects_an_odd_action_length():
    with pytest.raises(ValueError, match="even length"):
        ActionQuatConversion(RotOption.QUAT, is_biarm=True).convert_position(np.zeros(13))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("quat", RotOption.QUAT),
        ("matrix", RotOption.MATRIX),
        ("rot6d", RotOption.ROT6D),
        ("rotvec", RotOption.ROTVEC),
        ("euler_xyz", RotOption.xyz),
        ("euler_XYZ", RotOption.XYZ),
    ],
)
def test_from_string_accepts_every_documented_option(text, expected):
    """Every option named in configs/robot_profiles/template.yaml must parse."""
    assert RotOption.from_string(text) == expected


def test_from_string_rejects_an_unknown_euler_order():
    with pytest.raises(ValueError, match="Unsupported Euler format"):
        RotOption.from_string("euler_abc")


def test_from_string_rejects_an_unknown_representation():
    with pytest.raises(ValueError, match="Unsupported rotation option"):
        RotOption.from_string("axis_angle")
