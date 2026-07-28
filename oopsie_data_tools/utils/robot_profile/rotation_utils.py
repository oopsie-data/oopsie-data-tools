"""Convert the orientation half of a cartesian action into a scalar-last quaternion.

A robot profile declares how its cartesian actions encode orientation
(``orientation_representation``); the recorder normalizes everything to
``(x, y, z, qx, qy, qz, qw)`` before writing an episode, so downstream consumers only
ever see quaternions.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation as R

_EULER_FORMATS = ("xyz", "zyx", "xyx", "XYZ", "ZYX", "XYX")


class RotOption(Enum):
    # Upper- and lower-case Euler orders are distinct on purpose: scipy reads uppercase as
    # intrinsic rotations and lowercase as extrinsic ones.
    XYZ = 0
    ZYX = 1
    XYX = 2
    QUAT = 3
    MATRIX = 4
    ROT6D = 5
    ROTVEC = 6
    xyz = 7
    zyx = 8
    xyx = 9

    @staticmethod
    def from_string(s: str) -> "RotOption":
        if s.startswith("euler_"):
            euler_format = s[len("euler_") :]
            if euler_format in _EULER_FORMATS:
                return RotOption[euler_format]
            raise ValueError(
                f"Unsupported Euler format: {euler_format!r}. "
                f"Expected one of {', '.join(_EULER_FORMATS)}."
            )
        try:
            return {
                "quat": RotOption.QUAT,
                "matrix": RotOption.MATRIX,
                "rot6d": RotOption.ROT6D,
                "rotvec": RotOption.ROTVEC,
            }[s]
        except KeyError:
            raise ValueError(
                f"Unsupported rotation option: {s!r}. Expected 'quat', 'matrix', 'rot6d', "
                "'rotvec', or 'euler_<order>'."
            ) from None


# How many elements each representation contributes to a flat action vector.
_EXPECTED_SIZE = {
    RotOption.QUAT: 4,
    RotOption.MATRIX: 9,
    RotOption.ROT6D: 6,
    RotOption.ROTVEC: 3,
    **{RotOption[f]: 3 for f in _EULER_FORMATS},
}


class ActionQuatConversion:
    """Convert the rotation slice of a cartesian action to a scalar-last quaternion."""

    def __init__(self, source_format: RotOption, is_biarm: bool = False) -> None:
        self.rot_option = source_format
        self.is_biarm = is_biarm

    def _to_quat(self, arr: np.ndarray) -> np.ndarray:
        expected = _EXPECTED_SIZE.get(self.rot_option)
        if expected is None:
            raise ValueError(f"Unsupported rotation option: {self.rot_option}")
        if arr.size != expected:
            raise ValueError(
                f"{self.rot_option.name} orientation expects {expected} value(s), got "
                f"{arr.size}. Check orientation_representation in the robot profile against "
                "the action your policy emits."
            )

        if self.rot_option == RotOption.QUAT:
            return arr
        if self.rot_option == RotOption.MATRIX:
            # Flat in the action vector, so reshape before handing it to scipy.
            return R.from_matrix(arr.reshape(3, 3)).as_quat()
        if self.rot_option == RotOption.ROT6D:
            # First two columns of the rotation matrix; rebuild the third by Gram-Schmidt.
            rot6d = arr.reshape(2, 3)
            u1 = rot6d[0] / np.linalg.norm(rot6d[0])
            u2 = rot6d[1] - np.dot(rot6d[1], u1) * u1
            u2 /= np.linalg.norm(u2)
            u3 = np.cross(u1, u2)
            return R.from_matrix(np.stack([u1, u2, u3], axis=1)).as_quat()
        if self.rot_option == RotOption.ROTVEC:
            return R.from_rotvec(arr).as_quat()
        return R.from_euler(self.rot_option.name, arr).as_quat()  # (x, y, z, w) order

    def _convert_arm(self, action: np.ndarray) -> np.ndarray:
        arm_rot = self._to_quat(action[3:])
        return np.concatenate([action[:3], arm_rot])  # position + converted rotation

    def convert_position(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action)
        if not self.is_biarm:
            return self._convert_arm(action)

        action_dim = len(action)
        if action_dim % 2:
            raise ValueError(
                f"Biarm cartesian action must have an even length to split per arm, got "
                f"{action_dim}."
            )
        arm1_action = self._convert_arm(action[: action_dim // 2])
        arm2_action = self._convert_arm(action[action_dim // 2 :])
        return np.concatenate([arm1_action, arm2_action])
