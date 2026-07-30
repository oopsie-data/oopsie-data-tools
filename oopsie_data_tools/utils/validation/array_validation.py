"""Shared validation for numeric trajectory arrays.

These helpers raise ``ValueError`` so both the recorder (which validates individual
steps) and the semantic episode validator can reuse the same numerical rules while
presenting their own public exception types.
"""

from __future__ import annotations

from typing import Any

import numpy as np

QUATERNION_NORM_ATOL = 1e-2


def require_finite_real_array(value: Any, label: str) -> np.ndarray:
    """Return ``value`` as an array after requiring real, finite numeric values."""
    arr = np.asarray(value)
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"{label} must be a real numeric array, got dtype {arr.dtype}")

    finite = np.isfinite(arr)
    if not np.all(finite):
        bad_flat = int(np.flatnonzero(~finite.ravel())[0])
        index = tuple(int(i) for i in np.unravel_index(bad_flat, arr.shape))
        bad_value = arr[index] if index else arr[()]
        count = int(np.count_nonzero(~finite))
        raise ValueError(
            f"{label} contains {count} non-finite value(s); first at index "
            f"{index}: {bad_value!r}"
        )
    return arr


def validate_cartesian_quaternions(value: Any, label: str) -> np.ndarray:
    """Require one unit scalar-last quaternion per 7-DOF Cartesian arm pose."""
    arr = require_finite_real_array(value, label)
    if arr.ndim == 0 or arr.shape[-1] not in (7, 14):
        raise ValueError(
            f"{label} must end in 7 or 14 values — "
            f"[x, y, z, qx, qy, qz, qw] per arm, got shape {arr.shape}"
        )

    starts = (3,) if arr.shape[-1] == 7 else (3, 10)
    for arm_index, start in enumerate(starts, 1):
        stop = start + 4
        norms = np.linalg.norm(arr[..., start:stop], axis=-1)
        valid = np.isclose(norms, 1.0, rtol=0.0, atol=QUATERNION_NORM_ATOL)
        if np.all(valid):
            continue

        norms_array = np.asarray(norms)
        bad_flat = int(np.flatnonzero(~np.asarray(valid).ravel())[0])
        sample_index = tuple(
            int(i) for i in np.unravel_index(bad_flat, norms_array.shape)
        )
        norm = float(norms_array[sample_index] if sample_index else norms_array[()])
        location = f" at sample index {sample_index}" if sample_index else ""
        raise ValueError(
            f"{label}[{start}:{stop}] (arm {arm_index}) must be a unit scalar-last "
            f"quaternion (norm ≈ 1.0); first invalid quaternion{location} has norm "
            f"{norm:.6f}"
        )
    return arr


def validate_gripper_binary_trajectory(
    value: Any, *, is_biarm: bool, label: str
) -> np.ndarray:
    """Validate a stored ``gripper_binary`` trajectory's shape and domain."""
    arr = require_finite_real_array(value, label)
    max_channels = 2 if is_biarm else 1
    if arr.ndim == 1:
        channels = 1
    elif arr.ndim == 2:
        channels = arr.shape[1]
    else:
        raise ValueError(
            f"{label} must have shape (T,), (T, 1)"
            f"{' or (T, 2)' if is_biarm else ''}; got {arr.shape}"
        )
    if not (1 <= channels <= max_channels):
        raise ValueError(
            f"{label} has {channels} command channels, but the "
            f"{'biarm' if is_biarm else 'single-arm'} profile permits at most "
            f"{max_channels}; got shape {arr.shape}"
        )

    valid = np.isin(arr, (0, 1))
    if not np.all(valid):
        bad_flat = int(np.flatnonzero(~valid.ravel())[0])
        index = tuple(int(i) for i in np.unravel_index(bad_flat, arr.shape))
        bad_value = arr[index]
        raise ValueError(
            f"{label} must contain only binary 0 or 1 values; first invalid value "
            f"at index {index}: {bad_value!r}"
        )
    return arr


def validate_gripper_binary_step(
    value: Any, *, is_biarm: bool, label: str
) -> np.ndarray:
    """Validate one scalar or per-arm ``gripper_binary`` command."""
    arr = require_finite_real_array(value, label)
    max_channels = 2 if is_biarm else 1
    if arr.ndim == 0:
        channels = 1
    elif arr.ndim == 1:
        channels = arr.shape[0]
    else:
        raise ValueError(
            f"{label} must be a scalar or a one-dimensional command, got shape {arr.shape}"
        )
    if not (1 <= channels <= max_channels):
        raise ValueError(
            f"{label} has {channels} command channels, but the "
            f"{'biarm' if is_biarm else 'single-arm'} profile permits at most "
            f"{max_channels}; got shape {arr.shape}"
        )
    if not np.all(np.isin(arr, (0, 1))):
        raise ValueError(f"{label} must contain only binary 0 or 1 values, got {arr!r}")
    return arr
