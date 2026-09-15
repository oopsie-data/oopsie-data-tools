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
    arr = _require_real_array(value, label)
    _reject_values(arr, ~np.isfinite(arr), label, "non-finite")
    return arr


def require_real_array_without_inf(value: Any, label: str) -> np.ndarray:
    """Like :func:`require_finite_real_array`, but NaN is allowed; ±inf still is not.

    For sensor streams where NaN marks a dropped reading.
    """
    arr = _require_real_array(value, label)
    _reject_values(
        arr, np.isinf(arr), label, "infinite",
        hint=" NaN may mark a missing reading, infinity may not.",
    )
    return arr


def _require_real_array(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"{label} must be a real numeric array, got dtype {arr.dtype}")
    return arr


def _reject_values(
    arr: np.ndarray, bad: np.ndarray, label: str, kind: str, hint: str = ""
) -> None:
    """Raise naming the count and first index of the ``bad`` entries, if there are any."""
    if not np.any(bad):
        return
    index = _first_index(bad)
    bad_value = arr[index] if index else arr[()]
    count = int(np.count_nonzero(bad))
    raise ValueError(
        f"{label} contains {count} {kind} value(s); first at index {index}: "
        f"{bad_value!r}.{hint}"
    )


def _first_index(mask: np.ndarray) -> tuple[int, ...]:
    """Index of the first ``True`` entry of ``mask``; ``()`` for a 0-d mask."""
    mask = np.asarray(mask)
    flat = int(np.flatnonzero(mask.ravel())[0])
    return tuple(int(i) for i in np.unravel_index(flat, mask.shape))


def _require_channels(
    label: str, channels: int, shape: tuple[int, ...], *, is_biarm: bool
) -> None:
    max_channels = 2 if is_biarm else 1
    if not (1 <= channels <= max_channels):
        raise ValueError(
            f"{label} has {channels} command channels, but the "
            f"{'biarm' if is_biarm else 'single-arm'} profile permits at most "
            f"{max_channels}; got shape {shape}"
        )


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
        sample_index = _first_index(~np.asarray(valid))
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
    if arr.ndim == 1:
        channels = 1
    elif arr.ndim == 2:
        channels = arr.shape[1]
    else:
        raise ValueError(
            f"{label} must have shape (T,), (T, 1)"
            f"{' or (T, 2)' if is_biarm else ''}; got {arr.shape}"
        )
    _require_channels(label, channels, arr.shape, is_biarm=is_biarm)

    valid = np.isin(arr, (0, 1))
    if not np.all(valid):
        index = _first_index(~valid)
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
    if arr.ndim == 0:
        channels = 1
    elif arr.ndim == 1:
        channels = arr.shape[0]
    else:
        raise ValueError(
            f"{label} must be a scalar or a one-dimensional command, got shape {arr.shape}"
        )
    _require_channels(label, channels, arr.shape, is_biarm=is_biarm)
    if not np.all(np.isin(arr, (0, 1))):
        raise ValueError(f"{label} must contain only binary 0 or 1 values, got {arr!r}")
    return arr
