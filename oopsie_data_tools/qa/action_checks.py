"""Action-space data quality checks: read real per-timestep ``actions/*`` and
``observations/robot_states/*`` arrays and flag suspicious episodes.

These load full per-timestep arrays, so they are a separate pass (``oopsie-data qa``)
rather than part of ``oopsie-data validate``. A finding is a likely problem, not a schema
violation.

Checks are grouped into categories matching the sections of :data:`config.DEFAULT_CHECKS`:
``general`` (any action space) plus one per arm action space (``joint_position``,
``joint_velocity``, ``cartesian_position``, ``cartesian_velocity``). Every finding/skip
carries its ``category`` and ``test``.
"""

from __future__ import annotations

import json
import re
from itertools import combinations

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from oopsie_data_tools.utils.robot_profile.rotation_utils import RotOption

# Position-type keys for the general scale/tracking/delta checks. Excludes
# joint_position and cartesian_position (their own categories check them per joint / per
# position-and-rotation, in real units), base_position (the format allows deltas there)
# and all *_velocity / gripper_binary keys (derivative-like by nature).
GENERAL_POSITION_KEYS = ("gripper_position",)

# Arm position keys whose action can be compared value-for-value with the same-named state.
# Not gripper_position: many grippers have no position sensor and report the last command
# as their state, so an exact match there is expected, not a sign of rebuilt actions.
COPY_CHECK_KEYS = ("joint_position", "cartesian_position")

# The canonical action keys (``VALID_ACTION_KEYS``); undeclared ones are stored as h5py.Empty.
ALL_ACTION_KEYS = (
    "joint_position", "joint_velocity", "cartesian_position", "cartesian_velocity",
    "gripper_position", "gripper_velocity", "gripper_binary",
    "base_position", "base_velocity",
)

# The state each action key is observed through (velocity commands via the position state).
STATE_KEY_FOR_ACTION = {
    "joint_position": "joint_position",
    "joint_velocity": "joint_position",
    "cartesian_position": "cartesian_position",
    "cartesian_velocity": "cartesian_position",
    "gripper_position": "gripper_position",
    "gripper_velocity": "gripper_position",
    "gripper_binary": "gripper_position",
    "base_position": "base_position",
    "base_velocity": "base_position",
}

# Width of each orientation representation in a flat pose vector.
_ROT_WIDTH = {RotOption.QUAT: 4, RotOption.MATRIX: 9, RotOption.ROT6D: 6, RotOption.ROTVEC: 3}


def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        return _decode(v.flat[0]) if v.size else ""
    return str(v)


def load_action_state_arrays(
    f: h5py.File,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Read every non-Empty ``actions/<key>`` and ``observations/robot_states/<key>``
    as float64 ``(T, D)``. Returns ``(actions, states, robot_profile)``, the profile
    parsed from the ``robot_profile`` attr JSON (``{}`` if missing or unparseable).
    """
    try:
        profile = json.loads(_decode(f.attrs.get("robot_profile", "{}")))
    except (json.JSONDecodeError, TypeError):
        profile = {}
    if not isinstance(profile, dict):
        profile = {}

    actions: dict[str, np.ndarray] = {}
    actions_group = f.get("actions")
    if isinstance(actions_group, h5py.Group):
        for k in ALL_ACTION_KEYS:
            ds = actions_group.get(k)
            if isinstance(ds, h5py.Dataset) and ds.shape is not None:  # skip h5py.Empty
                actions[k] = np.asarray(ds[()], dtype=np.float64)

    states: dict[str, np.ndarray] = {}
    obs_group = f.get("observations")
    robot_states_group = obs_group.get("robot_states") if isinstance(obs_group, h5py.Group) else None
    if isinstance(robot_states_group, h5py.Group):
        for k in robot_states_group:
            ds = robot_states_group[k]
            if isinstance(ds, h5py.Dataset) and ds.shape is not None:
                states[k] = np.asarray(ds[()], dtype=np.float64)

    return actions, states, profile


def _finding(category: str, test: str, key: str, **details) -> dict:
    return {"category": category, "test": test, "key": key, **details}


def _skip(category: str, test: str, key: str, reason: str, **details) -> dict:
    return {"category": category, "test": test, "key": key, "reason": reason, **details}


def _params(check_cfg: dict) -> dict:
    """A check's config minus its ``enabled`` toggle, for passing as keyword arguments."""
    return {k: v for k, v in check_cfg.items() if k != "enabled"}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


# ── general ───────────────────────────────────────────────────────────────────


def find_duplicate_action_pairs(
    actions: dict[str, np.ndarray],
    declared_keys: list[str],
    *,
    atol: float = 1e-9,
    rtol: float = 0.0,
) -> list[dict]:
    """Compare every same-shaped pair of declared action arrays for exact/near-exact
    equality. One entry per *compared* pair (mismatched shapes are never compared,
    not reported as "not flagged")."""
    results = []
    keys = sorted({k for k in declared_keys if k in actions})
    for key_a, key_b in combinations(keys, 2):
        a, b = actions[key_a], actions[key_b]
        if a.shape != b.shape:
            continue
        max_abs_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        flagged = bool(np.allclose(a, b, atol=atol, rtol=rtol))
        results.append(
            {
                "key_a": key_a,
                "key_b": key_b,
                "shape": list(a.shape),
                "flagged": flagged,
                "max_abs_diff": max_abs_diff,
            }
        )
    return results


def _align_T(action: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncate both to ``min(T_action, T_state)`` (a mismatch is reported separately by
    the general ``length_mismatch`` check)."""
    t = min(action.shape[0], state.shape[0])
    return action[:t], state[:t]


def scale_consistency(action: np.ndarray, state_next: np.ndarray) -> dict:
    """``RMS(action) / RMS(state_next)`` - no thresholding here, caller applies
    config bounds."""
    action_rms = _rms(action)
    state_rms = _rms(state_next)
    ratio = action_rms / state_rms if state_rms > 0 else float("inf")
    return {"action_rms": action_rms, "state_rms": state_rms, "ratio": ratio}


def delta_vs_absolute_residuals(action: np.ndarray, state: np.ndarray) -> dict:
    """``resid_abs[t] = action[t] - state[t+1]`` (absolute-action hypothesis);
    ``resid_delta[t] = action[t] - (state[t+1] - state[t])`` (delta-action
    hypothesis). Computed over ``t = 0..T-2`` (``action``/``state`` must already be
    T-aligned by the caller, e.g. via :func:`_align_T`). Returns the RMS of both
    plus ``ratio = RMS(resid_delta) / RMS(resid_abs)`` - this same ``resid_abs`` is
    also what the tracking-residual check (a separate, simpler plausibility check)
    uses; computing it once here serves both.
    """
    if action.shape[0] < 2:
        return {"resid_abs_rms": 0.0, "resid_delta_rms": 0.0, "ratio": float("nan"), "action_rms": 0.0}
    a = action[:-1]
    resid_abs_rms = _rms(a - state[1:])
    resid_delta_rms = _rms(a - (state[1:] - state[:-1]))
    ratio = resid_delta_rms / resid_abs_rms if resid_abs_rms > 0 else float("inf")
    return {
        "resid_abs_rms": resid_abs_rms,
        "resid_delta_rms": resid_delta_rms,
        "ratio": ratio,
        "action_rms": _rms(a),
    }


def action_copies_state(action: np.ndarray, state: np.ndarray, atol: float) -> dict:
    """Share of steps where the action equals the state at the same step (lag 0) or the
    next step (lag 1) to within ``atol`` - a sign the "actions" were rebuilt from recorded
    states rather than logged commands, since a real controller never tracks exactly.
    T-aligned inputs. Returns the lag with the higher share."""
    best = {"lag": 0, "matching_fraction": 0.0}
    for lag in (0, 1):
        a, s = (action, state) if lag == 0 else (action[:-1], state[1:])
        if not len(a):
            continue
        fraction = float(np.mean(np.max(np.abs(a - s), axis=1) <= atol))
        if fraction > best["matching_fraction"]:
            best = {"lag": lag, "matching_fraction": fraction}
    return best


def gripper_range_violation(values: np.ndarray, atol: float) -> dict | None:
    """Min/max of ``values`` if any falls outside ``[0, 1]`` by more than ``atol``."""
    if not values.size:
        return None
    lo, hi = float(np.min(values)), float(np.max(values))
    return {"min": lo, "max": hi} if lo < -atol or hi > 1.0 + atol else None


def gripper_command_outside_state(action: np.ndarray, state: np.ndarray, max_fraction: float, min_margin: float) -> dict | None:
    """How far the commanded gripper range exceeds the observed state range. The allowed
    margin is ``max_fraction`` of the state's scale (the larger of its span and its largest
    absolute value), but at least ``min_margin``. Returns details if exceeded, else ``None``."""
    lo, hi = float(np.min(state)), float(np.max(state))
    scale = max(hi - lo, abs(lo), abs(hi))
    outside = max(lo - float(np.min(action)), float(np.max(action)) - hi, 0.0)
    margin = max(max_fraction * scale, min_margin)
    if outside <= margin:
        return None
    return {
        "action_min": float(np.min(action)), "action_max": float(np.max(action)),
        "state_min": lo, "state_max": hi, "outside": outside, "margin": margin,
    }


def _run_general_checks(
    actions: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    declared: list[str],
    cfg: dict,
    findings: list[dict],
    skips: list[dict],
    checked: set[tuple[str, str]],
) -> None:
    category = "general"

    if cfg["duplicate_pairs"]["enabled"] and len(declared) >= 2:
        dup_cfg = cfg["duplicate_pairs"]
        checked.add((category, "duplicate_pairs"))
        for pair in find_duplicate_action_pairs(actions, declared, atol=dup_cfg["atol"], rtol=dup_cfg["rtol"]):
            if pair["flagged"]:
                findings.append(_finding(category, "duplicate_pairs", f"{pair['key_a']}+{pair['key_b']}", **pair))

    if cfg["length_mismatch"]["enabled"]:
        for key in declared:
            state_key = STATE_KEY_FOR_ACTION.get(key)
            if key not in actions or state_key not in states:
                continue
            checked.add((category, "length_mismatch"))
            action_T, state_T = actions[key].shape[0], states[state_key].shape[0]
            if action_T != state_T:
                findings.append(
                    _finding(category, "length_mismatch", key, state_key=state_key, action_T=action_T, state_T=state_T)
                )

    if cfg["action_copies_state"]["enabled"]:
        copy_cfg = cfg["action_copies_state"]
        for key in declared:
            if key not in COPY_CHECK_KEYS or key not in actions or key not in states:
                continue
            action, state = actions[key], states[key]
            if action.ndim != 2 or action.shape[1:] != state.shape[1:]:
                skips.append(_skip(category, "action_copies_state", key, "feature_dim_mismatch"))
                continue
            if not (np.all(np.isfinite(action)) and np.all(np.isfinite(state))):
                skips.append(_skip(category, "action_copies_state", key, "non_finite_values"))
                continue
            action_t, state_t = _align_T(action, state)
            # A robot that never moves matches a constant action trivially.
            if not len(state_t) or np.max(np.ptp(state_t, axis=0)) <= copy_cfg["atol"]:
                skips.append(_skip(category, "action_copies_state", key, "state_static"))
                continue
            checked.add((category, "action_copies_state"))
            copy = action_copies_state(action_t, state_t, copy_cfg["atol"])
            if copy["matching_fraction"] >= copy_cfg["min_fraction"]:
                findings.append(_finding(category, "action_copies_state", key, **copy))

    if cfg["gripper_binary_present"]["enabled"]:
        checked.add((category, "gripper_binary_present"))
        if "gripper_binary" not in actions:
            findings.append(
                _finding(
                    category, "gripper_binary_present", "gripper_binary",
                    declared_gripper_keys=[k for k in declared if k.startswith("gripper")],
                    recorded_gripper_keys=sorted(k for k in actions if k.startswith("gripper")),
                )
            )

    if cfg["gripper_range"]["enabled"] and "gripper_binary" in declared:
        key = "gripper_binary"
        if key not in actions:
            skips.append(_skip(category, "gripper_range", key, "declared_but_empty"))
        elif not np.all(np.isfinite(actions[key])):
            skips.append(_skip(category, "gripper_range", key, "non_finite_values"))
        else:
            checked.add((category, "gripper_range"))
            result = gripper_range_violation(actions[key], cfg["gripper_range"]["atol"])
            if result is not None:
                findings.append(_finding(category, "gripper_range", key, **result))

    if cfg["gripper_command_range"]["enabled"] and "gripper_position" in declared:
        key = "gripper_position"
        if key not in actions:
            skips.append(_skip(category, "gripper_command_range", key, "declared_but_empty"))
        elif key not in states:
            skips.append(_skip(category, "gripper_command_range", key, "no_matching_state"))
        elif not (actions[key].size and states[key].size):
            skips.append(_skip(category, "gripper_command_range", key, "too_short"))
        elif not (np.all(np.isfinite(actions[key])) and np.all(np.isfinite(states[key]))):
            skips.append(_skip(category, "gripper_command_range", key, "non_finite_values"))
        else:
            checked.add((category, "gripper_command_range"))
            result = gripper_command_outside_state(actions[key], states[key], **_params(cfg["gripper_command_range"]))
            if result is not None:
                findings.append(_finding(category, "gripper_command_range", key, **result))

    tests = [t for t in ("scale_consistency", "tracking_residual", "delta_detection") if cfg[t]["enabled"]]
    if not tests:
        return

    for key in declared:
        if key not in GENERAL_POSITION_KEYS:
            continue

        def skip(reason: str, _key: str = key, **details) -> None:
            skips.extend(_skip(category, t, _key, reason, **details) for t in tests)

        if key not in actions:
            skip("declared_but_empty")
            continue
        if key not in states:
            skip("no_matching_state")
            continue

        action, state = actions[key], states[key]
        if action.shape[1:] != state.shape[1:]:
            # Different feature dimension (not just a different episode length) -
            # e.g. an action commands fewer/more DOF than the state reports for
            # some hand/gripper profiles. Elementwise comparison is meaningless
            # here; skip rather than guessing which columns correspond.
            skip("feature_dim_mismatch", action_shape=list(action.shape), state_shape=list(state.shape))
            continue
        if not (np.all(np.isfinite(action)) and np.all(np.isfinite(state))):
            skip("non_finite_values")
            continue

        action_t, state_t = _align_T(action, state)
        if action_t.shape[0] < 2:
            skip("too_short")
            continue

        checked.update((category, t) for t in tests)
        if "scale_consistency" in tests:
            scale_cfg = cfg["scale_consistency"]
            sc = scale_consistency(action_t, state_t)
            if sc["ratio"] > scale_cfg["max_ratio"] or sc["ratio"] < scale_cfg["min_ratio"]:
                findings.append(_finding(category, "scale_consistency", key, **sc))

        if "tracking_residual" in tests or "delta_detection" in tests:
            dv = delta_vs_absolute_residuals(action_t, state_t)
            if "tracking_residual" in tests and dv["action_rms"] > 0:
                normalized_rms = dv["resid_abs_rms"] / dv["action_rms"]
                if normalized_rms > cfg["tracking_residual"]["max_normalized_rms"]:
                    findings.append(_finding(category, "tracking_residual", key, normalized_rms=normalized_rms, **dv))
            # action_rms == 0 (a constant/empty action channel) makes resid_delta
            # trivially small relative to resid_abs purely because there's no real
            # signal to compare - that's a "degenerate action" problem, already
            # caught distinctly by scale_consistency's min_ratio, not a delta
            # encoding. Don't double-report it here under the wrong label.
            if (
                "delta_detection" in tests
                and dv["action_rms"] > 0
                and dv["ratio"] < cfg["delta_detection"]["ratio_threshold"]
            ):
                findings.append(_finding(category, "delta_detection", key, **dv))


# ── cartesian_position ────────────────────────────────────────────────────────


def _looks_scalar_first(quat: np.ndarray) -> bool:
    """Heuristic only - unit norm is invariant to component order, so this can't be a
    hard guarantee. Across most real trajectories (small/moderate rotations away from
    identity), the scalar (w) component stays closer to +-1 with lower variance than the
    vector (x, y, z) components, which center near 0. Compares how "scalar-like" index 0
    looks vs index 3 of the 4-slot quaternion; True suggests ``[qw, qx, qy, qz]`` was
    written where scalar-last ``[qx, qy, qz, qw]`` was required.
    """
    abs_mean = np.mean(np.abs(quat), axis=0)
    std = np.std(quat, axis=0)
    scalar_likeness = abs_mean - std  # higher = closer to a near-constant +-1 value
    return bool(scalar_likeness[0] > scalar_likeness[3])


def validate_cartesian_quaternion_shape(action: np.ndarray, *, is_biarm: bool, norm_atol: float = 1e-3) -> dict:
    """Check a ``cartesian_position`` action array against the mandated per-arm
    ``[x, y, z, qx, qy, qz, qw]`` (scalar-last) layout: dim must be 7 (single-arm) or 14
    (biarm), and each arm's 4-slot quaternion must be unit-norm. Mirrors the upload-time
    validator's ``validate_cartesian_quaternions`` (oopsie-data-tools), re-checked here
    independently since that validator only ever ran on files that went through it.

    Returns ``{"ok": True}`` or ``{"ok": False, "reason": ..., **details}``.
    """
    expected_dim = 14 if is_biarm else 7
    dim = action.shape[-1] if action.ndim else 0
    if dim != expected_dim:
        return {"ok": False, "reason": "wrong_dim", "expected_dim": expected_dim, "actual_dim": dim}

    starts = (3, 10) if is_biarm else (3,)
    bad_arms = []
    for arm_index, start in enumerate(starts, 1):
        quat = action[..., start : start + 4]
        norms = np.linalg.norm(quat, axis=-1)
        valid = np.isclose(norms, 1.0, rtol=0.0, atol=norm_atol)
        if not np.all(valid):
            bad_norms = norms[~valid]
            bad_arms.append(
                {"arm": arm_index, "min_norm": float(np.min(bad_norms)), "max_norm": float(np.max(bad_norms))}
            )
    if bad_arms:
        return {"ok": False, "reason": "bad_quaternion_norm", "bad_arms": bad_arms}
    return {"ok": True}


def check_quaternion_scalar_order(arm_quats: list[np.ndarray]) -> list[int]:
    """Heuristic only (see :func:`_looks_scalar_first`) - arms whose quaternion looks
    like it was written scalar-first (``[qw, qx, qy, qz]``) instead of the mandated
    scalar-last (``[qx, qy, qz, qw]``). Unreliable in general (unit norm is invariant to
    component order, so this can never be a hard guarantee); kept as its own toggle so it
    can be turned off without losing the dim/norm checks in ``quaternion_shape``."""
    return [arm for arm, q in enumerate(arm_quats, 1) if q.shape[0] >= 2 and _looks_scalar_first(q)]


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row rotation angle between unit quaternions ``a`` and ``b`` (degrees),
    sign-invariant (q and -q are the same rotation)."""
    dots = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def _rot_width(option: RotOption) -> int:
    return _ROT_WIDTH.get(option, 3)  # every Euler order is 3 wide


def _state_rotation_to_quat(rot: np.ndarray, declared: str | None) -> np.ndarray | None:
    """Convert one arm's ``(T, k)`` state rotation slice to scalar-last quaternions.

    The recorder converts states with a declared ``robot_state_orientation_representation``
    to quaternions on write, so a 4-wide slice is taken as a quaternion whatever the profile
    says; any other width is converted using the declared representation if its width
    matches. ``None`` when the representation can't be determined or converted.
    """
    width = rot.shape[1]
    option = None
    if declared:
        try:
            parsed = RotOption.from_string(declared)
        except ValueError:
            parsed = None
        if parsed is not None and _rot_width(parsed) == width:
            option = parsed
    if option is None and width == 4:
        option = RotOption.QUAT
    if option is None:
        return None

    try:
        if option == RotOption.QUAT:
            norms = np.linalg.norm(rot, axis=1, keepdims=True)
            return rot / norms if np.all(norms > 0) else None
        if option == RotOption.MATRIX:
            return Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat()
        if option == RotOption.ROT6D:
            # First two rotation-matrix columns; rebuild the third by Gram-Schmidt
            # (same convention as oopsie-data-tools' ActionQuatConversion).
            cols = rot.reshape(-1, 2, 3)
            u1 = cols[:, 0] / np.linalg.norm(cols[:, 0], axis=1, keepdims=True)
            u2 = cols[:, 1] - np.sum(cols[:, 1] * u1, axis=1, keepdims=True) * u1
            u2 /= np.linalg.norm(u2, axis=1, keepdims=True)
            return Rotation.from_matrix(np.stack([u1, u2, np.cross(u1, u2)], axis=-1)).as_quat()
        if option == RotOption.ROTVEC:
            return Rotation.from_rotvec(rot).as_quat()
        return Rotation.from_euler(option.name, rot).as_quat()
    except ValueError:
        return None


# Smaller units a metre position is commonly mistaken for, and their size in metres.
_UNIT_FACTORS = {"cm": 100.0, "mm": 1000.0}


def _unit_from_ratio(ratio: float) -> str | None:
    """Unit whose metre factor is within half a decade of the action/state scale ratio
    (so ~32-316x reads as cm, ~316-3162x as mm), assuming the state is in metres."""
    for unit, factor in _UNIT_FACTORS.items():
        if abs(np.log10(ratio / factor)) < 0.5:
            return unit
    return None


def _unit_from_reach(max_abs: float, reach_min_m: float, reach_max_m: float) -> str | None:
    """Unit that, converted to metres, puts ``max_abs`` within a typical arm reach;
    when both do, the one landing nearest the middle of that range (log scale)."""
    center = np.sqrt(reach_min_m * reach_max_m)
    fits = {
        unit: abs(np.log(max_abs / factor / center))
        for unit, factor in _UNIT_FACTORS.items()
        if reach_min_m <= max_abs / factor <= reach_max_m
    }
    return min(fits, key=fits.get) if fits else None


def check_position_units(
    action_xyz: np.ndarray,
    state_xyz: np.ndarray | None,
    *,
    max_abs_position_m: float,
    max_action_state_ratio: float,
    typical_reach_min_m: float,
    typical_reach_max_m: float,
) -> dict:
    """Positions must be in metres: flags action (and state) xyz beyond
    ``max_abs_position_m`` - mm/cm values or a far-away world frame - and an action xyz
    RMS more than ``max_action_state_ratio`` times the state's (action in smaller units
    than the state). The low side is left to delta detection, since delta-encoded
    actions are also much smaller than the state.

    Each flagged side gets a ``*_suspected_unit`` guess (``"cm"``, ``"mm"`` or ``None``):
    from the action/state ratio when the state looks like metres, otherwise from which
    unit puts the values within a typical arm reach.

    ``*_xyz`` hold every arm's xyz columns. Returns ``{"ok": bool, "reasons": [...], ...}``.
    """
    reasons = []
    result: dict = {"action_max_abs_m": float(np.max(np.abs(action_xyz)))}
    if result["action_max_abs_m"] > max_abs_position_m:
        reasons.append("action_out_of_range")
    state_in_metres = False
    if state_xyz is not None:
        result["state_max_abs_m"] = float(np.max(np.abs(state_xyz)))
        if result["state_max_abs_m"] > max_abs_position_m:
            reasons.append("state_out_of_range")
        else:
            state_in_metres = True
        state_rms = _rms(state_xyz)
        result["action_state_ratio"] = _rms(action_xyz) / state_rms if state_rms > 0 else float("inf")
        if result["action_state_ratio"] > max_action_state_ratio:
            reasons.append("action_state_scale_mismatch")

    if "action_out_of_range" in reasons or "action_state_scale_mismatch" in reasons:
        if state_in_metres and np.isfinite(result["action_state_ratio"]):
            result["action_suspected_unit"] = _unit_from_ratio(result["action_state_ratio"])
        else:
            result["action_suspected_unit"] = _unit_from_reach(
                result["action_max_abs_m"], typical_reach_min_m, typical_reach_max_m
            )
    if "state_out_of_range" in reasons:
        result["state_suspected_unit"] = _unit_from_reach(
            result["state_max_abs_m"], typical_reach_min_m, typical_reach_max_m
        )
    return {"ok": not reasons, "reasons": reasons, **result}


def check_quaternion_sign_flips(arm_quats: list[np.ndarray]) -> list[dict]:
    """Per arm, steps where ``dot(q[t], q[t+1]) < 0``: the same orientation written
    with the opposite sign, which breaks learning on raw quaternion values."""
    flagged = []
    for arm, q in enumerate(arm_quats, 1):
        flips = np.flatnonzero(np.sum(q[1:] * q[:-1], axis=1) < 0)
        if flips.size:
            flagged.append({"arm": arm, "count": int(flips.size), "first_step": int(flips[0])})
    return flagged


def check_rotation_jumps(arm_quats: list[np.ndarray], max_step_deg: float) -> list[dict]:
    """Per arm, steps whose rotation to the next step exceeds ``max_step_deg``."""
    flagged = []
    for arm, q in enumerate(arm_quats, 1):
        angles = _quat_angle_deg(q[1:], q[:-1])
        jumps = np.flatnonzero(angles > max_step_deg)
        if jumps.size:
            flagged.append(
                {
                    "arm": arm,
                    "count": int(jumps.size),
                    "first_step": int(jumps[0]),
                    "max_step_deg": float(np.max(angles)),
                }
            )
    return flagged


def rotation_tracking_errors(action_quats: list[np.ndarray], state_quats: list[np.ndarray]) -> list[dict]:
    """Per arm, angle between ``action[t]`` and ``state[t+1]`` (T-aligned inputs)."""
    errors = []
    for arm, (a, s) in enumerate(zip(action_quats, state_quats), 1):
        angles = _quat_angle_deg(a[:-1], s[1:])
        errors.append(
            {
                "arm": arm,
                "median_error_deg": float(np.median(angles)),
                "mean_error_deg": float(np.mean(angles)),
                "max_error_deg": float(np.max(angles)),
            }
        )
    return errors


def position_tracking_errors(action_xyz: list[np.ndarray], state_xyz: list[np.ndarray]) -> list[dict]:
    """Per arm, distance (metres) between ``action[t]`` and ``state[t+1]`` (T-aligned inputs) -
    the position half of :func:`rotation_tracking_errors`'s pose-tracking pair."""
    errors = []
    for arm, (a, s) in enumerate(zip(action_xyz, state_xyz), 1):
        dist = np.linalg.norm(a[:-1] - s[1:], axis=-1)
        errors.append(
            {
                "arm": arm,
                "median_error_m": float(np.median(dist)),
                "mean_error_m": float(np.mean(dist)),
                "max_error_m": float(np.max(dist)),
            }
        )
    return errors


def biarm_swap_ratio(action_xyz: list[np.ndarray], state_xyz: list[np.ndarray]) -> dict:
    """Position error of action[t] vs state[t+1] with the arms as given and with them
    swapped (T-aligned inputs). ``ratio = swapped / same``: well below 1 means the
    action's arm order is reversed relative to the state's."""
    a1, a2 = action_xyz[0][:-1], action_xyz[1][:-1]
    s1, s2 = state_xyz[0][1:], state_xyz[1][1:]
    same = _rms(np.concatenate([a1 - s1, a2 - s2]))
    swapped = _rms(np.concatenate([a1 - s2, a2 - s1]))
    ratio = swapped / same if same > 0 else float("inf")
    return {"same_arm_rms_m": same, "swapped_arm_rms_m": swapped, "ratio": ratio}


def _run_cartesian_position_checks(
    actions: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    cfg: dict,
    findings: list[dict],
    skips: list[dict],
    checked: set[tuple[str, str]],
    *,
    is_biarm: bool,
    control_freq: float | None,
    state_orientation: str | None,
) -> None:
    category = key = "cartesian_position"
    tests = [t for t, c in cfg.items() if c["enabled"]]

    def skip(reason: str, only: list[str] | None = None, **details) -> None:
        skips.extend(_skip(category, t, key, reason, **details) for t in (tests if only is None else only))

    if not tests:
        return
    if key not in actions:
        skip("declared_but_empty")
        return
    action = actions[key]
    if not np.all(np.isfinite(action)):
        skip("non_finite_values")
        return

    if "quaternion_shape" in tests:
        checked.add((category, "quaternion_shape"))
        qcfg = cfg["quaternion_shape"]
        shape_check = validate_cartesian_quaternion_shape(action, is_biarm=is_biarm, norm_atol=qcfg["norm_atol"])
        if not shape_check["ok"]:
            findings.append(_finding(category, "quaternion_shape", key, **shape_check))
        tests.remove("quaternion_shape")
    if not tests:
        return

    # Everything below relies on the mandated per-arm [x, y, z, qx, qy, qz, qw] layout;
    # a wrong width is already reported by quaternion_shape.
    n_arms = 2 if is_biarm else 1
    if action.ndim != 2 or action.shape[1] != 7 * n_arms:
        skip("wrong_action_dim", actual_shape=list(action.shape))
        return
    if action.shape[0] < 2:
        skip("too_short")
        return
    action_xyz = [action[:, 7 * a : 7 * a + 3] for a in range(n_arms)]
    action_quats = [action[:, 7 * a + 3 : 7 * a + 7] for a in range(n_arms)]
    # pose_tracking's orientation half needs a normalized action quaternion too, same as
    # the other rotation checks - a zero quaternion (already caught separately by
    # quaternion_shape's norm check) makes all of them meaningless, not just these two.
    rotation_tests = [
        t for t in ("quaternion_sign_flip", "rotation_jump", "pose_tracking", "quaternion_scalar_order") if t in tests
    ]
    if any(np.any(np.linalg.norm(q, axis=1) == 0) for q in action_quats):
        skip("zero_quaternion", rotation_tests)
        tests = [t for t in tests if t not in rotation_tests]
    else:
        action_quats = [q / np.linalg.norm(q, axis=1, keepdims=True) for q in action_quats]

    # Action-only checks.
    if "normalized_range" in tests:
        checked.add((category, "normalized_range"))
        xyz = np.concatenate(action_xyz, axis=1)
        if looks_clipped_to_unit_range(xyz):
            findings.append(_finding(category, "normalized_range", key, max_abs=float(np.max(np.abs(xyz)))))

    if "quaternion_sign_flip" in tests:
        checked.add((category, "quaternion_sign_flip"))
        flips = check_quaternion_sign_flips(action_quats)
        if flips:
            findings.append(_finding(category, "quaternion_sign_flip", key, arms=flips))

    if "rotation_jump" in tests:
        if not control_freq or control_freq <= 0:
            skip("no_control_freq", ["rotation_jump"])
        else:
            checked.add((category, "rotation_jump"))
            max_step_deg = cfg["rotation_jump"]["max_angular_speed_deg_s"] / control_freq
            jumps = check_rotation_jumps(action_quats, max_step_deg)
            if jumps:
                findings.append(
                    _finding(category, "rotation_jump", key, limit_step_deg=max_step_deg, control_freq=control_freq, arms=jumps)
                )

    if "quaternion_scalar_order" in tests:
        checked.add((category, "quaternion_scalar_order"))
        scalar_first_arms = check_quaternion_scalar_order(action_quats)
        if scalar_first_arms:
            findings.append(_finding(category, "quaternion_scalar_order", key, arms=scalar_first_arms))

    # Checks against the state; position_units still runs its action-only part without one.
    state_tests = [t for t in ("pose_tracking", "delta_detection", "biarm_ordering") if t in tests]
    if not is_biarm and "biarm_ordering" in state_tests:
        state_tests.remove("biarm_ordering")  # not applicable, not a skip

    state = states.get(key)
    state_problem = None
    if state is None:
        state_problem = {"reason": "no_matching_state"}
    elif state.ndim != 2 or state.shape[1] % n_arms or state.shape[1] // n_arms < 3:
        state_problem = {"reason": "unexpected_state_dim", "state_shape": list(state.shape)}
    elif not np.all(np.isfinite(state)):
        state_problem = {"reason": "non_finite_state"}
    if state_problem is not None:
        skip(only=state_tests, **state_problem)
        if "position_units" in tests:
            checked.add((category, "position_units"))
            units = check_position_units(
                np.concatenate(action_xyz, axis=1), None,
                **_params(cfg["position_units"]),
            )
            if not units["ok"]:
                findings.append(_finding(category, "position_units", key, **units))
        return

    arm_width = state.shape[1] // n_arms
    t = min(action.shape[0], state.shape[0])
    action_xyz = [x[:t] for x in action_xyz]
    action_quats = [q[:t] for q in action_quats]
    state = state[:t]
    state_xyz = [state[:, arm_width * a : arm_width * a + 3] for a in range(n_arms)]

    if "position_units" in tests:
        checked.add((category, "position_units"))
        units = check_position_units(
            np.concatenate(action_xyz, axis=1), np.concatenate(state_xyz, axis=1),
            **_params(cfg["position_units"]),
        )
        if not units["ok"]:
            findings.append(_finding(category, "position_units", key, **units))

    if t < 2:
        skip("too_short", state_tests)
        return

    if "delta_detection" in state_tests:
        checked.add((category, "delta_detection"))
        dv = delta_vs_absolute_residuals(np.concatenate(action_xyz, axis=1), np.concatenate(state_xyz, axis=1))
        if dv["action_rms"] > 0 and dv["ratio"] < cfg["delta_detection"]["ratio_threshold"]:
            findings.append(_finding(category, "delta_detection", key, **dv))

    if "biarm_ordering" in state_tests:
        checked.add((category, "biarm_ordering"))
        swap = biarm_swap_ratio(action_xyz, state_xyz)
        if swap["ratio"] < cfg["biarm_ordering"]["max_swap_ratio"]:
            findings.append(_finding(category, "biarm_ordering", key, **swap))

    if "pose_tracking" in state_tests:
        pose_cfg = cfg["pose_tracking"]
        position_limit = pose_cfg["max_median_position_error_m"]
        orientation_limit = pose_cfg["max_median_orientation_error_deg"]

        position_errors = position_tracking_errors(action_xyz, state_xyz)
        position_flagged = any(e["median_error_m"] > position_limit for e in position_errors)

        state_quats = [
            _state_rotation_to_quat(state[:, arm_width * a + 3 : arm_width * (a + 1)], state_orientation)
            for a in range(n_arms)
        ]
        orientation_errors = None
        orientation_flagged = False
        if any(q is None for q in state_quats):
            # Position still ran above - only the orientation half is unavailable here.
            skip(
                "unknown_state_orientation",
                ["pose_tracking"],
                state_rotation_width=arm_width - 3,
                declared=state_orientation,
            )
        else:
            orientation_errors = rotation_tracking_errors(action_quats, state_quats)
            orientation_flagged = any(e["median_error_deg"] > orientation_limit for e in orientation_errors)

        checked.add((category, "pose_tracking"))
        if position_flagged or orientation_flagged:
            findings.append(
                _finding(
                    category,
                    "pose_tracking",
                    key,
                    position_limit_m=position_limit,
                    orientation_limit_deg=orientation_limit,
                    position_flagged=position_flagged,
                    orientation_flagged=orientation_flagged,
                    position_arms=position_errors,
                    orientation_arms=orientation_errors,
                )
            )


# ── joint_position ────────────────────────────────────────────────────────────

# Which arm a joint name belongs to ("left_joint1", "waist_left", "l_joint_1", "follower_l/joint1").
_SIDE_PATTERNS = {
    "left": re.compile(r"(?:^|[^a-z])(?:left|l)(?:[^a-z]|$)"),
    "right": re.compile(r"(?:^|[^a-z])(?:right|r)(?:[^a-z]|$)"),
}
_LAST_NUMBER = re.compile(r"\d+(?!.*\d)")

# Where a joint sits along the arm (base -> wrist), by name keyword, for UR, ViperX/ALOHA
# and SO-100/101 naming. Matched in this order, so specific keywords win over "shoulder".
_CHAIN_KEYWORDS = (
    ("shoulder_pan", 0),
    ("waist", 0),
    ("shoulder_lift", 1),
    ("forearm_roll", 3),
    ("wrist_angle", 4),
    ("wrist_flex", 4),
    ("wrist_1", 4),
    ("wrist_2", 5),
    ("wrist_rotate", 6),
    ("wrist_roll", 6),
    ("wrist_3", 6),
    ("shoulder", 1),
    ("elbow", 2),
)


def _side(name: str) -> str:
    lowered = name.lower()
    for side, pattern in _SIDE_PATTERNS.items():
        if pattern.search(lowered):
            return side
    return ""


def _is_gripper_joint(name: str) -> bool:
    return "gripper" in name.lower()


def _name_list_problems(names: list[str]) -> list[str]:
    """Problems within one joint-name list: blank or duplicate names, numbered joints out
    of order (``joint3`` before ``joint2``), or standard joints out of base-to-wrist order
    (``elbow`` before ``shoulder_lift`` - e.g. a ROS ``/joint_states`` alphabetical list)."""
    problems = []
    if any(not n.strip() for n in names):
        problems.append("blank_names")
    if len(set(names)) != len(names):
        problems.append("duplicate_names")

    last_number: dict[tuple[str, str], int] = {}
    numbering_ok = True
    last_rank: dict[str, int] = {}
    chain_ok = True
    for name in names:
        side = _side(name)
        match = _LAST_NUMBER.search(name)
        if match:
            group = (side, name[: match.start()] + "#" + name[match.end() :])
            number = int(match.group())
            if group in last_number and number <= last_number[group]:
                numbering_ok = False
            last_number[group] = number
        lowered = name.lower()
        rank = next((r for keyword, r in _CHAIN_KEYWORDS if keyword in lowered), None)
        if rank is not None:
            if side in last_rank and rank < last_rank[side]:
                chain_ok = False
            last_rank[side] = rank
    if not numbering_ok:
        problems.append("numbering_out_of_order")
    if not chain_ok:
        problems.append("chain_out_of_order")
    return problems


def check_joint_names(
    action_names: list[str], state_names: list[str], *, has_joint_state: bool, is_biarm: bool
) -> dict:
    """The profile's ``action_joint_names``/``robot_state_joint_names`` must be present and
    plausible: see :func:`_name_list_problems`, plus every action joint appearing in the
    state, in the same relative order, and biarm names saying which arm each joint is on.
    Returns ``{"ok": bool, "reasons": [...], ...}``."""
    reasons: list[str] = []
    details: dict = {}
    if not action_names:
        reasons.append("missing_action_names")
    if has_joint_state and not state_names:
        reasons.append("missing_state_names")
    reasons += [f"action_{p}" for p in _name_list_problems(action_names)]
    reasons += [f"state_{p}" for p in _name_list_problems(state_names)]

    if action_names and state_names:
        missing = [n for n in action_names if n not in state_names]
        if missing:
            reasons.append("action_names_not_in_state")
            details["not_in_state"] = missing
        elif [n for n in state_names if n in set(action_names)] != action_names:
            reasons.append("reordered_vs_state")
    if is_biarm and action_names and not {"left", "right"} <= {_side(n) for n in action_names}:
        reasons.append("biarm_sides_not_named")
    return {"ok": not reasons, "reasons": reasons, **details}


def _joint_pairing(
    action_names: list[str], state_names: list[str], action_width: int, state_width: int
) -> list[tuple[int, int]] | None:
    """``(action column, state column)`` pairs: by name when both name lists match their
    arrays and cover every action joint, else by position when the widths agree."""
    if (
        len(action_names) == action_width
        and len(state_names) == state_width
        and len(set(state_names)) == len(state_names)
        and all(n in state_names for n in action_names)
    ):
        return [(i, state_names.index(n)) for i, n in enumerate(action_names)]
    if action_width == state_width:
        return [(i, i) for i in range(action_width)]
    return None


def _run_joint_position_checks(
    actions: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    cfg: dict,
    findings: list[dict],
    skips: list[dict],
    checked: set[tuple[str, str]],
    *,
    is_biarm: bool,
    control_freq: float | None,
    action_names: list[str],
    state_names: list[str],
) -> None:
    category = key = "joint_position"
    tests = [t for t, c in cfg.items() if c["enabled"]]

    def skip(reason: str, only: list[str] | None = None, **details) -> None:
        skips.extend(_skip(category, t, key, reason, **details) for t in (tests if only is None else only))

    state = states.get("joint_position")

    # Profile-only check: runs even when the action array is unusable.
    if "joint_names" in tests:
        checked.add((category, "joint_names"))
        names = check_joint_names(action_names, state_names, has_joint_state=state is not None, is_biarm=is_biarm)
        if not names["ok"]:
            findings.append(_finding(category, "joint_names", key, **names))
        tests.remove("joint_names")
    if not tests:
        return

    if key not in actions:
        skip("declared_but_empty")
        return
    action = actions[key]
    if action.ndim != 2 or not np.all(np.isfinite(action)):
        skip("non_finite_values" if action.ndim == 2 else "unexpected_action_dim")
        return
    width = action.shape[1]

    if "dof" in tests:
        checked.add((category, "dof"))
        problems = {}
        if action_names and len(action_names) != width:
            problems["action"] = {"width": width, "names": len(action_names)}
        if state is not None and state.ndim == 2:
            if state_names and len(state_names) != state.shape[1]:
                problems["state"] = {"width": state.shape[1], "names": len(state_names)}
            if not action_names and not state_names and state.shape[1] != width:
                problems["action_vs_state"] = {"action_width": width, "state_width": state.shape[1]}
        if problems:
            findings.append(_finding(category, "dof", key, **problems))

    names_fit = len(action_names) == width
    gripper_cols = [i for i, n in enumerate(action_names) if _is_gripper_joint(n)] if names_fit else []
    if "gripper_in_joints" in tests:
        if not names_fit:
            skip("no_usable_joint_names", ["gripper_in_joints"])
        else:
            checked.add((category, "gripper_in_joints"))
            if gripper_cols:
                findings.append(
                    _finding(category, "gripper_in_joints", key, joints=[action_names[i] for i in gripper_cols])
                )

    # Gripper columns are in their own units and saturate against objects, so they're left
    # out of the radian, jump and tracking math (gripper_in_joints reports them).
    cols = [i for i in range(width) if i not in gripper_cols]
    label = (lambda i: action_names[i]) if names_fit else (lambda i: f"col{i}")
    if not cols:
        skip("only_gripper_columns", [t for t in tests if t not in ("dof", "gripper_in_joints")])
        return

    if "normalized_range" in tests:
        checked.add((category, "normalized_range"))
        if looks_clipped_to_unit_range(action[:, cols]):
            findings.append(_finding(category, "normalized_range", key, max_abs=float(np.max(np.abs(action[:, cols])))))

    in_radians = True
    if "radians" in tests:
        checked.add((category, "radians"))
        limit = cfg["radians"]["max_abs_rad"]
        result: dict = {}
        arrays = {"action": action[:, cols]}
        if state is not None and state.ndim == 2 and np.all(np.isfinite(state)):
            arrays["state"] = state
        for side, values in arrays.items():
            max_abs = float(np.max(np.abs(values))) if values.size else 0.0
            if max_abs > limit:
                # LeRobot SO-100/101 arms store either degrees or a -100..100 range;
                # the two can't be told apart below 100.
                unit = "degrees_or_normalized_100" if max_abs <= 100.5 else "degrees" if max_abs <= 400 else None
                result[side] = {"max_abs": max_abs, "suspected_unit": unit}
        if result:
            in_radians = "action" not in result
            findings.append(_finding(category, "radians", key, limit_rad=limit, **result))

    if "step_jump" in tests:
        if not in_radians:
            skip("not_radians", ["step_jump"])
        elif not control_freq or control_freq <= 0:
            skip("no_control_freq", ["step_jump"])
        elif action.shape[0] < 2:
            skip("too_short", ["step_jump"])
        else:
            checked.add((category, "step_jump"))
            limit = cfg["step_jump"]["max_joint_speed_rad_s"] / control_freq
            steps = np.abs(np.diff(action[:, cols], axis=0))
            jumps = []
            for j, col in enumerate(cols):
                over = np.flatnonzero(steps[:, j] > limit)
                if over.size:
                    jumps.append(
                        {
                            "joint": label(col),
                            "count": int(over.size),
                            "first_step": int(over[0]),
                            "max_step_rad": float(np.max(steps[:, j])),
                        }
                    )
            if jumps:
                findings.append(
                    _finding(category, "step_jump", key, limit_step_rad=limit, control_freq=control_freq, joints=jumps)
                )

    # Checks against the state, joint by joint.
    state_tests = [t for t in ("delta_detection", "tracking") if t in tests]
    if not state_tests:
        return
    if state is None:
        skip("no_matching_state", state_tests)
        return
    if state.ndim != 2 or not np.all(np.isfinite(state)):
        skip("non_finite_state" if state.ndim == 2 else "unexpected_state_dim", state_tests)
        return
    pairing = _joint_pairing(action_names, state_names, width, state.shape[1])
    if pairing is None:
        skip("no_joint_pairing", state_tests, action_width=width, state_width=state.shape[1])
        return
    pairing = [(a, s) for a, s in pairing if a in cols]
    t = min(action.shape[0], state.shape[0])
    if t < 2:
        skip("too_short", state_tests)
        return
    paired_action = action[:t, [a for a, _ in pairing]]
    paired_state = state[:t, [s for _, s in pairing]]

    if "delta_detection" in state_tests:
        checked.add((category, "delta_detection"))
        dv = delta_vs_absolute_residuals(paired_action, paired_state)
        if dv["action_rms"] > 0 and dv["ratio"] < cfg["delta_detection"]["ratio_threshold"]:
            findings.append(_finding(category, "delta_detection", key, **dv))

    if "tracking" in state_tests:
        if not in_radians:
            skip("not_radians", ["tracking"])
        else:
            checked.add((category, "tracking"))
            limit = cfg["tracking"]["max_median_error_rad"]
            medians = np.median(np.abs(paired_action[:-1] - paired_state[1:]), axis=0)
            if np.max(medians) > limit:
                worst = np.argsort(medians)[::-1][:3]
                findings.append(
                    _finding(
                        category,
                        "tracking",
                        key,
                        limit_rad=limit,
                        worst_joints=[
                            {"joint": label(pairing[j][0]), "median_error_rad": float(medians[j])} for j in worst
                        ],
                    )
                )


# ── velocities (joint_velocity, cartesian_velocity) ───────────────────────────
#
# States only hold positions, so a velocity command is checked by integrating it over one
# control step (``v[t] / control_freq``) and comparing with how far the state moved.


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def velocity_integration(step: np.ndarray, moved: np.ndarray) -> dict:
    """Compare per-step commanded motion ``step`` (``v / control_freq``, ``(T, D)``) with
    the state's per-step motion ``moved`` (``(T-1, D)``) at lag 0 (``step[t]`` vs
    ``moved[t]``) and lag 1, keeping the better-correlated. ``ratio`` = RMS(step) /
    RMS(moved): ~1 is ideal, real controllers land within a few x. ``exact`` means the
    "command" reproduces the state's own motion - it was derived from it."""
    best: dict | None = None
    for lag in (0, 1):
        s, m = step[: len(moved) - lag], moved[lag:]
        if len(s) < 2:
            continue
        moved_rms = _rms(m)
        result = {
            "lag": lag,
            "ratio": _rms(s) / moved_rms if moved_rms > 0 else float("inf"),
            "correlation": _corr(s, m),
            "exact": bool(moved_rms > 0 and _rms(s - m) <= 1e-6 * moved_rms),
        }
        if best is None or np.nan_to_num(result["correlation"], nan=-2) > np.nan_to_num(best["correlation"], nan=-2):
            best = result
    return best or {"lag": 0, "ratio": float("nan"), "correlation": float("nan"), "exact": False}


def _integration_problems(result: dict, cfg: dict, control_freq: float) -> tuple[list[str], str | None]:
    """Reasons an integration result fails, plus a guess at the cause from the ratio:
    ~57x is deg/s, ~1/control_freq is a per-step change stored instead of a rate."""
    reasons: list[str] = []
    ratio = result["ratio"]
    if result["exact"]:
        reasons.append("copies_state_motion")
    if np.isfinite(ratio) and ratio > cfg["max_ratio"]:
        reasons.append("too_large")
    if np.isfinite(ratio) and ratio < cfg["min_ratio"]:
        reasons.append("too_small")
    if not result["exact"] and not result["correlation"] >= cfg["min_correlation"]:
        reasons.append("doesnt_match_state_motion")
    suspected = None
    if np.isfinite(ratio) and ratio > 0:
        if abs(np.log(ratio / np.degrees(1.0))) < np.log(2):
            suspected = "deg_per_s"
        elif abs(np.log(ratio * control_freq)) < np.log(2):
            suspected = "per_step_delta"
    return reasons, suspected


def sign_flip_rate(v: np.ndarray) -> float:
    """Share of consecutive steps where a meaningfully large velocity (> 10% of that
    column's RMS) flips sign - chatter from noise or misaligned timestamps."""
    rms = np.sqrt(np.mean(v**2, axis=0))
    big = (np.abs(v[1:]) > 0.1 * rms) & (np.abs(v[:-1]) > 0.1 * rms)
    flips = (np.sign(v[1:]) != np.sign(v[:-1])) & big
    return float(flips.sum() / big.sum()) if big.sum() else 0.0


def looks_clipped_to_unit_range(v: np.ndarray) -> bool:
    """Peaks exactly at +-1 - the signature of values normalized to [-1, 1] and
    clipped (DROID-style), not physical units."""
    return bool(v.size and np.isclose(np.max(np.abs(v)), 1.0, rtol=0, atol=1e-6))


def position_match(v: np.ndarray, state: np.ndarray) -> float:
    """RMS(v[t] - state[t+1]) / RMS(state[t+1]): near 0 means the "velocity" is really a
    position. T-aligned inputs."""
    s = state[1:]
    state_rms = _rms(s)
    return _rms(v[:-1] - s) / state_rms if state_rms > 0 else float("inf")


def _run_velocity_common(
    category: str,
    v: np.ndarray,
    cfg: dict,
    tests: list[str],
    findings: list[dict],
    checked: set[tuple[str, str]],
) -> None:
    """``normalized_range`` and ``smoothness``: action-only checks shared by both
    velocity categories."""
    if "normalized_range" in tests:
        checked.add((category, "normalized_range"))
        if looks_clipped_to_unit_range(v):
            findings.append(_finding(category, "normalized_range", category, max_abs=float(np.max(np.abs(v)))))
    if "smoothness" in tests and len(v) >= 2:
        checked.add((category, "smoothness"))
        rate = sign_flip_rate(v)
        if rate > cfg["smoothness"]["max_sign_flip_rate"]:
            findings.append(_finding(category, "smoothness", category, sign_flip_rate=rate))


def _run_joint_velocity_checks(
    actions: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    cfg: dict,
    findings: list[dict],
    skips: list[dict],
    checked: set[tuple[str, str]],
    *,
    control_freq: float | None,
    action_names: list[str],
    state_names: list[str],
) -> None:
    category = key = "joint_velocity"
    tests = [t for t, c in cfg.items() if c["enabled"]]

    def skip(reason: str, only: list[str] | None = None, **details) -> None:
        skips.extend(_skip(category, t, key, reason, **details) for t in (tests if only is None else only))

    if not tests:
        return
    if key not in actions:
        skip("declared_but_empty")
        return
    v = actions[key]
    if v.ndim != 2 or not np.all(np.isfinite(v)):
        skip("non_finite_values" if v.ndim == 2 else "unexpected_action_dim")
        return
    width = v.shape[1]

    if "dof" in tests:
        checked.add((category, "dof"))
        if action_names and len(action_names) != width:
            findings.append(_finding(category, "dof", key, width=width, names=len(action_names)))

    names_fit = len(action_names) == width
    cols = [i for i in range(width) if not (names_fit and _is_gripper_joint(action_names[i]))]
    label = (lambda i: action_names[i]) if names_fit else (lambda i: f"col{i}")
    v_arm = v[:, cols]

    if "speed_limit" in tests:
        checked.add((category, "speed_limit"))
        limit = cfg["speed_limit"]["max_joint_speed_rad_s"]
        peaks = np.max(np.abs(v_arm), axis=0)
        over = [{"joint": label(c), "max_speed": float(p)} for c, p in zip(cols, peaks) if p > limit]
        if over:
            findings.append(_finding(category, "speed_limit", key, limit_rad_s=limit, joints=over))

    _run_velocity_common(category, v_arm, cfg, tests, findings, checked)

    state_tests = [t for t in ("integration", "looks_like_position") if t in tests]
    if not state_tests:
        return
    state = states.get("joint_position")
    if state is None:
        skip("no_matching_state", state_tests)
        return
    if state.ndim != 2 or not np.all(np.isfinite(state)):
        skip("non_finite_state" if state.ndim == 2 else "unexpected_state_dim", state_tests)
        return
    pairing = _joint_pairing(action_names, state_names, width, state.shape[1])
    if pairing is None:
        skip("no_joint_pairing", state_tests, action_width=width, state_width=state.shape[1])
        return
    pairing = [(a, s) for a, s in pairing if a in cols]
    t = min(len(v), len(state))
    if t < 3:
        skip("too_short", state_tests)
        return
    pv = v[:t, [a for a, _ in pairing]]
    ps = state[:t, [s for _, s in pairing]]

    if "looks_like_position" in state_tests:
        checked.add((category, "looks_like_position"))
        match = position_match(pv, ps)
        if match < cfg["looks_like_position"]["max_position_match"]:
            findings.append(_finding(category, "looks_like_position", key, position_match=match))

    if "integration" in state_tests:
        motion = float(np.max(np.ptp(ps, axis=0))) if ps.size else 0.0
        if not control_freq or control_freq <= 0:
            skip("no_control_freq", ["integration"])
        elif motion < cfg["integration"]["min_motion_rad"]:
            # With barely any motion the comparison is all sensor noise.
            skip("too_little_motion", ["integration"], max_joint_range_rad=motion)
        else:
            checked.add((category, "integration"))
            result = velocity_integration(pv / control_freq, np.diff(ps, axis=0))
            reasons, suspected = _integration_problems(result, cfg["integration"], control_freq)
            if reasons:
                findings.append(
                    _finding(category, "integration", key, reasons=reasons, suspected=suspected, control_freq=control_freq, **result)
                )


def _rotation_steps(quats: np.ndarray, frame: str) -> np.ndarray:
    """Per-step rotation (rotation vector, rad) between consecutive orientations, expressed
    in the base frame (R[t+1] R[t]^-1) or the end-effector frame (R[t]^-1 R[t+1])."""
    r = Rotation.from_quat(quats)
    rel = r[1:] * r[:-1].inv() if frame == "base" else r[:-1].inv() * r[1:]
    return rel.as_rotvec()


def _run_cartesian_velocity_checks(
    actions: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    cfg: dict,
    findings: list[dict],
    skips: list[dict],
    checked: set[tuple[str, str]],
    *,
    is_biarm: bool,
    control_freq: float | None,
    state_orientation: str | None,
) -> None:
    category = key = "cartesian_velocity"
    tests = [t for t, c in cfg.items() if c["enabled"]]

    def skip(reason: str, only: list[str] | None = None, **details) -> None:
        skips.extend(_skip(category, t, key, reason, **details) for t in (tests if only is None else only))

    if not tests:
        return
    if key not in actions:
        skip("declared_but_empty")
        return
    v = actions[key]
    if v.ndim != 2 or not np.all(np.isfinite(v)):
        skip("non_finite_values" if v.ndim == 2 else "unexpected_action_dim")
        return

    n_arms = 2 if is_biarm else 1
    if "shape" in tests:
        checked.add((category, "shape"))
        if v.shape[1] != 6 * n_arms:
            findings.append(_finding(category, "shape", key, expected_dim=6 * n_arms, actual_dim=v.shape[1]))
    if v.shape[1] != 6 * n_arms:
        skip("wrong_action_dim", [t for t in tests if t != "shape"], actual_dim=v.shape[1])
        return
    lin = [v[:, 6 * a : 6 * a + 3] for a in range(n_arms)]
    ang = [v[:, 6 * a + 3 : 6 * a + 6] for a in range(n_arms)]

    if "speed_limit" in tests:
        checked.add((category, "speed_limit"))
        sl = cfg["speed_limit"]
        peak_lin = max(float(np.max(np.linalg.norm(x, axis=1))) for x in lin)
        peak_ang = max(float(np.max(np.linalg.norm(x, axis=1))) for x in ang)
        if peak_lin > sl["max_linear_m_s"] or peak_ang > sl["max_angular_rad_s"]:
            findings.append(_finding(category, "speed_limit", key, max_linear_m_s=peak_lin, max_angular_rad_s=peak_ang))

    _run_velocity_common(category, v, cfg, tests, findings, checked)

    state_tests = [t for t in ("integration", "linear_angular_order", "looks_like_position") if t in tests]
    if not state_tests:
        return
    state = states.get("cartesian_position")
    if state is None:
        skip("no_matching_state", state_tests)
        return
    if state.ndim != 2 or state.shape[1] % n_arms or state.shape[1] // n_arms < 3 or not np.all(np.isfinite(state)):
        skip("unusable_state", state_tests, state_shape=list(state.shape))
        return
    arm_width = state.shape[1] // n_arms
    t = min(len(v), len(state))
    if t < 3:
        skip("too_short", state_tests)
        return
    lin = [x[:t] for x in lin]
    ang = [x[:t] for x in ang]
    xyz = [state[:t, arm_width * a : arm_width * a + 3] for a in range(n_arms)]
    quats = [
        _state_rotation_to_quat(state[:t, arm_width * a + 3 : arm_width * (a + 1)], state_orientation)
        for a in range(n_arms)
    ]
    has_rotation = all(q is not None for q in quats)
    # How far the state moves / turns: with barely any motion the comparison is all noise.
    translation = max(float(np.max(np.linalg.norm(p - p[0], axis=1))) for p in xyz)
    rotation = (
        max(float(np.degrees(np.max((Rotation.from_quat(q) * Rotation.from_quat(q[0]).inv()).magnitude()))) for q in quats)
        if has_rotation
        else 0.0
    )

    if "looks_like_position" in state_tests:
        checked.add((category, "looks_like_position"))
        match = position_match(np.concatenate(lin, axis=1), np.concatenate(xyz, axis=1))
        if match < cfg["looks_like_position"]["max_position_match"]:
            findings.append(_finding(category, "looks_like_position", key, position_match=match))

    if not control_freq or control_freq <= 0:
        skip("no_control_freq", [s for s in state_tests if s != "looks_like_position"])
        return

    # State motion per step, in both frames a velocity can be expressed in; the frame
    # whose motion the command matches best is reported, not judged.
    frames = ("base", "ee") if has_rotation else ("base",)
    moved_lin, moved_ang = {}, {}
    for frame in frames:
        moved_lin[frame] = np.concatenate(
            [
                np.diff(p, axis=0) if frame == "base" else Rotation.from_quat(q[:-1]).inv().apply(np.diff(p, axis=0))
                for p, q in zip(xyz, quats)
            ],
            axis=1,
        )
        if has_rotation:
            moved_ang[frame] = np.concatenate([_rotation_steps(q, frame) for q in quats], axis=1)
    step_lin = np.concatenate(lin, axis=1) / control_freq
    step_ang = np.concatenate(ang, axis=1) / control_freq

    results = {}
    for frame in frames:
        lin_result = velocity_integration(step_lin, moved_lin[frame])
        ang_result = velocity_integration(step_ang, moved_ang[frame]) if has_rotation else None
        score = np.nan_to_num(lin_result["correlation"], nan=-1) + (
            np.nan_to_num(ang_result["correlation"], nan=-1) if ang_result else 0
        )
        results[frame] = (score, lin_result, ang_result)
    frame = max(results, key=lambda f: results[f][0])
    _, lin_result, ang_result = results[frame]

    icfg = cfg["integration"]
    moves = translation >= icfg["min_translation_m"]
    turns = has_rotation and rotation >= icfg["min_rotation_deg"]
    if "integration" in state_tests:
        details: dict = {"frame": frame, "control_freq": control_freq}
        reasons: list[str] = []
        if moves:
            lin_reasons, lin_suspected = _integration_problems(lin_result, icfg, control_freq)
            details["linear"] = {**lin_result, "suspected": lin_suspected}
            reasons += [f"linear_{r}" for r in lin_reasons]
        if turns:
            ang_reasons, ang_suspected = _integration_problems(ang_result, icfg, control_freq)
            details["angular"] = {**ang_result, "suspected": ang_suspected}
            reasons += [f"angular_{r}" for r in ang_reasons]
        if not moves and not turns:
            skip("too_little_motion", ["integration"], translation_m=translation, rotation_deg=rotation)
        else:
            checked.add((category, "integration"))
            if not has_rotation:
                skip("unknown_state_orientation", ["integration"], note="linear part checked only")
            if reasons:
                findings.append(_finding(category, "integration", key, reasons=reasons, **details))

    if "linear_angular_order" in state_tests:
        if not has_rotation:
            skip("unknown_state_orientation", ["linear_angular_order"])
        elif not (moves and turns):
            skip("too_little_motion", ["linear_angular_order"], translation_m=translation, rotation_deg=rotation)
        else:
            checked.add((category, "linear_angular_order"))
            as_given = (lin_result["correlation"] + ang_result["correlation"]) / 2
            swapped_lin = velocity_integration(step_ang, moved_lin[frame])
            swapped_ang = velocity_integration(step_lin, moved_ang[frame])
            swapped = (swapped_lin["correlation"] + swapped_ang["correlation"]) / 2
            if np.isfinite(swapped) and (
                not np.isfinite(as_given) or swapped > as_given + cfg["linear_angular_order"]["min_improvement"]
            ):
                findings.append(
                    _finding(category, "linear_angular_order", key, correlation_as_given=as_given, correlation_swapped=swapped)
                )


# ── episode entry points ──────────────────────────────────────────────────────


def run_action_qa_on_episode(
    actions: dict[str, np.ndarray],
    states: dict[str, np.ndarray],
    declared_action_space: list[str],
    config: dict,
    *,
    is_biarm: bool = False,
    control_freq: float | None = None,
    state_orientation: str | None = None,
    action_joint_names: list[str] | None = None,
    state_joint_names: list[str] | None = None,
) -> dict:
    """Run every enabled check, category by category, per ``config``.

    Never raises - NaN/Inf/missing-key cases become a skip reason, not an
    exception, so a batch run never dies on one bad file. Returns
    ``{"findings": [...], "skips": [...], "checked": [(category, test), ...]}`` - ``checked``
    lists the checks that actually ran, so "passed" can be told apart from "not applicable".
    """
    findings: list[dict] = []
    skips: list[dict] = []
    checked: set[tuple[str, str]] = set()

    _run_general_checks(actions, states, declared_action_space, config["general"], findings, skips, checked)
    if "cartesian_position" in declared_action_space:
        _run_cartesian_position_checks(
            actions,
            states,
            config["cartesian_position"],
            findings,
            skips,
            checked,
            is_biarm=is_biarm,
            control_freq=control_freq,
            state_orientation=state_orientation,
        )
    if "joint_position" in declared_action_space:
        _run_joint_position_checks(
            actions,
            states,
            config["joint_position"],
            findings,
            skips,
            checked,
            is_biarm=is_biarm,
            control_freq=control_freq,
            action_names=action_joint_names or [],
            state_names=state_joint_names or [],
        )
    if "joint_velocity" in declared_action_space:
        _run_joint_velocity_checks(
            actions,
            states,
            config["joint_velocity"],
            findings,
            skips,
            checked,
            control_freq=control_freq,
            action_names=action_joint_names or [],
            state_names=state_joint_names or [],
        )
    if "cartesian_velocity" in declared_action_space:
        _run_cartesian_velocity_checks(
            actions,
            states,
            config["cartesian_velocity"],
            findings,
            skips,
            checked,
            is_biarm=is_biarm,
            control_freq=control_freq,
            state_orientation=state_orientation,
        )

    return {"findings": findings, "skips": skips, "checked": sorted(checked)}


def profile_control_freq(profile: dict) -> float | None:
    try:
        return float(profile.get("control_freq"))
    except (TypeError, ValueError):
        return None


def run_action_qa_on_arrays(
    actions: dict[str, np.ndarray], states: dict[str, np.ndarray], profile: dict, config: dict
) -> dict | None:
    """:func:`run_action_qa_on_episode` with the declared action space and robot
    details taken from ``profile``. ``None`` if no action space is declared.
    """
    declared = profile.get("action_space") or []
    if isinstance(declared, str):
        declared = [declared]
    declared = [str(k) for k in declared]
    if not declared:
        return None

    return run_action_qa_on_episode(
        actions,
        states,
        declared,
        config,
        is_biarm=bool(profile.get("is_biarm", False)),
        control_freq=profile_control_freq(profile),
        state_orientation=profile.get("robot_state_orientation_representation"),
        action_joint_names=_str_list(profile.get("action_joint_names")),
        state_joint_names=_str_list(profile.get("robot_state_joint_names")),
    )


def _str_list(value) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def run_action_qa_on_h5(f: h5py.File, config: dict) -> dict | None:
    """:func:`load_action_state_arrays` + :func:`run_action_qa_on_arrays`."""
    return run_action_qa_on_arrays(*load_action_state_arrays(f), config)
