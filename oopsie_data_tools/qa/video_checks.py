"""Video quality checks, run as the ``video`` category of ``oopsie-data qa``.

Each camera video is decoded once with ffmpeg into small grayscale frames; all
checks run on that probe:

- ``resolution``: short side below ``min_short_side``.
- ``decoding``: missing file, ffmpeg errors, nothing decoded, or the decoded frame
  count differing from the container's.
- ``brightness``: median luma too dark or too bright, or a large share of
  near-black / near-white frames.
- ``fps``: container fps differs from ``control_freq``.
- ``duration``: video duration (decoded frames / fps) differs from the trajectory
  duration (T / ``control_freq``).
- ``camera_duration_mismatch``: the cameras of one episode disagree in duration.
- ``corrupted_frames``: a single frame unlike both its neighbours.
- ``state_jump`` / ``video_jump`` / ``recording_gap``: a sudden jump in only the robot
  state, only the video, or both at the same moment (time missing from the recording).
- ``choppy``: many duplicate frames while the robot moves.
- ``frozen``: a long run of static frames while the robot moves.
- ``periodic_skips``: recording pauses at a regular period (e.g. during each
  action-chunk inference) while the low-level controller keeps the robot moving,
  seen as robot state that jumps at that period throughout the motion, with the
  recorded robot position and the video both jumping perceptibly at the gaps.
  Episode-level.

Frame differences are mean absolute differences of 0-255 grayscale values.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from oopsie_data_tools.qa.action_checks import _finding, _params, _skip

CATEGORY = "video"
CHECKS = (
    "decoding", "corrupted_frames", "state_jump", "video_jump", "recording_gap", "periodic_skips", "choppy",
    "frozen", "resolution", "brightness", "fps", "duration", "camera_duration_mismatch",
)
ALL_CAMERAS = "all cameras"
PROBE_SIZE = 96
_ERROR_RE = re.compile(r"\[(error|fatal)\]")
SKIP_WINDOW = 8
SKIP_MAX_PERIOD = 30
POSITION_STATES = ("joint_position", "cartesian_position")
_DIFF_FLOOR = 0.05


@dataclass
class VideoProbe:
    width: int
    height: int
    fps: float
    container_frames: int
    returncode: int
    errors: list[str] = field(default_factory=list)
    luma: np.ndarray = field(default_factory=lambda: np.zeros(0))
    diff: np.ndarray = field(default_factory=lambda: np.zeros(0))
    skip_diff: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def decoded_frames(self) -> int:
        return len(self.luma)

    @property
    def duration(self) -> float | None:
        return self.decoded_frames / self.fps if self.fps > 0 and self.decoded_frames else None


def probe_video(path: Path, size: int = PROBE_SIZE) -> VideoProbe:
    """Read container metadata and decode the full stream to ``size``x``size`` gray."""
    cap = cv2.VideoCapture(str(path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        container_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "level+info",
        "-i",
        str(path),
        "-fps_mode",
        "passthrough",
        "-vf",
        f"scale={size}:{size},format=gray",
        "-f",
        "rawvideo",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    log = proc.stderr.decode("utf-8", errors="replace").splitlines()
    errors = [line.strip() for line in log if _ERROR_RE.search(line)]

    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = raw.size // (size * size)
    frames = raw[: n * size * size].reshape(n, size, size).astype(np.float32)
    return VideoProbe(
        width=width,
        height=height,
        fps=fps,
        container_frames=container_frames,
        returncode=proc.returncode,
        errors=errors,
        luma=frames.mean(axis=(1, 2)),
        diff=np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2)),
        skip_diff=np.abs(frames[2:] - frames[:-2]).mean(axis=(1, 2)),
    )


def _continuous_quaternions(pose: np.ndarray) -> np.ndarray:
    """Copy of a ``(T, 7 * arms)`` cartesian pose with each arm's quaternion sign-flipped
    where needed so consecutive quaternions agree (q and -q are the same orientation, so a
    sign flip is not motion, but as raw numbers it looks like a jump of up to 2)."""
    pose = pose.astype(float, copy=True)
    for start in range(3, pose.shape[1], 7):
        q = pose[:, start : start + 4]
        for t in range(1, len(q)):
            if np.dot(q[t], q[t - 1]) < 0:
                q[t] = -q[t]
    return pose


def _unwrap_angles(a: np.ndarray, cols: range) -> np.ndarray:
    """Copy with the angle columns ``cols`` unwrapped (a jump from +pi to -pi is not motion).
    Only columns that look like radians (all values within +-(pi + 0.1)) are touched, so
    degrees or other units are left alone."""
    a = a.astype(float, copy=True)
    for c in cols:
        if c < a.shape[1] and np.all(np.abs(a[:, c]) <= np.pi + 0.1):
            a[:, c] = np.unwrap(a[:, c])
    return a


def state_step(states: dict[str, np.ndarray]) -> np.ndarray | None:
    """Per-step size of the robot state change, over all non-gripper states, each
    scaled by its standard deviation. None without usable state."""
    arrays = []
    for k, a in states.items():
        if k == "gripper_position" or a.ndim < 1 or len(a) < 2:
            continue
        a = a.reshape(len(a), -1)
        if k == "cartesian_position" and a.shape[1] % 7 == 0:
            a = _continuous_quaternions(a)  # [x, y, z, qx, qy, qz, qw] per arm: remove q -> -q flips
        elif k == "cartesian_position" and a.shape[1] % 6 == 0:
            a = _unwrap_angles(a, [c for c in range(a.shape[1]) if c % 6 >= 3])  # [x, y, z, euler] per arm
        elif k == "joint_position":
            a = _unwrap_angles(a, range(a.shape[1]))  # continuous joints wrapping at +-pi
        arrays.append(a)
    if not arrays:
        return None
    n = min(len(a) for a in arrays)
    state = np.concatenate([a[:n] for a in arrays], axis=1)
    scale = state.std(axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return np.nan_to_num(np.linalg.norm(np.diff(state / scale, axis=0), axis=1))


def robot_moving(step: np.ndarray, moving_fraction_of_p95: float) -> np.ndarray:
    """Mask of steps above ``moving_fraction_of_p95`` of the 95th percentile step."""
    p95 = np.percentile(step, 95)
    if p95 <= 0:
        return np.zeros(len(step), dtype=bool)
    return step > moving_fraction_of_p95 * p95


def _resample(mask: np.ndarray, n: int) -> np.ndarray:
    if len(mask) == n or len(mask) == 0:
        return mask[:n]
    idx = np.round(np.linspace(0, len(mask) - 1, n)).astype(int)
    return mask[idx]


def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    padded = np.pad(x, w, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1), axis=1)


def _best_period(gaps: np.ndarray) -> tuple[int, float]:
    """Period ``p`` that best explains the gaps between events, as the share of gaps
    within one step of a multiple of ``p`` minus that share for random gaps (3/p).
    Among periods within 0.05 of the best score, the one with most exact multiples wins."""
    scored = []
    for p in range(3, SKIP_MAX_PERIOD + 1):
        r = gaps % p
        near = float((np.minimum(r, p - r) <= 1).mean()) - 3.0 / p
        scored.append((p, near, float((r == 0).mean())))
    top = max(s[1] for s in scored)
    if top <= 0:
        return 0, 0.0
    p, near, _ = max((s for s in scored if s[1] >= top - 0.05), key=lambda s: s[2])
    return p, near


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for x in mask:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def check_resolution(p: VideoProbe, min_short_side: int) -> list[str]:
    if 0 < min(p.width, p.height) < min_short_side:
        return [f"{p.width}x{p.height} (short side < {min_short_side})"]
    return []


def check_decoding(p: VideoProbe, max_frame_count_diff: int) -> list[str]:
    issues = []
    if p.decoded_frames == 0:
        issues.append(f"no frames decoded (ffmpeg exit {p.returncode})")
    elif p.returncode != 0:
        issues.append(f"ffmpeg exit {p.returncode}")
    if p.errors:
        issues.append(f"{len(p.errors)} decoder error(s), first: {p.errors[0][:200]}")
    if (
        p.decoded_frames
        and p.container_frames > 0
        and abs(p.decoded_frames - p.container_frames) > max_frame_count_diff
    ):
        issues.append(f"decoded {p.decoded_frames} frames, container reports {p.container_frames}")
    return issues


def check_brightness(
    p: VideoProbe,
    dark_median: float,
    bright_median: float,
    dark_frame: float,
    bright_frame: float,
    extreme_frame_fraction: float,
) -> list[str]:
    if not p.decoded_frames:
        return []
    issues = []
    median = float(np.median(p.luma))
    if median < dark_median:
        issues.append(f"too dark: median luma {median:.0f}")
    elif median > bright_median:
        issues.append(f"too bright: median luma {median:.0f}")
    dark = float((p.luma < dark_frame).mean())
    bright = float((p.luma > bright_frame).mean())
    if dark >= extreme_frame_fraction:
        issues.append(f"{dark:.0%} of frames near black")
    if bright >= extreme_frame_fraction:
        issues.append(f"{bright:.0%} of frames near white")
    return issues


def check_fps(p: VideoProbe, control_freq: float | None, fps_rel_tol: float) -> list[str]:
    """The container's fps is a header value: a mismatch alone misaligns nothing if the frame
    count still equals T (that is ``check_duration``), but playback runs at the wrong speed."""
    if control_freq and p.fps > 0 and abs(p.fps - control_freq) > fps_rel_tol * control_freq:
        return [f"video fps {p.fps:g} != control_freq {control_freq:g}"]
    return []


def check_duration(
    p: VideoProbe,
    n_steps: int | None,
    control_freq: float | None,
    duration_tol_s: float,
    duration_rel_tol: float,
) -> list[str]:
    """With fps == control_freq this is "decoded frames == T": frame i is no longer step i."""
    if control_freq and n_steps and p.duration is not None:
        expected = n_steps / control_freq
        if abs(p.duration - expected) > max(duration_tol_s, duration_rel_tol * expected):
            return [
                f"video lasts {p.duration:.1f}s, trajectory {expected:.1f}s "
                f"({p.decoded_frames} frames, {n_steps} steps)"
            ]
    return []


def check_frozen(p: VideoProbe, moving: np.ndarray | None, rate: float, static_diff: float, frozen_s: float) -> list[str]:
    """The longest run of static frames while the robot moves, if at least ``frozen_s``."""
    d = p.diff
    if len(d) < 3 or moving is None or not moving.any():
        return []
    rate = rate if rate > 0 else 1.0
    frozen = _longest_run((d < static_diff) & _resample(moving, len(d)))
    return [f"frozen for {frozen / rate:.1f}s while the robot moves"] if frozen / rate >= frozen_s else []


def check_choppy(
    p: VideoProbe, moving: np.ndarray | None, duplicate_diff: float, choppy_fraction: float
) -> list[str]:
    """Share of near-duplicate frames while the robot moves, if at least ``choppy_fraction``."""
    d = p.diff
    if len(d) < 3 or moving is None or not moving.any():
        return []
    mov = _resample(moving, len(d))
    dup = float(((d < duplicate_diff) & mov).sum() / mov.sum())
    return [f"{dup:.0%} duplicate frames while the robot moves"] if dup >= choppy_fraction else []


def find_corrupted_frames(p: VideoProbe, glitch_diff: float, glitch_ratio: float) -> list[int]:
    """Frames unlike both neighbours: differing by at least ``glitch_diff`` from each, while
    the neighbours differ ``glitch_ratio`` times less from each other. Frame indices."""
    d = p.diff
    return [
        t for t in range(1, len(d))
        if (a := min(d[t - 1], d[t])) >= glitch_diff and a / (p.skip_diff[t - 1] + 1.0) >= glitch_ratio
    ]


def find_video_jumps(p: VideoProbe, rate: float, jump_diff: float, jump_ratio: float, exclude: set[int]) -> list[float]:
    """Times (s) of sudden one-way image changes: a frame difference of at least
    ``jump_diff`` and ``jump_ratio`` times the median of the surrounding second. Frame
    differences in ``exclude`` (around corrupted frames) are skipped."""
    d = p.diff
    rate = rate if rate > 0 else 1.0
    window = max(2, round(rate))
    times = []
    for t in range(len(d)):
        if t in exclude or d[t] < jump_diff:
            continue
        neighbours = np.r_[d[max(0, t - window) : t], d[t + 1 : t + 1 + window]]
        if neighbours.size and d[t] / (np.median(neighbours) + 1.0) >= jump_ratio:
            times.append((t + 1) / rate)
    return times


def state_jumps_at(step: np.ndarray, time_s: float, control_freq: float | None, state_jump_ratio: float) -> bool:
    """Whether the robot state also changes abruptly within one step of ``time_s``: a step
    at least ``state_jump_ratio`` times the mean step of the surrounding second."""
    hz = control_freq if control_freq and control_freq > 0 else None
    if hz is None or len(step) < 3:
        return False
    k = int(round(time_s * hz)) - 1  # step[k] is the change from state k to k+1
    near = [i for i in (k - 1, k, k + 1) if 0 <= i < len(step)]
    if not near:
        return False
    window = max(2, round(hz))
    around = np.r_[step[max(0, near[0] - window) : near[0]], step[near[-1] + 1 : near[-1] + 1 + window]]
    mean = float(np.mean(around)) if around.size else 0.0
    return max(step[i] for i in near) > state_jump_ratio * mean


def _position_jump(states: dict[str, np.ndarray], idx: np.ndarray) -> tuple[str, np.ndarray] | None:
    """Extra robot motion at steps ``idx`` beyond the local median step: the largest
    joint change in rad from ``joint_position``, else the largest end-effector
    translation in cm from ``cartesian_position`` ([x, y, z, qx, qy, qz, qw] per arm)."""
    if "joint_position" in states:
        q = states["joint_position"].reshape(len(states["joint_position"]), -1)
        d, unit = np.abs(np.diff(q, axis=0)).max(axis=1), "rad"
    elif "cartesian_position" in states:
        c = states["cartesian_position"].reshape(len(states["cartesian_position"]), -1)
        arms = [c[:, k : k + 3] for k in range(0, c.shape[1], 7)] if c.shape[1] % 7 == 0 else [c[:, :3]]
        d, unit = 100 * np.max([np.linalg.norm(np.diff(a, axis=0), axis=1) for a in arms], axis=0), "cm"
    else:
        return None
    d = np.nan_to_num(d)
    return unit, (d - _rolling_median(d, SKIP_WINDOW))[np.minimum(idx, len(d) - 1)]


def _position_series(series: dict[str, np.ndarray], space: str) -> np.ndarray:
    x = series[space].reshape(len(series[space]), -1).astype(float)
    if space == "cartesian_position" and x.shape[1] % 7 == 0:
        x = np.concatenate([x[:, k : k + 3] for k in range(0, x.shape[1], 7)], axis=1)
    return np.nan_to_num(x)


def _command_explained(
    actions: dict[str, np.ndarray] | None, states: dict[str, np.ndarray], idx: np.ndarray, ratio: float
) -> np.ndarray:
    """Mask of state jumps at ``idx`` that a logged position command explains: the
    largest command change over the jump step and the two steps before it is at
    least ``ratio`` times the state change, in the same space (joint, else xyz).
    A held command changes by 0 and explains nothing."""
    space = next((k for k in POSITION_STATES if actions and k in actions and k in states), None)
    if space is None:
        return np.zeros(len(idx), dtype=bool)
    a, x = _position_series(actions, space), _position_series(states, space)
    n = min(len(a), len(x))
    da = np.linalg.norm(np.diff(a[:n], axis=0), axis=1)
    ds = np.linalg.norm(np.diff(x[:n], axis=0), axis=1)
    out = np.zeros(len(idx), dtype=bool)
    for j, i in enumerate(idx):
        if i < len(ds):
            out[j] = da[max(0, i - 2) : i + 1].max() >= ratio * ds[i] > 0
    return out


def _video_jump(diff: np.ndarray, idx: np.ndarray, n_steps: int) -> np.ndarray:
    """Largest frame difference within one frame of each state step in ``idx``,
    relative to the local median frame difference."""
    if len(diff) < 2 * SKIP_WINDOW + 1:
        return np.zeros(len(idx))
    frames = np.round(idx * (len(diff) - 1) / max(n_steps - 1, 1)).astype(int)
    med = _rolling_median(diff, SKIP_WINDOW)
    peak = np.array([diff[max(0, f - 1) : f + 2].max() for f in frames])
    return peak / (med[frames] + _DIFF_FLOOR)


def check_periodic_skips(
    step: np.ndarray,
    moving: np.ndarray,
    states: dict[str, np.ndarray],
    actions: dict[str, np.ndarray] | None,
    diffs: dict[str, np.ndarray],
    control_freq: float | None,
    jump_ratio: float,
    min_jumps: int,
    min_period_score: float,
    min_coverage: float,
    min_joint_jump_rad: float,
    min_ee_jump_cm: float,
    min_video_jump: float,
    min_confirmed_fraction: float,
    command_explains_ratio: float,
) -> list[str]:
    """Recording gaps at a regular period: isolated one-step jumps of the robot state
    throughout the motion that the logged command doesn't explain, where the robot
    position and the video both jump.

    A jump is a step above ``jump_ratio`` times the local median step (window of
    +-SKIP_WINDOW steps) while the robot moves, with no jump on either neighbour.
    Jumps where the position command changed at least ``command_explains_ratio``
    times as much as the state (over the jump step and the two before) are a
    legitimately large command and are dropped. The remaining jumps need a period score of at least ``min_period_score`` and must cover
    at least ``min_coverage`` of the periods while the robot moves (jumps x period
    / moving steps). A jump is confirmed when the robot position moves at least
    ``min_joint_jump_rad`` (or ``min_ee_jump_cm`` without joint state) beyond a
    normal step and some camera's frame difference reaches ``min_video_jump`` times
    its local median. Flags when at least ``min_jumps`` jumps are found and at least
    ``min_confirmed_fraction`` of them are confirmed. ``diffs`` maps camera name to
    consecutive-frame differences.
    """
    if len(step) < 2 * SKIP_WINDOW + 1 or not moving.any():
        return []
    med = _rolling_median(step, SKIP_WINDOW)
    jump = (step > jump_ratio * med) & moving
    jump &= ~(np.r_[False, jump[:-1]] | np.r_[jump[1:], False])
    idx = np.flatnonzero(jump)
    idx = idx[~_command_explained(actions, states, idx, command_explains_ratio)]
    if len(idx) < min_jumps:
        return []
    period, score = _best_period(np.diff(idx))
    if score < min_period_score:
        return []
    coverage = len(idx) * period / moving.sum()
    if coverage < min_coverage:
        return []
    position = _position_jump(states, idx)
    if position is None or not diffs:
        return []
    unit, extra = position
    state_ok = extra >= (min_joint_jump_rad if unit == "rad" else min_ee_jump_cm)
    video = np.max([_video_jump(d, idx, len(step) + 1) for d in diffs.values()], axis=0)
    confirmed = state_ok & (video >= min_video_jump)
    if confirmed.mean() < min_confirmed_fraction:
        return []
    every = f"{period} steps" + (f" ({period / control_freq:.1f}s)" if control_freq else "")
    issue = (
        f"periodic recording gaps every {every}: {len(idx)} jumps in robot state, "
        f"{confirmed.mean():.0%} with position and video jumping "
        f"(median +{np.median(extra[confirmed]):.3g} {unit}, video {np.median(video[confirmed]):.1f}x)"
    )
    return [issue]


def find_state_jumps(
    step: np.ndarray,
    control_freq: float | None,
    jump_ratio: float,
    min_fraction_of_p95: float,
    moving_fraction_of_p95: float,
) -> list[tuple[int, float]]:
    """Isolated one-step jumps of the robot state: ``(step index, size / local median)``.

    A jump is at least ``jump_ratio`` times the mean step of the surrounding second
    (excluding itself; the mean, not the median, so a state that updates only every few
    steps - zeros in between - doesn't make each update look like a jump), at least ``min_fraction_of_p95`` of the episode's 95th-percentile
    step (so jitter while the robot stands still doesn't count), and neither neighbour is
    one (a run of big steps is fast motion, not a gap). Most of the surrounding second must
    also be moving (at least half its steps above ``moving_fraction_of_p95`` of the 95th
    percentile, the "robot moving" test): missing time only shows as a jump while the robot
    moves, and a small twitch in an otherwise still second (e.g. the arm settling at the
    start) is not one.
    """
    if len(step) < 5:
        return []
    window = max(2, round(control_freq or 10))
    p95 = np.percentile(step, 95)
    floor = min_fraction_of_p95 * p95
    ratios = np.zeros(len(step))
    surroundings_moving = np.zeros(len(step), dtype=bool)
    for t in range(len(step)):
        neighbours = np.r_[step[max(0, t - window) : t], step[t + 1 : t + 1 + window]]
        mean = float(np.mean(neighbours)) if neighbours.size else 0.0
        ratios[t] = step[t] / mean if mean > 0 else np.inf
        surroundings_moving[t] = neighbours.size > 0 and np.mean(neighbours > moving_fraction_of_p95 * p95) >= 0.5
    candidate = (ratios >= jump_ratio) & (step >= floor) & (step > 0) & surroundings_moving
    isolated = candidate & ~(np.r_[False, candidate[:-1]] | np.r_[candidate[1:], False])
    return [(int(t), float(ratios[t])) for t in np.flatnonzero(isolated)]


def video_jumps_at(p: "VideoProbe", time_s: float, control_freq: float | None, video_jump_ratio: float) -> bool:
    """Whether this camera's frame difference also spikes within one frame of ``time_s``,
    compared with the median difference of the surrounding second."""
    rate = p.fps or control_freq or 0.0
    d = p.diff
    if rate <= 0 or len(d) < 3:
        return False
    k = int(round(time_s * rate)) - 1  # d[k] is the change from frame k to k+1
    near = range(max(0, k - 1), min(len(d), k + 2))
    if not near:
        return False
    window = max(2, round(rate))
    around = np.r_[d[max(0, k - 1 - window) : max(0, k - 1)], d[min(len(d), k + 2) : k + 2 + window]]
    base = float(np.median(around)) if around.size else 0.0
    return max(d[i] for i in near) / (base + 1.0) >= video_jump_ratio


def drop_ignored_cameras(videos: dict[str, Path], ignore_cameras) -> tuple[dict[str, Path], list[str]]:
    """Split off cameras named in ``ignore_cameras`` (case-insensitive): streams that are
    other sensors stored as cameras, e.g. CMU_Intent's ``tactile``."""
    ignore = {str(name).strip().lower() for name in ignore_cameras}
    kept = {cam: path for cam, path in videos.items() if cam.lower() not in ignore}
    return kept, sorted(set(videos) - set(kept))


def run_video_checks(
    videos: dict[str, Path],
    states: dict[str, np.ndarray],
    control_freq: float | None,
    cfg: dict,
    settings: dict,
    actions: dict[str, np.ndarray] | None = None,
) -> dict:
    """Run every enabled video check. ``videos`` maps camera name to mp4 path,
    ``cfg`` is the ``video`` section of the action QA config and ``settings`` its
    ``video_settings``: cameras named in ``ignore_cameras`` are left out of every check
    (recorded as an ``ignored_camera`` skip), and ``moving_fraction_of_p95`` sets when the
    robot counts as moving. ``actions`` are the logged actions, used by ``periodic_skips``
    to tell large commands from recording gaps.

    Returns ``{"findings", "skips", "checked"}`` like ``run_action_qa_on_episode``;
    findings are keyed by camera and carry the readable ``issue``.
    """
    enabled = [test for test in CHECKS if cfg[test]["enabled"]]
    findings: list[dict] = []
    skips: list[dict] = []
    if not enabled:
        return {"findings": findings, "skips": skips, "checked": []}
    videos, ignored = drop_ignored_cameras(videos, settings["ignore_cameras"])
    skips += [_skip(CATEGORY, t, cam, "ignored_camera") for cam in ignored for t in enabled]
    if not videos:
        skips += [_skip(CATEGORY, t, ALL_CAMERAS, "no_video_files") for t in enabled]
        return {"findings": findings, "skips": skips, "checked": []}

    def add(test: str, camera: str, issues: list[str], time_s: float | None = None) -> None:
        # time_s: when a single-moment issue happens, for the report's evidence panel.
        extra = {} if time_s is None else {"time_s": round(float(time_s), 3)}
        findings.extend(_finding(CATEGORY, test, camera, issue=issue, **extra) for issue in issues)

    n_steps = next((len(a) for a in states.values() if a.ndim >= 1), None)
    moving = None
    moving_fraction = settings["moving_fraction_of_p95"]
    state_tests = [t for t in ("state_jump", "recording_gap", "periodic_skips", "choppy", "frozen") if cfg[t]["enabled"]]
    # video_jump also reads the state, to leave out jumps the state shares; without
    # state it still runs and says so.
    step = state_step(states) if state_tests or cfg["video_jump"]["enabled"] else None
    if state_tests and step is None:
        skips += [_skip(CATEGORY, t, ALL_CAMERAS, "no_robot_state") for t in state_tests]
    if step is not None:
        moving = robot_moving(step, moving_fraction)
    if cfg["periodic_skips"]["enabled"] and step is not None and not any(k in states for k in POSITION_STATES):
        skips.append(_skip(CATEGORY, "periodic_skips", ALL_CAMERAS, "no_position_state"))

    durations: dict[str, float] = {}
    probes: dict[str, VideoProbe] = {}
    video_jumps: dict[str, list[float]] = {}
    diffs: dict[str, np.ndarray] = {}
    for camera, path in sorted(videos.items()):
        if not Path(path).is_file():
            if cfg["decoding"]["enabled"]:
                add("decoding", camera, [f"missing file {Path(path).name}"])
            continue
        p = probe_video(Path(path))
        probes[camera] = p
        if cfg["resolution"]["enabled"]:
            add("resolution", camera, check_resolution(p, **_params(cfg["resolution"])))
        if cfg["decoding"]["enabled"]:
            add("decoding", camera, check_decoding(p, **_params(cfg["decoding"])))
        if cfg["brightness"]["enabled"]:
            add("brightness", camera, check_brightness(p, **_params(cfg["brightness"])))
        if cfg["fps"]["enabled"]:
            add("fps", camera, check_fps(p, control_freq, **_params(cfg["fps"])))
        if cfg["duration"]["enabled"]:
            add("duration", camera, check_duration(p, n_steps, control_freq, **_params(cfg["duration"])))
        rate = p.fps or control_freq or 0.0
        if cfg["corrupted_frames"]["enabled"] or cfg["video_jump"]["enabled"]:
            corrupted = find_corrupted_frames(p, **_params(cfg["corrupted_frames"]))
            if cfg["corrupted_frames"]["enabled"]:
                for t in corrupted:
                    add("corrupted_frames", camera, [f"corrupted frame at {t / (rate or 1.0):.1f}s"], t / (rate or 1.0))
            if cfg["video_jump"]["enabled"]:
                vj = _params(cfg["video_jump"])
                around_corrupted = {i for t in corrupted for i in (t - 1, t)}
                video_jumps[camera] = find_video_jumps(p, rate, vj["jump_diff"], vj["jump_ratio"], around_corrupted)
        if cfg["choppy"]["enabled"]:
            add("choppy", camera, check_choppy(p, moving, **_params(cfg["choppy"])))
        if cfg["frozen"]["enabled"]:
            add("frozen", camera, check_frozen(p, moving, rate, **_params(cfg["frozen"])))
        if p.duration is not None:
            durations[camera] = p.duration
        diffs[camera] = p.diff

    # State jumps, split by whether the video jumps at the same moment: both -> time is
    # missing from the recording (recording_gap); state only -> a state glitch or the
    # state and video out of step (state_jump). Video-only jumps are video_jump, below.
    if step is not None and (cfg["recording_gap"]["enabled"] or cfg["state_jump"]["enabled"]):
        hz = control_freq if control_freq and control_freq > 0 else None
        for t, ratio in find_state_jumps(step, hz, **_params(cfg["state_jump"]), moving_fraction_of_p95=moving_fraction):
            time_s = (t + 1) / hz if hz else float(t + 1)
            when = f"{time_s:.1f}s" if hz else f"step {t + 1}"
            cams = [c for c, p in probes.items() if video_jumps_at(p, time_s, hz, **_params(cfg["recording_gap"]))]
            if cams and cfg["recording_gap"]["enabled"]:
                add("recording_gap", ALL_CAMERAS, [
                    f"robot state and video ({', '.join(cams)}) both jump at {when} "
                    f"({ratio:.0f}x the surrounding motion): time looks missing from the recording"
                ], time_s)
            elif not cams and cfg["state_jump"]["enabled"]:
                video_note = "but the video doesn't" if probes else "(no video to compare)"
                add("state_jump", ALL_CAMERAS, [
                    f"robot state jumps at {when} ({ratio:.0f}x the surrounding motion) {video_note}"
                ], time_s)

    # Image jumps the robot state doesn't share. Ones it does share are recording gaps
    # (reported above when the state jump is clear enough).
    if cfg["video_jump"]["enabled"]:
        hz = control_freq if control_freq and control_freq > 0 else None
        ratio = cfg["video_jump"]["state_jump_ratio"]
        for camera, times in video_jumps.items():
            for time_s in times:
                if step is None:
                    add("video_jump", camera, [f"image jump at {time_s:.1f}s (no robot state to compare)"], time_s)
                elif not state_jumps_at(step, time_s, hz, ratio):
                    add("video_jump", camera, [f"image jump at {time_s:.1f}s while the robot state changes smoothly"], time_s)

    checked = list(enabled)
    if cfg["camera_duration_mismatch"]["enabled"]:
        if len(durations) > 1:
            spread = max(durations.values()) - min(durations.values())
            if spread > cfg["camera_duration_mismatch"]["duration_tol_s"]:
                detail = ", ".join(f"{c} {d:.1f}s" for c, d in durations.items())
                add("camera_duration_mismatch", ALL_CAMERAS, [f"camera durations differ: {detail}"])
        else:
            # One camera has nothing to agree with - not applicable rather than passed.
            checked.remove("camera_duration_mismatch")
            skips.append(_skip(CATEGORY, "camera_duration_mismatch", ALL_CAMERAS, "fewer_than_two_cameras"))

    if cfg["periodic_skips"]["enabled"] and step is not None:
        skip_params = _params(cfg["periodic_skips"])
        add("periodic_skips", ALL_CAMERAS, check_periodic_skips(step, moving, states, actions, diffs, control_freq, **skip_params))

    return {"findings": findings, "skips": skips, "checked": [(CATEGORY, t) for t in checked]}
