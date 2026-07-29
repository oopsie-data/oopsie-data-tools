"""Convert the SOAR self-improvement dataset to ``oopsiedata_format_v1`` HDF5.

SOAR (https://auto-improvement.github.io) is a large WidowX / BridgeData-V2 dataset of
autonomously collected trajectories, ~68% of them failures auto-labeled by a VLM. A local
copy is laid out as:

  {source}/berkeley_robot_{0..4}/{task}/{policy}/{date}/{success|failure}/traj{N}/
    actions.npy        (T, 7) [dx, dy, dz, droll, dpitch, dyaw, gripper∈{0,1}] — delta EEF
    eef_poses.npy      (T, 7) [x, y, z, roll, pitch, yaw, gripper] — absolute proprio
    trajectory.mp4     256x256, 15 fps, frame_count == T — the robot camera
    goals.mp4 / combined.mp4   VLM goal images / viz — not robot cameras, ignored
    language_task.txt  natural-language instruction
    success.txt        "True" / "False"
    time.txt, task_list.txt, object_list.txt, robot_id.txt  (unused)

The source tree is treated as strictly read-only: this script only reads and copies out of it.

Action representation (dataset-owner decision): the oopsie action space is absolute
``cartesian_position``, while SOAR stores delta arm commands plus absolute proprio, so the
absolute action is reconstructed as the next proprio state —

    actions/cartesian_position[t] = eef_poses[t + 1, :6]   (last: eef_poses[-1,:6] + delta[-1,:6])
    actions/gripper_position[t]   = eef_poses[t + 1, 6]    (last: eef_poses[-1, 6])

Both the state and the action pose are euler_xyz in the source and are converted to
scalar-last quaternions here, since the validator requires a 7-DOF ``cartesian_position``.

There is no joint data in SOAR, and the profile does not claim any: ``joint_position`` is
only required when the action space is a joint space, which this one is not.

Output — one file per trajectory:

  {output_dir}/{episode_id}.h5
  {output_dir}/videos/{episode_id}_main.mp4

Usage:
    # Small sanity batch
    python convert_soar.py -s ~/soar-dataset-local -o ./converted \\
        --lab-id my_lab --operator-name SOAR_policy --annotator-name soar_vlm \\
        --max-episodes 5

    # Failures only, one robot
    python convert_soar.py -s ~/soar-dataset-local -o ./converted \\
        --lab-id my_lab --operator-name SOAR_policy --annotator-name soar_vlm \\
        --outcome failure --robot-id 1
"""

from __future__ import annotations

import argparse
import re
from functools import partial
from pathlib import Path

import h5py
import numpy as np
from _common import ConversionOutput, Skip, add_common_args, report, run_batch

from oopsie_data_tools.utils.conversion_utils import (
    to_quaternion_poses,
    write_actions,
    write_episode_annotations,
    write_robot_states,
    write_root_attrs,
    write_video_paths,
)
from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile
from oopsie_data_tools.utils.validation.episode_validator import (
    MAX_EPISODE_DURATION_S,
    MIN_EPISODE_DURATION_S,
)

CONTROL_FREQ = 15  # Hz — the rate trajectory.mp4 is encoded at
CAMERA_NAME = "main"
ACTION_SPACE = ["cartesian_position", "gripper_position"]
SOURCE_ORIENTATION = "euler_xyz"


def build_profile(policy_name: str) -> RobotProfile:
    return RobotProfile(
        policy_name=policy_name,
        robot_name="widowx_250s",
        gripper_name="widowx_250s_gripper",
        is_biarm=False,
        uses_mobile_base=False,
        control_freq=CONTROL_FREQ,
        camera_names=[CAMERA_NAME],
        # No joint_position: SOAR has none, and a cartesian action space does not need it.
        robot_state_keys=["cartesian_position", "gripper_position"],
        robot_state_joint_names=[],
        robot_state_orientation_representation="quat",
        action_space=list(ACTION_SPACE),
        # Poses are converted to quats below, so the profile describes what is on disk.
        orientation_representation="quat",
    )


def _sanitize(text: str) -> str:
    """Filesystem- and identifier-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "x"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _build_actions(eef_poses: np.ndarray, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute cartesian + gripper action = the next proprio state (see module docstring)."""
    length = eef_poses.shape[0]
    cartesian = np.empty((length, 6), dtype=np.float64)
    gripper = np.empty((length, 1), dtype=np.float64)

    if length > 1:
        cartesian[:-1] = eef_poses[1:, :6]
        gripper[:-1, 0] = eef_poses[1:, 6]
    cartesian[-1] = eef_poses[-1, :6] + deltas[-1, :6]
    gripper[-1, 0] = eef_poses[-1, 6]
    return cartesian, gripper


def episode_id_for(traj_dir_and_source: tuple[Path, Path], counter: int) -> str:
    """Descriptive, globally unique id built from the trajectory's path components."""
    traj_dir, source = traj_dir_and_source
    slug = "_".join(_sanitize(part) for part in traj_dir.relative_to(source).parts)
    return f"soar_{counter:06d}_{slug}"


def convert_one(
    item: tuple[Path, Path],
    episode_id: str,
    out: ConversionOutput,
    *,
    lab_id: str,
    operator_name: str,
    annotator_name: str,
) -> None:
    traj_dir, source = item

    deltas = np.load(traj_dir / "actions.npy")
    eef_poses = np.load(traj_dir / "eef_poses.npy")
    length = eef_poses.shape[0]

    duration_s = length / CONTROL_FREQ
    if not MIN_EPISODE_DURATION_S <= duration_s <= MAX_EPISODE_DURATION_S:
        raise Skip(
            f"T={length} is {duration_s:.2f}s, outside the validator's "
            f"[{MIN_EPISODE_DURATION_S}, {MAX_EPISODE_DURATION_S}]s"
        )

    # Path shape: .../{robot}/{task}/{policy}/{date}/{outcome}/traj{N}
    parts = traj_dir.relative_to(source).parts
    profile = build_profile(parts[2] if len(parts) >= 5 else "soar")

    # 256px at 15 fps is already in spec, so the video is copied rather than re-encoded.
    video = out.copy_video(traj_dir / "trajectory.mp4", episode_id, CAMERA_NAME)

    cartesian_action, gripper_action = _build_actions(eef_poses, deltas)
    success = 1.0 if _read_text(traj_dir / "success.txt").lower() == "true" else 0.0

    h5_path = out.episode_h5(episode_id)
    with h5py.File(h5_path, "w") as f:
        write_root_attrs(
            f,
            episode_id=episode_id,
            language_instruction=_read_text(traj_dir / "language_task.txt"),
            lab_id=lab_id,
            operator_name=operator_name,
            robot_profile=profile,
        )
        write_video_paths(f, {CAMERA_NAME: str(video)}, h5_path)
        write_robot_states(
            f,
            {
                "cartesian_position": to_quaternion_poses(
                    eef_poses[:, :6], SOURCE_ORIENTATION
                ),
                "gripper_position": eef_poses[:, 6:7],
            },
            profile.robot_state_keys,
        )
        write_actions(
            f,
            {
                "cartesian_position": to_quaternion_poses(
                    cartesian_action, SOURCE_ORIENTATION
                ),
                "gripper_position": gripper_action,
            },
            profile.action_space,
        )
        # The label is the VLM's, not a human's — hence a distinct --annotator-name.
        write_episode_annotations(f, annotator_name=annotator_name, success=success)


def discover(source: Path, robot_id: str | None, outcome: str) -> list[tuple[Path, Path]]:
    traj_dirs = sorted(p.parent for p in source.rglob("actions.npy"))
    if robot_id is not None:
        robot_dir = f"berkeley_robot_{robot_id}"
        traj_dirs = [d for d in traj_dirs if robot_dir in d.relative_to(source).parts]
    if outcome != "all":
        traj_dirs = [d for d in traj_dirs if outcome in d.relative_to(source).parts]
    return [(d, source) for d in traj_dirs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the SOAR dataset to oopsiedata_format_v1 HDF5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser, source_help="SOAR dataset root (read-only).")
    parser.add_argument(
        "--outcome",
        choices=["all", "success", "failure"],
        default="all",
        help="Filter by the VLM success label in the path (default: all).",
    )
    parser.add_argument(
        "--robot-id",
        default=None,
        help="Only convert berkeley_robot_<id> (e.g. 1). Default: every robot.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        parser.error(f"Source directory not found: {source}")

    out = ConversionOutput.create(args.output_dir)
    counts = run_batch(
        discover(source, args.robot_id, args.outcome),
        partial(
            convert_one,
            lab_id=args.lab_id,
            operator_name=args.operator_name,
            annotator_name=args.annotator_name,
        ),
        out=out,
        episode_id_for=episode_id_for,
        label=lambda item: item[0].name,
        start_id=args.start_id,
        desc="SOAR",
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
    )
    report(counts)


if __name__ == "__main__":
    main()
