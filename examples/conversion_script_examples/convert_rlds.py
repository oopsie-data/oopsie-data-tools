"""Convert RoboArena evaluation sessions to ``oopsiedata_format_v1`` HDF5.

Source layout (one session):

  {session_dir}/
    metadata.yaml                      — session-level metadata
    {letter}_{policy_name}/            — one directory per evaluated policy
      {policy_name}_{ts}_npz_file.npz  — per-step trajectory dicts
      {policy_name}_{ts}_video_left.mp4
      {policy_name}_{ts}_video_right.mp4
      {policy_name}_{ts}_video_wrist.mp4

``metadata.yaml`` keys used:

  language_instruction               → root attr
  policies.{letter}.policy_name      → robot_profile.policy_name
  policies.{letter}.binary_success   → episode_annotations success

``evaluation_location`` and ``evaluator_name`` are deliberately not carried over: ``lab_id``
has to be the id you registered with, so it comes from ``--lab-id``.

NPZ ``data`` array — a list of per-step dicts with ``cartesian_position`` (6,) as
xyz + euler_xyz, ``joint_position`` (7,), ``gripper_position`` (1,), and ``action`` (8,)
holding 7 joint velocities plus a gripper position.

The euler orientation is converted to a scalar-last quaternion before writing: the
validator requires a 7-DOF ``cartesian_position``, and nothing downstream of a
direct-to-HDF5 converter performs that conversion for you.

Output — one file per policy per session:

  {output_dir}/{episode_id}.h5
  {output_dir}/videos/{episode_id}_{camera}.mp4

Usage:
    # One session, or a parent directory of sessions — both are accepted
    python convert_rlds.py -s /data/roboarena/sessions -o ./converted \\
        --lab-id my_lab --operator-name alice --annotator-name roboarena_eval

    # Continue numbering an earlier run
    python convert_rlds.py -s /data/roboarena/sessions -o ./converted \\
        --lab-id my_lab --operator-name alice --annotator-name roboarena_eval \\
        --start-id 500
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import yaml
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

CONTROL_FREQ = 10  # Hz — matches the source MP4 frame rate
CAMERA_NAMES = ["left", "right", "wrist"]
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
ACTION_SPACE = ["joint_velocity", "gripper_position"]
SOURCE_ORIENTATION = "euler_xyz"


@dataclass(frozen=True)
class PolicyEpisode:
    """One evaluated policy inside one session — the unit this script converts."""

    session_dir: Path
    letter: str
    policy_name: str
    binary_success: float
    language_instruction: str

    @property
    def policy_dir(self) -> Path:
        return self.session_dir / f"{self.letter}_{self.policy_name}"

    def __str__(self) -> str:
        return f"{self.session_dir.name}/{self.letter}_{self.policy_name}"


def build_profile(policy_name: str, camera_names: list[str]) -> RobotProfile:
    """The profile documents the episode, so it lists only the cameras actually present."""
    return RobotProfile(
        policy_name=policy_name,
        robot_name="franka_research_3",
        gripper_name="robotiq_2f_85",
        is_biarm=False,
        uses_mobile_base=False,
        control_freq=CONTROL_FREQ,
        camera_names=camera_names,
        robot_state_keys=["joint_position", "gripper_position", "cartesian_position"],
        robot_state_joint_names=list(JOINT_NAMES),
        # Quat, not euler_xyz: the conversion happens here, so what lands on disk is a quat.
        robot_state_orientation_representation="quat",
        action_space=list(ACTION_SPACE),
        action_joint_names=list(JOINT_NAMES),
    )


def _load_npz_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    """Stack the per-step dicts in the NPZ ``data`` array into ``(T, D)`` float64 arrays."""
    data = np.load(npz_path, allow_pickle=True)["data"]
    length = len(data)
    if length == 0:
        raise Skip("empty trajectory")

    joint_position = np.empty((length, 7), dtype=np.float64)
    gripper_position = np.empty((length, 1), dtype=np.float64)
    cartesian_position = np.empty((length, 6), dtype=np.float64)
    joint_velocity = np.empty((length, 7), dtype=np.float64)
    gripper_action = np.empty((length, 1), dtype=np.float64)

    for t, step in enumerate(data):
        joint_position[t] = step["joint_position"]
        gripper_position[t, 0] = step["gripper_position"][0]
        cartesian_position[t] = step["cartesian_position"]
        action = step["action"]
        joint_velocity[t] = action[:7]
        gripper_action[t, 0] = action[7]

    return {
        "joint_position": joint_position,
        "gripper_position": gripper_position,
        "cartesian_position": cartesian_position,
        "joint_velocity": joint_velocity,
        "gripper_action": gripper_action,
    }


def _find_npz(policy_dir: Path) -> Path:
    matches = sorted(policy_dir.glob("*_npz_file.npz"))
    if not matches:
        raise Skip(f"no *_npz_file.npz in {policy_dir.name}")
    return matches[0]


def convert_one(
    item: PolicyEpisode,
    episode_id: str,
    out: ConversionOutput,
    *,
    lab_id: str,
    operator_name: str,
    annotator_name: str,
) -> None:
    if not item.policy_dir.is_dir():
        raise Skip(f"policy directory not found: {item.policy_dir.name}")

    # Not every episode has all three cameras; the profile has to describe what is there.
    sources = {
        camera: matches[0]
        for camera in CAMERA_NAMES
        if (matches := sorted(item.policy_dir.glob(f"*_video_{camera}.mp4")))
    }
    if not sources:
        raise Skip(f"no camera videos in {item.policy_dir.name}")

    arrays = _load_npz_arrays(_find_npz(item.policy_dir))
    profile = build_profile(item.policy_name, list(sources))

    video_paths = {
        camera: str(out.copy_video(source, episode_id, camera))
        for camera, source in sources.items()
    }

    h5_path = out.episode_h5(episode_id)
    with h5py.File(h5_path, "w") as f:
        write_root_attrs(
            f,
            episode_id=episode_id,
            language_instruction=item.language_instruction,
            lab_id=lab_id,
            operator_name=operator_name,
            robot_profile=profile,
        )
        write_video_paths(f, video_paths, h5_path)
        write_robot_states(
            f,
            {
                "joint_position": arrays["joint_position"],
                "gripper_position": arrays["gripper_position"],
                "cartesian_position": to_quaternion_poses(
                    arrays["cartesian_position"], SOURCE_ORIENTATION
                ),
            },
            profile.robot_state_keys,
        )
        write_actions(
            f,
            {
                "joint_velocity": arrays["joint_velocity"],
                "gripper_position": arrays["gripper_action"],
            },
            profile.action_space,
        )
        write_episode_annotations(
            f,
            annotator_name=annotator_name,
            success=item.binary_success,
        )


def discover(source: Path) -> list[PolicyEpisode]:
    """Flatten sessions into policy episodes, in a stable order."""
    if (source / "metadata.yaml").is_file():
        session_dirs = [source]
    else:
        session_dirs = sorted(
            d for d in source.iterdir() if d.is_dir() and (d / "metadata.yaml").is_file()
        )
    if not session_dirs:
        raise SystemExit(f"No session directories with metadata.yaml found under {source}")

    episodes: list[PolicyEpisode] = []
    for session_dir in session_dirs:
        meta = yaml.safe_load((session_dir / "metadata.yaml").read_text())
        for letter in sorted(meta["policies"]):
            policy = meta["policies"][letter]
            episodes.append(
                PolicyEpisode(
                    session_dir=session_dir,
                    letter=letter,
                    policy_name=policy["policy_name"],
                    binary_success=float(policy["binary_success"]),
                    language_instruction=meta["language_instruction"],
                )
            )
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RoboArena evaluation sessions to oopsiedata_format_v1 HDF5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(
        parser,
        source_help=(
            "A single session directory (containing metadata.yaml), or a parent "
            "directory of session directories."
        ),
    )
    args = parser.parse_args()

    out = ConversionOutput.create(args.output_dir)
    episodes = discover(args.source.resolve())

    counts = run_batch(
        episodes,
        partial(
            convert_one,
            lab_id=args.lab_id,
            operator_name=args.operator_name,
            annotator_name=args.annotator_name,
        ),
        out=out,
        start_id=args.start_id,
        desc="RoboArena",
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
    )
    report(counts)


if __name__ == "__main__":
    main()
