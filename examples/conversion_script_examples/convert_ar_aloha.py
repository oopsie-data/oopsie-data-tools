"""Convert ACT / ALOHA episode HDF5 files to ``oopsiedata_format_v1`` HDF5.

Source layout (one file per episode, as written by the ACT / ACT++ recording stack):

  /action                                  (T, 14)  target joint positions, both arms
  /observations/qpos                       (T, 14)  measured joint positions, both arms
  /observations/qvel, /observations/effort  (T, 14) optional, unused here
  /success                                 (T,)     per-timestep success flag
  /compress_len                            (3, T)   JPEG byte lengths, one row per camera
  /observations/images/camera_high         (T, ...) frames, JPEG-compressed or raw RGB
  /observations/images/camera_wrist_left   (T, ...)
  /observations/images/camera_wrist_right  (T, ...)

Both 14-vectors are ``[6 arm joints + gripper]`` per arm. They are written out whole as
``joint_position``, matching the 14 entries in the profile's joint-name lists, with the two
gripper columns additionally written as ``gripper_position`` — the layout of the bundled
``configs/robot_profiles/act_plus_plus_robot_profile.yaml``.

There are no cartesian data in the source and the profile does not claim any: a joint action
space requires ``joint_position`` state, not ``cartesian_position``. Earlier versions of this
script wrote zero-filled cartesian arrays, which the validator now rejects as undeclared.

The images are decoded and re-encoded to one MP4 per camera at ``control_freq`` fps, and
downscaled if a side exceeds the 1280 px the validator allows.

``language_instruction`` is not stored in the source, so it is passed on the command line and
applies to every episode in the run.

Output — one file per source episode:

  {output_dir}/{episode_id}.h5
  {output_dir}/videos/{episode_id}_{camera}.mp4

Usage:
    # A single episode
    python convert_ar_aloha.py -s /data/aloha/episode_0.hdf5 -o ./converted \\
        --lab-id my_lab --operator-name alice --annotator-name act_eval \\
        --language "Pick up the ball"

    # Every episode in a directory
    python convert_ar_aloha.py -s /data/aloha/rollouts -o ./converted \\
        --lab-id my_lab --operator-name alice --annotator-name act_eval \\
        --language "Pick up the ball"
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import cv2
import h5py
import numpy as np
from _common import ConversionOutput, Skip, add_common_args, report, run_batch

from oopsie_data_tools.annotation_tool.episode_recorder import write_mp4
from oopsie_data_tools.utils.conversion_utils import (
    resize_frames,
    write_actions,
    write_episode_annotations,
    write_robot_states,
    write_root_attrs,
    write_video_paths,
)
from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile

DEFAULT_CONTROL_FREQ = 50  # Hz — ALOHA's teleop rate, and the video fps

# Column indices within the 14D qpos / action vectors.
GRIPPER_IDX = [6, 13]

# Source dataset → output camera name. The order is also the row order of /compress_len.
CAMERA_MAP = {
    "camera_high": "top",
    "camera_wrist_left": "left_wrist",
    "camera_wrist_right": "right_wrist",
}

JOINT_NAMES = [
    f"{joint}_{side}"
    for side in ("left", "right")
    for joint in (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
    )
]
ACTION_SPACE = ["joint_position", "gripper_position"]


def build_profile(policy_name: str, control_freq: int) -> RobotProfile:
    return RobotProfile(
        policy_name=policy_name,
        robot_name="viperx_300",
        gripper_name="viperx_300_gripper",
        is_biarm=True,
        uses_mobile_base=False,
        control_freq=control_freq,
        camera_names=list(CAMERA_MAP.values()),
        robot_state_keys=["joint_position", "gripper_position"],
        robot_state_joint_names=list(JOINT_NAMES),
        action_space=list(ACTION_SPACE),
        action_joint_names=list(JOINT_NAMES),
    )


def _decode_frames(raw: np.ndarray, lengths: np.ndarray | None) -> np.ndarray:
    """Return ``(T, H, W, 3)`` RGB frames from a camera dataset.

    ``lengths`` is the camera's ``/compress_len`` row; without it the dataset is assumed to
    hold raw RGB frames already, which is how uncompressed ALOHA recordings are stored.
    """
    if lengths is None:
        if raw.ndim != 4 or raw.shape[-1] != 3:
            raise ValueError(
                f"Camera dataset has shape {raw.shape} and the file has no /compress_len, "
                "so it is neither raw RGB nor decodable JPEG."
            )
        return np.asarray(raw, dtype=np.uint8)

    frames = []
    for t in range(raw.shape[0]):
        buffer = np.frombuffer(raw[t, : int(lengths[t])], dtype=np.uint8)
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode JPEG frame {t}")
        frames.append(bgr[:, :, ::-1])  # BGR → RGB
    return np.stack(frames, axis=0)


def convert_one(
    source_h5: Path,
    episode_id: str,
    out: ConversionOutput,
    *,
    lab_id: str,
    operator_name: str,
    annotator_name: str,
    language_instruction: str,
    policy_name: str,
    control_freq: int,
    success_aggregation: str,
) -> None:
    profile = build_profile(policy_name, control_freq)

    with h5py.File(source_h5, "r") as src:
        action = src["action"][()]
        qpos = src["observations/qpos"][()]

        success_per_step = src["success"][()]
        if success_per_step.size == 0:
            raise Skip("no /success values to aggregate")
        success = float(
            success_per_step[-1] if success_aggregation == "last" else success_per_step.max()
        )

        compress_len = src["compress_len"][()] if "compress_len" in src else None
        frames = {
            camera: _decode_frames(
                src[f"observations/images/{key}"][()],
                None if compress_len is None else compress_len[row],
            )
            for row, (key, camera) in enumerate(CAMERA_MAP.items())
        }

    video_paths = {}
    for camera, camera_frames in frames.items():
        destination = out.video_path(episode_id, camera)
        write_mp4(destination, resize_frames(camera_frames), float(control_freq))
        video_paths[camera] = str(destination)

    h5_path = out.episode_h5(episode_id)
    with h5py.File(h5_path, "w") as f:
        write_root_attrs(
            f,
            episode_id=episode_id,
            language_instruction=language_instruction,
            lab_id=lab_id,
            operator_name=operator_name,
            robot_profile=profile,
        )
        write_video_paths(f, video_paths, h5_path)
        write_robot_states(
            f,
            {
                "joint_position": qpos,
                "gripper_position": qpos[:, GRIPPER_IDX],
            },
            profile.robot_state_keys,
        )
        write_actions(
            f,
            {
                "joint_position": action,
                "gripper_position": action[:, GRIPPER_IDX],
            },
            profile.action_space,
        )
        write_episode_annotations(f, annotator_name=annotator_name, success=success)


def discover(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    files = sorted(p for p in source.iterdir() if p.suffix in {".h5", ".hdf5"})
    if not files:
        raise SystemExit(f"No .h5 or .hdf5 files found in {source}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ACT/ALOHA HDF5 episodes to oopsiedata_format_v1 HDF5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(
        parser,
        source_help="A single ACT/ALOHA .hdf5 episode, or a directory of them.",
    )
    parser.add_argument(
        "--language",
        "-l",
        required=True,
        help="Language instruction for every episode (the source files do not store one).",
    )
    parser.add_argument(
        "--policy-name",
        default="act_plus_plus",
        help="Value for robot_profile.policy_name (default: act_plus_plus).",
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=DEFAULT_CONTROL_FREQ,
        help=f"Control frequency in Hz, also the video fps (default: {DEFAULT_CONTROL_FREQ}).",
    )
    parser.add_argument(
        "--success-aggregation",
        choices=["max", "last"],
        default="max",
        help="How to collapse the per-timestep /success flag to a scalar (default: max).",
    )
    args = parser.parse_args()

    out = ConversionOutput.create(args.output_dir)
    counts = run_batch(
        discover(args.source.resolve()),
        partial(
            convert_one,
            lab_id=args.lab_id,
            operator_name=args.operator_name,
            annotator_name=args.annotator_name,
            language_instruction=args.language,
            policy_name=args.policy_name,
            control_freq=args.control_freq,
            success_aggregation=args.success_aggregation,
        ),
        out=out,
        label=lambda path: path.name,
        start_id=args.start_id,
        desc="ACT/ALOHA",
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
    )
    report(counts)


if __name__ == "__main__":
    main()
