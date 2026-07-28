#!/usr/bin/env python3
"""Convert MolmoAct2 LeRobot eval rollouts to Oopsie HDF5 episodes.

The default source is ``allenai/eval_molmoact_candy_sorting_ood``.  The source
dataset stores trajectory rows in Parquet and shared long-form camera videos in
LeRobot format; this script writes one native ``oopsiedata_format_v1`` HDF5 file
and one clipped MP4 per camera for each episode.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2  # noqa: F401 - imported to keep video backend dependency explicit for users
import h5py
import imageio_ffmpeg
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from oopsie_data_tools.utils.robot_profile.robot_profile import (
    RobotProfile,
    robot_profile_to_json,
)

DEFAULT_REPO_ID = "allenai/eval_molmoact_candy_sorting_ood"
DEFAULT_REVISION = "main"
DEFAULT_SPLIT = "train"
DEFAULT_EPISODE_PREFIX = "episode"
SCHEMA_VERSION = "oopsiedata_format_v1"

CAMERA_KEYS = {
    "left": "observation.images.left",
    "right": "observation.images.right",
    "top": "observation.images.top",
}
CAMERA_NAMES = ["left", "right", "top"]

ALL_ACTION_KEYS = [
    "cartesian_position",
    "cartesian_velocity",
    "joint_position",
    "joint_velocity",
    "base_position",
    "base_velocity",
    "gripper_velocity",
    "gripper_position",
    "gripper_binary",
]

DEFAULT_JOINT_NAMES = [
    "left_joint_0.pos",
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_gripper.pos",
    "right_joint_0.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_gripper.pos",
]


DownloadFile = Callable[[str], Path]


def _import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required for MolmoAct2/LeRobot parquet conversion. "
            "Install it with `uv sync --extra lerobot`, or run this script with "
            "`uv run --with pyarrow python scripts/dataset_conversion/"
            "convert_molmoact2_eval.py ...`."
        ) from exc
    return pq


def _sorted_matching(files: Iterable[str], prefix: str, suffix: str) -> List[str]:
    return sorted(path for path in files if path.startswith(prefix) and path.endswith(suffix))


def _list_hf_files(repo_id: str, revision: str) -> List[str]:
    return HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)


def _make_hf_downloader(repo_id: str, revision: str) -> DownloadFile:
    def download_file(filename: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                revision=revision,
            )
        )

    return download_file


def _load_json(download_file: DownloadFile, filename: str) -> Dict[str, Any]:
    with open(download_file(filename), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_task_map(pq: Any, download_file: DownloadFile, files: Sequence[str]) -> Dict[int, str]:
    if "meta/tasks.parquet" not in files:
        return {}

    table = pq.read_table(download_file("meta/tasks.parquet"))
    task_map: Dict[int, str] = {}
    for row in table.to_pylist():
        task_index = row.get("task_index")
        if task_index is None:
            continue
        task_text = (
            row.get("task")
            or row.get("task_name")
            or row.get("language_instruction")
            or row.get("__index_level_0__")
        )
        if not task_text:
            for key, value in row.items():
                if key != "task_index" and isinstance(value, str) and value.strip():
                    task_text = value
                    break
        if task_text:
            task_map[int(task_index)] = str(task_text)
    return task_map


def _load_episode_records(
    pq: Any, download_file: DownloadFile, files: Sequence[str]
) -> List[Dict[str, Any]]:
    episode_files = _sorted_matching(files, "meta/episodes/", ".parquet")
    if not episode_files:
        raise FileNotFoundError("No meta/episodes/**/*.parquet files found in source repo")

    records: List[Dict[str, Any]] = []
    for filename in episode_files:
        table = pq.read_table(download_file(filename))
        records.extend(table.to_pylist())

    if not records:
        raise ValueError("Episode metadata parquet files contained no rows")
    return sorted(records, key=lambda row: int(row["episode_index"]))


def _load_data_by_episode(
    pq: Any, download_file: DownloadFile, files: Sequence[str]
) -> Dict[int, List[Dict[str, Any]]]:
    data_files = _sorted_matching(files, "data/", ".parquet")
    if not data_files:
        raise FileNotFoundError("No data/**/*.parquet files found in source repo")

    data_by_episode: Dict[int, List[Dict[str, Any]]] = {}
    columns = [
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    for filename in data_files:
        table = pq.read_table(download_file(filename), columns=columns)
        for row in table.to_pylist():
            episode_index = int(row["episode_index"])
            data_by_episode.setdefault(episode_index, []).append(row)

    for rows in data_by_episode.values():
        rows.sort(key=lambda row: int(row.get("frame_index", row.get("index", 0))))
    return data_by_episode


def _parse_split_bounds(info: Dict[str, Any], split: str) -> Optional[Tuple[int, int]]:
    splits = info.get("splits", {})
    if not splits:
        return None
    if split not in splits:
        raise ValueError(f"Split {split!r} not found in source metadata: {sorted(splits)}")

    raw = splits[split]
    if not isinstance(raw, str) or ":" not in raw:
        return None

    start_raw, end_raw = raw.split(":", 1)
    start = int(start_raw) if start_raw else 0
    end = int(end_raw) if end_raw else None
    if end is None:
        return None
    return start, end


def _filter_episodes_for_split(
    episodes: Sequence[Dict[str, Any]], info: Dict[str, Any], split: str
) -> List[Dict[str, Any]]:
    bounds = _parse_split_bounds(info, split)
    if bounds is None:
        return list(episodes)
    start, end = bounds
    return [
        episode
        for episode in episodes
        if start <= int(episode["episode_index"]) < end
    ]


def _feature_names(info: Dict[str, Any], feature_name: str) -> List[str]:
    features = info.get("features", {})
    names = features.get(feature_name, {}).get("names")
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return list(DEFAULT_JOINT_NAMES)


def _robot_profile_from_info(info: Dict[str, Any]) -> RobotProfile:
    fps = int(round(float(info.get("fps", 30))))
    robot_name = str(info.get("robot_type") or "bi_yam_follower")
    return RobotProfile(
        policy_name="molmoact2_eval",
        robot_name=robot_name,
        is_biarm=True,
        uses_mobile_base=False,
        gripper_name="bi_yam_follower_grippers",
        control_freq=fps,
        camera_names=list(CAMERA_NAMES),
        robot_state_keys=["joint_position", "gripper_position"],
        robot_state_joint_names=_feature_names(info, "observation.state"),
        action_space=["joint_position", "gripper_position"],
        action_joint_names=_feature_names(info, "action"),
    )


def _rows_to_array(rows: Sequence[Dict[str, Any]], key: str, episode_index: int) -> np.ndarray:
    arr = np.asarray([row[key] for row in rows], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 14:
        raise ValueError(
            f"Episode {episode_index} field {key!r} must have shape (T, 14), got {arr.shape}"
        )
    return arr


def _language_instruction(
    episode: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    task_map: Dict[int, str],
) -> str:
    tasks = episode.get("tasks")
    if isinstance(tasks, list) and tasks:
        text = str(tasks[0]).strip()
        if text:
            return text

    if rows:
        task_index = int(rows[0].get("task_index", -1))
        text = task_map.get(task_index, "").strip()
        if text:
            return text

    raise ValueError(
        f"Could not determine language_instruction for episode {episode.get('episode_index')}"
    )


def _source_video_path(info: Dict[str, Any], video_key: str, chunk_index: int, file_index: int) -> str:
    template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    return template.format(
        video_key=video_key,
        chunk_index=int(chunk_index),
        file_index=int(file_index),
    )


def _clip_video(
    source_path: Path,
    output_path: Path,
    start_seconds: float,
    frame_count: int,
    fps: float,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.9f}",
        "-i",
        str(source_path),
        "-frames:v",
        str(frame_count),
        "-an",
        "-vf",
        f"fps={fps:g}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed while clipping "
            f"{source_path} -> {output_path}:\n{result.stderr.strip()}"
        )
    os.replace(tmp_path, output_path)


def _write_h5(
    output_h5_path: Path,
    episode_id: str,
    language_instruction: str,
    lab_id: str,
    operator_name: str,
    robot_profile: RobotProfile,
    observations: np.ndarray,
    actions: np.ndarray,
    video_rel_paths: Dict[str, str],
    source_attrs: Dict[str, Any],
) -> None:
    str_dtype = h5py.string_dtype(encoding="utf-8")
    T = observations.shape[0]
    gripper_placeholder = np.zeros((T, 1), dtype=np.float64)

    with h5py.File(output_h5_path, "w") as f:
        f.attrs["schema"] = SCHEMA_VERSION
        f.attrs["episode_id"] = episode_id
        f.attrs["language_instruction"] = language_instruction
        f.attrs["lab_id"] = lab_id
        f.attrs["operator_name"] = operator_name
        f.attrs.create("robot_profile", robot_profile_to_json(robot_profile), dtype=str_dtype)
        f.attrs["timestamp"] = float(time.time())
        for key, value in source_attrs.items():
            f.attrs[key] = value

        obs_group = f.create_group("observations")
        video_group = obs_group.create_group("video_paths")
        for camera_name in CAMERA_NAMES:
            video_group.create_dataset(
                camera_name,
                data=video_rel_paths[camera_name],
                dtype=str_dtype,
            )

        robot_states = obs_group.create_group("robot_states")
        robot_states.create_dataset("joint_position", data=observations, dtype=np.float64)
        robot_states.create_dataset(
            "gripper_position", data=gripper_placeholder, dtype=np.float64
        )

        action_group = f.create_group("actions")
        for action_key in ALL_ACTION_KEYS:
            if action_key == "joint_position":
                action_group.create_dataset(action_key, data=actions, dtype=np.float64)
            elif action_key == "gripper_position":
                action_group.create_dataset(
                    action_key, data=gripper_placeholder, dtype=np.float64
                )
            else:
                action_group.create_dataset(action_key, data=h5py.Empty(dtype=np.float64))


def _convert_episode(
    *,
    episode: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    task_map: Dict[int, str],
    info: Dict[str, Any],
    download_file: DownloadFile,
    output_dir: Path,
    episode_id: str,
    lab_id: str,
    operator_name: str,
    robot_profile: RobotProfile,
    repo_id: str,
    revision: str,
    overwrite: bool,
) -> Path:
    episode_index = int(episode["episode_index"])
    expected_length = int(episode["length"])
    if len(rows) != expected_length:
        raise ValueError(
            f"Episode {episode_index} metadata length is {expected_length}, "
            f"but data rows contain {len(rows)} frames"
        )

    output_h5_path = output_dir / f"{episode_id}.h5"
    observations = _rows_to_array(rows, "observation.state", episode_index)
    actions = _rows_to_array(rows, "action", episode_index)
    language_instruction = _language_instruction(episode, rows, task_map)
    fps = float(info.get("fps", robot_profile.control_freq))

    videos_dir = output_dir / "videos"
    video_rel_paths: Dict[str, str] = {}
    for camera_name, video_key in CAMERA_KEYS.items():
        prefix = f"videos/{video_key}"
        source_video = _source_video_path(
            info=info,
            video_key=video_key,
            chunk_index=int(episode[f"{prefix}/chunk_index"]),
            file_index=int(episode[f"{prefix}/file_index"]),
        )
        source_video_path = download_file(source_video)
        output_video_path = videos_dir / f"{episode_id}_{camera_name}.mp4"
        _clip_video(
            source_path=source_video_path,
            output_path=output_video_path,
            start_seconds=float(episode[f"{prefix}/from_timestamp"]),
            frame_count=expected_length,
            fps=fps,
            overwrite=overwrite,
        )
        video_rel_paths[camera_name] = os.path.relpath(
            output_video_path.resolve(), start=output_h5_path.parent.resolve()
        ).replace(os.sep, "/")

    source_attrs = {
        "source_repo_id": repo_id,
        "source_revision": revision,
        "source_episode_index": episode_index,
        "source_dataset_from_index": int(episode.get("dataset_from_index", -1)),
        "source_dataset_to_index": int(episode.get("dataset_to_index", -1)),
    }
    _write_h5(
        output_h5_path=output_h5_path,
        episode_id=episode_id,
        language_instruction=language_instruction,
        lab_id=lab_id,
        operator_name=operator_name,
        robot_profile=robot_profile,
        observations=observations,
        actions=actions,
        video_rel_paths=video_rel_paths,
        source_attrs=source_attrs,
    )
    return output_h5_path


def convert_molmoact2_eval(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    split: str = DEFAULT_SPLIT,
    output_dir: Path,
    lab_id: str = "",
    operator_name: str = "",
    max_episodes: Optional[int] = None,
    start_id: int = 0,
    episode_prefix: str = DEFAULT_EPISODE_PREFIX,
    overwrite: bool = False,
    repo_files: Optional[Sequence[str]] = None,
    download_file: Optional[DownloadFile] = None,
) -> List[Path]:
    """Convert a MolmoAct2/LeRobot eval dataset into Oopsie episode files."""
    if max_episodes is not None and max_episodes < 0:
        raise ValueError("--max-episodes must be >= 0")

    pq = _import_pyarrow_parquet()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(repo_files) if repo_files is not None else _list_hf_files(repo_id, revision)
    downloader = download_file or _make_hf_downloader(repo_id, revision)

    info = _load_json(downloader, "meta/info.json")
    task_map = _load_task_map(pq, downloader, files)
    episodes = _filter_episodes_for_split(
        _load_episode_records(pq, downloader, files),
        info=info,
        split=split,
    )
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    data_by_episode = _load_data_by_episode(pq, downloader, files)
    robot_profile = _robot_profile_from_info(info)

    written: List[Path] = []
    for output_offset, episode in enumerate(episodes):
        output_index = start_id + output_offset
        episode_id = f"{episode_prefix}_{output_index:06d}"
        output_h5_path = output_dir / f"{episode_id}.h5"
        if output_h5_path.exists() and not overwrite:
            print(f"Skipping existing episode: {output_h5_path}")
            continue

        episode_index = int(episode["episode_index"])
        rows = data_by_episode.get(episode_index, [])
        print(
            f"[{output_offset + 1}/{len(episodes)}] "
            f"Converting source episode {episode_index} -> {episode_id}"
        )
        written.append(
            _convert_episode(
                episode=episode,
                rows=rows,
                task_map=task_map,
                info=info,
                download_file=downloader,
                output_dir=output_dir,
                episode_id=episode_id,
                lab_id=lab_id,
                operator_name=operator_name,
                robot_profile=robot_profile,
                repo_id=repo_id,
                revision=revision,
                overwrite=overwrite,
            )
        )

    print(f"Done. Wrote {len(written)} Oopsie episode file(s) to: {output_dir}")
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MolmoAct2 LeRobot eval rollouts to Oopsie HDF5 episodes."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo ID.")
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hugging Face revision, branch, tag, or commit SHA.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Source split to convert.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated .h5 files and clipped videos.",
    )
    parser.add_argument(
        "--lab-id",
        default="",
        help="Oopsie lab_id root attribute (default: empty string).",
    )
    parser.add_argument(
        "--operator-name",
        default="",
        help="Oopsie operator_name root attribute (default: empty string).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on converted episodes.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Starting numeric suffix for generated episode IDs.",
    )
    parser.add_argument(
        "--episode-prefix",
        default=DEFAULT_EPISODE_PREFIX,
        help="Prefix for generated episode IDs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5/video outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_molmoact2_eval(
        repo_id=args.repo_id,
        revision=args.revision,
        split=args.split,
        output_dir=args.output_dir,
        lab_id=args.lab_id,
        operator_name=args.operator_name,
        max_episodes=args.max_episodes,
        start_id=args.start_id,
        episode_prefix=args.episode_prefix,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
