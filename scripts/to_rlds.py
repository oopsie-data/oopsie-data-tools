"""Example script to convert OopsieData HDF5 episodes to an RLDS dataset.

The generated TensorFlow Datasets builder is written as::
    <output_dir>/<dataset_name>/1.0.0/


Example Usage::

    python scripts/to_rlds.py \
        /path/to/oopsie_hdf5 \
        /path/to/tfds_data \
        --dataset-name oopsie

-------------------------- PLEASE READ --------------------------

NOTE(zhouzypaul): this file assumes three input images. See _camera_mapping() below.
NOTE(zhouzypaul): this file logs action as joint_velocity (i.e. DROID setup)
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
import json
from pathlib import Path
import re
from typing import Any

import cv2
import h5py
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm


DATASET_NAME = "oopsie"
DATASET_VERSION = tfds.core.Version("1.0.0")
DEFAULT_IMAGE_HEIGHT = 180
DEFAULT_IMAGE_WIDTH = 320


def validate_dataset_name(dataset_name: str) -> str:
    """Validate and return a TFDS-compatible snake_case dataset name."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", dataset_name):
        raise ValueError(
            "dataset_name must start with a lowercase letter and contain only "
            "lowercase letters, digits, and underscores"
        )
    return dataset_name


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def discover_episodes(source_dir: Path, max_episodes: int | None = None) -> list[Path]:
    """Return a deterministic list of Oopsie episode files."""
    paths = sorted((*source_dir.rglob("*.h5"), *source_dir.rglob("*.hdf5")))
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise FileNotFoundError(f"No .h5 or .hdf5 files found under {source_dir}")
    return paths


def _dataset_array(group: h5py.Group, key: str, length: int, width: int) -> np.ndarray:
    """Read a trajectory array, using zeros for optional DROID compatibility fields."""
    dataset = group.get(key)
    if dataset is None or dataset.shape is None or dataset.size == 0:
        return np.zeros((length, width), dtype=np.float64)

    value = np.asarray(dataset, dtype=np.float64)
    if value.shape != (length, width):
        raise ValueError(
            f"Dataset {dataset.name} has shape {value.shape}; expected {(length, width)}"
        )
    return value


def _trajectory_length(file_handle: h5py.File) -> int:
    lengths: dict[str, int] = {}
    for group_name in ("observations/robot_states", "actions"):
        group = file_handle[group_name]
        for key, dataset in group.items():
            if dataset.shape is not None and dataset.ndim >= 1 and dataset.size > 0:
                lengths[f"{group_name}/{key}"] = int(dataset.shape[0])

    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Trajectory arrays have inconsistent lengths: {lengths}")
    if not unique_lengths:
        raise ValueError("Episode has no non-empty trajectory arrays")
    length = unique_lengths.pop()
    if length <= 0:
        raise ValueError("Episode has zero steps")
    return length


def _episode_success(file_handle: h5py.File) -> bool:
    annotations = file_handle.get("episode_annotations")
    values: list[bool] = []
    if annotations is not None:
        for annotation in annotations.values():
            if "success" in annotation.attrs:
                value = float(annotation.attrs["success"])
                # NOTE(zhouzypaul): this assumes the old oopsie format where success is binary
                if value not in (0.0, 1.0):
                    raise ValueError(f"Invalid success annotation {value!r}")
                values.append(bool(value))

    if not values:
        raise ValueError("Episode has no success annotation")
    if len(set(values)) != 1:
        raise ValueError(f"Episode has conflicting success annotations: {values}")
    return values[0]


def _video_paths(file_handle: h5py.File, h5_path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for camera_name, dataset in file_handle["observations/video_paths"].items():
        relative_path = Path(_decode_text(dataset[()]))
        video_path = relative_path if relative_path.is_absolute() else h5_path.parent / relative_path
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video for camera {camera_name!r} does not exist: {video_path}"
            )
        result[camera_name] = video_path
    return result


def _camera_mapping(video_paths: dict[str, Path]) -> dict[str, Path]:
    """Map arbitrary Oopsie camera names onto the three DROID image keys."""
    wrist_names = sorted(name for name in video_paths if "wrist" in name.lower())
    if not wrist_names:
        raise ValueError(f"Could not identify a wrist camera among {sorted(video_paths)}")

    exterior_names = sorted(
        (name for name in video_paths if name not in wrist_names),
        key=lambda name: (name.lower() != "left", name.lower() != "right", name),
    )
    if not exterior_names:
        raise ValueError(f"Could not identify an exterior camera among {sorted(video_paths)}")

    exterior_1 = exterior_names[0]
    exterior_2 = exterior_names[1] if len(exterior_names) > 1 else exterior_1
    return {
        "exterior_image_1_left": video_paths[exterior_1],
        "exterior_image_2_left": video_paths[exterior_2],
        "wrist_image_left": video_paths[wrist_names[0]],
    }


class _VideoReader:
    def __init__(self, path: Path, image_size: tuple[int, int]):
        self.path = path
        self.height, self.width = image_size
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video: {path}")

    def read(self, step: int) -> np.ndarray:
        ok, frame = self.capture.read()
        if not ok:
            raise ValueError(f"Video {self.path} ended before trajectory step {step}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (self.height, self.width):
            interpolation = (
                cv2.INTER_AREA
                if frame.shape[0] >= self.height and frame.shape[1] >= self.width
                else cv2.INTER_LINEAR
            )
            frame = cv2.resize(frame, (self.width, self.height), interpolation=interpolation)
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        self.capture.release()


def _read_video_frames(
    mapping: dict[str, Path], length: int, image_size: tuple[int, int]
) -> dict[str, list[np.ndarray]]:
    readers = {path: _VideoReader(path, image_size) for path in set(mapping.values())}
    frames = {key: [] for key in mapping}
    try:
        for step in range(length):
            by_path = {path: reader.read(step) for path, reader in readers.items()}
            for key, path in mapping.items():
                frames[key].append(by_path[path])
    finally:
        for reader in readers.values():
            reader.close()
    return frames


def _relative_metadata_path(source_dir: Path, h5_path: Path, success: bool) -> str:
    # oopsie-rl's DROID loader currently infers success from this string.
    outcome = "success" if success else "failure"
    return (Path(outcome) / h5_path.relative_to(source_dir)).as_posix()


def read_episode(
    h5_path: Path,
    source_dir: Path,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Read one Oopsie episode and return a DROID-shaped RLDS example."""
    with h5py.File(h5_path, "r") as file_handle:
        schema = _decode_text(file_handle.attrs.get("schema", ""))
        if schema != "oopsiedata_format_v1":
            raise ValueError(f"Unsupported schema {schema!r}; expected 'oopsiedata_format_v1'")

        length = _trajectory_length(file_handle)
        success = _episode_success(file_handle)
        robot_profile = json.loads(_decode_text(file_handle.attrs["robot_profile"]))
        language_instruction = _decode_text(file_handle.attrs["language_instruction"])

        observations = file_handle["observations/robot_states"]
        actions = file_handle["actions"]
        joint_position = _dataset_array(observations, "joint_position", length, 7)
        gripper_state = _dataset_array(observations, "gripper_position", length, 1)
        # cartesian_state = _dataset_array(observations, "cartesian_position", length, 6)

        joint_velocity = _dataset_array(actions, "joint_velocity", length, 7)
        joint_action = _dataset_array(actions, "joint_position", length, 7)
        gripper_action = _dataset_array(actions, "gripper_position", length, 1)
        # gripper_velocity = _dataset_array(actions, "gripper_velocity", length, 1)
        # cartesian_action = _dataset_array(actions, "cartesian_position", length, 6)
        # cartesian_velocity = _dataset_array(actions, "cartesian_velocity", length, 6)

        camera_mapping = _camera_mapping(_video_paths(file_handle, h5_path))
        images = _read_video_frames(camera_mapping, length, image_size)

        steps = []
        for step in range(length):
            is_last = step == length - 1
            steps.append(
                {
                    "observation": {
                        "exterior_image_1_left": images["exterior_image_1_left"][step],
                        "exterior_image_2_left": images["exterior_image_2_left"][step],
                        "wrist_image_left": images["wrist_image_left"][step],
                        "joint_position": joint_position[step],
                        "gripper_position": gripper_state[step],
                        # "cartesian_position": cartesian_state[step],
                    },
                    # Kept for standard RLDS tooling. oopsie-rl selects from action_dict.
                    "action": np.concatenate((joint_velocity[step], gripper_action[step])),
                    # NOTE(zhouzpaul): uncomment the below when you need different action spaces
                    "action_dict": {
                        "joint_position": joint_action[step],
                        "joint_velocity": joint_velocity[step],
                        "gripper_position": gripper_action[step],
                        # "gripper_velocity": gripper_velocity[step],
                        # "cartesian_position": cartesian_action[step],
                        # "cartesian_velocity": cartesian_velocity[step],
                    },
                    "reward": np.float32(1.0 if is_last and success else 0.0),
                    "is_first": step == 0,
                    "is_last": is_last,
                    "is_terminal": is_last and success,
                    "language_instruction": language_instruction,
                }
            )

        relative_parent = h5_path.parent.relative_to(source_dir).as_posix()
        return {
            "steps": steps,
            "episode_metadata": {
                "file_path": _relative_metadata_path(source_dir, h5_path, success),
                "recording_folderpath": relative_parent,
                "episode_id": _decode_text(file_handle.attrs.get("episode_id", h5_path.stem)),
                "lab_id": _decode_text(file_handle.attrs.get("lab_id", "")),
                "robot_name": _decode_text(robot_profile.get("robot_name", "")),
                "control_freq": np.float32(robot_profile.get("control_freq", 0.0)),
                "success": success,
            },
        }


class OopsieRLDSBuilder(tfds.core.GeneratorBasedBuilder):
    """TFDS builder backed by an existing tree of Oopsie HDF5 episodes."""

    VERSION = DATASET_VERSION
    RELEASE_NOTES = {"1.0.0": "Initial Oopsie HDF5 to RLDS conversion."}

    def __init__(
        self,
        *,
        source_dir: Path,
        episode_paths: Sequence[Path],
        image_size: tuple[int, int],
        dataset_name: str = DATASET_NAME,
        **kwargs: Any,
    ):
        self._source_dir = source_dir
        self._episode_paths = tuple(episode_paths)
        self._image_size = image_size
        # DatasetBuilder uses ``name`` while constructing both the output path
        # and DatasetInfo, so set the instance override before its initializer.
        self.name = validate_dataset_name(dataset_name)
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        height, width = self._image_size
        image = lambda doc: tfds.features.Image(  # noqa: E731
            shape=(height, width, 3),
            dtype=np.uint8,
            encoding_format="jpeg",
            doc=doc,
        )
        float_vector = lambda width, doc: tfds.features.Tensor(  # noqa: E731
            shape=(width,), dtype=np.float64, doc=doc
        )
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": {
                                "exterior_image_1_left": image("First exterior camera."),
                                "exterior_image_2_left": image("Second exterior camera."),
                                "wrist_image_left": image("Wrist camera."),
                                "joint_position": float_vector(7, "Observed joint positions."),
                                "gripper_position": float_vector(1, "Observed gripper position."),
                                # "cartesian_position": float_vector(6, "Observed Cartesian pose."),
                            },
                            "action": float_vector(8, "Joint velocity plus gripper position."),
                            "action_dict": {
                                "joint_position": float_vector(7, "Commanded joint positions."),
                                "joint_velocity": float_vector(7, "Commanded joint velocities."),
                                "gripper_position": float_vector(1, "Commanded gripper position."),
                                # "gripper_velocity": float_vector(1, "Commanded gripper velocity."),
                                # "cartesian_position": float_vector(6, "Commanded Cartesian pose."),
                                # "cartesian_velocity": float_vector(6, "Commanded Cartesian velocity."),
                            },
                            "reward": tfds.features.Scalar(np.float32),
                            "is_first": tfds.features.Scalar(np.bool_),
                            "is_last": tfds.features.Scalar(np.bool_),
                            "is_terminal": tfds.features.Scalar(np.bool_),
                            "language_instruction": tfds.features.Text(),
                            # "language_instruction_2": tfds.features.Text(),
                            # "language_instruction_3": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": {
                        "file_path": tfds.features.Text(),
                        "recording_folderpath": tfds.features.Text(),
                        "episode_id": tfds.features.Text(),
                        "lab_id": tfds.features.Text(),
                        "robot_name": tfds.features.Text(),
                        "control_freq": tfds.features.Scalar(np.float32),
                        "success": tfds.features.Scalar(np.bool_),
                    },
                }
            ),
            supervised_keys=None,
            homepage="https://oopsie-data.com/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        return {"train": self._generate_examples()}

    def _generate_examples(self) -> Iterator[tuple[str, dict[str, Any]]]:
        for h5_path in tqdm(self._episode_paths, desc="Converting episodes", unit="episode"):
            relative_path = h5_path.relative_to(self._source_dir).as_posix()
            try:
                yield relative_path, read_episode(h5_path, self._source_dir, self._image_size)
            except Exception as exc:
                raise RuntimeError(f"Failed to convert {h5_path}: {exc}") from exc


def convert(
    source_dir: Path,
    output_dir: Path,
    *,
    dataset_name: str = DATASET_NAME,
    max_episodes: int | None = None,
    image_size: tuple[int, int] = (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH),
    num_shards: int | None = None,
) -> Path:
    """Convert an Oopsie directory and return the generated builder directory."""
    source_dir = source_dir.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output directory must not be the source directory or one of its children")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image dimensions must be positive")
    dataset_name = validate_dataset_name(dataset_name)

    episode_paths = discover_episodes(source_dir, max_episodes=max_episodes)
    builder_dir = output_dir / dataset_name / str(DATASET_VERSION)
    if builder_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing RLDS builder directory: {builder_dir}"
        )

    builder = OopsieRLDSBuilder(
        data_dir=output_dir,
        source_dir=source_dir,
        episode_paths=episode_paths,
        image_size=image_size,
        dataset_name=dataset_name,
    )
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            try_download_gcs=False,
            num_shards=num_shards,
        )
    )
    return builder_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Directory containing Oopsie HDF5 episodes")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="TFDS data root; writes output_dir/<dataset-name>/1.0.0",
    )
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help=f"TFDS dataset name (default: {DATASET_NAME})",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert only the first N sorted episodes (useful for validation)",
    )
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Optional fixed number of TFRecord shards",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    builder_dir = convert(
        args.source_dir,
        args.output_dir,
        dataset_name=args.dataset_name,
        max_episodes=args.max_episodes,
        image_size=(args.image_height, args.image_width),
        num_shards=args.num_shards,
    )
    print(f"RLDS dataset written to {builder_dir}")


if __name__ == "__main__":
    main()
