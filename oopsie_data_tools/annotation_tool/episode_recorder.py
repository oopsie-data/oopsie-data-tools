"""Episode recorder for saving robot evaluation data in HDF5 format."""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

import h5py
import imageio
import numpy as np

from oopsie_data_tools.annotation_tool.annotation_schema import write_annotation_attrs
from oopsie_data_tools.utils.contributor_config import read_contributor_config
from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile, robot_profile_to_json
from oopsie_data_tools.utils.robot_profile.rotation_utils import ActionQuatConversion
from oopsie_data_tools.utils.validation.episode_data import EpisodeData, VideoInfo
from oopsie_data_tools.utils.validation.episode_validator import validate_episode

logger = logging.getLogger(__name__)

REQUIRED_OBSERVATION_KEYS = ["robot_state", "image_observation"]

VALID_ACTION_KEYS = {
    "cartesian_position",
    "cartesian_velocity",
    "joint_position",
    "joint_velocity",
    "base_position",
    "base_velocity",
    "gripper_velocity",
    "gripper_position",
    "gripper_binary",
}


def write_mp4(video_path: Path, frames: np.ndarray, fps: float) -> None:
    """Write RGB frames to an MP4 file.

    Args:
        video_path (Path): Destination path for the MP4 file.
        frames (np.ndarray): Video frames with shape ``(T, H, W, 3)``.
        fps (float): Output video frame rate.

    Raises:
        ValueError: If ``frames`` does not have shape ``(T, H, W, 3)`` or
            contains zero timesteps.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape (T,H,W,3), got {frames.shape}")
    if frames.shape[0] == 0:
        raise ValueError("Cannot write video with zero frames")

    with imageio.get_writer(
        str(video_path),
        format="FFMPEG",
        mode="I",
        fps=float(fps),
        codec="libx264",
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


class EpisodeRecorder:
    """Record rollout observations and persist them in HDF5 format.

    The recorder stores proprioception, optional video references, and
    annotation metadata under the robotic failure upload schema.
    """

    def __init__(
        self,
        robot_profile: RobotProfile,
        data_root_dir: Path | str,
        operator_name: str,
        resume_session_name: str | None = None,
    ) -> None:
        """Initialize a recorder instance.

        Args:
            robot_profile (RobotProfile): Robot profile.
            data_root_dir (str): Base output directory for saved artifacts.
            operator_name (str): Name of the operator recording the episode.
            session_name (str | None): Optional unique session name

        Raises:
            ValueError: If ``data_root_dir`` is not a valid directory.
        """
        self.data_root_dir = Path(data_root_dir)
        self.session_name = (
            resume_session_name
            if resume_session_name is not None
            else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.session_dir = self.data_root_dir / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.operator_name = operator_name

        self.robot_profile = robot_profile
        self.camera_names = robot_profile.camera_names

        self.quat_conversion = (
            ActionQuatConversion(
                self.robot_profile.get_rot_option(),
                is_biarm=self.robot_profile.is_biarm,
            )
            if self.robot_profile.orientation_representation
            else None
        )
        self.robot_state_quat_conversion = (
            ActionQuatConversion(
                self.robot_profile.get_robot_state_rot_option(),
                is_biarm=self.robot_profile.is_biarm,
            )
            if self.robot_profile.robot_state_orientation_representation
            else None
        )
        self.frames: dict[str, list[np.ndarray]] = {}
        self.timesteps: list[dict[str, Any]] = []
        self.timestamp: float = 0.0
        self.save_fname: str = ""
        # A fresh recorder is ready to record. This used to be left to an explicit
        # reset_episode_recorder() call, so going straight to record_step() raised
        # AttributeError on save_fname — an undocumented ordering requirement.
        self.reset_episode_recorder()

        self.lab_id, _ = read_contributor_config()

    def reset_episode_recorder(self) -> None:
        """Reset the buffers and start a new episode."""
        ts = datetime.datetime.now()
        self.timestamp = ts.timestamp()
        self.save_fname = f"{ts.strftime('%Y%m%d_%H%M%S')}"
        self.frames = {cam: [] for cam in self.camera_names}
        self.timesteps = []

    def record_step(
        self, observation: dict[str, Any], action: dict[str, np.ndarray]
    ) -> None:
        """Append one rollout timestep to in-memory buffers.

        Args:
            observation (dict[str, Any]): Observation payload containing state
                and optional images.
            action (dict[str, np.ndarray]): Dictionary of action vector applied at this timestep.

        Returns:
            None: This method only updates in-memory buffers.
        """
        # TODO: Make sure all checks are present here
        # Returns normalized copies; the caller's dicts are left untouched.
        robot_state, action = self._check_and_normalize_step_data(observation, action)

        # Buffer frames for each configured camera (if available)
        for cam in self.camera_names:
            frame = self._get_camera_frame(observation["image_observation"], cam)
            if frame is not None:
                self.frames[cam].append(np.asarray(frame, dtype=np.uint8))

        # Buffer timestep data
        step_data = {"robot_state": {}, "action_dict": {}}
        for key in self.robot_profile.robot_state_keys:
            step_data["robot_state"][key] = np.asarray(robot_state[key], dtype=np.float32)
        step_data["action_dict"] = {
            "cartesian_position": action.get("cartesian_position", None),
            "cartesian_velocity": action.get("cartesian_velocity", None),
            "joint_position": action.get("joint_position", None),
            "joint_velocity": action.get("joint_velocity", None),
            "base_position": action.get("base_position", None),
            "base_velocity": action.get("base_velocity", None),
            "gripper_velocity": action.get("gripper_velocity", None),
            "gripper_position": action.get("gripper_position", None),
            "gripper_binary": action.get("gripper_binary", None),
        }
        self.timesteps.append(step_data)

    def _save_videos(self) -> dict[str, str]:
        """Write one MP4 per camera into the session directory.

        Returns:
            ``{camera: absolute mp4 path}``. Cameras that buffered no frames are skipped
            rather than crashing in ``np.stack`` on an empty list.
        """
        video_paths: dict[str, str] = {}
        self.session_dir.mkdir(parents=True, exist_ok=True)
        fps = float(self.robot_profile.control_freq)
        for cam_name, frames in self.frames.items():
            if not frames:
                continue
            video_path = self.session_dir / f"{self.save_fname}_{cam_name}.mp4"
            write_mp4(video_path=video_path, frames=np.asarray(frames), fps=fps)
            video_paths[cam_name] = str(video_path.resolve())
        return video_paths

    def finish_rollout(self, instruction: str, success: float | None = None) -> None:
        data = {
                "language_instruction": instruction,
                "metadata": {
                    "episode_id": self.save_fname,
                    "operator_name": self.operator_name,
                },
        }
        if success is not None:
            data["episode_annotations"] = {
                self.operator_name: {
                    "schema": "oopsie_failure_taxonomy_v1",
                    "source": "human",
                    "timestamp": datetime.datetime.now().timestamp(),
                    "failure_description": "",
                    "taxonomy_schema": "none",
                    "taxonomy": json.dumps(
                        {}, ensure_ascii=False,
                    ),
                    "additional_notes": "",
                    "success": success,
                },
            }
        self._validate_pre_save(data)
        # 1. Save videos under the recorder's per-session folder
        video_paths = self._save_videos()

        data["video_paths"] = video_paths

        self.save(
            data
        )

    def save(self, data: dict[str, Any]) -> Path:
        """Persist the currently buffered episode to disk.

        Args:
            metadata (dict[str, Any]): Save metadata containing language and
                annotation fields.

        Returns:
            Path: Path to the written HDF5 episode file.

        Raises:
            ValueError: If no rollout steps were recorded.
        """
        if len(self.timesteps) == 0:
            raise ValueError("No steps recorded. Call record_step() first.")

        # Converts the absolute paths finish_rollout() hands over into paths relative to
        # the episode file, which is what the HDF5 stores. Batch 4 folds this together with
        # _save_videos so there is one writer.
        data["video_paths"] = self._resolve_video_paths(
            output_dir=self.session_dir,
            provided_video_paths=data.get("video_paths", {}),
        )

        # Save HDF5 file to disk
        h5_filename = f"{self.save_fname}.h5"
        h5_path = self.session_dir / h5_filename
        self._save_h5(h5_path, data)

        return h5_path

    def _check_and_normalize_step_data(
        self, observation: dict[str, Any], action: dict[str, np.ndarray]
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Validate one rollout step and return its normalized robot state and action.

        Normalizing means converting any ``cartesian_position`` orientation into a
        scalar-last quaternion, per the robot profile's ``orientation_representation``.

        The converted values are *returned* rather than written back into the caller's
        dicts. They used to be written back, which made this validator quietly responsible
        for feeding the recording buffer — and made "stop mutating the caller's input" a
        change that would have silently recorded raw euler/rot6d values labelled as
        quaternions.

        Args:
            observation (dict[str, Any]): Observation payload to validate.
            action (dict[str, np.ndarray]): Action dict to validate.

        Returns:
            A ``(robot_state, action)`` pair of new dicts, safe to buffer.

        Raises:
            ValueError: If required observation keys are missing, action is
                empty, action contains unrecognized keys, or all action values
                are None.
        """

        # Make sure the observation is a dictionary
        if not isinstance(observation, dict):
            raise ValueError(
                f"observation must be a dictionary, got {type(observation)}"
            )

        missing_obs = [k for k in REQUIRED_OBSERVATION_KEYS if k not in observation]
        if missing_obs:
            raise ValueError(
                f"observation is missing required keys: {missing_obs}. "
                f"Required: {REQUIRED_OBSERVATION_KEYS}. Please pass it in your record_step() call"
            )

        robot_state = observation["robot_state"]
        missing_robot_state_keys = [
            k for k in self.robot_profile.robot_state_keys if k not in robot_state
        ]
        if missing_robot_state_keys:
            raise ValueError(
                f"robot_state is missing required keys: {missing_robot_state_keys}. "
                f"Required: {self.robot_profile.robot_state_keys}. Please pass it in your record_step() call. Double check that the passed keys match the robot profile you initialized the recorder with."
            )

        image_observation = observation["image_observation"]
        missing_image_observation_keys = [
            k for k in self.robot_profile.camera_names if k not in image_observation
        ]
        if missing_image_observation_keys:
            raise ValueError(
                f"image_observation is missing required keys: {missing_image_observation_keys}. "
                f"Required: {self.robot_profile.camera_names} that you provided in the robot setup. Please pass it in your record_step() call. Double check that the passed keys match the robot profile you initialized the recorder with."
            )

        # Make sure the action is a dictionary
        if not isinstance(action, dict):
            raise ValueError(f"action must be a dictionary, got {type(action)}")

        # Make sure action has at least one key
        if not action:
            raise ValueError(
                f"action must not be empty. Valid keys: {VALID_ACTION_KEYS}. Please pass it in your record_step() call. Double check that the passed keys match the robot profile you initialized the recorder with."
            )

        # Make sure the action keys are valid
        invalid_action = set(action.keys()) - set(VALID_ACTION_KEYS)
        if invalid_action:
            raise ValueError(
                f"action contains unrecognized keys: {sorted(invalid_action)}. "
                f"Valid keys: {VALID_ACTION_KEYS}"
            )

        # Make sure the action keys agree between robot_profile and the action dict
        profile_action_keys = set(self.robot_profile.action_space)
        action_keys = set(action.keys())
        if action_keys != profile_action_keys:
            raise ValueError(
                f"action keys {action_keys} must match the robot profile action_space {profile_action_keys}"
            )

        # Make sure action values are not None
        if any(v is None for v in action.values()):
            raise ValueError(
                f"action contains None values for keys: {[k for k, v in action.items() if v is None]}. "
            )

        # Shallow copies from here on: everything below normalizes values, and the caller's
        # dicts must come back out of record_step() exactly as they went in.
        action = dict(action)
        robot_state = dict(robot_state)

        # TODO: only cartesian_position is normalized and shape-checked; the other action
        # keys are recorded as given.
        if "cartesian_position" in action:
            action["cartesian_position"] = self._normalize_cartesian(
                action["cartesian_position"], self.quat_conversion, "action"
            )
            self._check_quaternion_convention(action["cartesian_position"])

        if "cartesian_position" in robot_state:
            robot_state["cartesian_position"] = self._normalize_cartesian(
                robot_state["cartesian_position"],
                self.robot_state_quat_conversion,
                "observation",
            )

        return robot_state, action

    @staticmethod
    def _normalize_cartesian(
        value: Any, conversion: ActionQuatConversion | None, label: str
    ) -> np.ndarray:
        """Convert a cartesian pose to ``(x, y, z, qx, qy, qz, qw)`` and check its shape."""
        arr = np.asarray(value)
        if conversion is not None:
            arr = np.asarray(conversion.convert_position(arr))
        if arr.shape not in ((7,), (14,)):
            raise ValueError(
                f"{label}['cartesian_position'] must have shape (7,) or (14,) — "
                f"[x, y, z, qx, qy, qz, qw], got shape {arr.shape}"
            )
        quat_norm = np.linalg.norm(arr[3:7])
        if not np.isclose(quat_norm, 1.0, atol=1e-2):
            raise ValueError(
                f"{label}['cartesian_position'][3:7] must be a unit scalar-last quaternion "
                f"(norm ≈ 1.0), got norm {quat_norm:.6f}"
            )
        return arr

    @staticmethod
    def _check_quaternion_convention(arr: np.ndarray) -> None:
        """Warn on a pose that looks scalar-first. A heuristic, so it never raises."""
        if abs(arr[6]) > 0.99 and np.linalg.norm(arr[3:6]) < 0.1:
            logger.warning(
                "action['cartesian_position'][3:7] looks like (w,x,y,z) order rather than "
                "the expected (x,y,z,w)."
            )

    # TODO: Polish this function!
    def _save_h5(self, path: Path, data: dict[str, Any]) -> None:
        """Write buffered rollout data and metadata into one HDF5 file.

        Args:
            path (Path): Target HDF5 file path.
            metadata (dict[str, Any]): Normalized metadata payload.

        Returns:
            None: This method only performs file I/O side effects.
        """
        str_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(path, "w") as f:
            # 1. Save the metadata attributes
            f.attrs["schema"] = "oopsiedata_format_v1"
            f.attrs["episode_id"] = data["metadata"]["episode_id"]
            f.attrs.create(
                "robot_profile",
                robot_profile_to_json(self.robot_profile),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            f.attrs["language_instruction"] = data.get("language_instruction", "")
            f.attrs["operator_name"] = data["metadata"]["operator_name"]
            f.attrs["lab_id"] = self.lab_id
            f.attrs["timestamp"] = self.timestamp

            # 2. Save episode annotations if any
            if "episode_annotations" in data:
                ea_group = f.create_group("episode_annotations")
                for annotator_name, annotation in data["episode_annotations"].items():
                    ag = ea_group.require_group(annotator_name)
                    for attr_key, attr_val in annotation.items():
                        if attr_val is None:
                            continue
                        ag.attrs[attr_key] = attr_val

            # 3. Save the per-camera MP4 file paths (strings), not inlined frame tensors.
            observations_group = f.create_group("observations")
            video_paths_group = observations_group.create_group("video_paths")
            video_paths = data.get("video_paths", {})
            if not isinstance(video_paths, dict):
                video_paths = {}
            for cam in self.camera_names:
                raw_video_path = str(video_paths.get(cam, "")).strip()
                if not raw_video_path:
                    continue

                episode_dir = path.parent.resolve()
                video_path_obj = Path(raw_video_path).expanduser()
                if not video_path_obj.is_absolute():
                    video_path_obj = episode_dir / video_path_obj
                rel_video_path = os.path.relpath(video_path_obj.resolve(), start=episode_dir)

                video_paths_group.create_dataset(
                    cam,
                    data=rel_video_path.replace(os.sep, "/"),
                    dtype=str_dtype,
                )

            # 3. Save the robot state data
            robot_states = observations_group.create_group("robot_states")
            for key in self.robot_profile.robot_state_keys:
                robot_states.create_dataset(
                    key,
                    data=np.stack(
                        [t["robot_state"][key] for t in self.timesteps], axis=0
                    ),
                    dtype=np.float64,
                )

            # 4. Save the action data
            action_group = f.create_group("actions")

            for action_key in self.timesteps[0]["action_dict"]:
                action_values = [t["action_dict"][action_key] for t in self.timesteps]
                if all(v is None for v in action_values):
                    action_group.create_dataset(
                        action_key, data=h5py.Empty(dtype=np.float64)
                    )
                else:
                    action_group.create_dataset(
                        action_key,
                        data=np.stack(action_values, axis=0),
                        dtype=np.float64,
                    )

    def _resolve_video_paths(
        self,
        output_dir: Path,
        provided_video_paths: Any,
    ) -> dict[str, str]:
        """Resolve per-camera MP4 paths to paths relative to the episode file.

        On the ``finish_rollout`` path every camera arrives in ``provided_video_paths``
        (absolute, already written by :meth:`_save_videos`) and this only relativizes them —
        which is what the HDF5 stores. A recorder driven directly through :meth:`save`
        supplies nothing, and the buffered frames are written here instead, through the same
        :func:`write_mp4`.

        Args:
            output_dir (Path): Directory used as the base for relative path
                storage.
            provided_video_paths (Any): Optional external mapping from camera to
                MP4 path.

        Returns:
            dict[str, str]: Mapping from camera name to relative MP4 path.
        """
        paths: dict[str, str] = {}
        provided = (
            provided_video_paths if isinstance(provided_video_paths, dict) else {}
        )
        # Both sides of every relpath below must be resolved. Measuring a resolved video
        # against an unresolved base is what produced stored paths like
        # "../../../private/tmp/<session>/x.mp4" for a video sitting right next to the
        # episode, because /tmp is a symlink to /private/tmp on macOS. Those still resolve,
        # so nothing failed — but they break the moment the session directory is moved.
        base = Path(output_dir).resolve()

        for cam in self.camera_names:
            provided_path = str(provided.get(cam, ""))
            if provided_path:
                abs_path = Path(provided_path).expanduser().resolve()
                if abs_path.suffix.lower() != ".mp4":
                    abs_path = abs_path.with_suffix(".mp4")
                paths[cam] = os.path.relpath(abs_path, start=base)
                continue

            frames = self.frames.get(cam, [])
            if len(frames) == 0:
                continue

            video_path = output_dir / f"{self.save_fname}_{cam}.mp4"
            fps = float(self.robot_profile.control_freq)
            write_mp4(video_path=video_path, frames=np.asarray(frames), fps=fps)
            paths[cam] = os.path.relpath(video_path.resolve(), start=base)

        return paths

    def _get_camera_frame(
        self, observation: dict[str, Any], cam_name: str
    ) -> np.ndarray | None:
        """Extract a camera frame from supported observation key patterns.

        Args:
            observation (dict[str, Any]): Observation dictionary for one
                timestep.
            cam_name (str): Canonical camera name to resolve.

        Returns:
            np.ndarray | None: Camera frame array when available, otherwise
                ``None``.
        """
        candidates = (cam_name, f"image_{cam_name}", f"{cam_name}_image")
        for key in candidates:
            if key in observation:
                return np.asarray(observation[key])
        return None

    def _validate_pre_save(self, data: dict[str, Any]) -> None:
        """Perform final validation checks before saving the episode.

        Args:
            data (dict[str, Any]): Full episode data payload to validate.

        Raises:
            ValueError: If any required fields are missing or invalid.
        """
        episode_data = EpisodeData(
            robot_profile=self.robot_profile,
            language_instruction=data.get("language_instruction", ""),
            episode_id=data["metadata"]["episode_id"],
            lab_id=self.lab_id,
            operator_name=data["metadata"]["operator_name"],
            trajectory_length=len(self.timesteps),
            control_freq=float(self.robot_profile.control_freq),
            observations={
                key: np.stack([t["robot_state"][key] for t in self.timesteps], axis=0)
                for key in self.robot_profile.robot_state_keys
            },
            actions={
                key: np.stack(
                    [t["action_dict"][key] for t in self.timesteps], axis=0
                )
                for key in self.timesteps[0]["action_dict"]
            },
            videos={
                cam: VideoInfo.from_frames(self.frames[cam], fps=self.robot_profile.control_freq) for cam in self.camera_names
            },
            annotations=data.get("episode_annotations", None),
        )
        validate_episode(episode_data)

    @staticmethod
    def patch_h5_failure_annotation(
        h5_path: Path,
        annotation: dict[str, Any],
    ) -> None:
        """Patch an existing episode HDF5 with a human annotation.

        This is used when the episode is saved immediately after rollout and the
        human annotation arrives later.
        """
        if not h5_path.exists():
            raise FileNotFoundError(str(h5_path))

        with h5py.File(h5_path, "r+") as f:
            episode_annotations = f.require_group("episode_annotations")
            annotation_group = episode_annotations.require_group(
                annotation["annotator"]
            )
            write_annotation_attrs(annotation_group, annotation)

    @property
    def num_steps(self) -> int:
        """Return the number of recorded timesteps.

        Returns:
            int: Number of timesteps currently buffered.
        """
        return len(self.timesteps)
