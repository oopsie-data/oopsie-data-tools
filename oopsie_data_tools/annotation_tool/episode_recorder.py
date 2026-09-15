"""Episode recorder for saving robot evaluation data in HDF5 format."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

import h5py
import imageio
import numpy as np

from oopsie_data_tools.annotation_tool.annotation_schema import (
    annotation_attrs_dict,
    success_to_outcome,
    write_annotation_attrs,
)
from oopsie_data_tools.utils import contributor_config
from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile, robot_profile_to_json
from oopsie_data_tools.utils.robot_profile.rotation_utils import ActionQuatConversion
from oopsie_data_tools.utils.validation.array_validation import (
    require_finite_real_array,
    require_real_array_without_inf,
    validate_cartesian_quaternions,
    validate_gripper_binary_step,
)
from oopsie_data_tools.utils.validation.episode_data import EpisodeData, VideoInfo
from oopsie_data_tools.utils.validation.episode_validator import (
    check_additional_data_size,
    validate_episode,
)
from oopsie_data_tools.utils.validation.errors import EpisodeValidationError
from oopsie_data_tools.utils.video_encoding import VIDEO_CRF
from oopsie_data_tools.utils.video_paths import write_video_path

logger = logging.getLogger(__name__)

REQUIRED_OBSERVATION_KEYS = ["robot_state", "image_observation"]

#: How record_step treats NaN in array-format additional_data, which usually marks a
#: dropped sensor reading. ±inf is always rejected.
ADDITIONAL_DATA_NAN_POLICIES = ("ignore", "warn", "error")

#: Smallest frame side accepted for a video-format sensor. libx264 via imageio upscales
#: anything smaller to 16 px, which distorts a coarse taxel grid.
MIN_SENSOR_FRAME_SIZE = 16

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
        quality=None,
        output_params=["-crf", str(VIDEO_CRF)],
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
        additional_data_nan_policy: str = "warn",
    ) -> None:
        """Initialize a recorder instance.

        Args:
            robot_profile (RobotProfile): Robot profile.
            data_root_dir (str): Base output directory for saved artifacts.
            operator_name (str): Name of the operator recording the episode.
            resume_session_name (str | None): Optional unique session name
            additional_data_nan_policy (str): What a NaN in array-format additional_data
                does: ``"ignore"`` records it silently, ``"warn"`` records it and logs a
                warning the first time per key and a summary when the episode is saved,
                ``"error"`` rejects the step.

        Raises:
            ValueError: If ``data_root_dir`` is not a valid directory, or
                ``additional_data_nan_policy`` is not one of
                :data:`ADDITIONAL_DATA_NAN_POLICIES`.
        """
        if additional_data_nan_policy not in ADDITIONAL_DATA_NAN_POLICIES:
            raise ValueError(
                f"additional_data_nan_policy must be one of {list(ADDITIONAL_DATA_NAN_POLICIES)}, "
                f"got {additional_data_nan_policy!r}"
            )
        self.additional_data_nan_policy = additional_data_nan_policy
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
        # Per-step values of each profile.additional_data key; frames for video-format keys.
        self.additional_buffers: dict[str, list[np.ndarray]] = {}
        # additional_data key -> (NaN values, steps containing NaN) in the current episode.
        self.nan_counts: dict[str, tuple[int, int]] = {}
        self.timesteps: list[dict[str, Any]] = []
        self.timestamp: float = 0.0
        self.save_fname: str = ""
        # Names handed out by this recorder, so back-to-back episodes stay distinct even
        # before the first one has been written to disk.
        self._used_names: set[str] = set()
        # A fresh recorder is ready to record. This used to be left to an explicit
        # reset_episode_recorder() call, so going straight to record_step() raised
        # AttributeError on save_fname — an undocumented ordering requirement.
        self.reset_episode_recorder()

        self.lab_id, _ = contributor_config.read_contributor_config()

    def reset_episode_recorder(self) -> None:
        """Reset the buffers and start a new episode."""
        ts = datetime.datetime.now()
        self.timestamp = ts.timestamp()
        self.save_fname = self._unused_episode_name(ts)
        self.frames = {cam: [] for cam in self.camera_names}
        self.additional_buffers = {key: [] for key in self.robot_profile.additional_data}
        self.nan_counts = {}
        self.timesteps = []

    def _unused_episode_name(self, ts: datetime.datetime) -> str:
        """A ``%Y%m%d_%H%M%S`` name, suffixed if that second is already taken.

        Episode names are second-resolution, so a fast rollout loop can start two episodes
        within the same second and have the later one overwrite the earlier one's HDF5 and
        MP4s. Callers used to work around this by sleeping between episodes.
        """
        base = ts.strftime("%Y%m%d_%H%M%S")
        candidate = base
        attempt = 1
        while (self.session_dir / f"{candidate}.h5").exists() or candidate in self._used_names:
            attempt += 1
            candidate = f"{base}_{attempt}"
        self._used_names.add(candidate)
        return candidate

    def record_step(
        self,
        observation: dict[str, Any],
        action: dict[str, np.ndarray],
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """Append one rollout timestep to in-memory buffers.

        Args:
            observation (dict[str, Any]): Observation payload containing state
                and optional images.
            action (dict[str, np.ndarray]): Dictionary of action vector applied at this timestep.
            additional_data (dict[str, Any] | None): One value per key declared in
                ``robot_profile.additional_data``; required exactly when the profile
                declares any.

        Returns:
            None: This method only updates in-memory buffers.
        """
        # TODO: Make sure all checks are present here
        # Returns normalized copies; the caller's dicts are left untouched.
        robot_state, action = self._check_and_normalize_step_data(observation, action)
        additional = self._check_additional_data(additional_data)

        # Buffer frames for each configured camera (if available)
        for cam in self.camera_names:
            frame = self._get_camera_frame(observation["image_observation"], cam)
            self.frames[cam].append(np.asarray(frame, dtype=np.uint8))

        # Buffer timestep data
        step_data = {"robot_state": {}, "action_dict": {}}
        for key in self.robot_profile.robot_state_keys:
            step_data["robot_state"][key] = np.asarray(robot_state[key], dtype=np.float32)
        step_data["action_dict"] = action
        self.timesteps.append(step_data)
        for key, value in additional.items():
            self.additional_buffers[key].append(value)
        self._count_nans(additional)

    def _count_nans(self, additional: dict[str, np.ndarray]) -> None:
        """Tally NaN in a buffered step, warning on the first one per key if configured."""
        for key, value in additional.items():
            if value.dtype.kind != "f":
                continue
            n_nan = int(np.count_nonzero(np.isnan(value)))
            if not n_nan:
                continue
            values, steps = self.nan_counts.get(key, (0, 0))
            if steps == 0 and self.additional_data_nan_policy == "warn":
                logger.warning(
                    "additional_data[%r] contains NaN at step %d; recording continues. "
                    "Further NaNs in this episode are summarized when it is saved.",
                    key, len(self.timesteps) - 1,
                )
            self.nan_counts[key] = (values + n_nan, steps + 1)

    def _log_nan_summary(self) -> None:
        if self.additional_data_nan_policy != "warn":
            return
        for key, (values, steps) in sorted(self.nan_counts.items()):
            logger.warning(
                "Episode %s: additional_data/%s has %d NaN value(s) in %d of %d step(s).",
                self.save_fname, key, values, steps, len(self.timesteps),
            )

    def _check_additional_data(self, additional_data: Any) -> dict[str, np.ndarray]:
        """Validate one step's additional sensor data and return copies safe to buffer."""
        declared = self.robot_profile.additional_data
        if not declared:
            if additional_data:
                raise ValueError(
                    f"additional_data was passed with keys {sorted(additional_data)}, but the "
                    "robot profile declares no additional_data. Declare the sensors in the "
                    "profile first."
                )
            return {}
        if additional_data is None:
            raise ValueError(
                f"additional_data is required: the robot profile declares {sorted(declared)}. "
                "Pass record_step(..., additional_data={key: value})."
            )
        if not isinstance(additional_data, dict):
            raise ValueError(
                f"additional_data must be a dictionary, got {type(additional_data)}"
            )
        if set(additional_data) != set(declared):
            raise ValueError(
                f"additional_data keys {sorted(additional_data)} must match the robot profile "
                f"additional_data {sorted(declared)}"
            )

        checked: dict[str, np.ndarray] = {}
        for key, source in declared.items():
            label = f"additional_data[{key!r}]"
            # A copy, so a caller reusing one buffer across steps cannot rewrite history.
            value = np.array(additional_data[key])
            if source.is_video:
                if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                    raise ValueError(
                        f"{label} is declared with format: video and must be one (H, W, 3) "
                        f"uint8 frame, got shape {value.shape} and dtype {value.dtype}"
                    )
                if min(value.shape[:2]) < MIN_SENSOR_FRAME_SIZE:
                    raise ValueError(
                        f"{label} is a {value.shape[0]}x{value.shape[1]} frame; video frames "
                        f"must be at least {MIN_SENSOR_FRAME_SIZE}x{MIN_SENSOR_FRAME_SIZE} or "
                        "the encoder upscales them. Declare small sensor grids with "
                        "'format: array' instead."
                    )
            else:
                require_real_array_without_inf(value, label)
                if self.additional_data_nan_policy == "error" and np.isnan(value).any():
                    raise ValueError(
                        f"{label} contains NaN (additional_data_nan_policy='error'). Pass "
                        "additional_data_nan_policy='warn' or 'ignore' to EpisodeRecorder to "
                        "record dropped readings as NaN."
                    )
            buffered = self.additional_buffers[key]
            if buffered and buffered[0].shape != value.shape:
                raise ValueError(
                    f"{label} has shape {value.shape}, but earlier steps had "
                    f"{buffered[0].shape}; every step must have the same shape"
                )
            checked[key] = value

        # Fail during the rollout rather than at save time once the episode is too big.
        steps = len(self.timesteps) + 1
        try:
            check_additional_data_size(
                {
                    key: steps * checked[key].nbytes
                    for key in self.robot_profile.additional_array_keys()
                }
            )
        except EpisodeValidationError as e:
            raise ValueError(str(e)) from e
        return checked

    def _write_videos(self, buffers: dict[str, list[np.ndarray]]) -> dict[str, str]:
        """Write ``<episode>_<name>.mp4`` into the session directory for each buffer.

        Returns:
            ``{name: absolute mp4 path}``. Empty buffers are skipped rather than crashing in
            ``np.stack`` on an empty list.
        """
        paths: dict[str, str] = {}
        self.session_dir.mkdir(parents=True, exist_ok=True)
        fps = float(self.robot_profile.control_freq)
        for name, frames in buffers.items():
            if not frames:
                continue
            video_path = self.session_dir / f"{self.save_fname}_{name}.mp4"
            write_mp4(video_path=video_path, frames=np.asarray(frames), fps=fps)
            paths[name] = str(video_path.resolve())
        return paths

    def _save_videos(self) -> dict[str, str]:
        """Write one MP4 per camera; returns ``{camera: absolute mp4 path}``."""
        return self._write_videos(self.frames)

    def _additional_video_buffers(self) -> dict[str, list[np.ndarray]]:
        return {
            key: self.additional_buffers[key]
            for key in self.robot_profile.additional_video_keys()
        }

    def _stacked_robot_states(self) -> dict[str, np.ndarray]:
        return {
            key: np.stack([t["robot_state"][key] for t in self.timesteps], axis=0)
            for key in self.robot_profile.robot_state_keys
        }

    def _stacked_actions(self) -> dict[str, np.ndarray]:
        return {
            key: np.stack([t["action_dict"][key] for t in self.timesteps], axis=0)
            for key in self.robot_profile.action_space
        }

    def _stacked_additional_arrays(self) -> dict[str, np.ndarray]:
        return {
            key: np.stack(self.additional_buffers[key], axis=0)
            for key in self.robot_profile.additional_array_keys()
        }

    def _video_infos(self, buffers: dict[str, list[np.ndarray]]) -> dict[str, VideoInfo]:
        return {
            name: VideoInfo.from_frames(
                frames, fps=self.robot_profile.control_freq, crf=float(VIDEO_CRF)
            )
            for name, frames in buffers.items()
        }

    def finish_rollout(self, instruction: str, success: float | None = None) -> None:
        data = {
                "language_instruction": instruction,
                "metadata": {
                    "episode_id": self.save_fname,
                    "operator_name": self.operator_name,
                },
        }
        if success is not None:
            # A stub the human annotator fills in later: the outcome the float implies, and
            # nothing else. Built through annotation_attrs_dict so it cannot drift from what
            # the annotation tool writes.
            stub = annotation_attrs_dict(
                {
                    "outcome": success_to_outcome(success),
                    "timestamp": datetime.datetime.now().timestamp(),
                }
            )
            stub["success"] = success
            data["episode_annotations"] = {self.operator_name: stub}
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
            data (dict[str, Any]): Save metadata containing language and
                annotation fields.

        Returns:
            Path: Path to the written HDF5 episode file.

        Raises:
            ValueError: If no rollout steps were recorded.
        """
        if len(self.timesteps) == 0:
            raise ValueError("No steps recorded. Call record_step() first.")

        # finish_rollout() has already written the camera videos and passes their paths;
        # a direct save() call writes whichever cameras were not supplied.
        provided = data.get("video_paths")
        provided = provided if isinstance(provided, dict) else {}
        unwritten = {
            cam: self.frames[cam]
            for cam in self.camera_names
            if not str(provided.get(cam, "")).strip()
        }
        data["video_paths"] = {**provided, **self._write_videos(unwritten)}
        data["additional_video_paths"] = self._write_videos(self._additional_video_buffers())

        # Save HDF5 file to disk
        h5_filename = f"{self.save_fname}.h5"
        h5_path = self.session_dir / h5_filename
        self._save_h5(h5_path, data)
        self._log_nan_summary()

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

        if "cartesian_position" in robot_state:
            robot_state["cartesian_position"] = self._normalize_cartesian(
                robot_state["cartesian_position"],
                self.robot_state_quat_conversion,
                "observation",
            )

        for key, value in action.items():
            require_finite_real_array(value, f"action[{key!r}]")
        for key, value in robot_state.items():
            require_finite_real_array(value, f"observation['robot_state'][{key!r}]")
        if "gripper_binary" in action:
            validate_gripper_binary_step(
                action["gripper_binary"],
                is_biarm=self.robot_profile.is_biarm,
                label="action['gripper_binary']",
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
        return validate_cartesian_quaternions(
            arr, f"{label}['cartesian_position']"
        )

    # TODO: Polish this function!
    def _save_h5(self, path: Path, data: dict[str, Any]) -> None:
        """Write buffered rollout data and metadata into one HDF5 file.

        Args:
            path (Path): Target HDF5 file path.
            data (dict[str, Any]): Normalized metadata payload.

        Returns:
            None: This method only performs file I/O side effects.
        """
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
            for cam in self.camera_names:
                video_path = str(video_paths.get(cam, "")).strip()
                if video_path:
                    write_video_path(video_paths_group, cam, video_path, path.parent)

            # 3. Save the robot state data
            robot_states = observations_group.create_group("robot_states")
            for key, values in self._stacked_robot_states().items():
                robot_states.create_dataset(key, data=values, dtype=np.float64)

            # 4. Save the action data: every valid key, h5py.Empty outside the action space.
            action_group = f.create_group("actions")
            actions = self._stacked_actions()
            for action_key in sorted(VALID_ACTION_KEYS):
                if action_key in actions:
                    action_group.create_dataset(
                        action_key, data=actions[action_key], dtype=np.float64
                    )
                else:
                    action_group.create_dataset(
                        action_key, data=h5py.Empty(dtype=np.float64)
                    )

            # 5. Save additional sensor data: arrays keep their dtype, videos are paths.
            if self.robot_profile.additional_data:
                additional_group = f.create_group("additional_data")
                for key, values in self._stacked_additional_arrays().items():
                    additional_group.create_dataset(key, data=values)
                for key, video_path in data.get("additional_video_paths", {}).items():
                    write_video_path(additional_group, key, video_path, path.parent)

    def _get_camera_frame(
        self, observation: dict[str, Any], cam_name: str
    ) -> np.ndarray | None:
        """The frame for *cam_name*; the key is guaranteed present by the step check."""
        return np.asarray(observation[cam_name])

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
            observations=self._stacked_robot_states(),
            actions=self._stacked_actions(),
            videos=self._video_infos(self.frames),
            annotations=data.get("episode_annotations", None),
            additional_data=self._stacked_additional_arrays(),
            additional_videos=self._video_infos(self._additional_video_buffers()),
        )
        validate_episode(episode_data)

    @staticmethod
    def patch_h5_failure_annotation(
        h5_path: Path,
        annotation: dict[str, Any],
    ) -> None:
        """Patch an existing episode HDF5 with a human annotation.

        This is used when the episode is saved immediately after rollout and the
        human annotation arrives later. *annotation* is the annotation-tool dict —
        ``annotator``, ``outcome`` and whatever taxonomy fields that outcome asks about.
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
