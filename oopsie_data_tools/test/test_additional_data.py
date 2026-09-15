"""Additional sensor data: profile declaration, recording, validation and restructure."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pytest

from oopsie_data_tools.annotation_tool.episode_recorder import EpisodeRecorder, write_mp4
from oopsie_data_tools.cli import main
from oopsie_data_tools.test.fixtures.make_valid import write_valid_episode
from oopsie_data_tools.utils import hf_upload, restructure
from oopsie_data_tools.utils.conversion_utils import write_additional_data
from oopsie_data_tools.utils.robot_profile.robot_profile import (
    AdditionalDataSource,
    robot_profile_from_json,
    robot_profile_from_raw,
    robot_profile_to_json,
)
from oopsie_data_tools.utils.validation import episode_validator
from oopsie_data_tools.utils.validation.errors import EpisodeValidationError
from oopsie_data_tools.utils.validation.validation_utils import validate_h5_file

N_STEPS = 20

_BASE_RAW = {
    "policy_name": "test_policy",
    "robot_name": "test_robot",
    "is_biarm": False,
    "uses_mobile_base": False,
    "gripper_name": "test_gripper",
    "control_freq": 10,
    "camera_names": ["front"],
    "robot_state_keys": ["joint_position", "gripper_position"],
    "robot_state_joint_names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
    "action_space": ["joint_velocity", "gripper_position"],
    "action_joint_names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
}

_FT = {"sensor": "ATI Mini45", "sensor_info": {"units": "N, Nm", "frame": "sensor"}}
_TACTILE = {"sensor": "GelSight Mini", "sensor_info": "left finger", "format": "video"}


def _raw(additional_data) -> dict:
    return {**_BASE_RAW, "additional_data": additional_data}


# ── profile ───────────────────────────────────────────────────────────────────


class TestProfile:
    def test_declared_sources_are_parsed(self):
        profile = robot_profile_from_raw(_raw({"wrist_ft": _FT, "tactile": _TACTILE}))

        assert profile.additional_data == {
            "wrist_ft": AdditionalDataSource(
                sensor="ATI Mini45", sensor_info=_FT["sensor_info"], format="array"
            ),
            "tactile": AdditionalDataSource(
                sensor="GelSight Mini", sensor_info="left finger", format="video"
            ),
        }
        assert profile.additional_array_keys() == ["wrist_ft"]
        assert profile.additional_video_keys() == ["tactile"]

    def test_survives_the_json_round_trip_through_the_h5_attr(self):
        profile = robot_profile_from_raw(_raw({"wrist_ft": _FT, "tactile": _TACTILE}))
        assert robot_profile_from_json(robot_profile_to_json(profile)) == profile

    @pytest.mark.parametrize("value", [None, {}], ids=["blank", "empty"])
    def test_absent_or_blank_declares_nothing(self, value):
        assert robot_profile_from_raw(_raw(value)).additional_data == {}
        assert robot_profile_from_raw(_BASE_RAW).additional_data == {}

    @pytest.mark.parametrize(
        "additional_data,match",
        [
            (["wrist_ft"], "additional_data must be a mapping"),
            ({"wrist/ft": _FT}, "keys may only contain"),
            ({"front": _FT}, "collides with camera 'front'"),
            ({"Front": _TACTILE}, "collides with camera 'front'"),
            ({"tactile": _TACTILE, "Tactile": _TACTILE}, "collides with additional_data key"),
            ({"wrist_ft": "ATI Mini45"}, r"must be a mapping with at least 'sensor'"),
            ({"wrist_ft": {**_FT, "sensr": "x"}}, r"unknown field\(s\) \['sensr'\]"),
            ({"wrist_ft": {"sensor_info": "x"}}, "sensor must be a non-empty string"),
            ({"wrist_ft": {"sensor": "  "}}, "sensor must be a non-empty string"),
            ({"wrist_ft": {"sensor": "x", "sensor_info": [1, 2]}}, "string or a mapping"),
            ({"wrist_ft": {"sensor": "x", "format": "image"}}, "format must be one of"),
        ],
    )
    def test_malformed_declarations_are_rejected(self, additional_data, match):
        with pytest.raises(ValueError, match=match):
            robot_profile_from_raw(_raw(additional_data))


# ── recording ─────────────────────────────────────────────────────────────────


def _recorder(
    tmp_path: Path, additional_data: dict, nan_policy: str = "warn"
) -> EpisodeRecorder:
    profile = robot_profile_from_raw(_raw(additional_data))
    return EpisodeRecorder(
        robot_profile=profile,
        data_root_dir=tmp_path,
        operator_name="test_operator",
        additional_data_nan_policy=nan_policy,
    )


def _obs() -> dict:
    return {
        "robot_state": {
            "joint_position": np.zeros(7, dtype=np.float32),
            "gripper_position": np.zeros(1, dtype=np.float32),
        },
        "image_observation": {"front": np.zeros((224, 224, 3), dtype=np.uint8)},
    }


def _action() -> dict:
    return {
        "joint_velocity": np.zeros(7, dtype=np.float32),
        "gripper_position": np.zeros(1, dtype=np.float32),
    }


def _ft(step: int = 0) -> np.ndarray:
    return np.full(6, step, dtype=np.float32)


def _tactile() -> np.ndarray:
    return np.full((64, 64, 3), 90, dtype=np.uint8)


class TestRecording:
    def test_recorded_episode_round_trips_and_validates(self, tmp_path):
        rec = _recorder(tmp_path, {"wrist_ft": _FT, "tactile": _TACTILE})
        for step in range(N_STEPS):
            rec.record_step(
                _obs(), _action(), additional_data={"wrist_ft": _ft(step), "tactile": _tactile()}
            )
        rec.finish_rollout("pick up the block", success=1.0)

        h5_path = rec.session_dir / f"{rec.save_fname}.h5"
        with h5py.File(h5_path, "r") as f:
            ft = f["additional_data/wrist_ft"][()]
            tactile_rel = f["additional_data/tactile"][()].decode()
        assert ft.dtype == np.float32, "array data keeps its dtype"
        assert ft.shape == (N_STEPS, 6)
        np.testing.assert_array_equal(ft[:, 0], np.arange(N_STEPS))
        assert tactile_rel == f"{rec.save_fname}_tactile.mp4"
        assert (rec.session_dir / tactile_rel).is_file()

        validate_h5_file(str(h5_path), strict_annotation_check=True)

    def test_one_recorder_records_consecutive_episodes(self, tmp_path):
        rec = _recorder(tmp_path, {"wrist_ft": _FT, "tactile": _TACTILE})
        h5_paths = []
        for n_steps, width in ((N_STEPS, 6), (N_STEPS + 5, 3)):
            rec.reset_episode_recorder()
            for _ in range(n_steps):
                rec.record_step(
                    _obs(), _action(),
                    additional_data={"wrist_ft": np.zeros(width), "tactile": _tactile()},
                )
            rec.finish_rollout("pick up the block", success=1.0)
            h5_paths.append(rec.session_dir / f"{rec.save_fname}.h5")

        assert h5_paths[0] != h5_paths[1]
        for h5_path, n_steps, width in zip(h5_paths, (N_STEPS, N_STEPS + 5), (6, 3)):
            with h5py.File(h5_path, "r") as f:
                assert f["additional_data/wrist_ft"].shape == (n_steps, width)
            validate_h5_file(str(h5_path), strict_annotation_check=True)

    def test_web_annotator_forwards_additional_data_and_nan_policy(self, tmp_path):
        from oopsie_data_tools.annotation_tool.rollout_annotator import WebRolloutAnnotator

        annotator = WebRolloutAnnotator(
            robot_profile=robot_profile_from_raw(_raw({"wrist_ft": _FT})),
            data_root_dir=tmp_path,
            operator_name="test_operator",
            wait_for_annotation=False,
            open_browser=False,
            additional_data_nan_policy="error",
        )
        annotator.record_step(_obs(), _action(), additional_data={"wrist_ft": _ft()})
        with pytest.raises(ValueError, match="additional_data_nan_policy='error'"):
            annotator.record_step(
                _obs(), _action(), additional_data={"wrist_ft": np.full(6, np.nan)}
            )
        assert len(annotator._active_recorder.additional_buffers["wrist_ft"]) == 1

    def test_profile_without_additional_data_writes_no_group(self, tmp_path):
        rec = _recorder(tmp_path, None)
        for _ in range(N_STEPS):
            rec.record_step(_obs(), _action())
        rec.finish_rollout("pick up the block", success=1.0)

        with h5py.File(rec.session_dir / f"{rec.save_fname}.h5", "r") as f:
            assert "additional_data" not in f

    def test_buffer_reused_by_the_caller_is_copied(self, tmp_path):
        rec = _recorder(tmp_path, {"wrist_ft": _FT})
        reading = np.zeros(6, dtype=np.float32)
        rec.record_step(_obs(), _action(), additional_data={"wrist_ft": reading})
        reading[:] = 5.0
        assert rec.additional_buffers["wrist_ft"][0].sum() == 0.0

    @pytest.mark.parametrize(
        "declared,additional_data,match",
        [
            ({"wrist_ft": _FT}, None, "additional_data is required"),
            ({"wrist_ft": _FT}, {}, "must match the robot profile additional_data"),
            ({"wrist_ft": _FT}, {"wrist_ft": _ft(), "imu": _ft()}, "must match the robot"),
            (None, {"wrist_ft": _ft()}, "robot profile declares no additional_data"),
            ({"wrist_ft": _FT}, [_ft()], "additional_data must be a dictionary"),
            ({"wrist_ft": _FT}, {"wrist_ft": np.array([np.inf] * 6)}, "infinite value"),
            ({"wrist_ft": _FT}, {"wrist_ft": np.array(["a"] * 6)}, "real numeric array"),
            (
                {"tactile": _TACTILE},
                {"tactile": _tactile().astype(np.float32)},
                r"\(H, W, 3\) uint8 frame",
            ),
            ({"tactile": _TACTILE}, {"tactile": _tactile()[..., 0]}, r"\(H, W, 3\) uint8"),
            (
                {"tactile": _TACTILE},
                {"tactile": np.zeros((4, 4, 3), dtype=np.uint8)},
                "4x4 frame.*'format: array'",
            ),
        ],
    )
    def test_malformed_step_is_rejected_and_not_buffered(
        self, tmp_path, declared, additional_data, match
    ):
        rec = _recorder(tmp_path, declared)
        with pytest.raises(ValueError, match=match):
            rec.record_step(_obs(), _action(), additional_data=additional_data)
        assert rec.num_steps == 0
        assert all(not buf for buf in rec.additional_buffers.values())

    def test_shape_change_between_steps_is_rejected(self, tmp_path):
        rec = _recorder(tmp_path, {"wrist_ft": _FT})
        rec.record_step(_obs(), _action(), additional_data={"wrist_ft": _ft()})
        with pytest.raises(ValueError, match=r"has shape \(7,\), but earlier steps had \(6,\)"):
            rec.record_step(
                _obs(), _action(), additional_data={"wrist_ft": np.zeros(7, np.float32)}
            )
        assert rec.num_steps == 1

    def _record_with_dropouts(self, rec: EpisodeRecorder) -> None:
        for step in range(N_STEPS):
            reading = _ft(step)
            if step in (3, 7):
                reading[:2] = np.nan
            rec.record_step(_obs(), _action(), additional_data={"wrist_ft": reading})

    def test_nan_warns_once_then_summarizes_and_still_validates(self, tmp_path, caplog):
        rec = _recorder(tmp_path, {"wrist_ft": _FT}, nan_policy="warn")
        with caplog.at_level(logging.WARNING):
            self._record_with_dropouts(rec)
            step_warnings = [r for r in caplog.records if "contains NaN at step" in r.message]
            rec.finish_rollout("pick up the block", success=1.0)

        assert len(step_warnings) == 1 and "step 3" in step_warnings[0].message
        assert f"additional_data/wrist_ft has 4 NaN value(s) in 2 of {N_STEPS} step(s)" in (
            caplog.text
        )
        h5_path = rec.session_dir / f"{rec.save_fname}.h5"
        with h5py.File(h5_path, "r") as f:
            assert np.isnan(f["additional_data/wrist_ft"][3, :2]).all()
        validate_h5_file(str(h5_path), strict_annotation_check=True)

    def test_nan_is_silent_when_ignored(self, tmp_path, caplog):
        rec = _recorder(tmp_path, {"wrist_ft": _FT}, nan_policy="ignore")
        with caplog.at_level(logging.WARNING):
            self._record_with_dropouts(rec)
            rec.finish_rollout("pick up the block", success=1.0)
        assert "NaN" not in caplog.text
        assert rec.nan_counts == {"wrist_ft": (4, 2)}

    def test_nan_rejects_the_step_when_policy_is_error(self, tmp_path):
        rec = _recorder(tmp_path, {"wrist_ft": _FT}, nan_policy="error")
        reading = _ft()
        reading[0] = np.nan
        with pytest.raises(ValueError, match=r"contains NaN \(additional_data_nan_policy='error'\)"):
            rec.record_step(_obs(), _action(), additional_data={"wrist_ft": reading})
        assert rec.num_steps == 0

    def test_nan_counts_reset_between_episodes(self, tmp_path):
        rec = _recorder(tmp_path, {"wrist_ft": _FT})
        self._record_with_dropouts(rec)
        rec.reset_episode_recorder()
        assert rec.nan_counts == {}

    def test_unknown_nan_policy_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="additional_data_nan_policy must be one of"):
            _recorder(tmp_path, {"wrist_ft": _FT}, nan_policy="drop")

    def test_size_cap_stops_the_rollout_at_the_step_that_crosses_it(
        self, tmp_path, monkeypatch
    ):
        # 24 bytes per step; the cap admits two steps.
        monkeypatch.setattr(episode_validator, "MAX_ADDITIONAL_DATA_BYTES", 50)
        rec = _recorder(tmp_path, {"wrist_ft": _FT})
        for _ in range(2):
            rec.record_step(_obs(), _action(), additional_data={"wrist_ft": _ft()})
        with pytest.raises(ValueError, match="additional_data is too large"):
            rec.record_step(_obs(), _action(), additional_data={"wrist_ft": _ft()})
        assert rec.num_steps == 2


# ── validation of files on disk ───────────────────────────────────────────────


def _declare(h5_path: Path, additional_data: dict) -> None:
    with h5py.File(h5_path, "r+") as f:
        profile = json.loads(f.attrs["robot_profile"])
        profile["additional_data"] = additional_data
        f.attrs["robot_profile"] = json.dumps(profile)


def _install_array(h5_path: Path, key: str, value: np.ndarray) -> None:
    with h5py.File(h5_path, "r+") as f:
        f.require_group("additional_data").create_dataset(key, data=value)


# write_valid_episode's camera video has 25 frames; video frame counts must agree within 1.
_FIXTURE_CAMERA_FRAMES = 25


def _install_video(h5_path: Path, key: str, n_frames: int = _FIXTURE_CAMERA_FRAMES) -> None:
    mp4 = h5_path.parent / f"{h5_path.stem}_{key}.mp4"
    write_mp4(mp4, np.full((n_frames, 64, 64, 3), 90, dtype=np.uint8), fps=10)
    with h5py.File(h5_path, "r+") as f:
        f.require_group("additional_data").create_dataset(
            key, data=mp4.name, dtype=h5py.string_dtype(encoding="utf-8")
        )


@pytest.fixture
def episode(tmp_path) -> Path:
    return write_valid_episode(tmp_path, "episode", n=N_STEPS)


class TestValidation:
    def test_declared_array_and_video_pass(self, episode):
        _declare(episode, {"wrist_ft": _FT, "tactile": _TACTILE})
        _install_array(episode, "wrist_ft", np.zeros((N_STEPS, 6), dtype=np.float32))
        _install_video(episode, "tactile")
        validate_h5_file(str(episode), strict_annotation_check=True)

    def test_undeclared_dataset_is_rejected(self, episode):
        _install_array(episode, "wrist_ft", np.zeros((N_STEPS, 6)))
        with pytest.raises(
            EpisodeValidationError,
            match=r"additional_data contains 1 key\(s\) the robot profile does not declare",
        ):
            validate_h5_file(str(episode), strict_annotation_check=True)

    def test_declared_array_missing_from_the_file_is_rejected(self, episode):
        _declare(episode, {"wrist_ft": _FT})
        with pytest.raises(EpisodeValidationError, match=r"Missing additional_data key\(s\) required by .*: \['wrist_ft'\]"):
            validate_h5_file(str(episode), strict_annotation_check=True)

    def test_declared_video_missing_from_the_file_is_rejected(self, episode):
        _declare(episode, {"tactile": _TACTILE})
        with pytest.raises(
            EpisodeValidationError,
            match=r"Missing additional_data video key\(s\) required by .*: \['tactile'\]",
        ):
            validate_h5_file(str(episode), strict_annotation_check=True)

    def test_leading_axis_must_be_the_trajectory(self, episode):
        _declare(episode, {"wrist_ft": _FT})
        _install_array(episode, "wrist_ft", np.zeros((N_STEPS - 1, 6)))
        with pytest.raises(
            EpisodeValidationError, match=rf"leading axis must be the {N_STEPS} recorded steps"
        ):
            validate_h5_file(str(episode), strict_annotation_check=True)

    def test_infinite_values_are_rejected(self, episode):
        _declare(episode, {"wrist_ft": _FT})
        values = np.zeros((N_STEPS, 6))
        values[3, 2] = np.inf
        _install_array(episode, "wrist_ft", values)
        with pytest.raises(
            EpisodeValidationError, match="additional_data/wrist_ft contains 1 infinite value"
        ):
            validate_h5_file(str(episode), strict_annotation_check=True)

    def test_nan_marks_a_dropped_reading_and_is_accepted(self, episode):
        _declare(episode, {"wrist_ft": _FT})
        values = np.zeros((N_STEPS, 6))
        values[3, :] = np.nan
        _install_array(episode, "wrist_ft", values)
        validate_h5_file(str(episode), strict_annotation_check=True)

    def test_oversized_additional_data_is_rejected(self, episode, monkeypatch):
        _declare(episode, {"wrist_ft": _FT, "imu": {"sensor": "BMI088"}})
        _install_array(episode, "wrist_ft", np.zeros((N_STEPS, 6)))
        _install_array(episode, "imu", np.zeros((N_STEPS, 6)))
        # Each key is 960 bytes; neither alone crosses the cap, both together do.
        monkeypatch.setattr(episode_validator, "MAX_ADDITIONAL_DATA_BYTES", 1500)
        with pytest.raises(
            EpisodeValidationError, match=r"additional_data is too large: .*imu=.*wrist_ft="
        ):
            validate_h5_file(str(episode), strict_annotation_check=True)

    def test_video_declared_key_stored_as_numbers_is_rejected(self, episode):
        _declare(episode, {"tactile": _TACTILE})
        _install_array(episode, "tactile", np.zeros((N_STEPS, 6)))
        with pytest.raises(EpisodeValidationError, match="must hold the MP4 path as a string"):
            validate_h5_file(str(episode), strict_annotation_check=True)

    def test_video_frame_count_must_match_the_trajectory(self, episode):
        _declare(episode, {"tactile": _TACTILE})
        _install_video(episode, "tactile", n_frames=N_STEPS * 3)
        with pytest.raises(
            EpisodeValidationError,
            match="Frame count / trajectory mismatch for additional_data/tactile",
        ):
            validate_h5_file(str(episode), strict_annotation_check=True)


# ── restructure and conversion ────────────────────────────────────────────────


def test_restructure_moves_additional_videos_with_their_episode(tmp_path, monkeypatch):
    monkeypatch.setattr(restructure, "FILE_LIMIT", 1)
    source = tmp_path / "session"
    source.mkdir()
    for stem in ("ep_a", "ep_b"):
        h5_path = write_valid_episode(source, stem, n=N_STEPS)
        _declare(h5_path, {"tactile": _TACTILE})
        _install_video(h5_path, "tactile")
    output = tmp_path / "out"

    assert main(["restructure", "--source", str(source), "--output", str(output), "--yes"]) == 0

    copies = sorted(output.rglob("*.h5"))
    assert [p.name for p in copies] == ["ep_a.h5", "ep_b.h5"]
    for h5_path in copies:
        with h5py.File(h5_path, "r") as f:
            rel = f["additional_data/tactile"][()].decode()
        assert (h5_path.parent / rel).is_file()
        validate_h5_file(str(h5_path), strict_annotation_check=True)


class TestConversionWriter:
    def test_writes_arrays_and_relative_video_paths(self, tmp_path):
        profile = robot_profile_from_raw(_raw({"wrist_ft": _FT, "tactile": _TACTILE}))
        h5_path = tmp_path / "ep.h5"
        with h5py.File(h5_path, "w") as f:
            write_additional_data(
                f,
                profile,
                {"wrist_ft": np.zeros((N_STEPS, 6), dtype=np.float32)},
                {"tactile": str(tmp_path / "ep_tactile.mp4")},
                h5_path,
            )
        with h5py.File(h5_path, "r") as f:
            assert f["additional_data/wrist_ft"].dtype == np.float32
            assert f["additional_data/tactile"][()].decode() == "ep_tactile.mp4"

    def test_keys_must_match_the_profile(self, tmp_path):
        profile = robot_profile_from_raw(_raw({"wrist_ft": _FT}))
        with h5py.File(tmp_path / "ep.h5", "w") as f:
            with pytest.raises(ValueError, match=r"missing: \['wrist_ft'\]"):
                write_additional_data(f, profile, {})


class TestUploadSizeSummary:
    def test_logs_the_total_and_the_additional_data_share(self, tmp_path, caplog):
        h5_path = write_valid_episode(tmp_path, "ep", n=N_STEPS)
        _declare(h5_path, {"wrist_ft": _FT, "tactile": _TACTILE})
        _install_array(h5_path, "wrist_ft", np.zeros((N_STEPS, 6)))
        _install_video(h5_path, "tactile")

        with caplog.at_level(logging.INFO):
            total = hf_upload.summarize_upload_size(str(tmp_path))

        assert total == sum(p.stat().st_size for p in tmp_path.iterdir())
        assert "[size] Upload total:" in caplog.text
        assert "of which additional data" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_session_without_additional_data_logs_only_the_total(self, tmp_path, caplog):
        write_valid_episode(tmp_path, "ep", n=N_STEPS)
        with caplog.at_level(logging.INFO):
            hf_upload.summarize_upload_size(str(tmp_path))
        assert "[size] Upload total:" in caplog.text
        assert "additional data" not in caplog.text

    def test_large_upload_warns_but_the_upload_goes_ahead(self, tmp_path, monkeypatch, caplog):
        write_valid_episode(tmp_path, "ep", n=N_STEPS)
        monkeypatch.setattr(hf_upload, "LARGE_UPLOAD_WARN_BYTES", 1)

        with caplog.at_level(logging.INFO):
            assert main(["upload", "--path", str(tmp_path), "--skip-upload"]) == 0

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("above the 0 GB we expect" in r.message for r in warnings)
