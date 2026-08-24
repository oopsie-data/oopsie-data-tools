"""Tests for episode validation against oopsiedata_format_v1.

Fixtures are generated once per session via conftest.py (tmp_path_factory) and
cleaned up automatically.  Per-test files use the built-in ``tmp_path`` fixture.

Every ``pytest.raises`` here names the message it expects. Without that, an invalid
fixture that fails for an unrelated reason still passes its test — which is what happened
for a long time: the fixtures declared videos nobody wrote, so most of them failed on
"Video file does not exist" and never reached the defect they were named for.

Sections
--------
TestReadable              – file-level guard (exists, is HDF5)
TestRequiredAttrs         – missing root attrs
TestRobotProfile          – malformed / inconsistent robot_profile JSON
TestRequiredGroups        – missing top-level or nested groups
TestTrajectoryLengths     – mismatched / zero trajectory lengths
TestVideos                – missing or too-small video files
TestValidEpisodes         – happy-path: all registered tests pass
TestProfileDocumentsTheEpisode – profile and file must agree in both directions
TestValidateSessionDir    – directory-level validation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pytest

from oopsie_data_tools.cli import _CliFormatter
from oopsie_data_tools.test.fixtures.make_valid import write_valid_episode
from oopsie_data_tools.utils.hf_upload import run_validation
from oopsie_data_tools.utils.validation.episode_loader import detect_x264_crf
from oopsie_data_tools.utils.validation.errors import EpisodeValidationError
from oopsie_data_tools.utils.validation.validation_utils import (
    VALIDATION_LOGGER_NAME,
    validate_h5_file,
    validate_session_dir,
)


def _install_cartesian_trajectories(h5_path: Path, *, is_biarm: bool = False) -> None:
    """Convert a valid joint-action fixture into a valid Cartesian one."""
    width = 14 if is_biarm else 7
    poses = np.zeros((20, width), dtype=np.float64)
    poses[:, 6] = 1.0
    if is_biarm:
        poses[:, 13] = 1.0

    with h5py.File(h5_path, "r+") as f:
        profile = json.loads(f.attrs["robot_profile"])
        profile["is_biarm"] = is_biarm
        profile["robot_state_keys"] = [
            "cartesian_position",
            "joint_position",
            "gripper_position",
        ]
        profile["action_space"] = ["cartesian_position", "gripper_position"]
        profile["action_joint_names"] = None
        f.attrs["robot_profile"] = json.dumps(profile)

        f["observations/robot_states"].create_dataset(
            "cartesian_position", data=poses
        )
        del f["actions/joint_velocity"]
        del f["actions/cartesian_position"]
        f["actions"].create_dataset("cartesian_position", data=poses)


def _install_gripper_binary_trajectory(
    h5_path: Path, values: np.ndarray, *, is_biarm: bool = False
) -> None:
    with h5py.File(h5_path, "r+") as f:
        profile = json.loads(f.attrs["robot_profile"])
        profile["is_biarm"] = is_biarm
        profile["action_space"] = ["joint_velocity", "gripper_binary"]
        f.attrs["robot_profile"] = json.dumps(profile)

        del f["actions/gripper_position"]
        del f["actions/gripper_binary"]
        f["actions"].create_dataset("gripper_binary", data=values)

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidEpisodes:
    @pytest.mark.parametrize("strict", [False, True])
    def test_minimally_annotated_episode_passes(self, valid_episode, strict):
        assert validate_h5_file(str(valid_episode), strict_annotation_check=strict) is True

    def test_episode_without_annotations_passes_lenient(self, episode_without_annotations):
        """No annotation group is structurally fine — this is a just-recorded episode."""
        assert validate_h5_file(str(episode_without_annotations)) is True

    def test_episode_without_annotations_fails_strict(self, episode_without_annotations):
        """...but the CLI always checks annotations, so it cannot be uploaded yet.

        This pins the asymmetry between EpisodeRecorder.finish_rollout (lenient, so an
        episode can be saved before anyone has annotated it) and run_validation, which
        both 'oopsie-data validate' and 'oopsie-data upload' call with strict=True.
        """
        with pytest.raises(EpisodeValidationError, match="Annotations dict is empty"):
            validate_h5_file(str(episode_without_annotations), strict_annotation_check=True)

    def test_cli_validation_rejects_an_episode_without_annotations(
        self, episode_without_annotations
    ):
        """The strict flag is not optional at the CLI boundary — assert on run_validation."""
        assert run_validation(str(episode_without_annotations.parent), None, None) == 1


# ---------------------------------------------------------------------------
# File-level guard
# ---------------------------------------------------------------------------


class TestReadable:
    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(AssertionError, match="does not exist"):
            validate_h5_file(str(tmp_path / "ghost.h5"))

    def test_not_h5_raises(self, invalid_fixtures):
        # (AssertionError, Exception) is just Exception, i.e. "something went wrong".
        with pytest.raises(AssertionError, match="not readable"):
            validate_h5_file(str(invalid_fixtures["invalid_not_h5"]))

    def test_empty_h5_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="Unsupported or missing schema"):
            validate_h5_file(str(invalid_fixtures["invalid_empty_h5"]))


# ---------------------------------------------------------------------------
# Missing root attrs
# ---------------------------------------------------------------------------


class TestRequiredAttrs:
    def test_missing_attrs_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="Missing root attr"):
            validate_h5_file(str(invalid_fixtures["invalid_missing_attrs"]))


# ---------------------------------------------------------------------------
# robot_profile JSON validation
# ---------------------------------------------------------------------------


class TestRobotProfile:
    def test_malformed_robot_profile_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="robot_profile"):
            validate_h5_file(str(invalid_fixtures["invalid_malformed_profile"]))

    @pytest.mark.parametrize(
        "fixture_key,match",
        [
            ("invalid_profile_missing_key", "Robot profile missing keys"),
            ("invalid_profile_no_gripper", "Invalid action_space"),
            ("invalid_profile_joint_no_names", "action_joint_names is required"),
            ("invalid_profile_missing_rs_key", "missing robot state keys"),
        ],
    )
    def test_invalid_profile_semantics_raise(self, invalid_fixtures, fixture_key, match):
        with pytest.raises(AssertionError, match=match):
            validate_h5_file(str(invalid_fixtures[fixture_key]))

    def test_invalid_control_freq_zero_raises(self, invalid_fixtures):
        # "control_freq" alone used to match the fixture's own *filename* in the
        # "Video file does not exist: .../invalid_control_freq_zero_front.mp4" message,
        # so this passed without the check under test ever running.
        with pytest.raises(AssertionError, match=r"control_freq must be > 0"):
            validate_h5_file(str(invalid_fixtures["invalid_control_freq_zero"]))

# ---------------------------------------------------------------------------
# Missing groups
# ---------------------------------------------------------------------------


class TestRequiredGroups:
    @pytest.mark.parametrize(
        "fixture_key,match",
        [
            ("invalid_actions_missing", "Missing group: actions"),
            ("invalid_robot_states_missing", "Missing group: observations/robot_states"),
            # observations/video_paths group entirely absent
            ("invalid_no_video_group", "Missing group: observations/video_paths"),
        ],
    )
    def test_missing_group_raises(self, invalid_fixtures, fixture_key, match):
        with pytest.raises(AssertionError, match=match):
            validate_h5_file(str(invalid_fixtures[fixture_key]))

    def test_missing_robot_state_key_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="Missing observations/robot_states/"):
            validate_h5_file(str(invalid_fixtures["invalid_robot_state_missing_key"]))


# ---------------------------------------------------------------------------
# Trajectory length violations
# ---------------------------------------------------------------------------


class TestTrajectoryLengths:
    def test_mismatched_lengths_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="Inconsistent trajectory lengths"):
            validate_h5_file(str(invalid_fixtures["invalid_mismatched_steps"]))

    def test_zero_steps_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="episode duration 0.00s out of range"):
            validate_h5_file(str(invalid_fixtures["invalid_zero_steps"]))


class TestTrajectoryValues:
    @pytest.mark.parametrize(
        "dataset,index,value",
        [
            ("actions/joint_velocity", (4, 2), np.nan),
            ("observations/robot_states/joint_position", (7, 3), np.inf),
            ("actions/gripper_position", (9, 0), -np.inf),
        ],
        ids=["nan-action", "inf-observation", "negative-inf-action"],
    )
    def test_non_finite_trajectory_values_are_rejected(
        self, tmp_path, dataset, index, value
    ):
        h5_path = write_valid_episode(tmp_path, "non_finite")
        with h5py.File(h5_path, "r+") as f:
            f[dataset][index] = value

        with pytest.raises(EpisodeValidationError, match=rf"{dataset}.*non-finite") as exc:
            validate_h5_file(str(h5_path), strict_annotation_check=True)
        assert str(index) in str(exc.value)

    def test_non_numeric_trajectory_array_is_a_validation_error(self, tmp_path):
        h5_path = write_valid_episode(tmp_path, "strings")
        with h5py.File(h5_path, "r+") as f:
            del f["actions/joint_velocity"]
            f["actions"].create_dataset(
                "joint_velocity",
                data=np.full((20, 7), "not-a-number", dtype="S12"),
            )

        with pytest.raises(
            EpisodeValidationError,
            match="actions/joint_velocity must be a real numeric array",
        ):
            validate_h5_file(str(h5_path), strict_annotation_check=True)

    @pytest.mark.parametrize(
        "group,arm_slice",
        [
            ("actions", slice(3, 7)),
            ("observations/robot_states", slice(3, 7)),
        ],
        ids=["action", "observation"],
    )
    def test_non_unit_cartesian_quaternion_is_rejected(
        self, tmp_path, group, arm_slice
    ):
        h5_path = write_valid_episode(tmp_path, "bad_quaternion")
        _install_cartesian_trajectories(h5_path)
        with h5py.File(h5_path, "r+") as f:
            f[f"{group}/cartesian_position"][5, arm_slice] = [0.0, 0.0, 0.0, 2.0]

        with pytest.raises(
            EpisodeValidationError,
            match=rf"{group}/cartesian_position.*sample index \(5,\).*norm 2.000000",
        ):
            validate_h5_file(str(h5_path), strict_annotation_check=True)

    def test_second_biarm_quaternion_is_validated(self, tmp_path):
        h5_path = write_valid_episode(tmp_path, "bad_second_arm")
        _install_cartesian_trajectories(h5_path, is_biarm=True)
        with h5py.File(h5_path, "r+") as f:
            f["actions/cartesian_position"][8, 10:14] = [0.0, 0.0, 0.0, 0.0]

        with pytest.raises(
            EpisodeValidationError,
            match=r"actions/cartesian_position\[10:14\] \(arm 2\).*index \(8,\)",
        ):
            validate_h5_file(str(h5_path), strict_annotation_check=True)

    def test_valid_unit_quaternions_pass(self, tmp_path):
        h5_path = write_valid_episode(tmp_path, "unit_quaternions")
        _install_cartesian_trajectories(h5_path, is_biarm=True)

        assert validate_h5_file(str(h5_path), strict_annotation_check=True) is True

    @pytest.mark.parametrize(
        "shape,is_biarm",
        [((20,), False), ((20, 1), False), ((20,), True), ((20, 2), True)],
    )
    def test_documented_gripper_binary_shapes_pass(self, tmp_path, shape, is_biarm):
        h5_path = write_valid_episode(tmp_path, "binary_shape")
        _install_gripper_binary_trajectory(
            h5_path, np.zeros(shape, dtype=np.float32), is_biarm=is_biarm
        )

        assert validate_h5_file(str(h5_path), strict_annotation_check=True) is True

    def test_non_binary_gripper_value_is_rejected(self, tmp_path):
        h5_path = write_valid_episode(tmp_path, "bad_binary_value")
        values = np.zeros((20,), dtype=np.float32)
        values[6] = 2.5
        _install_gripper_binary_trajectory(h5_path, values)

        with pytest.raises(
            EpisodeValidationError,
            match=r"actions/gripper_binary must contain only binary 0 or 1.*2.5",
        ):
            validate_h5_file(str(h5_path), strict_annotation_check=True)

    def test_gripper_width_larger_than_arm_count_is_rejected(self, tmp_path):
        h5_path = write_valid_episode(tmp_path, "bad_binary_width")
        _install_gripper_binary_trajectory(
            h5_path, np.zeros((20, 3), dtype=np.float32)
        )

        with pytest.raises(
            EpisodeValidationError,
            match=r"actions/gripper_binary has 3 command channels.*permits at most 1",
        ):
            validate_h5_file(str(h5_path), strict_annotation_check=True)


# ---------------------------------------------------------------------------
# Video checks
# ---------------------------------------------------------------------------


class TestVideos:
    def test_broken_video_ref_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="does not exist"):
            validate_h5_file(str(invalid_fixtures["invalid_broken_video_ref"]))

    @pytest.mark.parametrize(
        "fixture_key",
        ["invalid_inconsistent_video_lengths", "invalid_video_length_step_mismatch"],
    )
    def test_frame_count_must_match_the_trajectory(self, invalid_fixtures, fixture_key):
        with pytest.raises(AssertionError, match=r"Frame count / trajectory mismatch"):
            validate_h5_file(str(invalid_fixtures[fixture_key]))

    def test_video_below_the_minimum_size_is_rejected(self, tmp_path):
        """The check that used to mask every other video assertion, now on its own."""
        from oopsie_data_tools.test.fixtures.make_invalid import _write_video

        h5_path = write_valid_episode(tmp_path, "tiny")
        with h5py.File(h5_path, "r") as f:
            rel = f["observations/video_paths/front"][()].decode("utf-8")
        _write_video(tmp_path / rel, (10, 10, 10), size=64)

        with pytest.raises(AssertionError, match="Video too small"):
            validate_h5_file(str(h5_path), strict_annotation_check=True)

    @pytest.mark.parametrize(
        "crf, expected_message",
        [
            (18.0, None),
            (19.0, None),
            (23.0, "detected CRF 23.0"),
            (None, "CRF could not be determined"),
        ],
        ids=["better-than-target", "at-target", "too-lossy", "unknown"],
    )
    def test_crf_check_is_advisory(
        self, valid_episode, monkeypatch, caplog, crf, expected_message
    ):
        monkeypatch.setattr(
            "oopsie_data_tools.utils.validation.episode_loader.detect_x264_crf",
            lambda _path: crf,
        )
        caplog.set_level(logging.WARNING)

        assert validate_h5_file(str(valid_episode), strict_annotation_check=True) is True

        if expected_message is None:
            assert "[video quality]" not in caplog.text
        else:
            assert expected_message in caplog.text
            assert "validation will continue" in caplog.text

    def test_detect_x264_crf_reads_embedded_encoder_options(self, tmp_path):
        video = tmp_path / "encoder-options.mp4"
        video.write_bytes(
            b"binary prefix x264 - core 164 - H.264 encoder - options: "
            b"cabac=1 rc=crf mbtree=1 crf=23.0 qcomp=0.60 binary suffix"
        )

        assert detect_x264_crf(str(video)) == 23.0

    def test_detect_x264_crf_reads_an_encoded_fixture(self, valid_episode):
        with h5py.File(valid_episode, "r") as f:
            rel = f["observations/video_paths/front"][()].decode("utf-8")

        assert detect_x264_crf(str(valid_episode.parent / rel)) == 19.0

    def test_detect_x264_crf_does_not_guess_from_unrelated_metadata(self, tmp_path):
        video = tmp_path / "no-encoder-options.mp4"
        video.write_bytes(b"description=export crf=28.0 bitrate=1234")

        assert detect_x264_crf(str(video)) is None

    @pytest.mark.parametrize(
        "level, is_yellow",
        [
            (logging.INFO, False),
            (logging.WARNING, True),
            (logging.ERROR, False),
        ],
        ids=["info", "warning", "error"],
    )
    def test_cli_formatter_only_colors_warning_level(self, level, is_yellow):
        record = logging.LogRecord(
            name="validator",
            level=level,
            pathname=__file__,
            lineno=1,
            msg="log message",
            args=(),
            exc_info=None,
        )

        rendered = _CliFormatter("%(message)s").format(record)

        assert ("\x1b[33m" in rendered) is is_yellow
        assert "log message" in rendered


# ---------------------------------------------------------------------------
# Profile/file consistency
# ---------------------------------------------------------------------------


class TestProfileFileConsistency:
    @pytest.mark.parametrize(
        "fixture_key,match",
        [
            (
                "invalid_joint_names_length_mismatch",
                "robot_state_joint_names count does not match",
            ),
            (
                "invalid_action_names_length_mismatch",
                "action_joint_names count does not match",
            ),
            (
                "invalid_profile_camera_not_in_obs",
                "Missing observations/video_paths/",
            ),
            (
                "invalid_profile_action_not_in_recorded",
                "Missing actions/",
            ),
            (
                "invalid_profile_rs_key_not_in_recorded",
                "Missing observations/robot_states/",
            ),
        ],
    )
    def test_profile_file_consistency_raises(
        self, invalid_fixtures, fixture_key, match
    ):
        with pytest.raises(AssertionError, match=match):
            validate_h5_file(str(invalid_fixtures[fixture_key]))

# ---------------------------------------------------------------------------
# validate_session_dir
# ---------------------------------------------------------------------------


class TestMalformedAnnotations:
    """``episode_annotations`` stored as anything but a group of per-annotator subgroups
    used to escape as an AttributeError from ``.keys()`` rather than a validation error."""

    @pytest.mark.parametrize(
        "fixture_key,match",
        [
            ("invalid_annotation_dataset",
             "episode_annotations must be a group of per-annotator subgroups"),
            ("invalid_taxonomy_not_json", "taxonomy is not valid JSON"),
        ],
    )
    def test_fixture_fails_on_its_own_defect(self, invalid_fixtures, fixture_key, match):
        with pytest.raises(AssertionError, match=match):
            validate_h5_file(str(invalid_fixtures[fixture_key]), strict_annotation_check=True)


class TestProfileDocumentsTheEpisode:
    """The profile is what documents an episode, so the two must agree in both directions."""

    def test_undeclared_robot_state_key_is_rejected(self, invalid_fixtures):
        """Data the profile does not declare has no joint names, units or expected DOF."""
        with pytest.raises(AssertionError, match="does not declare.*velocity_hack"):
            validate_h5_file(str(invalid_fixtures["invalid_robot_state_extra_key"]))

    def test_biarm_profile_rejects_a_single_arm_cartesian_pose(self, invalid_fixtures):
        """7 DOF is one [x,y,z,qx,qy,qz,qw]; a bimanual robot records 14."""
        with pytest.raises(AssertionError, match="is_biarm"):
            validate_h5_file(str(invalid_fixtures["invalid_profile_biarm_mismatch"]))


class TestValidateSessionDir:
    """Return codes follow the shell convention: 0 = all passed, 1 = failure."""

    def test_valid_session_passes(self, valid_session_dir):
        assert validate_session_dir(str(valid_session_dir)) == 0

    @pytest.mark.parametrize("subdir", ["no_such_dir", ""], ids=["missing", "empty"])
    def test_a_directory_with_no_episodes_returns_1(self, tmp_path, subdir):
        assert validate_session_dir(str(tmp_path / subdir)) == 1

    def test_mixed_dir_returns_1(self, tmp_path):
        write_valid_episode(tmp_path, "good")
        (tmp_path / "bad.h5").write_text("not hdf5")
        assert validate_session_dir(str(tmp_path)) == 1


# ---------------------------------------------------------------------------
# Better error messaging (#21)
# ---------------------------------------------------------------------------


class TestBetterErrors:
    def test_validate_accepts_log_path(self, valid_episode, tmp_path):
        # Regression: upload.py passes log_path to validate_h5_file for single files.
        log_path = tmp_path / "validate.log"
        assert validate_h5_file(str(valid_episode), log_path=str(log_path)) is True
        assert log_path.exists(), "a log path that is accepted but never written is not a log"

    def test_crf_advisory_is_written_to_log_path(
        self, valid_episode, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "oopsie_data_tools.utils.validation.episode_loader.detect_x264_crf",
            lambda _path: 23.0,
        )
        log_path = tmp_path / "validate.log"
        validation_logger = logging.getLogger(VALIDATION_LOGGER_NAME)
        previous_handlers = set(validation_logger.handlers)

        try:
            assert (
                validate_h5_file(
                    str(valid_episode),
                    strict_annotation_check=True,
                    log_path=log_path,
                )
                is True
            )
            log_text = log_path.read_text(encoding="utf-8")
        finally:
            for handler in set(validation_logger.handlers) - previous_handlers:
                handler.close()
                validation_logger.removeHandler(handler)

        assert "WARNING" in log_text
        assert "detected CRF 23.0" in log_text
        assert "validation will continue" in log_text


# ---------------------------------------------------------------------------
# Annotation semantics (#29 qualified success, #26 multi/non-human annotators)
# ---------------------------------------------------------------------------


class TestAnnotationSemantics:
    def test_known_legacy_v1_vocabulary_passes(self, legacy_v1_episode) -> None:
        assert (
            validate_h5_file(
                str(legacy_v1_episode), strict_annotation_check=True
            )
            is True
        )

    def test_success_side_effect_with_severity_passes(self, tmp_path: Path) -> None:
        """A qualified success may set severity without a full taxonomy (#29)."""
        h5_path = write_valid_episode(tmp_path, stem="ep")
        with h5py.File(h5_path, "r+") as f:
            g = f["episode_annotations"]["test_annotator"]
            g.attrs["success"] = 1.0
            g.attrs["taxonomy"] = json.dumps(
                {
                    "outcome": "success_side_effect",
                    "failure_category": [],
                    "severity": "low",
                }
            )
        assert validate_h5_file(str(h5_path), strict_annotation_check=True) is True

    def test_partial_failure_taxonomy_is_now_valid(self, tmp_path: Path) -> None:
        """v2 relaxed the rule: only the outcome is required, so a partial trio passes.

        Under v1 this exact file was rejected ("all filled or all empty"). Keeping it
        rejected would mean an annotator could not save work in progress.
        """
        h5_path = write_valid_episode(tmp_path, stem="ep")
        with h5py.File(h5_path, "r+") as f:
            g = f["episode_annotations"]["test_annotator"]
            g.attrs["success"] = 0.0
            g.attrs["episode_description"] = ""
            g.attrs["taxonomy"] = json.dumps(
                {"outcome": "failure", "failure_category": [], "severity": "low"}
            )
        assert validate_h5_file(str(h5_path), strict_annotation_check=True) is True

    def test_failure_with_no_taxonomy_at_all_is_valid(self, tmp_path: Path) -> None:
        h5_path = write_valid_episode(tmp_path, stem="ep")
        with h5py.File(h5_path, "r+") as f:
            g = f["episode_annotations"]["test_annotator"]
            g.attrs["success"] = 0.0
            g.attrs["episode_description"] = ""
            g.attrs["taxonomy"] = json.dumps(
                {"outcome": "failure", "failure_category": [], "severity": ""}
            )
        assert validate_h5_file(str(h5_path), strict_annotation_check=True) is True

    def test_outcome_disagreeing_with_success_is_rejected(self, tmp_path: Path) -> None:
        """A float and a slug that contradict each other would split downstream readers."""
        h5_path = write_valid_episode(tmp_path, stem="ep")
        with h5py.File(h5_path, "r+") as f:
            g = f["episode_annotations"]["test_annotator"]
            g.attrs["success"] = 1.0
            g.attrs["taxonomy"] = json.dumps({"outcome": "failure", "severity": ""})
        with pytest.raises(AssertionError, match="disagrees with success"):
            validate_h5_file(str(h5_path), strict_annotation_check=True)

    def test_unrecognized_outcome_is_rejected(self, tmp_path: Path) -> None:
        h5_path = write_valid_episode(tmp_path, stem="ep")
        with h5py.File(h5_path, "r+") as f:
            g = f["episode_annotations"]["test_annotator"]
            g.attrs["taxonomy"] = json.dumps({"outcome": "sort_of_worked"})
        with pytest.raises(AssertionError, match="unrecognized outcome"):
            validate_h5_file(str(h5_path), strict_annotation_check=True)

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("failure_category", ["grasp_failure"], "unrecognized failure_category"),
            ("severity", "major", "severity must be one of"),
        ],
    )
    def test_unrecognized_taxonomy_values_are_rejected(
        self, tmp_path: Path, field, value, match
    ) -> None:
        h5_path = write_valid_episode(tmp_path, stem=f"unknown_{field}")
        with h5py.File(h5_path, "r+") as f:
            group = f["episode_annotations"]["test_annotator"]
            taxonomy = json.loads(group.attrs["taxonomy"])
            taxonomy[field] = value
            group.attrs["taxonomy"] = json.dumps(taxonomy)

        with pytest.raises(EpisodeValidationError, match=match):
            validate_h5_file(str(h5_path), strict_annotation_check=True)

    def test_unrecognized_legacy_v1_taxonomy_is_rejected_but_not_relabelled(
        self, tmp_path: Path
    ) -> None:
        h5_path = write_valid_episode(tmp_path, stem="unknown_v1")
        with h5py.File(h5_path, "r+") as f:
            group = f["episode_annotations"]["test_annotator"]
            group.attrs["success"] = 0.0
            group.attrs["taxonomy"] = json.dumps(
                {
                    "failure_category": ["mystery failure"],
                    "severity": "major",
                }
            )

        with pytest.raises(
            EpisodeValidationError, match="unrecognized failure_category.*mystery failure"
        ):
            validate_h5_file(str(h5_path), strict_annotation_check=True)

    def test_incomplete_extra_subgroup_fails(self, tmp_path: Path) -> None:
        """Every annotator subgroup must be complete: a stray incomplete one fails the episode."""
        h5_path = write_valid_episode(tmp_path, stem="ep")
        with h5py.File(h5_path, "r+") as f:
            extra = f["episode_annotations"].require_group("cosmos-7b")
            extra.attrs["source"] = "cosmos-7b"  # no 'success' → incomplete subgroup
        with pytest.raises(AssertionError, match="missing 'success'"):
            validate_h5_file(str(h5_path), strict_annotation_check=True)

    def test_extra_complete_subgroup_passes(self, tmp_path: Path) -> None:
        """An additional fully-annotated subgroup (any source) is fine."""
        h5_path = write_valid_episode(tmp_path, stem="ep")
        with h5py.File(h5_path, "r+") as f:
            extra = f["episode_annotations"].require_group("cosmos-7b")
            extra.attrs["source"] = "cosmos-7b"
            extra.attrs["success"] = 1.0
        assert validate_h5_file(str(h5_path), strict_annotation_check=True) is True
