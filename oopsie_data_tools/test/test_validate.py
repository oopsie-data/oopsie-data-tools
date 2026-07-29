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
from pathlib import Path

import h5py
import numpy as np
import pytest

from oopsie_data_tools.test.fixtures.make_valid import write_valid_episode
from oopsie_data_tools.utils.hf_upload import run_validation
from oopsie_data_tools.utils.validation.errors import EpisodeValidationError
from oopsie_data_tools.utils.validation.validation_utils import (
    validate_h5_file,
    validate_session_dir,
)

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidEpisodes:
    def test_minimally_annotated_episode_passes(self, valid_episode):
        assert validate_h5_file(str(valid_episode)) is True

    def test_minimally_annotated_episode_passes_strict(self, valid_episode):
        assert validate_h5_file(str(valid_episode), strict_annotation_check=True) is True

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

    def test_success_episode_passes(self, valid_success_episode):
        assert validate_h5_file(str(valid_success_episode)) is True

    def test_failure_episode_passes(self, valid_failure_episode):
        assert validate_h5_file(str(valid_failure_episode)) is True


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
    @pytest.mark.parametrize(
        "fixture_key",
        [
            "invalid_missing_attrs",  # all attrs absent
        ],
    )
    def test_missing_attrs_raises(self, invalid_fixtures, fixture_key):
        with pytest.raises(AssertionError, match="Missing root attr"):
            validate_h5_file(str(invalid_fixtures[fixture_key]))


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
            ("invalid_profile_unsupported_action", "Invalid action_space"),
            ("invalid_profile_missing_rs_key", "missing robot state keys"),
            # An empty camera_names list is not itself rejected; the episode fails because
            # it then carries no video group. Worth knowing which rule is doing the work.
            ("invalid_profile_empty_cameras", "Missing group: observations/video_paths"),
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

    def test_zero_trajectory_via_tmp_path(self, tmp_path):
        h5_path = write_valid_episode(tmp_path, "zero", n=0)
        with pytest.raises(AssertionError, match="No trajectory data|out of range"):
            validate_h5_file(str(h5_path), strict_annotation_check=True)


# ---------------------------------------------------------------------------
# Video checks
# ---------------------------------------------------------------------------


class TestVideos:
    def test_broken_video_ref_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="does not exist"):
            validate_h5_file(str(invalid_fixtures["invalid_broken_video_ref"]))

    def test_array_in_video_path_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="does not exist"):
            validate_h5_file(str(invalid_fixtures["invalid_image_obs_float"]))

    # Both of these used to allow "|Video too small" as an alternative, and every fixture
    # video was 64px against a 180px minimum — so that branch always won and neither
    # frame-count check ran even once.
    def test_inconsistent_video_lengths_raise(self, invalid_fixtures):
        with pytest.raises(AssertionError, match=r"Frame count / trajectory mismatch"):
            validate_h5_file(str(invalid_fixtures["invalid_inconsistent_video_lengths"]))

    def test_video_length_step_mismatch_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match=r"Frame count / trajectory mismatch"):
            validate_h5_file(str(invalid_fixtures["invalid_video_length_step_mismatch"]))

    def test_video_below_the_minimum_size_is_rejected(self, tmp_path):
        """The check that used to mask every other video assertion, now on its own."""
        from oopsie_data_tools.test.fixtures.make_invalid import _write_video

        h5_path = write_valid_episode(tmp_path, "tiny")
        with h5py.File(h5_path, "r") as f:
            rel = f["observations/video_paths/front"][()].decode("utf-8")
        _write_video(tmp_path / rel, (10, 10, 10), size=64)

        with pytest.raises(AssertionError, match="Video too small"):
            validate_h5_file(str(h5_path), strict_annotation_check=True)


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
            (
                "invalid_multiple_promised_fields_missing",
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


class TestPreviouslyUnreferencedFixtures:
    """Fixtures that were generated every session and asserted on by nothing.

    All four detect a real defect; they simply had no test. The annotation-dataset one is
    the sharpest: it was written to catch ``episode_annotations`` stored as a dataset, which
    used to escape as ``AttributeError`` from ``.keys()`` rather than as a validation error.
    """

    @pytest.mark.parametrize(
        "fixture_key,match",
        [
            (
                "invalid_annotation_dataset",
                "episode_annotations must be a group of per-annotator subgroups",
            ),
            (
                "invalid_joint_pos_wrong_dof",
                "robot_state_joint_names count does not match",
            ),
            (
                "invalid_joint_vel_wrong_dof",
                "action_joint_names count does not match",
            ),
            (
                "invalid_taxonomy_not_json",
                "taxonomy is not valid JSON",
            ),
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

    def test_a_declared_key_may_be_absent_is_still_the_other_error(self, invalid_fixtures):
        """The reverse direction was always enforced; check the two messages stay distinct."""
        with pytest.raises(AssertionError, match="Missing observations/robot_states/"):
            validate_h5_file(str(invalid_fixtures["invalid_robot_state_missing_key"]))

    def test_biarm_profile_rejects_a_single_arm_cartesian_pose(self, invalid_fixtures):
        """7 DOF is one [x,y,z,qx,qy,qz,qw]; a bimanual robot records 14."""
        with pytest.raises(AssertionError, match=r"has 7 DOF.*is_biarm=True.*means 14"):
            validate_h5_file(str(invalid_fixtures["invalid_profile_biarm_mismatch"]))

    def test_joint_count_is_not_constrained_by_is_biarm(self, tmp_path):
        """Deliberately not a rule: two arms need not share a DOF count, so 7+6 is valid."""
        h5_path = write_valid_episode(tmp_path, "asymmetric")
        with h5py.File(h5_path, "r+") as f:
            profile = json.loads(f.attrs["robot_profile"])
            profile["is_biarm"] = True
            f.attrs["robot_profile"] = json.dumps(profile)

        assert validate_h5_file(str(h5_path), strict_annotation_check=True) is True

    @pytest.mark.parametrize("gripper_dof", [1, 2, 3])
    def test_gripper_dof_is_whatever_the_profile_owner_records(self, tmp_path, gripper_dof):
        """Deliberately not a rule.

        Nothing in RobotProfile declares how many gripper DOF to expect — only
        ``gripper_name`` — so there is nothing to check against. Inferring 1-or-2 from
        ``is_biarm`` would reject a multi-finger hand, which ``gripper_name`` explicitly
        anticipates (``robotiq_3f`` and friends). Constraining this needs a
        ``gripper_joint_names`` field mirroring ``robot_state_joint_names``; until then the
        recorded width is accepted as given.
        """
        h5_path = write_valid_episode(tmp_path, f"gripper{gripper_dof}")
        with h5py.File(h5_path, "r+") as f:
            group = f["observations/robot_states"]
            n = group["gripper_position"].shape[0]
            del group["gripper_position"]
            group.create_dataset(
                "gripper_position", data=np.zeros((n, gripper_dof), dtype=np.float64)
            )

        assert validate_h5_file(str(h5_path), strict_annotation_check=True) is True


class TestValidateSessionDir:
    """Return codes follow the shell convention: 0 = all passed, 1 = failure."""

    def test_valid_session_passes(self, valid_session_dir):
        assert validate_session_dir(str(valid_session_dir)) == 0

    def test_nonexistent_dir_returns_1(self, tmp_path):
        assert validate_session_dir(str(tmp_path / "no_such_dir")) == 1

    def test_empty_dir_returns_1(self, tmp_path):
        assert validate_session_dir(str(tmp_path)) == 1

    def test_mixed_dir_returns_1(self, tmp_path):
        write_valid_episode(tmp_path, "good")
        (tmp_path / "bad.h5").write_text("not hdf5")
        assert validate_session_dir(str(tmp_path)) == 1

    def test_all_valid_returns_0(self, tmp_path):
        write_valid_episode(tmp_path, "ep_a")
        write_valid_episode(tmp_path, "ep_b")
        assert validate_session_dir(str(tmp_path)) == 0


# ---------------------------------------------------------------------------
# Better error messaging (#21)
# ---------------------------------------------------------------------------


class TestBetterErrors:
    def test_validate_accepts_log_path(self, valid_success_episode, tmp_path):
        # Regression: upload.py passes log_path to validate_h5_file for single files.
        log_path = tmp_path / "validate.log"
        assert validate_h5_file(str(valid_success_episode), log_path=str(log_path)) is True
        assert log_path.exists(), "a log path that is accepted but never written is not a log"

    def test_robot_state_joint_dof_message(self, invalid_fixtures):
        with pytest.raises(
            AssertionError, match="robot_state_joint_names count does not match"
        ):
            validate_h5_file(str(invalid_fixtures["invalid_joint_names_length_mismatch"]))

    def test_action_joint_dof_message(self, invalid_fixtures):
        with pytest.raises(
            AssertionError, match="action_joint_names count does not match"
        ):
            validate_h5_file(str(invalid_fixtures["invalid_action_names_length_mismatch"]))


# ---------------------------------------------------------------------------
# Annotation semantics (#29 qualified success, #26 multi/non-human annotators)
# ---------------------------------------------------------------------------


class TestAnnotationSemantics:
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
