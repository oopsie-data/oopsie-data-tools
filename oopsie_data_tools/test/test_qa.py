"""``oopsie-data qa``: the data-quality checks and the report they produce.

The checks run on synthetic arrays built to be clean, then broken in one named way, so each
test fails on the defect it is named for. The CLI tests run on the shared valid fixtures,
whose random-noise data, placeholder names and static videos are flagged by several checks;
they pin the shape of the report rather than which checks fire.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from oopsie_data_tools import cli
from oopsie_data_tools.qa.action_checks import placeholder_reason, run_action_qa_on_arrays
from oopsie_data_tools.qa.config import (
    CHECK_DESCRIPTIONS,
    CONFIDENCE_LEVELS,
    DEFAULT_CHECKS,
    FEEDBACK,
    SUGGESTED_ACTIONS,
    build_checks,
    disable_category,
    load_qa_config,
)
from oopsie_data_tools.qa.runner import (
    _episode_record,
    episode_cameras,
    run_qa,
    select_episodes,
)
from oopsie_data_tools.qa.video_checks import run_video_checks

T = 100
JOINTS = [f"joint{i + 1}" for i in range(7)]


def _tested(result: dict) -> set[str]:
    return {f"{f['category']}.{f['test']}" for f in result["findings"]}


@pytest.fixture
def joint_episode():
    """A clean joint-position episode: actions lead the state by one step, plus noise."""
    rng = np.random.default_rng(0)
    t = np.arange(T + 1)[:, None] / 10.0
    q = 0.5 * np.sin(t * np.linspace(0.5, 1.2, len(JOINTS)))
    grip = (np.sin(t[:, 0]) > 0).astype(float)[:, None]
    states = {"joint_position": q[:-1], "gripper_position": 0.04 * grip[:-1]}
    actions = {
        "joint_position": q[1:] + 1e-3 * rng.standard_normal((T, len(JOINTS))),
        "gripper_binary": grip[1:],
    }
    profile = {
        "action_space": ["joint_position", "gripper_binary"],
        "control_freq": 10,
        "is_biarm": False,
        "action_joint_names": JOINTS,
        "robot_state_joint_names": JOINTS,
    }
    return actions, states, profile


@pytest.fixture
def cartesian_episode():
    """A clean Cartesian-position episode with a unit-quaternion orientation."""
    t = np.arange(T + 1)[:, None] / 10.0
    xyz = np.c_[0.4 + 0.1 * np.sin(t), 0.1 * np.cos(t), 0.3 + 0 * t]
    euler = np.c_[np.pi + 0 * t, 0.2 * np.sin(t[:, 0]), 0.3 * t[:, 0]]
    pose = np.c_[xyz, Rotation.from_euler("xyz", euler).as_quat()]
    grip = (np.sin(t[:, 0]) > 0).astype(float)[:, None]
    states = {"cartesian_position": pose[:-1], "gripper_position": 0.04 * grip[:-1]}
    actions = {"cartesian_position": pose[1:] + 1e-4, "gripper_binary": grip[1:]}
    profile = {
        "action_space": ["cartesian_position", "gripper_binary"],
        "control_freq": 10,
        "is_biarm": False,
    }
    return actions, states, profile


# ── checks ────────────────────────────────────────────────────────────────────


def test_every_check_has_a_description():
    """The report prints a description next to every finding; a new check needs one."""
    checks = {(c, t) for c, tests in DEFAULT_CHECKS.items() for t in tests}
    assert checks == set(CHECK_DESCRIPTIONS)


def test_every_check_has_a_confidence_and_suggested_action():
    checks = {f"{c}.{t}" for c, tests in DEFAULT_CHECKS.items() for t in tests}
    assert checks == set(FEEDBACK)
    for confidence, action in FEEDBACK.values():
        assert confidence in CONFIDENCE_LEVELS
        assert action in SUGGESTED_ACTIONS


def test_clean_joint_episode_has_no_findings(joint_episode):
    result = run_action_qa_on_arrays(*joint_episode, load_qa_config())
    assert result["findings"] == []
    assert ("joint_position", "tracking") in result["checked"]


def test_joint_positions_in_degrees_are_flagged(joint_episode):
    actions, states, profile = joint_episode
    actions = {**actions, "joint_position": np.degrees(actions["joint_position"])}
    states = {**states, "joint_position": np.degrees(states["joint_position"])}
    result = run_action_qa_on_arrays(actions, states, profile, load_qa_config())
    assert _tested(result) == {"joint_position.radians"}


def test_missing_gripper_binary_is_flagged(joint_episode):
    actions, states, profile = joint_episode
    actions = {k: v for k, v in actions.items() if k != "gripper_binary"}
    profile = {**profile, "action_space": ["joint_position"]}
    result = run_action_qa_on_arrays(actions, states, profile, load_qa_config())
    assert _tested(result) == {"general.gripper_binary_present"}


def test_clean_cartesian_episode_has_no_findings(cartesian_episode):
    result = run_action_qa_on_arrays(*cartesian_episode, load_qa_config())
    assert result["findings"] == []


def test_quaternion_sign_flip_is_flagged_when_enabled(cartesian_episode):
    """Off by default; the check itself still has to work when a user turns it on."""
    actions, states, profile = cartesian_episode
    pose = actions["cartesian_position"].copy()
    pose[T // 2 :, 3:] *= -1
    config = build_checks({"cartesian_position": {"quaternion_sign_flip": {"enabled": True}}})
    result = run_action_qa_on_arrays(
        {**actions, "cartesian_position": pose}, states, profile, config
    )
    assert _tested(result) == {"cartesian_position.quaternion_sign_flip"}


def test_cartesian_positions_in_millimetres_are_flagged(cartesian_episode):
    actions, states, profile = cartesian_episode
    pose = actions["cartesian_position"].copy()
    pose[:, :3] *= 1000
    result = run_action_qa_on_arrays(
        {**actions, "cartesian_position": pose}, states, profile, load_qa_config()
    )
    assert "cartesian_position.position_units" in _tested(result)


def test_constant_action_is_flagged(joint_episode):
    actions, states, profile = joint_episode
    held = np.repeat(actions["joint_position"][:1], T, axis=0)
    result = run_action_qa_on_arrays(
        {**actions, "joint_position": held}, states, profile, load_qa_config()
    )
    assert "general.constant_action" in _tested(result)


def test_actions_held_for_several_steps_are_flagged_as_command_rate(joint_episode):
    actions, states, profile = joint_episode
    held = np.repeat(actions["joint_position"][::3], 3, axis=0)[:T]
    result = run_action_qa_on_arrays(
        {**actions, "joint_position": held}, states, profile, load_qa_config()
    )
    (finding,) = [f for f in result["findings"] if f["test"] == "command_rate"]
    assert finding["period"] == 3


def test_action_chunk_recorded_per_step_is_flagged(joint_episode):
    actions, states, profile = joint_episode
    chunked = np.repeat(actions["joint_position"][:, None, :], 4, axis=1)
    result = run_action_qa_on_arrays(
        {**actions, "joint_position": chunked}, states, profile, load_qa_config()
    )
    assert "general.action_chunk_shape" in _tested(result)


def test_missing_gripper_state_is_flagged(joint_episode):
    actions, states, profile = joint_episode
    states = {k: v for k, v in states.items() if k != "gripper_position"}
    result = run_action_qa_on_arrays(actions, states, profile, load_qa_config())
    assert {f["key"] for f in result["findings"] if f["test"] == "required_state_keys"} == {
        "gripper_position"
    }


@pytest.mark.parametrize(
    "field, value, reason",
    [
        ("operator_name", "", "blank"),
        ("operator_name", "pending_operator", "placeholder"),
        ("operator_name", "autoeval_orchestrator", "not_a_person"),
        ("robot_name", "your_robot", "template"),
        ("policy_name", "Unknown", "placeholder"),
        ("operator_name", "Jane Doe", None),
        ("robot_name", "franka_panda", None),
    ],
)
def test_placeholder_metadata(field, value, reason):
    cfg = load_qa_config()["general"]["placeholder_metadata"]
    assert placeholder_reason(field, value, cfg) == reason

# ── config ────────────────────────────────────────────────────────────────────


def test_config_file_overrides_a_threshold(tmp_path):
    path = tmp_path / "qa.yaml"
    path.write_text("video:\n  resolution:\n    min_short_side: 100\n")
    config = load_qa_config(path)
    assert config["video"]["resolution"]["min_short_side"] == 100
    assert config["video"]["brightness"] == DEFAULT_CHECKS["video"]["brightness"]


def test_config_file_rejects_an_unknown_check(tmp_path):
    path = tmp_path / "qa.yaml"
    path.write_text("video:\n  resoluton:\n    enabled: false\n")
    with pytest.raises(ValueError, match="unknown check"):
        load_qa_config(path)


def test_disable_category_leaves_the_input_untouched():
    config = load_qa_config()
    off = disable_category(config, "video")
    assert not any(c["enabled"] for c in off["video"].values())
    assert config["video"]["decoding"]["enabled"]


def test_config_file_category_switch_turns_off_every_check(tmp_path):
    path = tmp_path / "qa.yaml"
    path.write_text("video:\n  enabled: false\n  decoding:\n    enabled: true\n")
    config = load_qa_config(path)
    assert not any(c["enabled"] for c in config["video"].values())


def test_config_file_sets_video_settings(tmp_path):
    path = tmp_path / "qa.yaml"
    path.write_text("video:\n  ignore_cameras: [wrist]\n")
    assert load_qa_config(path)["video_settings"]["ignore_cameras"] == ["wrist"]


# ── runner ────────────────────────────────────────────────────────────────────


def test_episode_cameras_resolves_relative_video_paths(valid_episode):
    cameras = episode_cameras(valid_episode)
    assert cameras
    for path in cameras.values():
        assert path.is_absolute() and path.is_file()


def test_ignored_camera_is_skipped_by_every_video_check(valid_episode):
    cameras = episode_cameras(valid_episode)
    config = load_qa_config()
    settings = {**config["video_settings"], "ignore_cameras": list(cameras)}
    result = run_video_checks(cameras, {}, 10.0, config["video"], settings)
    assert result["findings"] == []
    assert {s["reason"] for s in result["skips"]} >= {"ignored_camera"}


def test_informational_findings_do_not_flag_an_episode(tmp_path):
    finding = {"category": "joint_velocity", "test": "smoothness", "key": "joint_velocity"}
    result = {"findings": [finding], "skips": [], "checked": [("joint_velocity", "smoothness")]}
    record = _episode_record(tmp_path / "e.h5", tmp_path, result, None)
    assert record["status"] == "clean"
    assert record["findings"][0]["suggested_action"] == "informational"


def test_select_episodes_is_reproducible_with_a_seed():
    paths = [Path(f"{i:06d}.h5") for i in range(20)]
    first = select_episodes(paths, 5, seed=3)
    assert first == select_episodes(paths, 5, seed=3)
    assert len(first) == 5 and first == sorted(first)
    assert select_episodes(paths, 0, seed=3) == paths


def test_unreadable_episode_is_reported_as_an_error(tmp_path):
    (tmp_path / "broken.h5").write_bytes(b"not an hdf5 file")
    report = run_qa(tmp_path, load_qa_config())
    (episode,) = report["episodes"]
    assert episode["status"] == "error"
    assert report["summary"]["episodes_errored"] == 1


# ── CLI ───────────────────────────────────────────────────────────────────────


def _qa(tmp_path, *args: str) -> tuple[int, Path]:
    output = tmp_path / "report.yaml"
    code = cli.main(["qa", "--workers", "1", "--output", str(output), *args])
    return code, output


def test_qa_writes_a_parseable_report(tmp_path, valid_session_dir):
    code, output = _qa(tmp_path, "--path", str(valid_session_dir))
    assert code == 0

    report = yaml.safe_load(output.read_text())
    summary, episodes = report["summary"], report["episodes"]
    assert summary["schema"] == "oopsie_qa_report_v1"
    assert summary["episodes_checked"] == len(episodes) == 4
    assert summary["episodes_flagged"] + summary["episodes_clean"] == 4
    for episode in episodes:
        for finding in episode.get("findings", []):
            assert set(finding) == {
                "check",
                "key",
                "description",
                "confidence",
                "suggested_action",
                "details",
            }
            assert finding["check"] in episode["flagged_checks"]


def test_qa_no_video_runs_no_video_check(tmp_path, valid_session_dir):
    code, output = _qa(tmp_path, "--path", str(valid_session_dir), "--no-video")
    assert code == 0
    report = yaml.safe_load(output.read_text())
    assert not any(c.startswith("video.") for c in report["summary"]["checks_enabled"])
    assert report["summary"]["config_overrides"]["video"]["decoding"] == {"enabled": False}


def test_qa_strict_exits_1_on_findings(tmp_path, valid_episode):
    """The fixture's operator is ``test_operator``, a placeholder name."""
    code, output = _qa(tmp_path, "--path", str(valid_episode), "--strict")
    assert code == 1
    report = yaml.safe_load(output.read_text())
    assert "general.placeholder_metadata" in report["episodes"][0]["flagged_checks"]


def test_qa_exits_1_when_nothing_could_be_checked(tmp_path):
    (tmp_path / "broken.h5").write_bytes(b"not an hdf5 file")
    code, _ = _qa(tmp_path, "--path", str(tmp_path))
    assert code == 1


def test_qa_missing_path_exits_1(tmp_path):
    code, output = _qa(tmp_path, "--path", str(tmp_path / "nope"))
    assert code == 1
    assert not output.exists()


def test_qa_json_matches_the_report_file(tmp_path, capsys, valid_episode):
    code, output = _qa(tmp_path, "--path", str(valid_episode), "--json")
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["report_path"] == str(output.resolve())
    assert payload["episodes"] == yaml.safe_load(output.read_text())["episodes"]
