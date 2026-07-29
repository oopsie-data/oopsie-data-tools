"""``--json`` on the read-only commands.

Agents and scripts drive ``validate``, ``show-config`` and ``inspect`` far more often than
they drive anything that writes, and all three previously answered only in prose. What
matters here is that the document is the *whole* of stdout — a stray log line makes it
unparseable, which is the one failure mode that defeats the purpose of the flag — and that
the exit code still means what it meant before.
"""

from __future__ import annotations

import json

import pytest

from oopsie_data_tools import cli


def _run_json(capsys, argv: list[str]) -> tuple[int, dict]:
    """Run a command and parse its stdout, failing loudly if it is not pure JSON."""
    code = cli.main(argv)
    out = capsys.readouterr().out
    try:
        return code, json.loads(out)
    except json.JSONDecodeError as e:
        pytest.fail(f"stdout was not parseable JSON ({e}):\n{out}")


# ── validate ──────────────────────────────────────────────────────────────────


def test_validate_json_reports_a_passing_episode(capsys, valid_episode):
    code, payload = _run_json(capsys, ["validate", "--path", str(valid_episode), "--json"])

    assert code == 0
    assert payload["total"] == 1
    assert payload["failed"] == 0
    assert payload["episodes"][0]["passed"] is True
    assert payload["episodes"][0]["episode"] == valid_episode.name


def test_validate_json_names_the_episode_and_the_error(capsys, episode_without_annotations):
    """The point of the flag: which episode failed, and on what, without parsing prose."""
    code, payload = _run_json(
        capsys, ["validate", "--path", str(episode_without_annotations), "--json"]
    )

    assert code == 1
    assert payload["failed"] == 1
    (episode,) = payload["episodes"]
    assert episode["passed"] is False
    assert episode["error_type"] == "validation"
    assert "annotation" in episode["error"].lower()


def test_validate_json_covers_a_whole_session_directory(capsys, valid_session_dir):
    code, payload = _run_json(capsys, ["validate", "--path", str(valid_session_dir), "--json"])

    assert code == 0
    assert payload["total"] == payload["passed"] > 1
    assert {e["passed"] for e in payload["episodes"]} == {True}


def test_validate_json_separates_failures_from_passes_in_one_run(
    capsys, tmp_path, valid_session_dir, invalid_fixtures
):
    """A mixed directory is the case an agent actually has: triage, not a yes/no."""
    import shutil

    session = tmp_path / "mixed"
    shutil.copytree(valid_session_dir, session)
    bad = next(iter(invalid_fixtures.values()))
    shutil.copy(bad, session / bad.name)

    code, payload = _run_json(capsys, ["validate", "--path", str(session), "--json"])

    assert code == 1
    assert payload["failed"] >= 1
    assert payload["passed"] >= 1
    failed = [e["episode"] for e in payload["episodes"] if not e["passed"]]
    assert bad.name in failed


def test_validate_json_reports_a_missing_path_as_a_document(capsys, tmp_path):
    """Exiting 1 with empty stdout would leave the caller nothing to read."""
    code, payload = _run_json(capsys, ["validate", "--path", str(tmp_path / "nope"), "--json"])

    assert code == 1
    assert payload["error"]


def test_json_does_not_silence_later_commands(capsys, caplog, valid_episode):
    """--json quiets logging for its own run only; main is called repeatedly in-process."""
    assert cli.main(["validate", "--path", str(valid_episode), "--json"]) == 0
    capsys.readouterr()

    with caplog.at_level("INFO"):
        assert cli.main(["validate", "--path", str(valid_episode)]) == 0

    assert caplog.text.strip(), "prose output went missing after an earlier --json run"


def test_validate_exit_code_is_the_same_with_and_without_json(
    capsys, valid_episode, episode_without_annotations
):
    for path, expected in ((valid_episode, 0), (episode_without_annotations, 1)):
        assert cli.main(["validate", "--path", str(path)]) == expected
        capsys.readouterr()
        assert cli.main(["validate", "--path", str(path), "--json"]) == expected
        capsys.readouterr()


# ── show-config ───────────────────────────────────────────────────────────────


def test_show_config_json_reports_both_search_chains(capsys):
    code, payload = _run_json(capsys, ["show-config", "--json"])

    assert code == 0
    assert payload["credentials"]["search_path"]
    assert payload["robot_profiles"]["search_path"]
    assert payload["credentials"]["env_var"] == "OOPSIE_CONFIG_DIR"


def test_show_config_json_masks_the_token_by_default(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value_1234")

    code, payload = _run_json(capsys, ["show-config", "--json"])

    assert code == 0
    assert payload["credentials"]["hf_token_source"] == "HF_TOKEN"
    assert payload["credentials"]["hf_token"] != "hf_secret_value_1234"

    code, payload = _run_json(capsys, ["show-config", "--json", "--show-token"])

    assert payload["credentials"]["hf_token"] == "hf_secret_value_1234"


def test_show_config_json_survives_an_unparseable_config(capsys, tmp_path, monkeypatch):
    """The command's job is to report what it found, including that it could not read it."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "contributor_config.yaml").write_text("lab_id: [unclosed\n", encoding="utf-8")
    monkeypatch.setenv("OOPSIE_CONFIG_DIR", str(config_dir))

    code, payload = _run_json(capsys, ["show-config", "--json"])

    assert code == 0
    assert payload["credentials"]["parse_error"]


# ── inspect ───────────────────────────────────────────────────────────────────


def test_inspect_json_exposes_the_tree_and_the_root_attrs(capsys, valid_episode):
    code, payload = _run_json(capsys, ["inspect", str(valid_episode), "--json"])

    assert code == 0
    root = payload["root"]
    assert root["type"] == "group"
    assert root["attrs"]["episode_id"]
    assert "observations" in root["children"]
    assert "actions" in root["children"]


def test_inspect_json_reports_dataset_shapes(capsys, valid_episode):
    _, payload = _run_json(capsys, ["inspect", str(valid_episode), "--json"])

    states = payload["root"]["children"]["observations"]["children"]["robot_states"]["children"]
    gripper = states["gripper_position"]

    assert gripper["type"] == "dataset"
    assert gripper["shape"]
    assert gripper["empty"] is False


def test_inspect_json_marks_empty_placeholder_datasets(capsys, valid_episode):
    """Undeclared action keys are h5py.Empty, and telling them apart is the usual question."""
    _, payload = _run_json(capsys, ["inspect", str(valid_episode), "--json"])

    actions = payload["root"]["children"]["actions"]["children"]

    assert any(a["empty"] for a in actions.values()), "expected at least one Empty placeholder"


def test_inspect_json_works_on_a_file_validate_rejects(capsys, invalid_fixtures):
    """inspect makes no schema assumptions; that has to hold in the structured form too."""
    bad = next(iter(invalid_fixtures.values()))

    code, payload = _run_json(capsys, ["inspect", str(bad), "--json"])

    assert code == 0
    assert payload["root"]["type"] == "group"


def test_inspect_json_reports_a_missing_file_as_a_document(capsys, tmp_path):
    code, payload = _run_json(capsys, ["inspect", str(tmp_path / "nope.h5"), "--json"])

    assert code == 1
    assert payload["error"]
