"""The contributor config holds a HuggingFace token, so where it lands matters.

``configs/contributor_config.yaml`` used to be tracked by git *and* listed in .gitignore
(which does nothing for an already-tracked file), while the wizard defaulted to writing
there. These tests pin down the two halves of the fix.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from oopsie_data_tools import init_wizard
from oopsie_data_tools.utils import contributor_config, paths

# conftest._test_contributor_config rebinds the module attribute for the whole session, so
# `contributor_config.read_contributor_config` resolves to a stub at call time. Bind the real
# function here at import, which happens during collection, before that fixture runs.
# Batch 5 switches conftest to monkeypatch, after which this can go back to a plain call.
_real_read_contributor_config = contributor_config.read_contributor_config

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No env override, a fake home, and an interactive stdin."""
    monkeypatch.delenv(paths.ENV_CONFIG_DIR, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(init_wizard.sys.stdin, "isatty", lambda: True)
    return tmp_path


def test_config_file_is_not_tracked_by_git():
    """A tracked file makes its .gitignore entry inert, which is how tokens got committed."""
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "configs/contributor_config.yaml"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert tracked.returncode != 0, (
        "configs/contributor_config.yaml is tracked again. While tracked, .gitignore does "
        "not apply to it and a contributor's HuggingFace token shows up in git status."
    )


def test_config_file_is_ignored():
    ignored = subprocess.run(
        ["git", "check-ignore", "configs/contributor_config.yaml"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert ignored.returncode == 0, "configs/contributor_config.yaml must stay gitignored"


def test_wizard_defaults_to_the_user_dir(isolated, monkeypatch):
    """Pressing Enter must not put a token where a commit can pick it up."""
    answers = iter([""])  # accept the default
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    assert init_wizard.choose_target_dir() == paths.user_config_dir()


def test_wizard_offers_the_working_directory(isolated, tmp_path, monkeypatch):
    """A second lab or a shared machine wants a config that is local to the project."""
    project = tmp_path / "my_robot_code"
    project.mkdir()
    monkeypatch.chdir(project)
    answers = iter(["2"])  # the second option is this directory
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    target = init_wizard.choose_target_dir()

    assert target == project
    assert target in paths.config_search_dirs(), "init must not write where reads cannot find"


def test_the_working_directory_is_not_offered_when_it_is_the_user_dir(isolated, monkeypatch):
    """Running init from ~/.config/oopsie-data must not print the same path twice."""

    def refuse(*_):
        raise AssertionError("there is only one option, so nothing should be asked")

    paths.user_config_dir().mkdir(parents=True)
    monkeypatch.chdir(paths.user_config_dir())
    monkeypatch.setattr("builtins.input", refuse)

    assert init_wizard.choose_target_dir() == paths.user_config_dir()


def test_wizard_uses_user_dir_when_not_interactive(isolated, monkeypatch):
    monkeypatch.setattr(init_wizard.sys.stdin, "isatty", lambda: False)

    assert init_wizard.choose_target_dir() == paths.user_config_dir()


def test_reading_from_a_git_working_tree_warns(tmp_path, monkeypatch, caplog):
    """A token under version control can be committed; say so whichever repo it is."""
    monkeypatch.setattr(contributor_config, "_warned_about_config_in_worktree", False)
    worktree = tmp_path / "some_project"
    (worktree / ".git").mkdir(parents=True)
    config = worktree / "contributor_config.yaml"
    config.write_text("lab_id: MyLab\nhuggingface_token: hf_secret\n", encoding="utf-8")

    lab_id, token = _real_read_contributor_config(config)

    assert (lab_id, token) == ("MyLab", "hf_secret"), "must still work, only warn"
    assert "git working tree" in caplog.text
    assert "oopsie-data init" in caplog.text
    assert "hf_secret" not in caplog.text, "never log the token itself"


def test_the_warning_does_not_depend_on_which_repo_it_is(tmp_path, monkeypatch, caplog):
    """Nothing privileges this toolkit's clone; a nested worktree counts just the same."""
    monkeypatch.setattr(contributor_config, "_warned_about_config_in_worktree", False)
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "deep" / "inside"
    nested.mkdir(parents=True)
    config = nested / "contributor_config.yaml"
    config.write_text("lab_id: MyLab\n", encoding="utf-8")

    _real_read_contributor_config(config)

    assert str(tmp_path) in caplog.text, "the warning names the worktree root it found"


def test_reading_from_the_user_dir_is_silent(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(contributor_config, "_warned_about_config_in_worktree", False)
    config = tmp_path / "contributor_config.yaml"
    config.write_text("lab_id: MyLab\n", encoding="utf-8")

    _real_read_contributor_config(config)

    assert caplog.text == ""
