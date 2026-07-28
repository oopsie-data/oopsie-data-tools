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


def test_wizard_defaults_away_from_the_checkout(isolated, monkeypatch):
    """Pressing Enter must not put a token inside a git working tree."""
    answers = iter([""])  # accept the default
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    target = init_wizard.choose_target_dir()

    assert target == paths.user_config_dir()
    assert target != paths.repo_config_dir()


def test_wizard_still_allows_the_checkout_when_asked(isolated, monkeypatch):
    answers = iter(["2"])  # the second option is the checkout
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    assert init_wizard.choose_target_dir() == paths.repo_config_dir()


def test_wizard_uses_user_dir_when_not_interactive(isolated, monkeypatch):
    monkeypatch.setattr(init_wizard.sys.stdin, "isatty", lambda: False)

    assert init_wizard.choose_target_dir() == paths.user_config_dir()


def test_reading_from_the_checkout_warns(tmp_path, monkeypatch, caplog):
    """Existing clones still have a filled-in copy; point them at the user config dir."""
    monkeypatch.setattr(contributor_config, "_warned_about_checkout_config", False)
    repo_configs = tmp_path / "configs"
    repo_configs.mkdir()
    config = repo_configs / "contributor_config.yaml"
    config.write_text("lab_id: MyLab\nhuggingface_token: hf_secret\n", encoding="utf-8")
    monkeypatch.setattr(paths, "_REPO_CONFIG_DIR", repo_configs)

    lab_id, token = _real_read_contributor_config(config)

    assert (lab_id, token) == ("MyLab", "hf_secret"), "must still work, only warn"
    assert "inside the repository working tree" in caplog.text
    assert "oopsie-data init" in caplog.text
    assert "hf_secret" not in caplog.text, "never log the token itself"


def test_reading_from_the_user_dir_is_silent(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(contributor_config, "_warned_about_checkout_config", False)
    config = tmp_path / "contributor_config.yaml"
    config.write_text("lab_id: MyLab\n", encoding="utf-8")

    _real_read_contributor_config(config)

    assert caplog.text == ""
