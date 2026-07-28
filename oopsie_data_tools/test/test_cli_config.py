"""Tests for ``oopsie-data show-config`` (``cli.cmd_show_config``).

The command is purely diagnostic: it prints the lookup chains, marks the entry that wins,
and shows the credentials in effect. It must never raise on a missing or half-filled
config — reporting that state is exactly what it is for.
"""

from __future__ import annotations

import pytest

from oopsie_data_tools import cli
from oopsie_data_tools.utils import paths

TOKEN = "hf_secrettokenvalue"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolate both chains and the working directory; returns the credential dir."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setenv(paths.ENV_CONFIG_DIR, str(config_dir))
    monkeypatch.delenv(paths.ENV_PROFILES_DIR, raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    return config_dir


def _write_config(config_dir, lab_id="MyLab", token=TOKEN):
    (config_dir / "contributor_config.yaml").write_text(
        f"lab_id: {lab_id}\nhuggingface_token: {token}\n"
    )


def test_reports_active_config_and_masks_the_token(env, capsys):
    _write_config(env)

    assert cli.main(["show-config"]) == 0

    out = capsys.readouterr().out
    assert f"Reading:    {env / 'contributor_config.yaml'}" in out
    assert "lab_id:     MyLab" in out
    assert "OopsieData-Submissions/MyLab" in out
    assert TOKEN not in out, "the token must not be printed in full by default"


def test_show_token_reveals_it(env, capsys):
    _write_config(env)

    assert cli.main(["show-config", "--show-token"]) == 0
    assert TOKEN in capsys.readouterr().out


def test_env_token_is_reported_as_the_override(env, capsys, monkeypatch):
    _write_config(env)
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_environment")

    assert cli.main(["show-config", "--show-token"]) == 0

    out = capsys.readouterr().out
    assert "hf_from_the_environment" in out
    assert "overrides the config" in out


def test_missing_config_says_where_init_would_write(env, capsys, monkeypatch):
    # Hide the checkout so the chain has no fallback, as for a pip install without a clone.
    monkeypatch.setattr(paths, "_REPO_CONFIG_DIR", env / "no_such_checkout")

    assert cli.main(["show-config"]) == 0

    out = capsys.readouterr().out
    assert "No config found" in out
    assert str(env / "contributor_config.yaml") in out
    assert "lab_id:     (not set)" in out


def test_placeholder_config_is_reported_not_raised(env, capsys):
    """read_contributor_config raises on this; the diagnostic command must not."""
    _write_config(env, lab_id="your_lab_id")

    assert cli.main(["show-config"]) == 0
    assert "your_lab_id" in capsys.readouterr().out


def test_project_local_profiles_are_listed(env, tmp_path, capsys):
    profiles = tmp_path / "robot_profiles"
    profiles.mkdir()
    (profiles / "my_robot.yaml").write_text("policy_name: p\n")
    _write_config(env)

    assert cli.main(["show-config"]) == 0

    out = capsys.readouterr().out
    assert f"Reading:    {profiles}" in out
    assert "profiles:   my_robot.yaml" in out
    assert "1 profile\n" in out, "singular when there is exactly one"


def test_chain_lists_each_directory_once(env, monkeypatch, tmp_path, capsys):
    """From a checkout root, ./configs/robot_profiles and the repo's dir are the same path."""
    checkout = tmp_path / "checkout"
    (checkout / "configs" / paths.PROFILES_DIR_NAME).mkdir(parents=True)
    monkeypatch.setattr(paths, "_REPO_CONFIG_DIR", checkout / "configs")
    monkeypatch.chdir(checkout)

    assert cli.main(["show-config"]) == 0

    listed = capsys.readouterr().out.count(str(checkout / "configs" / paths.PROFILES_DIR_NAME))
    assert listed == 2, f"expected one chain entry plus the 'Reading:' line, got {listed}"
