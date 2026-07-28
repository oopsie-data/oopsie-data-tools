"""Tests for config path resolution (``oopsie_data_tools/utils/paths.py``).

Credentials and robot profiles are resolved through separate chains: credentials are
per-user, profiles live next to the robot code and never in the user config directory.
"""

from __future__ import annotations

import pytest

from oopsie_data_tools.utils import paths


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point the user config dir at a scratch directory and clear both env overrides."""
    monkeypatch.delenv(paths.ENV_CONFIG_DIR, raising=False)
    monkeypatch.delenv(paths.ENV_PROFILES_DIR, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg" / "oopsie-data"


# ── credentials ───────────────────────────────────────────────────────────────


def test_user_config_dir_follows_xdg(isolated_home):
    assert paths.user_config_dir() == isolated_home


def test_credentials_fall_back_to_repo_when_user_dir_is_empty(isolated_home):
    repo_dir = paths.repo_config_dir()
    assert repo_dir is not None, "these tests run from a checkout"
    assert paths.contributor_config_path() == repo_dir / "contributor_config.yaml"


def test_user_dir_wins_over_repo_for_credentials(isolated_home):
    isolated_home.mkdir(parents=True)
    (isolated_home / "contributor_config.yaml").write_text("lab_id: UserLab\n")

    assert paths.contributor_config_path() == isolated_home / "contributor_config.yaml"
    assert paths.config_dir_source() == "user config dir"


def test_env_var_wins_when_it_holds_the_config(isolated_home, tmp_path, monkeypatch):
    env_dir = tmp_path / "explicit"
    env_dir.mkdir()
    (env_dir / "contributor_config.yaml").write_text("lab_id: EnvLab\n")
    isolated_home.mkdir(parents=True)
    (isolated_home / "contributor_config.yaml").write_text("lab_id: UserLab\n")
    monkeypatch.setenv(paths.ENV_CONFIG_DIR, str(env_dir))

    assert paths.contributor_config_path() == env_dir / "contributor_config.yaml"
    assert paths.write_config_dir() == env_dir
    assert paths.config_dir_source() == f"${paths.ENV_CONFIG_DIR}"


def test_env_var_is_the_write_target_but_reads_still_fall_back(
    isolated_home, tmp_path, monkeypatch
):
    """An override that does not hold a config yet still reads the one that exists."""
    env_dir = tmp_path / "explicit"
    env_dir.mkdir()
    isolated_home.mkdir(parents=True)
    (isolated_home / "contributor_config.yaml").write_text("lab_id: UserLab\n")
    monkeypatch.setenv(paths.ENV_CONFIG_DIR, str(env_dir))

    assert paths.write_config_dir() == env_dir
    assert paths.contributor_config_path() == isolated_home / "contributor_config.yaml"


def test_write_config_dir_defaults_to_user_dir(isolated_home):
    assert paths.write_config_dir() == isolated_home


# ── robot profiles ────────────────────────────────────────────────────────────


def test_profiles_never_resolve_to_the_user_config_dir(isolated_home, tmp_path, monkeypatch):
    """Credentials live in ~/.config; profiles must not be expected there."""
    profiles_in_user_dir = isolated_home / paths.PROFILES_DIR_NAME
    profiles_in_user_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    assert paths.robot_profiles_dir() != profiles_in_user_dir
    assert profiles_in_user_dir not in paths.profiles_search_dirs()


def test_project_local_profiles_win(isolated_home, tmp_path, monkeypatch):
    project = tmp_path / "my_robot_code"
    (project / paths.PROFILES_DIR_NAME).mkdir(parents=True)
    monkeypatch.chdir(project)

    assert paths.robot_profiles_dir() == project / paths.PROFILES_DIR_NAME


def test_project_configs_subdir_is_also_searched(isolated_home, tmp_path, monkeypatch):
    project = tmp_path / "my_robot_code"
    (project / "configs" / paths.PROFILES_DIR_NAME).mkdir(parents=True)
    monkeypatch.chdir(project)

    assert paths.robot_profiles_dir() == project / "configs" / paths.PROFILES_DIR_NAME


def test_profiles_fall_back_to_the_checkout(isolated_home, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # nothing project-local here

    assert paths.robot_profiles_dir() == paths.repo_config_dir() / paths.PROFILES_DIR_NAME


def test_profiles_env_override_wins(isolated_home, tmp_path, monkeypatch):
    override = tmp_path / "shared_profiles"
    override.mkdir()
    project = tmp_path / "my_robot_code"
    (project / paths.PROFILES_DIR_NAME).mkdir(parents=True)
    monkeypatch.chdir(project)
    monkeypatch.setenv(paths.ENV_PROFILES_DIR, str(override))

    assert paths.robot_profiles_dir() == override
    assert paths.write_profiles_dir() == override


def test_write_profiles_dir_is_project_local_by_default(isolated_home, tmp_path, monkeypatch):
    project = tmp_path / "my_robot_code"
    project.mkdir()
    monkeypatch.chdir(project)

    assert paths.write_profiles_dir() == project / paths.PROFILES_DIR_NAME


def test_config_dir_override_does_not_move_profiles(isolated_home, tmp_path, monkeypatch):
    """$OOPSIE_CONFIG_DIR is about credentials only."""
    env_dir = tmp_path / "explicit"
    (env_dir / paths.PROFILES_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv(paths.ENV_CONFIG_DIR, str(env_dir))
    monkeypatch.chdir(tmp_path)

    assert paths.robot_profiles_dir() == paths.repo_config_dir() / paths.PROFILES_DIR_NAME
