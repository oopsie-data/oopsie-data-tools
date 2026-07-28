"""Tests for ``oopsie-data new-profile``.

The command replaces the old "copy configs/robot_profiles/template.yaml" instruction,
which only worked from a source checkout. Nothing is read out of the installed package,
so the template text lives in Python.
"""

from __future__ import annotations

import pytest

from oopsie_data_tools import cli
from oopsie_data_tools.utils import paths
from oopsie_data_tools.utils.robot_profile.robot_profile import load_robot_profile
from oopsie_data_tools.utils.robot_profile.template import PROFILE_TEMPLATE


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project directory with no profiles, as the working directory."""
    monkeypatch.delenv(paths.ENV_PROFILES_DIR, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_checked_in_template_matches_the_python_source():
    """configs/robot_profiles/template.yaml is a copy for browsing; it must not drift."""
    repo_config = paths.repo_config_dir()
    assert repo_config is not None, "this test runs from a checkout"
    checked_in = repo_config / paths.PROFILES_DIR_NAME / "template.yaml"

    assert checked_in.read_text(encoding="utf-8") == PROFILE_TEMPLATE, (
        "configs/robot_profiles/template.yaml has drifted from "
        "oopsie_data_tools/utils/robot_profile/template.py, which is the source of truth. "
        "Regenerate the checked-in copy from PROFILE_TEMPLATE."
    )


def test_writes_into_project_local_robot_profiles(project):
    assert cli.main(["new-profile"]) == 0

    written = project / paths.PROFILES_DIR_NAME / "robot_profile.yaml"
    assert written.is_file()
    assert written.read_text(encoding="utf-8") == PROFILE_TEMPLATE


def test_honours_name_and_dir(project):
    assert cli.main(["new-profile", "--name", "yam_pro", "--dir", str(project / "cfg")]) == 0

    assert (project / "cfg" / "yam_pro.yaml").is_file()
    assert not (project / paths.PROFILES_DIR_NAME).exists()


def test_refuses_to_clobber_without_force(project, caplog):
    assert cli.main(["new-profile"]) == 0
    written = project / paths.PROFILES_DIR_NAME / "robot_profile.yaml"
    written.write_text("policy_name: mine\n", encoding="utf-8")

    assert cli.main(["new-profile"]) == 1
    assert written.read_text(encoding="utf-8") == "policy_name: mine\n", "must not overwrite"
    assert "already exists" in caplog.text

    assert cli.main(["new-profile", "--force"]) == 0
    assert written.read_text(encoding="utf-8") == PROFILE_TEMPLATE


def test_emitted_profile_does_not_load_until_filled_in(project):
    """The skeleton must fail loudly rather than record placeholder metadata."""
    assert cli.main(["new-profile"]) == 0
    written = project / paths.PROFILES_DIR_NAME / "robot_profile.yaml"

    # Which check fires first is an ordering detail of robot_profile_from_raw; all that
    # matters here is that a blank skeleton is rejected rather than silently loaded.
    with pytest.raises(ValueError, match="Invalid action_space|robot state keys|missing keys"):
        load_robot_profile(written)


def test_written_profile_is_not_picked_up_from_site_packages(project):
    """The profile lookup must resolve to the project, never to the installed package."""
    assert cli.main(["new-profile"]) == 0

    resolved = paths.robot_profiles_dir()
    assert resolved == project / paths.PROFILES_DIR_NAME
    assert "site-packages" not in str(resolved)
