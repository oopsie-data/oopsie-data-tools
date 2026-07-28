"""Tests for ``oopsie-data install-skill``.

The command is a directory copy, so what is worth pinning down is the destination for
each scope, that an existing directory is never clobbered without --force, and that the
payload really is inside the package rather than only in a source checkout.
"""

from __future__ import annotations

import pytest

from oopsie_data_tools import cli
from oopsie_data_tools.utils import claude_skill


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty home directory, with an empty project as the working directory."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(claude_skill.Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.chdir(project)
    return fake_home


def test_payload_ships_inside_the_package():
    """The skill must resolve through the installed package, not the repo layout."""
    assert (claude_skill.bundled_skill_dir() / "SKILL.md").is_file()


def test_payload_ships_the_reference_files_skill_md_points_at():
    """SKILL.md is a router; the pages it defers to have to be there to be read."""
    reference = claude_skill.bundled_skill_dir() / "reference"
    linked = {"setup.md", "robot-profile.md", "format.md", "troubleshooting.md"}

    assert {p.name for p in reference.glob("*.md")} == linked

    skill_md = (claude_skill.bundled_skill_dir() / "SKILL.md").read_text(encoding="utf-8")
    for name in linked:
        assert f"reference/{name}" in skill_md


def test_skill_declares_frontmatter_claude_can_discover():
    text = (claude_skill.bundled_skill_dir() / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    frontmatter = text.split("---\n")[1]
    assert f"name: {claude_skill.SKILL_NAME}\n" in frontmatter
    assert "description:" in frontmatter


def test_installs_into_a_visible_project_directory_by_default(home, tmp_path):
    """Default scope is the project, and deliberately not a hidden .claude directory."""
    assert cli.main(["install-skill"]) == 0

    assert (tmp_path / "project" / "skills" / "oopsie-data" / "SKILL.md").is_file()
    assert not (tmp_path / "project" / ".claude").exists()
    assert not (home / ".claude").exists()


def test_install_copies_the_whole_payload_tree(home, tmp_path):
    """Subdirectories install too, so a page added to the payload cannot go missing."""
    assert cli.main(["install-skill"]) == 0

    source = claude_skill.bundled_skill_dir()
    dest = tmp_path / "project" / "skills" / "oopsie-data"
    expected = {p.relative_to(source) for p in source.rglob("*") if p.is_file()}

    assert expected == {p.relative_to(dest) for p in dest.rglob("*") if p.is_file()}


def test_user_scope_installs_where_claude_code_scans(home, tmp_path):
    assert cli.main(["install-skill", "--user"]) == 0

    assert (home / ".claude" / "skills" / "oopsie-data" / "SKILL.md").is_file()
    assert not (tmp_path / "project" / "skills").exists()


def test_default_install_explains_how_to_activate_the_skill(home, caplog):
    """./skills is not scanned by Claude Code, so the command must not imply it is live."""
    with caplog.at_level("INFO"):
        assert cli.main(["install-skill"]) == 0

    assert ".claude/skills" in caplog.text


def test_refuses_to_overwrite_without_force(home, tmp_path):
    assert cli.main(["install-skill"]) == 0
    edited = tmp_path / "project" / "skills" / "oopsie-data" / "SKILL.md"
    edited.write_text("local edits", encoding="utf-8")

    assert cli.main(["install-skill"]) == 1
    assert edited.read_text(encoding="utf-8") == "local edits"


def test_force_replaces_the_existing_installation(home, tmp_path):
    assert cli.main(["install-skill"]) == 0
    installed = tmp_path / "project" / "skills" / "oopsie-data"
    (installed / "SKILL.md").write_text("local edits", encoding="utf-8")
    stale = installed / "stale.md"
    stale.write_text("removed by --force", encoding="utf-8")

    assert cli.main(["install-skill", "--force"]) == 0

    assert "local edits" not in (installed / "SKILL.md").read_text(encoding="utf-8")
    assert not stale.exists()


def test_nothing_is_written_when_the_command_is_not_run(home, tmp_path):
    assert cli.main(["show-config"]) == 0
    assert not (home / ".claude").exists()
    assert not (tmp_path / "project" / "skills").exists()
