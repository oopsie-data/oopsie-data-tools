"""Tests for ``oopsie-data install-skill``.

The command is a directory copy, so what is worth pinning down is the destination for
each scope, that an existing directory is never clobbered without --force, and that the
payload really is inside the package rather than only in a source checkout.
"""

from __future__ import annotations

from pathlib import Path

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
    linked = {
        "setup.md",
        "robot-profile.md",
        "format.md",
        "conversion.md",
        "troubleshooting.md",
    }

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


def test_default_install_lands_where_an_agent_actually_scans(home, tmp_path):
    """The default has to produce a working skill, not a directory nothing reads."""
    assert cli.main(["install-skill"]) == 0

    assert (tmp_path / "project" / ".claude" / "skills" / "oopsie-data" / "SKILL.md").is_file()
    assert not (tmp_path / "project" / "skills").exists()
    assert not (home / ".claude").exists()


@pytest.mark.parametrize(
    "agent,subdir",
    [
        ("claude", ".claude/skills"),
        ("cursor", ".cursor/skills"),
        ("codex", ".codex/skills"),
        ("agents", ".agents/skills"),
        ("none", "skills"),
    ],
)
def test_agent_selects_the_destination(home, tmp_path, agent, subdir):
    assert cli.main(["install-skill", "--agent", agent]) == 0

    assert (tmp_path / "project" / subdir / "oopsie-data" / "SKILL.md").is_file()


def test_install_copies_the_whole_payload_tree(home, tmp_path):
    """Subdirectories install too, so a page added to the payload cannot go missing."""
    assert cli.main(["install-skill"]) == 0

    source = claude_skill.bundled_skill_dir()
    dest = tmp_path / "project" / ".claude" / "skills" / "oopsie-data"
    expected = {p.relative_to(source) for p in source.rglob("*") if p.is_file()}
    # The version stamp is written at install time, so it is in the copy and not the source.
    installed = {p.relative_to(dest) for p in dest.rglob("*") if p.is_file()}

    assert expected == installed - {Path(claude_skill.VERSION_STAMP)}


def test_user_scope_installs_into_the_home_directory(home, tmp_path):
    assert cli.main(["install-skill", "--user"]) == 0

    assert (home / ".claude" / "skills" / "oopsie-data" / "SKILL.md").is_file()
    assert not (tmp_path / "project" / ".claude").exists()


def test_agent_none_explains_how_to_activate_the_skill(home, caplog):
    """./skills is not scanned by any agent, so the command must not imply it is live."""
    with caplog.at_level("INFO"):
        assert cli.main(["install-skill", "--agent", "none"]) == 0

    # A plain copy: one directory, no link to reason about, and trivially undone. A relative
    # symlink out of the agent's skills directory reads as broken even when it resolves.
    assert "cp -r" in caplog.text
    assert "ln -s" not in caplog.text
    assert "No agent scans this directory" in caplog.text


def test_activation_advice_covers_more_agents_than_claude(home, caplog):
    """The payload is a shared format; the advice must not read as Claude-only."""
    with caplog.at_level("INFO"):
        assert cli.main(["install-skill", "--agent", "none"]) == 0

    for _label, subdir in claude_skill.AGENT_SKILL_DIRS.values():
        if subdir == "skills":
            continue
        assert subdir in caplog.text

    assert ".agents/skills" in caplog.text, "the vendor-neutral location must be offered"
    assert "Cursor" in caplog.text


def test_refuses_to_overwrite_without_force(home, tmp_path):
    assert cli.main(["install-skill"]) == 0
    edited = tmp_path / "project" / ".claude" / "skills" / "oopsie-data" / "SKILL.md"
    edited.write_text("local edits", encoding="utf-8")

    assert cli.main(["install-skill"]) == 1
    assert edited.read_text(encoding="utf-8") == "local edits"


def test_force_replaces_the_existing_installation(home, tmp_path):
    assert cli.main(["install-skill"]) == 0
    installed = tmp_path / "project" / ".claude" / "skills" / "oopsie-data"
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


# ── --check ───────────────────────────────────────────────────────────────────


def test_check_reports_nothing_installed(home, caplog):
    with caplog.at_level("INFO"):
        assert cli.main(["install-skill", "--check"]) == 1

    assert "No installed copy" in caplog.text


def test_check_passes_on_a_fresh_install(home, caplog):
    assert cli.main(["install-skill"]) == 0

    with caplog.at_level("INFO"):
        assert cli.main(["install-skill", "--check"]) == 0

    assert "up to date" in caplog.text


def test_check_flags_a_copy_installed_from_another_version(home, tmp_path, caplog):
    """The whole point: an upgraded package must not leave a stale skill looking fine."""
    assert cli.main(["install-skill"]) == 0
    installed = tmp_path / "project" / ".claude" / "skills" / "oopsie-data"
    (installed / claude_skill.VERSION_STAMP).write_text("0.0.1\n", encoding="utf-8")

    with caplog.at_level("INFO"):
        assert cli.main(["install-skill", "--check"]) == 1

    assert "stale" in caplog.text
    assert "0.0.1" in caplog.text


def test_check_reports_an_unstamped_copy_without_claiming_it_is_old(home, tmp_path, caplog):
    assert cli.main(["install-skill"]) == 0
    installed = tmp_path / "project" / ".claude" / "skills" / "oopsie-data"
    (installed / claude_skill.VERSION_STAMP).unlink()

    with caplog.at_level("INFO"):
        assert cli.main(["install-skill", "--check"]) == 1

    assert "unstamped" in caplog.text


def test_check_finds_copies_installed_for_any_agent(home, tmp_path, caplog):
    assert cli.main(["install-skill", "--agent", "cursor"]) == 0

    with caplog.at_level("INFO"):
        assert cli.main(["install-skill", "--check"]) == 0

    assert ".cursor" in caplog.text
