"""Install the bundled agent skill into a user's project or home directory.

Agents discover skills by scanning the filesystem for ``<dir>/SKILL.md``, so "installing"
one is a directory copy — no agent CLI, and no network access, is involved. Two scopes
exist:

* project (default) — ``<cwd>/skills/<name>/``, checked in and shared with collaborators,
  and versioned alongside the data-collection code it describes
* personal          — ``~/.claude/skills/<name>/``, available in every project

Project scope is the default because the skill describes one project's workflow, and
because writing inside the working directory is the change a contributor can most easily
see and undo. It is a plain ``skills/`` directory rather than any one agent's config
location on purpose: the payload is just markdown, and Claude Code is not the only agent
that can read it. Whichever agent a contributor uses, the last step is theirs — copy or
move the directory to wherever that agent scans. This is also deliberately an explicit
opt-in subcommand rather than anything that runs at install time: contributors who do not
use an agent never have files written for them at all.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_NAME = "oopsie-data"

# Where the common agents scan, for the "now move it into place" hint. SKILL.md is a shared
# format — Cursor reads the same payload, and also loads Claude's and Codex's directories —
# so this is only ever about the destination, never about the files themselves. Kept as data
# rather than prose so adding an agent is one line and cannot drift from what is printed.
# ``.agents/skills`` is the vendor-neutral location, hence first.
AGENT_SKILL_DIRS: list[tuple[str, str]] = [
    ("any agent following the convention", ".agents/skills/"),
    ("Claude Code", ".claude/skills/"),
    ("Cursor", ".cursor/skills/"),
    ("Codex", ".codex/skills/"),
]


def bundled_skill_dir() -> Path:
    """Path to the skill payload shipped inside the installed package."""
    if sys.version_info >= (3, 9):
        from importlib.resources import files

        # The payload is plain files in a normal wheel, so this is always a real path.
        return Path(str(files("oopsie_data_tools") / "skill"))
    return Path(__file__).resolve().parent.parent / "skill"


def skill_destination(user: bool = False, root: Path | None = None) -> Path:
    """Where the skill would be installed for the given scope."""
    if root is not None:
        base = root
    elif user:
        base = Path.home() / ".claude" / "skills"
    else:
        base = Path.cwd() / "skills"
    return base / SKILL_NAME


def install_skill(
    user: bool = False,
    force: bool = False,
    root: Path | None = None,
) -> int:
    source = bundled_skill_dir()
    if not (source / "SKILL.md").is_file():
        logger.error("The installed package does not contain a skill payload at %s.", source)
        return 1

    dest = skill_destination(user, root)
    if dest.exists():
        if not force:
            logger.error(
                "%s already exists. Pass --force to overwrite it.\n"
                "Anything you added inside that directory would be lost, so it is not "
                "overwritten by default.",
                dest,
            )
            return 1
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest)

    logger.info("Installed the '%s' skill to %s", SKILL_NAME, dest)
    if user:
        # ~/.claude/skills is the one personal location more than one agent reads: Claude Code
        # owns it, and Cursor loads it too, so it covers the most people without asking them
        # which agent they use. Anyone else copies it on to their own directory from here.
        logger.info(
            "Claude Code picks it up from ~/.claude/skills/ in every project, and Cursor "
            "reads that directory too; start a new session, then run /%s to invoke it "
            "directly. For another agent, copy it on to ~/.agents/skills/, ~/.cursor/skills/ "
            "or ~/.codex/skills/.",
            SKILL_NAME,
        )
    else:
        # ./skills/ is a plain project directory: visible, committable, and not tied to any
        # one agent. No agent scans it, though, so say that rather than let the skill sit
        # there looking installed. The last step stays manual because only the contributor
        # knows which agent they use, and copying is something they can inspect and undo.
        agents = "\n".join(f"      {where:<18} {label}" for label, where in AGENT_SKILL_DIRS)
        # A command the user can paste as-is. Project scope writes under the working
        # directory, so the relative form is both shorter and correct; --root can point
        # elsewhere, and then the absolute path is the only one that works.
        try:
            copy_from = Path(dest).relative_to(Path.cwd())
        except ValueError:
            copy_from = dest
        logger.info(
            "This is a plain project directory, so no agent scans it yet. Copy it to "
            "whichever directory your agent reads, then start a new session there:\n\n"
            "    mkdir -p .agents/skills && cp -r %s .agents/skills/\n\n"
            "  Substitute the directory your agent uses:\n\n"
            "%s\n\n"
            "  The payload is the same in every case — SKILL.md is a shared format, and "
            "Cursor also reads Claude's and Codex's directories. Prefix any of these with "
            "~/ to install it for yourself in every project instead.",
            copy_from,
            agents,
        )
    return 0
