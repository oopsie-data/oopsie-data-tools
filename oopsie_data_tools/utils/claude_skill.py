"""Install the bundled Claude Code skill into a user's Claude configuration.

Claude Code discovers skills by scanning the filesystem for ``<dir>/SKILL.md``, so
"installing" one is a directory copy — no Claude CLI, and no network access, is involved.
Two scopes exist:

* project (default) — ``<cwd>/.claude/skills/<name>/``, checked in and shared with
  collaborators, and versioned alongside the data-collection code it describes
* personal          — ``~/.claude/skills/<name>/``, available in every project

Project scope is the default because the skill describes one project's workflow, and
because writing inside the working directory is the change a contributor can most easily
see and undo. This is also deliberately an explicit opt-in subcommand rather than anything
that runs at install time: contributors who do not use Claude Code never have files
written for them at all.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_NAME = "oopsie-data"


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
        logger.info(
            "Claude Code picks it up from ~/.claude/skills/ in every project; start a new "
            "session, then run /%s to invoke it directly.",
            SKILL_NAME,
        )
    else:
        # ./skills/ is a plain project directory, so it is visible and committable, but it is
        # not one of the two locations Claude Code scans. Say so rather than let the skill sit
        # there looking installed.
        logger.info(
            "This is a plain project directory, so Claude Code does not scan it yet. Link it "
            "into the project's skills directory to activate it:\n\n"
            "    mkdir -p .claude/skills && ln -s ../../skills/%s .claude/skills/%s\n\n"
            "Or install it for yourself in every project instead: "
            "oopsie-data install-skill --user",
            SKILL_NAME,
            SKILL_NAME,
        )
    return 0
