"""Install the bundled Claude Code skill into a user's Claude configuration.

Claude Code discovers skills by scanning the filesystem for ``<dir>/SKILL.md``, so
"installing" one is a directory copy — no Claude CLI, and no network access, is involved.
Two scopes exist:

* personal — ``~/.claude/skills/<name>/``, available in every project
* project  — ``<cwd>/.claude/skills/<name>/``, checked in and shared with collaborators

This is deliberately an explicit opt-in subcommand rather than anything that runs at
install time: contributors who do not use Claude Code never have files written into their
home directory.
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


def skill_destination(project: bool, root: Path | None = None) -> Path:
    """Where the skill would be installed for the given scope."""
    if root is not None:
        base = root
    elif project:
        base = Path.cwd() / ".claude"
    else:
        base = Path.home() / ".claude"
    return base / "skills" / SKILL_NAME


def install_skill(
    project: bool = False,
    force: bool = False,
    root: Path | None = None,
) -> int:
    source = bundled_skill_dir()
    if not (source / "SKILL.md").is_file():
        logger.error("The installed package does not contain a skill payload at %s.", source)
        return 1

    dest = skill_destination(project, root)
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
    logger.info(
        "Start a new Claude Code session to pick it up, then run /%s to invoke it directly.",
        SKILL_NAME,
    )
    if project:
        logger.info("Commit .claude/skills/ to share the skill with the rest of the project.")
    return 0
