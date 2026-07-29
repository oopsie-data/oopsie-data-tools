"""Install the bundled agent skill into a user's project or home directory.

Agents discover skills by scanning the filesystem for ``<dir>/SKILL.md``, so "installing"
one is a directory copy — no agent CLI, and no network access, is involved.

``--agent`` picks that directory and defaults to Claude Code's, because a skill that lands
somewhere nothing scans is not installed in any useful sense. Claude Code uses
``.claude/skills/``; Codex and Cursor share ``.agents/skills/``; ``--agent none`` writes
to plain ``skills/``. ``--user`` swaps the working directory for the home directory,
making the skill available in every project.

Installing is an explicit opt-in subcommand rather than anything that runs at install time:
contributors who do not use an agent never have files written for them at all.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_NAME = "oopsie-data"

#: Records the package version a copy was installed from, so ``--check`` can spot a stale
#: one. A dotfile so it does not read as part of the skill's content, and inside the
#: installed directory so it survives being copied onward.
VERSION_STAMP = ".skill-version"

# Where each agent scans, keyed by the --agent value. Kept as data rather than prose so
# adding an agent is one line and cannot drift from what is printed.
AGENT_SKILL_DIRS: dict[str, str] = {
    "claude": ".claude/skills",
    "codex": ".agents/skills",
    "cursor": ".agents/skills",
    "none": "skills",
}

DEFAULT_AGENT = "claude"


def bundled_skill_dir() -> Path:
    """Path to the skill payload shipped inside the installed package."""
    if sys.version_info >= (3, 9):
        from importlib.resources import files

        # The payload is plain files in a normal wheel, so this is always a real path.
        return Path(str(files("oopsie_data_tools") / "skill"))
    return Path(__file__).resolve().parent.parent / "skill"


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("oopsie-data-tools")
    except Exception:
        return "unknown"


def skill_destination(
    user: bool = False, agent: str = DEFAULT_AGENT, root: Path | None = None
) -> Path:
    """Where the skill would be installed for the given scope and agent."""
    if root is not None:
        return root / SKILL_NAME
    base = Path.home() if user else Path.cwd()
    return base / AGENT_SKILL_DIRS[agent] / SKILL_NAME


def installed_version(dest: Path) -> str | None:
    """The package version a copy at *dest* was installed from, if it says."""
    stamp = dest / VERSION_STAMP
    if not stamp.is_file():
        return None
    return stamp.read_text(encoding="utf-8").strip() or None


def find_installations() -> list[Path]:
    """Every installed copy of the skill under the working directory or the home directory."""
    found = []
    subdirs = dict.fromkeys(AGENT_SKILL_DIRS.values())
    for base in (Path.cwd(), Path.home()):
        for subdir in subdirs:
            candidate = base / subdir / SKILL_NAME
            if (candidate / "SKILL.md").is_file() and candidate not in found:
                found.append(candidate)
    return found


def check_installations() -> int:
    """Report every installed copy and whether it matches the running package."""
    current = _package_version()
    installs = find_installations()
    if not installs:
        logger.info(
            "No installed copy of the '%s' skill found under %s or %s.\n"
            "Run 'oopsie-data install-skill' to install one.",
            SKILL_NAME,
            Path.cwd(),
            Path.home(),
        )
        return 1

    stale = 0
    logger.info("Installed package version: %s\n", current)
    for dest in installs:
        found = installed_version(dest)
        if found == current:
            logger.info("  up to date   %s", dest)
        elif found is None:
            # Copies predating the stamp, and hand-made copies, land here. The version is
            # genuinely unknown rather than known-old, so say that instead of guessing.
            stale += 1
            logger.info("  unstamped    %s   (installed before versions were recorded)", dest)
        else:
            stale += 1
            logger.info("  stale        %s   (from %s)", dest, found)

    if stale:
        logger.info(
            "\nRe-install the copies above with 'oopsie-data install-skill --force' "
            "(add --agent/--user to match where each one lives)."
        )
    return 1 if stale else 0


def install_skill(
    user: bool = False,
    force: bool = False,
    agent: str = DEFAULT_AGENT,
    root: Path | None = None,
) -> int:
    source = bundled_skill_dir()
    if not (source / "SKILL.md").is_file():
        logger.error("The installed package does not contain a skill payload at %s.", source)
        return 1

    dest = skill_destination(user, agent, root)
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
    (dest / VERSION_STAMP).write_text(_package_version() + "\n", encoding="utf-8")

    logger.info("Installed the '%s' skill to %s", SKILL_NAME, dest)

    if agent == "none":
        # A plain directory: visible, committable, and not tied to any one agent. No agent
        # scans it, though, so say that rather than let the skill sit there looking
        # installed. Copying is something the contributor can inspect and undo.
        agents = "\n".join(
            f"      {subdir + '/':<18} {label}"
            for key, (label, subdir) in AGENT_SKILL_DIRS.items()
            if key != "none"
        )
        try:
            copy_from = dest.relative_to(Path.cwd())
        except ValueError:
            copy_from = dest
        logger.info(
            "No agent scans this directory. Copy it to whichever directory your agent "
            "reads, then start a new session there:\n\n"
            "    mkdir -p .claude/skills && cp -r %s .claude/skills/\n\n"
            "  Substitute the directory your agent uses:\n\n"
            "%s\n\n"
            "  The payload is the same in every case — SKILL.md is a shared format, and "
            "Cursor also reads Claude's and Codex's directories. Prefix any of these with "
            "~/ to install it for yourself in every project instead.",
            copy_from,
            agents,
        )
        return 0

    label = AGENT_SKILL_DIRS[agent][0]
    scope = "in every project" if user else "in this project"
    logger.info(
        "%s picks it up from there %s. Start a new session, then run /%s to invoke it "
        "directly.\n"
        "For another agent, re-run with --agent {%s}; the payload is identical in every "
        "case, and Cursor reads Claude's and Codex's directories too.",
        label,
        scope,
        SKILL_NAME,
        ",".join(key for key in AGENT_SKILL_DIRS if key != agent),
    )
    return 0
