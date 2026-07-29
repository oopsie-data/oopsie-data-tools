"""Resolve where oopsie configs live, so the package works when pip-installed.

The two kinds of config are looked up separately, because they belong to different things.

**Credentials** (``contributor_config.yaml``) usually belong to the person, so the per-user
directory is where ``init`` puts them by default and where they are found from any project:

1. ``$OOPSIE_CONFIG_DIR`` — explicit override
2. ``.`` or ``./configs``, relative to the working directory
3. ``$XDG_CONFIG_HOME/oopsie-data`` (default ``~/.config/oopsie-data``)

The project-local legs make a per-project identity possible — a second lab, a shared
machine, a CI checkout — without exporting anything. They come first so that a config sitting
in front of you wins over the personal one; ``oopsie-data show-config`` prints the whole chain
with the winner marked, because a token that changes with the working directory is otherwise
a confusing thing to debug. A local config also holds a token inside whatever repository it
sits in, which ``contributor_config`` warns about on read.

**Robot profiles** belong to the robot code that uses them, so they are looked up next to
that code and never in the user config directory:

1. ``$OOPSIE_ROBOT_PROFILES_DIR`` — explicit override
2. ``./robot_profiles`` or ``./configs/robot_profiles``, relative to the working directory

Neither chain knows anything about a source checkout. A clone is just another directory:
working inside one, its ``configs/`` is found by the ordinary cwd-relative legs, and nothing
else about it is privileged.

In each chain the first location that actually exists wins; when none do, the first
*writable* location is returned, which is where new files should be created. Note that
profiles are usually loaded by explicit path (``load_robot_profile(path)``), so this lookup
mainly backs a project's own profile directory.

``oopsie-data --config-dir <dir> <command>`` sets the credential override for one invocation.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_CONFIG_DIR = "OOPSIE_CONFIG_DIR"
ENV_PROFILES_DIR = "OOPSIE_ROBOT_PROFILES_DIR"

# No location below is derived from the package's own path. Config found that way is
# read-only under a wheel, invisible to the user, and — under an editable install — follows
# them into every unrelated directory. Each candidate is one the user picked, by exporting
# an environment variable or by choosing a working directory.

PROFILES_DIR_NAME = "robot_profiles"


def _env_dir(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else None


def _unique(candidates: list[Path | None]) -> list[Path]:
    """Drop Nones and repeats, keeping order.

    Two entries can name the same directory — ``$OOPSIE_ROBOT_PROFILES_DIR`` pointing at
    the project-local one, say — and a chain that lists it twice is confusing when printed
    by ``oopsie-data show-config``.
    """
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate is None:
            continue
        key = candidate.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


# ── credentials ───────────────────────────────────────────────────────────────


def env_config_dir() -> Path | None:
    """``$OOPSIE_CONFIG_DIR`` if set, else None."""
    return _env_dir(ENV_CONFIG_DIR)


def user_config_dir() -> Path:
    """Per-user config location, where ``init`` stores credentials."""
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "oopsie-data"


def project_config_dirs() -> list[Path]:
    """Credential directories relative to the working directory, in precedence order."""
    cwd = Path.cwd()
    return [cwd, cwd / "configs"]


def config_search_dirs() -> list[Path]:
    """Candidate directories for the contributor config, highest precedence first."""
    return _unique([env_config_dir(), *project_config_dirs(), user_config_dir()])


def write_config_dir() -> Path:
    """Where a new contributor config should be written."""
    return env_config_dir() or user_config_dir()


def contributor_config_path() -> Path:
    """The contributor config that will be read, or where to create one."""
    for candidate in config_search_dirs():
        target = candidate / "contributor_config.yaml"
        if target.is_file():
            return target
    return write_config_dir() / "contributor_config.yaml"


def config_dir() -> Path:
    """Directory the contributor config resolves to."""
    return contributor_config_path().parent


def config_dir_source() -> str:
    """Human-readable reason the resolved config dir was chosen (for messages)."""
    resolved = contributor_config_path().parent
    if resolved == env_config_dir():
        return f"${ENV_CONFIG_DIR}"
    if resolved in project_config_dirs():
        return "project directory"
    if resolved == user_config_dir():
        return "user config dir"
    return str(resolved)


# ── robot profiles ────────────────────────────────────────────────────────────


def env_profiles_dir() -> Path | None:
    """``$OOPSIE_ROBOT_PROFILES_DIR`` if set, else None."""
    return _env_dir(ENV_PROFILES_DIR)


def project_profiles_dirs() -> list[Path]:
    """Profile directories relative to the working directory, in precedence order."""
    cwd = Path.cwd()
    return [cwd / PROFILES_DIR_NAME, cwd / "configs" / PROFILES_DIR_NAME]


def profiles_search_dirs() -> list[Path]:
    """Candidate directories for robot profiles, highest precedence first."""
    return _unique([env_profiles_dir(), *project_profiles_dirs()])


def write_profiles_dir() -> Path:
    """Where a new robot profile should be written: the override, else project-local."""
    return env_profiles_dir() or project_profiles_dirs()[0]


def robot_profiles_dir() -> Path:
    """The robot profile directory that will be read, or where to create one."""
    for candidate in profiles_search_dirs():
        if candidate.is_dir():
            return candidate
    return write_profiles_dir()
