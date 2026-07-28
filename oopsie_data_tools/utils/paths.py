"""Resolve where oopsie configs live, so the package works when pip-installed.

The two kinds of config are looked up separately, because they belong to different things.

**Credentials** (``contributor_config.yaml``) belong to the person, so they are stored once
per user and shared by every project:

1. ``$OOPSIE_CONFIG_DIR`` — explicit override
2. ``$XDG_CONFIG_HOME/oopsie-data`` (default ``~/.config/oopsie-data``)
3. the repo's ``configs/`` directory, when running from a source checkout

**Robot profiles** belong to the robot code that uses them, so they are looked up next to
that code and never in the user config directory:

1. ``$OOPSIE_ROBOT_PROFILES_DIR`` — explicit override
2. ``./robot_profiles`` or ``./configs/robot_profiles``, relative to the working directory
3. the repo's ``configs/robot_profiles``, when running from a source checkout

In each chain the first location that actually exists wins; when none do, the first
*writable* location is returned, which is where new files should be created. Note that
profiles are usually loaded by explicit path (``load_robot_profile(path)``), so this lookup
mainly backs the bundled example profiles.

``oopsie-data --config-dir <dir> <command>`` sets the credential override for one invocation.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_CONFIG_DIR = "OOPSIE_CONFIG_DIR"
ENV_PROFILES_DIR = "OOPSIE_ROBOT_PROFILES_DIR"

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_REPO_CONFIG_DIR = _PACKAGE_ROOT.parent / "configs"

PROFILES_DIR_NAME = "robot_profiles"


def _env_dir(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else None


def _unique(candidates: list[Path | None]) -> list[Path]:
    """Drop Nones and repeats, keeping order.

    Two entries can name the same directory — running from the checkout root makes
    ``./configs/robot_profiles`` and the repo's own profile dir identical — and a chain
    that lists it twice is confusing when printed by ``oopsie-data config``.
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


def repo_config_dir() -> Path | None:
    """The checkout's ``configs/`` directory, or None when not running from source."""
    return _REPO_CONFIG_DIR if _REPO_CONFIG_DIR.is_dir() else None


def config_search_dirs() -> list[Path]:
    """Candidate directories for the contributor config, highest precedence first."""
    return _unique([env_config_dir(), user_config_dir(), repo_config_dir()])


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
    if resolved == user_config_dir():
        return "user config dir"
    if resolved == repo_config_dir():
        return "repo checkout"
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
    repo_dir = repo_config_dir()
    return _unique(
        [
            env_profiles_dir(),
            *project_profiles_dirs(),
            repo_dir / PROFILES_DIR_NAME if repo_dir is not None else None,
        ]
    )


def write_profiles_dir() -> Path:
    """Where a new robot profile should be written: the override, else project-local."""
    return env_profiles_dir() or project_profiles_dirs()[0]


def robot_profiles_dir() -> Path:
    """The robot profile directory that will be read, or where to create one."""
    for candidate in profiles_search_dirs():
        if candidate.is_dir():
            return candidate
    return write_profiles_dir()
