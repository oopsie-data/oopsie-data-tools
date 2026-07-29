"""Read the contributor config (lab_id + HuggingFace token) with clear errors.

Shared by the episode recorder, the upload pipeline, and the repo-stats script so a
missing/blank config gives one actionable message instead of a cryptic crash.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from oopsie_data_tools.utils.paths import contributor_config_path, user_config_dir

logger = logging.getLogger(__name__)

_REGISTER_HINT = (
    "Register at https://forms.gle/9arwZHAvRjvbozoT7 to obtain your lab_id and "
    "HuggingFace token, then set them in the contributor config:\n"
    "    lab_id: <YOUR_LAB_ID>\n"
    "    huggingface_token: <YOUR_HF_TOKEN>\n"
    "Use the exact lab_id you were given (capitalization matters)."
)

# One warning per process; this is read once per recorder and once per upload.
_warned_about_config_in_worktree = False


def _enclosing_git_worktree(path: Path) -> Path | None:
    """The root of the git working tree containing ``path``, if there is one."""
    for parent in path.resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def _warn_if_inside_git_worktree(path: Path) -> None:
    """Nudge users whose token sits in a git working tree towards the user config dir.

    The file holds a HuggingFace token, and a token in a working tree is one ``git add -A``
    away from being published. Which repository it is does not matter — a clone of this
    toolkit and the user's own robot project carry the same risk — so this asks git rather
    than comparing against any particular directory.
    """
    global _warned_about_config_in_worktree
    if _warned_about_config_in_worktree:
        return
    worktree = _enclosing_git_worktree(path)
    if worktree is None:
        return
    _warned_about_config_in_worktree = True
    logger.warning(
        "Reading credentials from %s, inside the git working tree at %s.\n"
        "That holds your HuggingFace token where an accidental commit can pick it up, and "
        "it is lost if you re-clone.\n"
        "Move it with:  oopsie-data init   (writes to %s)",
        path,
        worktree,
        user_config_dir(),
    )


def read_contributor_config(config_path: Path | str | None = None) -> tuple[str, str]:
    """Return ``(lab_id, huggingface_token)`` from the contributor config.

    Args:
        config_path: Optional override for the config location.

    Returns:
        A ``(lab_id, huggingface_token)`` tuple; the token may be empty.

    Raises:
        RuntimeError: If the file is missing/unparseable, or ``lab_id`` is unset or
            still the placeholder — always with an actionable message.
    """
    path = Path(config_path) if config_path is not None else contributor_config_path()
    if not path.exists():
        raise RuntimeError(f"Contributor config not found at {path}.\n{_REGISTER_HINT}")

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise RuntimeError(f"Could not parse {path}: {e}\n{_REGISTER_HINT}") from e
    if not isinstance(config, dict):
        config = {}

    # ``config.get("lab_id", "")`` returns None when the key is present but blank
    # (``lab_id:``), which used to crash with ``None.strip()`` — normalize first.
    lab_id = str(config.get("lab_id") or "").strip()
    huggingface_token = str(config.get("huggingface_token") or "").strip()

    if not lab_id:
        raise RuntimeError(f"lab_id is not set in {path}.\n{_REGISTER_HINT}")
    if lab_id == "your_lab_id":
        raise RuntimeError(
            f"lab_id in {path} is still the placeholder 'your_lab_id'.\n{_REGISTER_HINT}"
        )

    _warn_if_inside_git_worktree(path)
    return lab_id, huggingface_token
