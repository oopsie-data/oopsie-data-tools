"""Read the contributor config (lab_id + HuggingFace token) with clear errors.

Shared by the episode recorder, the upload pipeline, and the repo-stats script so a
missing/blank config gives one actionable message instead of a cryptic crash.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from oopsie_data_tools.utils.paths import contributor_config_path, repo_config_dir, user_config_dir

logger = logging.getLogger(__name__)

_REGISTER_HINT = (
    "Register at https://forms.gle/9arwZHAvRjvbozoT7 to obtain your lab_id and "
    "HuggingFace token, then set them in the contributor config:\n"
    "    lab_id: <YOUR_LAB_ID>\n"
    "    huggingface_token: <YOUR_HF_TOKEN>\n"
    "Use the exact lab_id you were given (capitalization matters)."
)

# One warning per process; this is read once per recorder and once per upload.
_warned_about_checkout_config = False


def _warn_if_inside_checkout(path: Path) -> None:
    """Nudge users whose token still lives in the checkout towards the user config dir.

    ``configs/contributor_config.yaml`` used to be tracked by git, so a filled-in copy sat
    in the working tree of every contributor. It is untracked and ignored now, but existing
    clones still have one, and a token in a working tree is a token that can be committed.
    """
    global _warned_about_checkout_config
    if _warned_about_checkout_config:
        return
    repo_dir = repo_config_dir()
    if repo_dir is None or path.parent.resolve() != repo_dir.resolve():
        return
    _warned_about_checkout_config = True
    logger.warning(
        "Reading credentials from %s, inside the repository working tree.\n"
        "That file is no longer tracked by git, but it still holds your HuggingFace token "
        "where an accidental commit can pick it up, and it is lost if you re-clone.\n"
        "Move it with:  oopsie-data init   (writes to %s)",
        path,
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

    _warn_if_inside_checkout(path)
    return lab_id, huggingface_token
