"""Read the contributor config (lab_id + HuggingFace token) with clear errors.

Shared by the episode recorder, the upload pipeline, and the repo-stats script so a
missing/blank config gives one actionable message instead of a cryptic crash.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "contributor_config.yaml"

_REGISTER_HINT = (
    "Register at https://forms.gle/9arwZHAvRjvbozoT7 to obtain your lab_id and "
    "HuggingFace token, then set them in configs/contributor_config.yaml:\n"
    "    lab_id: <YOUR_LAB_ID>\n"
    "    huggingface_token: <YOUR_HF_TOKEN>\n"
    "Use the exact lab_id you were given (capitalization matters)."
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
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise RuntimeError(f"Contributor config not found at {path}.\n{_REGISTER_HINT}")

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise RuntimeError(f"Could not parse {path}: {e}\n{_REGISTER_HINT}")
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
    return lab_id, huggingface_token
