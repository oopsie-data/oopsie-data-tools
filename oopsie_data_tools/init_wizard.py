"""Interactive setup wizard behind ``oopsie-data init``.

Writes ``contributor_config.yaml`` — the lab id and HuggingFace token every contributor
needs before uploading — into the config directory chosen in :func:`choose_target_dir`,
which follows the lookup order documented in :mod:`oopsie_data_tools.utils.paths`.

Robot profiles are not created here: they are hand-written YAML, documented in
``configs/robot_profiles/template.yaml`` and on the docs site.

Every question can be answered ahead of time with a flag, so a fully-flagged invocation
runs unattended (see ``oopsie-data init --help``).
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from oopsie_data_tools.utils import paths
from oopsie_data_tools.utils.contributor_config import read_contributor_config

logger = logging.getLogger(__name__)

CONTRIBUTOR_CONFIG_NAME = "contributor_config.yaml"
PLACEHOLDER_LAB_ID = "your_lab_id"
REGISTRATION_URL = "https://forms.gle/9arwZHAvRjvbozoT7"


class WizardAbort(Exception):
    """Raised when the wizard cannot continue (no TTY, or the user gave up on a question)."""


# ── prompt helpers ────────────────────────────────────────────────────────────
#
# Same contract as cli._prompt_annotator_name: never block on a non-interactive stdin,
# re-ask a bounded number of times, strip every answer.

_MAX_ATTEMPTS = 3


def _require_tty(what: str) -> None:
    if not sys.stdin.isatty():
        raise WizardAbort(
            f"Cannot ask for {what}: stdin is not a terminal. "
            "Pass the value as a command line flag to run non-interactively."
        )


def _with_default(prompt: str, default: str | None) -> str:
    return f"{prompt} [{default}]: " if default else f"{prompt}: "


def ask(
    prompt: str,
    default: str | None = None,
    validate: Callable[[str], str | None] | None = None,
) -> str:
    """Ask for a string. ``validate`` returns an error message, or None if the value is fine."""
    _require_tty(prompt)
    for _ in range(_MAX_ATTEMPTS):
        answer = input(_with_default(prompt, default)).strip()
        if not answer and default is not None:
            answer = default
        if not answer:
            print("  Please enter a value.")
            continue
        error = validate(answer) if validate else None
        if error:
            print(f"  {error}")
            continue
        return answer
    raise WizardAbort(f"No valid answer for: {prompt}")


def ask_optional(prompt: str, default: str | None = None) -> str:
    """Ask for a string that may be left empty."""
    _require_tty(prompt)
    answer = input(_with_default(prompt, default)).strip()
    if not answer and default is not None:
        return default
    return answer


def ask_bool(prompt: str, default: bool = False) -> bool:
    _require_tty(prompt)
    hint = "[Y/n]" if default else "[y/N]"
    for _ in range(_MAX_ATTEMPTS):
        answer = input(f"{prompt} {hint}: ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please answer y or n.")
    raise WizardAbort(f"No valid answer for: {prompt}")


def ask_choice(prompt: str, options: Sequence[str], default: str | None = None) -> str:
    _require_tty(prompt)
    print(f"\n{prompt}")
    for index, option in enumerate(options, start=1):
        marker = "  <- default" if option == default else ""
        print(f"  {index}) {option}{marker}")

    def _validate(value: str) -> str | None:
        if value in options:
            return None
        if value.isdigit() and 1 <= int(value) <= len(options):
            return None
        return f"Please enter a number between 1 and {len(options)}."

    answer = ask("Choice", default, _validate)
    return options[int(answer) - 1] if answer.isdigit() else answer


# ── config directory ──────────────────────────────────────────────────────────


def choose_target_dir() -> Path:
    """Pick the directory to write the config into.

    ``$OOPSIE_CONFIG_DIR`` (set directly or via ``--config-dir``) wins outright. Otherwise
    the per-user config directory is the default everywhere; inside a checkout we still
    offer the checkout's ``configs/``, but never pick it for you.

    The default deliberately points away from the checkout. The file holds a HuggingFace
    token, and a token inside a git working tree is one ``git add -A`` away from being
    published — which is exactly how it used to be committed.
    """
    env_dir = paths.env_config_dir()
    if env_dir is not None:
        logger.info("Using config directory from $%s: %s", paths.ENV_CONFIG_DIR, env_dir)
        return env_dir

    repo_dir = paths.repo_config_dir()
    user_dir = paths.user_config_dir()
    if repo_dir is None or not sys.stdin.isatty():
        logger.info("Using config directory: %s", user_dir)
        return user_dir

    user_option = f"{user_dir}  (your user config directory)"
    repo_option = f"{repo_dir}  (this checkout — holds a token inside a git working tree)"
    choice = ask_choice(
        "Where should your config be saved? Both are found automatically.",
        [user_option, repo_option],
        default=user_option,
    )
    return repo_dir if choice.startswith(str(repo_dir)) else user_dir


def _shell_rc_path() -> Path:
    shell = os.path.basename(os.environ.get("SHELL", "")).strip()
    name = ".zshrc" if shell == "zsh" else ".bashrc"
    return Path.home() / name


def advise_persisting_config_dir(target_dir: Path, from_flag: bool = False) -> None:
    """Tell the user how to make ``target_dir`` stick, if it would not be found next run.

    ``from_flag`` marks a location that came from ``--config-dir``: the CLI exports that into
    the environment for the current process only, so it must not count as persisted.
    """
    if target_dir in (paths.user_config_dir(), paths.repo_config_dir()):
        return  # found by the normal lookup order
    if paths.env_config_dir() == target_dir and not from_flag:
        return  # already exported in the user's environment

    export_line = f'export {paths.ENV_CONFIG_DIR}="{target_dir}"'
    rc_path = _shell_rc_path()
    logger.warning(
        "\nOne more step: %s is not a location oopsie-data looks in by default, so a new "
        "terminal will not find it.\nAdd this line to %s to make it stick:\n\n    %s\n",
        target_dir, rc_path, export_line,
    )
    if sys.stdin.isatty() and ask_bool(f"Append that line to {rc_path} now?", default=False):
        with rc_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n# Added by oopsie-data init\n{export_line}\n")
        logger.info("Done. Run 'source %s' or open a new terminal.", rc_path)


# ── contributor config ────────────────────────────────────────────────────────


def _read_existing_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _mask(token: str) -> str:
    return f"{token[:5]}…{token[-4:]}" if len(token) > 12 else "…"


def _validate_lab_id(value: str) -> str | None:
    if value == PLACEHOLDER_LAB_ID:
        return f"'{PLACEHOLDER_LAB_ID}' is the placeholder value — use the lab id you were given."
    return None


def _verify_token(token: str) -> None:
    """Confirm the token works, without ``login()`` writing it to the HF cache."""
    try:
        from huggingface_hub import whoami

        info = whoami(token=token)
        logger.info("  Token verified — logged in as %s", info["name"])
    except Exception as e:
        logger.warning(
            "  Could not verify the token (%s). Saving it anyway; "
            "'oopsie-data upload' will tell you if it is wrong.", e
        )


def step_credentials(
    target_dir: Path,
    lab_id: str | None = None,
    hf_token: str | None = None,
    verify_token: bool = True,
    force: bool = False,
) -> Path | None:
    """Write ``contributor_config.yaml``. Returns the path, or None if nothing was written."""
    path = target_dir / CONTRIBUTOR_CONFIG_NAME
    existing = _read_existing_yaml(path)
    existing_lab_id = str(existing.get("lab_id") or "").strip()
    existing_token = str(existing.get("huggingface_token") or "").strip()

    logger.info("\n── Contributor config ─────────────────────────────────")
    logger.info("Writing to %s", path)
    if path.is_file() and not force:
        if existing_lab_id:
            logger.info("This file already has lab_id '%s'.", existing_lab_id)
        if not sys.stdin.isatty():
            logger.error(
                "%s already exists. Re-run with --force to overwrite it non-interactively.", path
            )
            return None
        if not ask_bool("Update it?", default=True):
            logger.info("Keeping the existing contributor config.")
            return path

    if lab_id is None:
        logger.info(
            "Your lab id and HuggingFace token come from the registration form: %s", REGISTRATION_URL
        )
        logger.info("Use the exact lab id you were given — capitalization matters.")
        lab_id = ask("Lab id", existing_lab_id or None, _validate_lab_id)
    else:
        error = _validate_lab_id(lab_id)
        if error:
            raise WizardAbort(error)

    if hf_token is None:
        hf_token = ask_optional(
            "HuggingFace token (leave blank to set $HF_TOKEN instead)",
            _mask(existing_token) if existing_token else None,
        )
        if existing_token and hf_token == _mask(existing_token):
            hf_token = existing_token  # user accepted the masked default: keep the real value

    if hf_token and verify_token:
        _verify_token(hf_token)

    target_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"lab_id": lab_id, "huggingface_token": hf_token}, sort_keys=False),
        encoding="utf-8",
    )
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # contains a credential

    # Prove the file round-trips through the reader the rest of the toolkit uses.
    saved_lab_id, _ = read_contributor_config(path)
    logger.info("Saved. Episodes will be uploaded to OopsieData-Submissions/%s", saved_lab_id)
    if not hf_token:
        logger.info("No token saved — set $HF_TOKEN in your environment before uploading.")
    return path


# ── entry point ───────────────────────────────────────────────────────────────


def run_init(
    config_dir_from_flag: bool = False,
    lab_id: str | None = None,
    hf_token: str | None = None,
    verify_token: bool = True,
    force: bool = False,
) -> int:
    """Run the wizard. Returns a shell-style exit code."""
    try:
        target_dir = choose_target_dir()
        written = step_credentials(
            target_dir,
            lab_id=lab_id,
            hf_token=hf_token,
            verify_token=verify_token,
            force=force,
        )
        if written is None:
            return 1  # refused to touch an existing config; the reason is already logged
        advise_persisting_config_dir(target_dir, from_flag=config_dir_from_flag)

        # Profiles deliberately do not live next to the credentials: they belong with the
        # robot code that loads them (see oopsie_data_tools.utils.paths).
        logger.info("\nSetup complete. Your credentials are saved in %s", target_dir)
        logger.info("Run 'oopsie-data show-config' at any time to see what is in use.")
        logger.info(
            "Next: write a robot profile and keep it next to your robot code, e.g. %s.\n"
            "Copy one of the examples or the commented template from the repo's "
            "configs/robot_profiles/, then load it with load_robot_profile(<path>).",
            paths.write_profiles_dir(),
        )
        return 0
    except WizardAbort as e:
        logger.error("%s", e)
        return 1
    except (KeyboardInterrupt, EOFError):
        logger.info("\nAborted — nothing further was written.")
        return 130
