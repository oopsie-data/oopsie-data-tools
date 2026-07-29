"""``oopsie-data`` command line interface.

Single entry point for the contributor workflow; every capability lives here rather than
in a parallel collection of scripts.

Credentials and robot profiles are looked up through separate chains (see
:mod:`oopsie_data_tools.utils.paths`); ``--config-dir`` overrides the credential location
for one invocation, and ``oopsie-data show-config`` prints what is in effect right now.

Note that this docstring is *not* the ``--help`` epilog: help text is user-facing and lives
in :data:`_EPILOG` and the per-command ``description``/``epilog`` strings, so developer
notes and reStructuredText markup cannot leak into a terminal.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import logging
import os
import re
import sys
import textwrap
from pathlib import Path

# Imported eagerly, unlike the heavier command modules below: the parser needs the agent
# names to build --agent's choices, and this module costs nothing but the standard library.
from oopsie_data_tools.utils.claude_skill import AGENT_SKILL_DIRS as _AGENT_CHOICES
from oopsie_data_tools.utils.claude_skill import DEFAULT_AGENT as _DEFAULT_AGENT
from oopsie_data_tools.utils.hf_limits import BATCH_SIZE, FILE_LIMIT
from oopsie_data_tools.utils.paths import ENV_CONFIG_DIR

logger = logging.getLogger(__name__)


def _version() -> str:
    """Installed package version, for ``--version``.

    A source tree that was never installed has no distribution metadata, and reporting the
    version is never worth failing an invocation over.
    """
    try:
        from importlib.metadata import version

        return version("oopsie-data-tools")
    except Exception:
        return "unknown (not installed)"


# ── machine-readable output ───────────────────────────────────────────────────

#: Commands whose result is a structure rather than a narrative accept ``--json``, for
#: callers — scripts, CI, coding agents — that would otherwise have to parse prose. The
#: prose output is unchanged and stays the default; exit codes mean the same thing either
#: way. JSON goes to stdout on its own, so logs must not be interleaved with it.
def _emit_json(payload: dict) -> None:
    import json

    print(json.dumps(payload, indent=2, default=str))


@contextlib.contextmanager
def _quiet_logging(enabled: bool):
    """Silence log output so --json emits a parseable document and nothing else.

    Restores the previous level on the way out: ``main`` is called more than once in a
    single process by the test suite and by anything embedding the CLI, and a root logger
    left at CRITICAL would silence every later command.
    """
    root = logging.getLogger()
    previous = root.level
    if enabled:
        root.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        root.setLevel(previous)


# ── show-config ───────────────────────────────────────────────────────────────


def _describe_credential_dir(candidate: Path) -> str:
    if (candidate / "contributor_config.yaml").is_file():
        return "has contributor_config.yaml"
    if candidate.is_dir():
        return "exists, no contributor_config.yaml"
    return "does not exist"


def _describe_profiles_dir(candidate: Path) -> str:
    if not candidate.is_dir():
        return "does not exist"
    count = len(list(candidate.glob("*.yaml")))
    if not count:
        return "exists, no profiles"
    return f"{count} profile{'' if count == 1 else 's'}"


def _print_chain(candidates: list[Path], active: Path | None, describe) -> None:
    """Print a lookup chain as two aligned columns, marking the entry that wins."""
    print("  Searched in order:")
    width = max(len(str(candidate)) for candidate in candidates)
    for candidate in candidates:
        marker = "->" if candidate == active else "  "
        print(f"  {marker} {str(candidate).ljust(width)}   {describe(candidate)}")


_FIELD_INDENT = 14  # two leading spaces + a 12-wide label column


def _print_field(label: str, value: str) -> None:
    print(f"  {(label + ':').ljust(_FIELD_INDENT - 2)}{value}")


def _print_wrapped_field(label: str, value: str) -> None:
    """A field whose value may be long (a list of file names), wrapped under the label."""
    lines = textwrap.wrap(value, width=88, subsequent_indent=" " * _FIELD_INDENT)
    _print_field(label, lines[0] if lines else "")
    for line in lines[1:]:
        print(line)


def _read_config_yaml(config_path: Path) -> tuple[dict, str | None]:
    """The contributor config as a plain dict, plus a parse error if there was one.

    Read directly rather than through ``read_contributor_config``, which raises on a
    missing or placeholder lab_id — this command must still report what it found.
    """
    import yaml

    if not config_path.is_file():
        return {}, None
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return (loaded if isinstance(loaded, dict) else {}), None
    except yaml.YAMLError as e:
        return {}, str(e)


def _show_config_json(args: argparse.Namespace) -> int:
    from oopsie_data_tools.init_wizard import _mask
    from oopsie_data_tools.utils import paths

    config_path = paths.contributor_config_path()
    profiles_dir = paths.robot_profiles_dir()
    data, parse_error = _read_config_yaml(config_path)

    lab_id = str(data.get("lab_id") or "").strip()
    config_token = str(data.get("huggingface_token") or "").strip()
    env_token = os.environ.get("HF_TOKEN", "").strip()
    token = env_token or config_token

    _emit_json(
        {
            "credentials": {
                "env_var": paths.ENV_CONFIG_DIR,
                "env_value": os.environ.get(paths.ENV_CONFIG_DIR),
                "search_path": [str(p) for p in paths.config_search_dirs()],
                "active": str(config_path) if config_path.is_file() else None,
                "parse_error": parse_error,
                "lab_id": lab_id or None,
                # Masked unless asked, so --json is safe to paste into a bug report.
                "hf_token": (token if args.show_token else _mask(token)) if token else None,
                "hf_token_source": (
                    None if not token else ("HF_TOKEN" if env_token else "contributor_config.yaml")
                ),
                "uploads_to": f"OopsieData-Submissions/{lab_id}" if lab_id else None,
            },
            "robot_profiles": {
                "env_var": paths.ENV_PROFILES_DIR,
                "env_value": os.environ.get(paths.ENV_PROFILES_DIR),
                "search_path": [str(p) for p in paths.profiles_search_dirs()],
                "active": str(profiles_dir) if profiles_dir.is_dir() else None,
                "profiles": (
                    sorted(p.name for p in profiles_dir.glob("*.yaml"))
                    if profiles_dir.is_dir()
                    else []
                ),
                "write_target": str(paths.write_profiles_dir()),
            },
        }
    )
    return 0


def cmd_show_config(args: argparse.Namespace) -> int:
    """Show where every config is read from right now, and what is in it. Read-only."""
    from oopsie_data_tools.init_wizard import _mask
    from oopsie_data_tools.utils import paths

    if args.json:
        return _show_config_json(args)

    config_path = paths.contributor_config_path()
    profiles_dir = paths.robot_profiles_dir()

    print("\nCredentials — contributor_config.yaml")
    print(f"  ${paths.ENV_CONFIG_DIR} = {os.environ.get(paths.ENV_CONFIG_DIR) or '(not set)'}\n")
    _print_chain(
        paths.config_search_dirs(),
        config_path.parent if config_path.is_file() else None,
        _describe_credential_dir,
    )
    print()
    if config_path.is_file():
        _print_field("Reading", str(config_path))
    else:
        print(f"  No config found. 'oopsie-data init' would write {config_path}")

    data, parse_error = _read_config_yaml(config_path)
    if parse_error:
        print(f"  Could not parse the file: {parse_error}")

    lab_id = str(data.get("lab_id") or "").strip()
    config_token = str(data.get("huggingface_token") or "").strip()
    env_token = os.environ.get("HF_TOKEN", "").strip()
    token = env_token or config_token

    _print_field("lab_id", lab_id or "(not set)")
    if token:
        shown = token if args.show_token else _mask(token)
        source = "$HF_TOKEN, overrides the config" if env_token else "contributor_config.yaml"
        _print_field("hf_token", f"{shown}   (from {source})")
    else:
        _print_field("hf_token", "(not set — set one in the config or export $HF_TOKEN)")
    if lab_id:
        _print_field("uploads to", f"OopsieData-Submissions/{lab_id}")

    print("\nRobot profiles")
    print(
        f"  ${paths.ENV_PROFILES_DIR} = "
        f"{os.environ.get(paths.ENV_PROFILES_DIR) or '(not set)'}\n"
    )
    _print_chain(
        paths.profiles_search_dirs(),
        profiles_dir if profiles_dir.is_dir() else None,
        _describe_profiles_dir,
    )
    print()
    if profiles_dir.is_dir():
        _print_field("Reading", str(profiles_dir))
        names = sorted(profile.name for profile in profiles_dir.glob("*.yaml"))
        if names:
            _print_wrapped_field("profiles", ", ".join(names))
    else:
        print(f"  No profile directory found. A new one belongs in {paths.write_profiles_dir()}")
    print(
        "\n  Profiles are usually loaded by explicit path — load_robot_profile(<path>) —\n"
        "  which ignores the lookup above. Paths here are relative to the current directory."
    )

    if not args.show_token and token:
        print("\nRe-run with --show-token to print the token in full.")
    return 0


# ── init ──────────────────────────────────────────────────────────────────────


def cmd_init(args: argparse.Namespace) -> int:
    from oopsie_data_tools.init_wizard import run_init

    return run_init(
        # --config-dir only lives for this process, so init should offer to persist it.
        config_dir_from_flag=args.config_dir is not None,
        lab_id=args.lab_id,
        hf_token=args.hf_token,
        verify_token=not args.no_verify_token,
        force=args.force,
    )


# ── new-profile ───────────────────────────────────────────────────────────────


def cmd_new_profile(args: argparse.Namespace) -> int:
    """Write a starter robot profile next to the user's robot code."""
    from oopsie_data_tools.utils import paths
    from oopsie_data_tools.utils.robot_profile.template import PROFILE_TEMPLATE

    target_dir = Path(args.dir) if args.dir is not None else paths.write_profiles_dir()
    target = target_dir / f"{args.name}.yaml"

    if target.exists() and not args.force:
        logger.error("%s already exists. Pass --force to overwrite it.", target)
        return 1

    target_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(PROFILE_TEMPLATE, encoding="utf-8")

    logger.info("Wrote %s", target)
    logger.info(
        "\nEvery required field is blank, so the profile will not load until you fill it in —\n"
        "that is deliberate, so a half-edited profile cannot stamp placeholder metadata into\n"
        "your recorded episodes. Check your work with:\n\n"
        "    python -c \"from oopsie_data_tools.utils.robot_profile.robot_profile import "
        'load_robot_profile; load_robot_profile(\'%s\')"',
        target,
    )
    return 0


# ── annotate ──────────────────────────────────────────────────────────────────


def _prompt_annotator_name() -> str | None:
    """Ask for the annotator name interactively; None if unavailable or left blank."""
    if not sys.stdin.isatty():
        logger.error("--annotator-name is required when stdin is not a terminal.")
        return None
    for _ in range(3):
        name = input("Annotator name (stamped into saved annotations): ").strip()
        if name:
            return name
        print("Please enter a non-empty name.")
    logger.error("No annotator name given.")
    return None


def cmd_annotate(args: argparse.Namespace) -> int:
    from oopsie_data_tools.annotation_tool.annotator_server import run_server

    annotator_name = (args.annotator_name or "").strip() or _prompt_annotator_name()
    if not annotator_name:
        return 1

    return run_server(
        samples_dir=args.samples_dir,
        annotator_name=annotator_name,
        port=args.port,
        open_browser=not args.no_browser,
        with_rollouts=args.with_rollouts,
    )


# ── validate ──────────────────────────────────────────────────────────────────


def cmd_validate(args: argparse.Namespace) -> int:
    from oopsie_data_tools.utils.hf_upload import run_validation

    target = os.path.abspath(os.path.normpath(args.path))
    if args.episode_id:
        target = os.path.join(target, f"{args.episode_id}.h5")
    if not os.path.exists(target):
        if args.json:
            _emit_json({"path": target, "error": "path does not exist", "episodes": []})
            return 1
        logger.error("Path does not exist: %s", target)
        return 1

    if not args.json:
        return run_validation(args.path, args.episode_id, args.log_path)

    from oopsie_data_tools.utils.validation.validation_utils import collect_validation_results

    episodes = collect_validation_results(target)
    failed = [e for e in episodes if not e["passed"]]
    _emit_json(
        {
            "path": target,
            "episodes": episodes,
            "total": len(episodes),
            "passed": len(episodes) - len(failed),
            "failed": len(failed),
        }
    )
    # An empty directory is a failure in the prose path too — nothing was checked, so
    # nothing can be said to have passed.
    return 1 if failed or not episodes else 0


# ── upload ────────────────────────────────────────────────────────────────────


def _restructure_for_upload(samples_dir: str) -> str | None:
    """Split an oversized session so it can be uploaded. Returns the copy, or None.

    Passing --with-restructure is the consent the standalone command prompts for, so this
    does not ask again — but ``run_restructure`` still logs the size estimate first, and
    the source is left untouched either way.
    """
    from oopsie_data_tools.utils.hf_upload import check_folder_size
    from oopsie_data_tools.utils.restructure import run_restructure

    source = Path(samples_dir)
    output = source.parent / f"{source.name}_restructured"
    logger.info(
        "[restructure] --with-restructure is set. Writing a restructured copy to\n"
        "              %s\n"
        "              The original is not modified. Run 'oopsie-data restructure' "
        "yourself to choose a different destination.",
        output,
    )

    if run_restructure(source, output, assume_yes=True) != 0:
        logger.error("Aborting: the session could not be restructured.")
        return None

    # Confirm the layout is actually fixed rather than assuming it. A directory holding no
    # episodes cannot be split by episode, so it survives the copy still oversized — and
    # uploading it would fail at the Hub for the reason we just tried to fix.
    if check_folder_size(str(output), suggest_fix=False):
        logger.error(
            "Aborting: %s is still over the file limit after restructuring. "
            "Reorganise the directories listed above by hand.",
            output,
        )
        return None

    logger.info("[restructure] Continuing with %s", output)
    return str(output)


def cmd_upload(args: argparse.Namespace) -> int:
    from oopsie_data_tools.utils.hf_upload import (
        check_folder_size,
        hf_login,
        require_repo,
        resolve_hf_target,
        run_validation,
        upload_dataset,
    )
    from oopsie_data_tools.utils.validation.diversity import check_diversity

    logger.info("=" * 60)
    logger.info("  Robotic Failure Dataset — End-to-End Upload Pipeline")
    logger.info("=" * 60)

    samples_dir = os.path.abspath(os.path.normpath(args.path))
    if not os.path.isdir(samples_dir):
        logger.error("Not a directory: %s", samples_dir)
        return 1

    # Pre-upload folder-size check (HF enforces a per-directory file limit). Only relevant
    # when something is actually going to be uploaded — with --skip-upload this is a
    # validation run, and a layout HF would reject is not a reason to fail it. Asking for
    # --with-restructure is asking for the layout to be fixed, so it runs either way.
    if not args.skip_upload or args.with_restructure:
        if check_folder_size(samples_dir, suggest_fix=not args.with_restructure):
            if not args.with_restructure:
                return 1
            restructured = _restructure_for_upload(samples_dir)
            if restructured is None:
                return 1
            # Everything downstream must see the copy: validation reads the rewritten
            # video paths, and it is the copy that gets uploaded.
            samples_dir = restructured

    if not args.skip_validate:
        if run_validation(samples_dir, args.episode_id, args.log_path) != 0:
            logger.error(
                "Aborting upload due to validation failure. Fix the dataset format and retry."
            )
            return 1
    else:
        logger.info("[validate] Skipped.")

    # Low-diversity warning (advisory unless --strict-diversity). Reads attrs only.
    diversity_warnings = check_diversity(samples_dir)
    if diversity_warnings and args.strict_diversity:
        logger.error("Aborting: low-diversity warnings present and --strict-diversity is set.")
        return 1

    if args.skip_upload:
        logger.info("[upload] Skipped (--skip-upload).")
        return 0

    from huggingface_hub import HfApi

    hf_token, hf_repo = resolve_hf_target()
    try:
        hf_login(hf_token)
    except Exception as e:
        logger.error(
            "HuggingFace authentication failed: %s\n"
            "Check the huggingface_token in your contributor config (or $HF_TOKEN).",
            e,
        )
        return 1
    api = HfApi(token=hf_token)
    if not require_repo(api, hf_repo):
        return 1
    upload_dataset(api, hf_repo, samples_dir)
    return 0


# ── submissions ───────────────────────────────────────────────────────────────


def cmd_submissions(args: argparse.Namespace) -> int:
    from oopsie_data_tools.utils.hf_upload import query_submissions

    return query_submissions(args.lab_id.strip() if args.lab_id else None)


# ── inspect ───────────────────────────────────────────────────────────────────


def cmd_inspect(args: argparse.Namespace) -> int:
    from oopsie_data_tools.utils.h5_inspect import inspect_h5, inspect_h5_structure

    if not os.path.isfile(args.path):
        if args.json:
            _emit_json({"path": args.path, "error": "not a file"})
            return 1
        logger.error("Not a file: %s", args.path)
        return 1

    if args.json:
        _emit_json(inspect_h5_structure(args.path))
    else:
        inspect_h5(args.path)
    return 0


# ── restructure ───────────────────────────────────────────────────────────────


def cmd_restructure(args: argparse.Namespace) -> int:
    from oopsie_data_tools.utils.restructure import run_restructure

    return run_restructure(args.source, args.output, assume_yes=args.yes)


# ── install-skill ─────────────────────────────────────────────────────────────


def cmd_install_skill(args: argparse.Namespace) -> int:
    from oopsie_data_tools.utils.claude_skill import check_installations, install_skill

    if args.check:
        return check_installations()
    return install_skill(user=args.user, force=args.force, agent=args.agent)


# ── parser ────────────────────────────────────────────────────────────────────

#: Shown under every top-level ``--help``. argparse already lists the commands and what each
#: one does, so this adds only what that list cannot: the order they go in, and the two
#: things people get stuck on.
_EPILOG = """\
A typical session, in order:

  oopsie-data init                       # once: lab id + HuggingFace token
  oopsie-data new-profile                # once per robot, then fill the file in
  oopsie-data annotate --samples-dir ./samples --annotator-name "your name"
  oopsie-data validate --path ./samples
  oopsie-data upload   --path ./samples

If upload refuses because a directory holds too many files for HuggingFace, split it with
'oopsie-data restructure --source ./samples', or pass --with-restructure to upload. Run
'oopsie-data show-config' to see where credentials and robot profiles are being read from,
and 'oopsie-data <command> --help' for one command's options.
"""


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Wrap help prose to the terminal, but leave indented example blocks alone.

    ``RawDescriptionHelpFormatter`` would print every description as one unwrapped line,
    and the plain formatter would reflow the example blocks into paragraphs. A paragraph
    is treated as preformatted if any of its lines is indented.
    """

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        chunks = []
        for para in text.split("\n\n"):
            if any(line[:1].isspace() for line in para.splitlines()):
                chunks.append(textwrap.indent(para.strip("\n"), indent))
            else:
                chunks.append(
                    textwrap.fill(
                        " ".join(para.split()),
                        width,
                        initial_indent=indent,
                        subsequent_indent=indent,
                    )
                )
        return "\n\n".join(chunks)


class _ArgumentParser(argparse.ArgumentParser):
    """Replaces argparse's invalid-choice wall of text with a typo suggestion."""

    def error(self, message: str):  # noqa: D102 - argparse API
        match = re.search(r"argument <command>: invalid choice: '([^']*)'", message)
        if match:
            given = match.group(1)
            close = difflib.get_close_matches(given, _COMMANDS, n=1)
            hint = (
                f"did you mean '{close[0]}'?"
                if close
                else "run 'oopsie-data --help' for the list of commands."
            )
            message = f"unknown command '{given}' — {hint}"
        super().error(message)


def _add_json_flag(parser: argparse.ArgumentParser, what: str) -> None:
    """Give a command a ``--json`` mode. Exit codes are unchanged; only the format differs."""
    parser.add_argument(
        "--json",
        action="store_true",
        help=f"Print {what} as JSON on stdout instead of prose, for scripts and agents",
    )


def _add_command(
    sub, name: str, *, summary: str, description: str, examples: str = ""
) -> argparse.ArgumentParser:
    """Add a subcommand whose help renders consistently with every other one."""
    return sub.add_parser(
        name,
        help=summary,
        description=description,
        epilog=f"Examples:\n\n{examples}" if examples else None,
        formatter_class=_HelpFormatter,
    )


#: Kept next to the parser so the typo suggestion and the metavar cannot drift apart.
_COMMANDS = [
    "init",
    "show-config",
    "new-profile",
    "annotate",
    "validate",
    "upload",
    "submissions",
    "inspect",
    "restructure",
    "install-skill",
]


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="oopsie-data",
        description=(
            "Record, annotate, validate and publish robotic manipulation rollouts. "
            "Start with 'oopsie-data init'."
        ),
        formatter_class=_HelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument(
        "--version", action="version", version=f"oopsie-data {_version()}"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory holding contributor_config.yaml "
            f"(overrides ${ENV_CONFIG_DIR} for this invocation; robot profiles are unaffected)"
        ),
    )
    # Not required=True: a bare 'oopsie-data' should print the full help, which lists the
    # commands, rather than a usage line that only says a command was missing.
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # init
    p_init = _add_command(
        sub,
        "init",
        summary="Set up the contributor config (lab id + HuggingFace token)",
        description=(
            "Interactive setup: picks a config directory and writes contributor_config.yaml "
            "with your lab id and HuggingFace token, verifying the token against the "
            "HuggingFace API. Values passed as flags are not prompted for, so a fully-flagged "
            "invocation runs unattended."
        ),
        examples=(
            "  oopsie-data init\n"
            "  oopsie-data init --lab-id my_lab --hf-token hf_xxx   # unattended\n"
            "  oopsie-data --config-dir ./cfg init                  # somewhere else\n"
        ),
    )
    p_init.add_argument("--lab-id", default=None, help="Lab id from the registration form")
    p_init.add_argument("--hf-token", default=None, help="HuggingFace token to store")
    p_init.add_argument(
        "--no-verify-token",
        action="store_true",
        help="Do not check the token against the HuggingFace API",
    )
    p_init.add_argument(
        "--force", action="store_true", help="Overwrite an existing config without asking"
    )
    p_init.set_defaults(func=cmd_init)

    # show-config
    p_config = _add_command(
        sub,
        "show-config",
        summary="Show where configs are read from, and the current lab id and token",
        description=(
            "Print every location searched for the contributor config and for robot profiles, "
            "which one is currently being used, and the lab id and HuggingFace token in effect. "
            "Read-only: use 'oopsie-data init' to change anything. The token is masked unless "
            "--show-token is given."
        ),
        examples=(
            "  oopsie-data show-config\n"
            "  oopsie-data --config-dir ./cfg show-config   # try a different location\n"
        ),
    )
    p_config.add_argument(
        "--show-token", action="store_true", help="Print the HuggingFace token in full"
    )
    _add_json_flag(p_config, "the resolved config, both search paths, and what is in effect")
    p_config.set_defaults(func=cmd_show_config)

    # new-profile
    p_new_profile = _add_command(
        sub,
        "new-profile",
        summary="Write a starter robot profile you can fill in",
        description=(
            "Write a commented robot-profile skeleton into your project, by default "
            "./robot_profiles/. Profiles belong next to the robot code that loads them, not in "
            "your user config directory, and are never read out of the installed package. The "
            "skeleton does not load until you fill in the required fields — that is deliberate."
        ),
        examples=(
            "  oopsie-data new-profile\n"
            "  oopsie-data new-profile --name franka_pick --dir ./configs/robot_profiles\n"
        ),
    )
    p_new_profile.add_argument(
        "--name", default="robot_profile", help="File name without .yaml (default: robot_profile)"
    )
    p_new_profile.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory to write into (default: ./robot_profiles, or $OOPSIE_ROBOT_PROFILES_DIR)",
    )
    p_new_profile.add_argument(
        "--force", action="store_true", help="Overwrite an existing profile of the same name"
    )
    p_new_profile.set_defaults(func=cmd_new_profile)

    # annotate
    p_annotate = _add_command(
        sub,
        "annotate",
        summary="Launch the web annotation UI over a directory of episodes",
        description=(
            "Serve the annotation UI for the episodes in --samples-dir and open it in a "
            "browser. Annotations are written back into the HDF5 files under your annotator "
            "name, so the same session can be annotated by several people. Runs until you "
            "stop it with Ctrl-C."
        ),
        examples=(
            "  oopsie-data annotate --samples-dir ./samples --annotator-name alex\n"
            "  oopsie-data annotate --samples-dir ./samples --port 8080 --no-browser\n"
        ),
    )
    p_annotate.add_argument(
        "--samples-dir",
        "--path",
        type=Path,
        default=Path("samples"),
        metavar="DIR",
        help="Directory containing saved MP4s (and HDF5 episodes) (default: ./samples)",
    )
    p_annotate.add_argument(
        "--annotator-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Annotator name to stamp into saved annotations (prompted for if omitted)",
    )
    p_annotate.add_argument(
        "--port", type=int, default=5001, metavar="N", help="Server port (default: 5001)"
    )
    p_annotate.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser window on start"
    )
    p_annotate.add_argument(
        "--with-rollouts",
        action="store_true",
        help="Enable in-the-loop rollout mode (HDF5 browser + annotation form + rollouts)",
    )
    p_annotate.set_defaults(func=cmd_annotate)

    # validate
    p_validate = _add_command(
        sub,
        "validate",
        summary="Validate episodes against the oopsiedata schema",
        description=(
            "Check a single .h5 episode, or every episode in a session directory, against the "
            "oopsiedata_format_v1 schema — the same check 'oopsie-data upload' runs before "
            "publishing. Every failure is reported with the episode and field that caused it. "
            "Exits 0 if everything passed, 1 otherwise, so it is usable in a script."
        ),
        examples=(
            "  oopsie-data validate --path ./samples\n"
            "  oopsie-data validate --path ./samples --episode-id 000001\n"
            "  oopsie-data validate --path ./samples --log-path validate.log\n"
        ),
    )
    p_validate.add_argument(
        "--path",
        "-p",
        "--samples-dir",
        required=True,
        metavar="PATH",
        help="A single .h5 file, or a session directory containing .h5 files",
    )
    p_validate.add_argument(
        "--episode-id",
        "--episode_id",
        "-e",
        default=None,
        metavar="ID",
        help="Zero-padded episode id (e.g. 000001) to validate just that episode inside --path",
    )
    p_validate.add_argument(
        "--log-path", "-l", default=None, metavar="FILE", help="Also write logs to this file"
    )
    _add_json_flag(p_validate, "one record per episode, with the error that rejected it")
    p_validate.set_defaults(func=cmd_validate)

    # upload
    p_upload = _add_command(
        sub,
        "upload",
        summary="Validate and upload a session directory to HuggingFace",
        description=(
            "Validate a session directory and upload it to your lab's HuggingFace dataset repo, "
            "OopsieData-Submissions/<lab_id>, creating the repo on first use. Credentials come "
            "from contributor_config.yaml, with $HF_TOKEN overriding the stored token. Nothing "
            "is uploaded unless validation passes and the directory layout is within "
            "HuggingFace's per-directory file limit."
        ),
        examples=(
            "  oopsie-data upload --path ./samples\n"
            "  oopsie-data upload --path ./samples --skip-upload    # dry run: checks only\n"
            "  oopsie-data upload --path ./samples --with-restructure\n"
        ),
    )
    p_upload.add_argument(
        "--path",
        "-p",
        "--samples-dir",
        required=True,
        metavar="DIR",
        help="Session directory containing formatted episode files",
    )
    p_upload.add_argument(
        "--episode-id",
        "--episode_id",
        "-e",
        default=None,
        metavar="ID",
        help="Zero-padded episode id to validate; if omitted, every *.h5 in --path is validated",
    )
    p_upload.add_argument(
        "--skip-validate",
        "--skip_validate",
        action="store_true",
        help="Skip validation before uploading",
    )
    p_upload.add_argument(
        "--skip-upload",
        "--skip_upload",
        action="store_true",
        help="Run validation and checks only, do not upload",
    )
    p_upload.add_argument(
        "--with-restructure",
        action="store_true",
        help=(
            "If a directory exceeds the HuggingFace file limit, write a restructured copy "
            "to <path>_restructured and upload that instead of aborting (the original is "
            "left untouched, so this needs room for a second copy)"
        ),
    )
    p_upload.add_argument(
        "--log-path", "-l", default=None, metavar="FILE", help="Also write logs to this file"
    )
    p_upload.add_argument(
        "--strict-diversity",
        action="store_true",
        help="Treat low task/annotation diversity warnings as a hard error",
    )
    p_upload.set_defaults(func=cmd_upload)

    # submissions
    p_submissions = _add_command(
        sub,
        "submissions",
        summary="Show what your lab has already uploaded to HuggingFace",
        description=(
            "Report episode, video and file counts in OopsieData-Submissions/<lab_id> without "
            "downloading anything. A repo that does not exist yet is not an error — it is "
            "created on your first successful upload."
        ),
        examples=(
            "  oopsie-data submissions\n"
            "  oopsie-data submissions --lab-id some_other_lab\n"
        ),
    )
    p_submissions.add_argument(
        "--lab-id",
        default=None,
        metavar="ID",
        help="Lab id to query (default: the lab_id in your contributor config)",
    )
    p_submissions.set_defaults(func=cmd_submissions)

    # inspect
    p_inspect = _add_command(
        sub,
        "inspect",
        summary="Dump the structure of an HDF5 episode",
        description=(
            "Print every group, dataset, shape, dtype and attribute in an HDF5 file. This is a "
            "debugging aid, not a validator: it makes no assumptions about the schema, so it "
            "works just as well on a file that 'oopsie-data validate' rejects — which is "
            "usually how you find out why it was rejected."
        ),
        examples=(
            "  oopsie-data inspect ./samples/000000.h5\n"
            "  oopsie-data inspect ./samples/000000.h5 | less\n"
        ),
    )
    p_inspect.add_argument("path", metavar="FILE", help="Path to a .h5 / .hdf5 file")
    _add_json_flag(p_inspect, "the full group/dataset/attribute tree")
    p_inspect.set_defaults(func=cmd_inspect)

    # restructure
    p_restructure = _add_command(
        sub,
        "restructure",
        summary="Split an oversized session directory into numbered subfolders",
        description=(
            "Copy a session into a new folder in which every directory that exceeds "
            f"HuggingFace's per-directory limit of {FILE_LIMIT:,} files — the one "
            "'oopsie-data upload' refuses on — has been split into numbered subfolders of at "
            f"most {BATCH_SIZE} episodes each, wherever in the tree that directory sits. "
            "Directories already under the limit are copied through unchanged.\n\n"
            "Non-destructive: the source is never modified, not even the HDF5 files, whose "
            "stored video paths are rewritten only in the copy. That means you need room for "
            "a second copy of the session, and you delete the original yourself once you have "
            "checked the output. 'oopsie-data upload --with-restructure' does all of this as "
            "one step."
        ),
        examples=(
            "  oopsie-data restructure --source ./samples\n"
            "  oopsie-data restructure --source ./samples --output /big/disk/samples_split\n"
            "  oopsie-data restructure --source ./samples --yes   # no prompt, for scripts\n"
        ),
    )
    p_restructure.add_argument(
        "--source", "-s", type=Path, required=True, metavar="DIR",
        help="Session directory to restructure (its whole tree is scanned)",
    )
    p_restructure.add_argument(
        "--output", "-o", type=Path, default=None, metavar="DIR",
        help="Destination for the restructured copy (default: <source>_restructured alongside it)",
    )
    p_restructure.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the confirmation prompt (this copies the whole dataset)",
    )
    p_restructure.set_defaults(func=cmd_restructure)

    # install-skill
    p_skill = _add_command(
        sub,
        "install-skill",
        summary="Install the bundled agent skill for oopsie-data",
        description=(
            "Copy the skill that ships with this package into a directory your coding agent "
            "scans, so it knows how to drive the contributor workflow. SKILL.md is a shared "
            "format, so the same payload works for Claude Code, Cursor, Codex and anything "
            "else that reads it — --agent only chooses the destination. Defaults to "
            "./.claude/skills/oopsie-data/, which Claude Code and Cursor both read; --user "
            "installs into your home directory instead, making it available in every project. "
            "Entirely optional: nothing else in oopsie-data needs an agent, and no files are "
            "written anywhere unless you run this command."
        ),
        examples=(
            "  oopsie-data install-skill\n"
            "  oopsie-data install-skill --user             # available in every project\n"
            "  oopsie-data install-skill --agent cursor\n"
            "  oopsie-data install-skill --agent none       # a plain ./skills/ to copy onward\n"
            "  oopsie-data install-skill --check            # is an installed copy out of date?\n"
        ),
    )
    p_skill.add_argument(
        "--agent",
        choices=list(_AGENT_CHOICES),
        default=_DEFAULT_AGENT,
        help=(
            f"Which agent's skills directory to install into (default: {_DEFAULT_AGENT}). "
            "'agents' is the vendor-neutral .agents/skills/; 'none' writes a plain ./skills/ "
            "that no agent scans, for copying onward yourself"
        ),
    )
    p_skill.add_argument(
        "--user",
        action="store_true",
        help="Install under your home directory instead of the working directory",
    )
    p_skill.add_argument(
        "--force", action="store_true", help="Overwrite an existing installation of the skill"
    )
    p_skill.add_argument(
        "--check",
        action="store_true",
        help=(
            "Report installed copies and whether they came from this version of the package, "
            "instead of installing anything. Exits 1 if any copy is stale or missing"
        ),
    )
    p_skill.set_defaults(func=cmd_install_skill)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    # A bare 'oopsie-data' is someone looking for the commands, so show them: the full help
    # answers that, where argparse's "the following arguments are required" does not.
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 2

    # Export before any command runs, so library code resolving config paths sees it.
    if args.config_dir is not None:
        os.environ[ENV_CONFIG_DIR] = str(Path(args.config_dir).expanduser().resolve())

    # --json owns stdout: a log line interleaved with the document would make it
    # unparseable, which is the whole point of the flag.
    as_json = getattr(args, "json", False)
    with _quiet_logging(as_json):
        try:
            return args.func(args)
        except KeyboardInterrupt:
            logger.info("Interrupted.")
            return 130
        except RuntimeError as e:
            # Config/auth problems already carry an actionable message; don't dump a
            # traceback. Under --json logging is silenced, so the message has to go into
            # the document or the caller gets an empty stdout and a bare exit code.
            if as_json:
                _emit_json({"error": str(e)})
            else:
                logger.error("%s", e)
            return 1


if __name__ == "__main__":
    sys.exit(main())
