"""``oopsie-data`` command line interface.

Single entry point for the contributor workflow::

    oopsie-data init                                 # first-time setup
    oopsie-data show-config                          # where configs are read from
    oopsie-data new-profile                          # starter robot profile to fill in
    oopsie-data annotate --samples-dir ./samples --annotator-name "your_name"
    oopsie-data validate --path ./samples
    oopsie-data upload   --path ./samples
    oopsie-data submissions                          # what has landed on HuggingFace
    oopsie-data inspect episode.h5                   # dump an episode's structure
    oopsie-data restructure --source ./samples       # split a folder HF would reject
    oopsie-data install-skill                        # optional: teach Claude Code this workflow

This is the only entry point; every capability lives here rather than in a parallel
collection of scripts.

Credentials and robot profiles are looked up through separate chains (see
oopsie_data_tools.utils.paths); --config-dir overrides the credential location for one
invocation, and 'oopsie-data show-config' prints what is in effect right now.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from pathlib import Path

from oopsie_data_tools.utils.paths import ENV_CONFIG_DIR

logger = logging.getLogger(__name__)


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


def cmd_show_config(args: argparse.Namespace) -> int:
    """Show where every config is read from right now, and what is in it. Read-only."""
    import yaml

    from oopsie_data_tools.init_wizard import _mask
    from oopsie_data_tools.utils import paths

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

    # Read the YAML directly: read_contributor_config raises on a missing or placeholder
    # lab_id, and this command must still report what it found.
    data = {}
    if config_path.is_file():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            data = loaded if isinstance(loaded, dict) else {}
        except yaml.YAMLError as e:
            print(f"  Could not parse the file: {e}")

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
    if not os.path.exists(target):
        logger.error("Path does not exist: %s", target)
        return 1
    return run_validation(target, args.episode_id, args.log_path)


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
        ensure_repo,
        hf_login,
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
    ensure_repo(api, hf_repo)
    upload_dataset(api, hf_repo, samples_dir)
    return 0


# ── submissions ───────────────────────────────────────────────────────────────


def cmd_submissions(args: argparse.Namespace) -> int:
    from oopsie_data_tools.utils.hf_upload import query_submissions

    return query_submissions(args.lab_id.strip() if args.lab_id else None)


# ── inspect ───────────────────────────────────────────────────────────────────


def cmd_inspect(args: argparse.Namespace) -> int:
    from oopsie_data_tools.utils.h5_inspect import inspect_h5

    if not os.path.isfile(args.path):
        logger.error("Not a file: %s", args.path)
        return 1
    inspect_h5(args.path)
    return 0


# ── restructure ───────────────────────────────────────────────────────────────


def cmd_restructure(args: argparse.Namespace) -> int:
    from oopsie_data_tools.utils.restructure import run_restructure

    return run_restructure(args.source, args.output, assume_yes=args.yes)


# ── install-skill ─────────────────────────────────────────────────────────────


def cmd_install_skill(args: argparse.Namespace) -> int:
    from oopsie_data_tools.utils.claude_skill import install_skill

    return install_skill(project=args.project, force=args.force)


# ── parser ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oopsie-data",
        description="Annotate, validate, and upload robotic rollout data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding contributor_config.yaml "
            f"(overrides ${ENV_CONFIG_DIR} for this invocation; robot profiles are unaffected)"
        ),
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
        metavar=(
            "{init,show-config,new-profile,annotate,validate,upload,"
            "submissions,inspect,restructure,install-skill}"
        ),
    )

    # init
    p_init = sub.add_parser(
        "init",
        help="Set up the contributor config (lab id + HuggingFace token)",
        description=(
            "Interactive setup: picks a config directory and writes contributor_config.yaml "
            "with your lab id and HuggingFace token, verifying the token against the "
            "HuggingFace API. Values passed as flags are not prompted for, so a fully-flagged "
            "invocation runs unattended."
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
    p_config = sub.add_parser(
        "show-config",
        help="Show where configs are read from, and the current lab id and token",
        description=(
            "Print every location searched for the contributor config and for robot profiles, "
            "which one is currently being used, and the lab id and HuggingFace token in effect. "
            "Read-only: use 'oopsie-data init' to change anything. The token is masked unless "
            "--show-token is given."
        ),
    )
    p_config.add_argument(
        "--show-token", action="store_true", help="Print the HuggingFace token in full"
    )
    p_config.set_defaults(func=cmd_show_config)

    # new-profile
    p_new_profile = sub.add_parser(
        "new-profile",
        help="Write a starter robot profile you can fill in",
        description=(
            "Write a commented robot-profile skeleton into your project, by default "
            "./robot_profiles/. Profiles belong next to the robot code that loads them, not in "
            "your user config directory, and are never read out of the installed package. The "
            "skeleton does not load until you fill in the required fields — that is deliberate."
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
    p_annotate = sub.add_parser(
        "annotate",
        help="Launch the web annotation UI over a directory of episodes",
        description="Launch the Flask annotation UI for the episodes in --samples-dir.",
    )
    p_annotate.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("samples"),
        help="Directory containing saved MP4s (and HDF5 episodes) (default: ./samples)",
    )
    p_annotate.add_argument(
        "--annotator-name",
        type=str,
        default=None,
        help="Annotator name to stamp into saved annotations (prompted for if omitted)",
    )
    p_annotate.add_argument("--port", type=int, default=5001, help="Server port (default: 5001)")
    p_annotate.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser window on start"
    )
    p_annotate.add_argument(
        "--with-rollouts",
        action="store_true",
        help="Enable in-the-loop rollout mode (HDF5 browser + questionnaire + rollouts)",
    )
    p_annotate.set_defaults(func=cmd_annotate)

    # validate
    p_validate = sub.add_parser(
        "validate",
        help="Validate episodes against the oopsiedata schema",
        description="Validate a single .h5 episode or every episode in a session directory.",
    )
    p_validate.add_argument(
        "--path",
        "-p",
        required=True,
        help="Path to a single .h5 file or a session directory containing .h5 files",
    )
    p_validate.add_argument(
        "--episode-id",
        "--episode_id",
        "-e",
        default=None,
        help="Zero-padded episode id (e.g. 000001) to validate just that episode inside --path",
    )
    p_validate.add_argument("--log-path", "-l", default=None, help="Also write logs to this file")
    p_validate.set_defaults(func=cmd_validate)

    # upload
    p_upload = sub.add_parser(
        "upload",
        help="Validate and upload a session directory to HuggingFace",
        description=(
            "Validate a session directory and upload it to the lab's HuggingFace dataset repo. "
            "Credentials come from configs/contributor_config.yaml (HF_TOKEN overrides the token)."
        ),
    )
    p_upload.add_argument(
        "--path",
        "-p",
        "-o",
        required=True,
        help="Session directory containing formatted episode files",
    )
    p_upload.add_argument(
        "--episode-id",
        "--episode_id",
        "-e",
        default=None,
        help="Zero-padded episode id to validate; if omitted, all *.h5 files in --path are validated",
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
    p_upload.add_argument("--log-path", "-l", default=None, help="Also write logs to this file")
    p_upload.add_argument(
        "--strict-diversity",
        action="store_true",
        help="Treat low task/annotation diversity warnings as a hard error",
    )
    p_upload.set_defaults(func=cmd_upload)

    # submissions
    p_submissions = sub.add_parser(
        "submissions",
        help="Show what your lab has already uploaded to HuggingFace",
        description=(
            "Report episode, video and file counts in OopsieData-Submissions/<lab_id> without "
            "downloading anything. A repo that does not exist yet is not an error — it is "
            "created on your first successful upload."
        ),
    )
    p_submissions.add_argument(
        "--lab-id",
        default=None,
        help="Lab id to query (default: the lab_id in your contributor config)",
    )
    p_submissions.set_defaults(func=cmd_submissions)

    # inspect
    p_inspect = sub.add_parser(
        "inspect",
        help="Dump the structure of an HDF5 episode",
        description=(
            "Print every group, dataset, shape, dtype and attribute in an HDF5 file. This is a "
            "debugging aid, not a validator: it makes no assumptions about the schema, so it "
            "works just as well on a file that 'oopsie-data validate' rejects."
        ),
    )
    p_inspect.add_argument("path", help="Path to a .h5 / .hdf5 file")
    p_inspect.set_defaults(func=cmd_inspect)

    # restructure
    p_restructure = sub.add_parser(
        "restructure",
        help="Split an oversized session directory into numbered subfolders",
        description=(
            "Copy a session into a new folder in which every directory that exceeds the "
            "HuggingFace per-directory file limit — the one 'oopsie-data upload' refuses on — "
            "has been split into numbered subfolders of at most 500 episodes each. The whole "
            "tree is copied, so directories already under the limit come through unchanged and "
            "nesting works. Non-destructive: the source is never modified, and video paths are "
            "rewritten only inside the HDF5 copies."
        ),
    )
    p_restructure.add_argument(
        "--source", "-s", type=Path, required=True,
        help="Directory to restructure (must contain .h5 / .hdf5 files at its root)",
    )
    p_restructure.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Destination for the restructured copy (default: <source>_restructured alongside it)",
    )
    p_restructure.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the confirmation prompt (this copies the whole dataset)",
    )
    p_restructure.set_defaults(func=cmd_restructure)

    # install-skill
    p_skill = sub.add_parser(
        "install-skill",
        help="Install the bundled Claude Code skill for oopsie-data",
        description=(
            "Copy the skill that ships with this package into your Claude Code configuration, "
            "so Claude knows how to drive the contributor workflow. Entirely optional: nothing "
            "else in oopsie-data needs Claude, and no files are written anywhere unless you run "
            "this command. Installs to ~/.claude/skills/ by default, or ./.claude/skills/ with "
            "--project, where it can be committed and shared with collaborators."
        ),
    )
    p_skill.add_argument(
        "--project",
        action="store_true",
        help="Install into ./.claude/skills/ instead of your home directory",
    )
    p_skill.add_argument(
        "--force", action="store_true", help="Overwrite an existing installation of the skill"
    )
    p_skill.set_defaults(func=cmd_install_skill)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    # Export before any command runs, so library code resolving config paths sees it.
    if args.config_dir is not None:
        os.environ[ENV_CONFIG_DIR] = str(Path(args.config_dir).expanduser().resolve())
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        return 130
    except RuntimeError as e:
        # Config/auth problems already carry an actionable message; don't dump a traceback.
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
