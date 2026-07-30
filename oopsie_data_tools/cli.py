"""``oopsie-data`` command line interface.

Single entry point for the contributor workflow; every capability lives here rather than
in a parallel collection of scripts.

Credentials and robot profiles are looked up through separate chains (see
:mod:`oopsie_data_tools.utils.paths`); ``--config-dir`` overrides the credential location
for one invocation, and ``oopsie-data show-config`` prints what is in effect right now.

Command modules are imported inside the command bodies, so ``--help`` does not pay for
h5py, huggingface_hub or flask.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import sys
import textwrap
from pathlib import Path

import click

# Imported eagerly, unlike the heavier command modules below: the parser needs the agent
# names to build --agent's choices, and this module costs nothing but the standard library.
from oopsie_data_tools.utils.claude_skill import AGENT_SKILL_DIRS as _AGENT_CHOICES
from oopsie_data_tools.utils.claude_skill import DEFAULT_AGENT as _DEFAULT_AGENT
from oopsie_data_tools.utils.hf_limits import BATCH_SIZE, FILE_LIMIT
from oopsie_data_tools.utils.paths import ENV_CONFIG_DIR

logger = logging.getLogger(__name__)

DIR = click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path)


# ── machine-readable output ───────────────────────────────────────────────────


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


def json_option(what: str):
    """Give a command a ``--json`` mode. Exit codes are unchanged; only the format differs.

    Commands whose result is a structure rather than a narrative accept this, for callers —
    scripts, CI, coding agents — that would otherwise have to parse prose. ``--json`` owns
    stdout, so logging is silenced for the duration and a RuntimeError that would normally
    be logged goes into the document instead.
    """

    def decorate(f):
        @functools.wraps(f)
        def wrapper(*args, as_json: bool, **kwargs):
            with _quiet_logging(as_json):
                try:
                    return f(*args, as_json=as_json, **kwargs)
                except RuntimeError as e:
                    if not as_json:
                        raise
                    _emit_json({"error": str(e)})
                    return 1

        return click.option(
            "--json",
            "as_json",
            is_flag=True,
            help=f"Print {what} as JSON on stdout instead of prose, for scripts and agents",
        )(wrapper)

    return decorate


# ── group ─────────────────────────────────────────────────────────────────────

_EPILOG = """\
A typical session, in order:

\b
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


@click.group(invoke_without_command=True, epilog=_EPILOG)
@click.version_option(package_name="oopsie-data-tools", prog_name="oopsie-data")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path),
    default=None,
    metavar="DIR",
    help=(
        "Directory holding contributor_config.yaml "
        f"(overrides ${ENV_CONFIG_DIR} for this invocation; robot profiles are unaffected)"
    ),
)
@click.pass_context
def cli(ctx: click.Context, config_dir: Path | None) -> None:
    """Record, annotate, validate and publish robotic manipulation rollouts.

    Start with 'oopsie-data init'.
    """
    # Export before any command runs, so library code resolving config paths sees it.
    if config_dir is not None:
        os.environ[ENV_CONFIG_DIR] = str(config_dir.expanduser().resolve())
    ctx.obj = {"config_dir_from_flag": config_dir is not None}

    # A bare 'oopsie-data' is someone looking for the commands, so show them.
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(2)


# ── init ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--lab-id", default=None, help="Lab id from the registration form")
@click.option("--hf-token", default=None, help="HuggingFace token to store")
@click.option(
    "--no-verify-token",
    is_flag=True,
    help="Do not check the token against the HuggingFace API",
)
@click.option("--force", is_flag=True, help="Overwrite an existing config without asking")
@click.pass_context
def init(ctx, lab_id, hf_token, no_verify_token, force):
    """Set up the contributor config (lab id + HuggingFace token).

    Interactive setup: picks a config directory and writes contributor_config.yaml with
    your lab id and HuggingFace token, verifying the token against the HuggingFace API.
    Values passed as flags are not prompted for, so a fully-flagged invocation runs
    unattended.

    \b
    Examples:
      oopsie-data init
      oopsie-data init --lab-id my_lab --hf-token hf_xxx   # unattended
      oopsie-data --config-dir ./cfg init                  # somewhere else
    """
    from oopsie_data_tools.init_wizard import run_init

    return run_init(
        # --config-dir only lives for this process, so init should offer to persist it.
        config_dir_from_flag=ctx.obj["config_dir_from_flag"],
        lab_id=lab_id,
        hf_token=hf_token,
        verify_token=not no_verify_token,
        force=force,
    )


# ── show-config ───────────────────────────────────────────────────────────────


def _config_state() -> dict:
    """Everything show-config reports, resolved once for both renderers below."""
    from oopsie_data_tools.utils import paths
    from oopsie_data_tools.utils.contributor_config import load_config_yaml

    config_path = paths.contributor_config_path()
    profiles_dir = paths.robot_profiles_dir()
    data, parse_error = load_config_yaml(config_path)
    env_token = os.environ.get("HF_TOKEN", "").strip()
    return {
        "config_path": config_path,
        "config_dirs": paths.config_search_dirs(),
        "parse_error": parse_error,
        "lab_id": str(data.get("lab_id") or "").strip(),
        "env_token": env_token,
        "token": env_token or str(data.get("huggingface_token") or "").strip(),
        "profiles_dir": profiles_dir,
        "profile_dirs": paths.profiles_search_dirs(),
        "profiles": (
            sorted(p.name for p in profiles_dir.glob("*.yaml")) if profiles_dir.is_dir() else []
        ),
        "write_profiles_dir": paths.write_profiles_dir(),
    }


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


def _print_config_prose(state: dict, show_token: bool) -> None:
    from oopsie_data_tools.init_wizard import _mask
    from oopsie_data_tools.utils import paths

    config_path, profiles_dir = state["config_path"], state["profiles_dir"]

    print("\nCredentials — contributor_config.yaml")
    print(f"  ${paths.ENV_CONFIG_DIR} = {os.environ.get(paths.ENV_CONFIG_DIR) or '(not set)'}\n")
    _print_chain(
        state["config_dirs"],
        config_path.parent if config_path.is_file() else None,
        _describe_credential_dir,
    )
    print()
    if config_path.is_file():
        _print_field("Reading", str(config_path))
    else:
        print(f"  No config found. 'oopsie-data init' would write {config_path}")
    if state["parse_error"]:
        print(f"  Could not parse the file: {state['parse_error']}")

    _print_field("lab_id", state["lab_id"] or "(not set)")
    if state["token"]:
        shown = state["token"] if show_token else _mask(state["token"])
        source = (
            "$HF_TOKEN, overrides the config" if state["env_token"] else "contributor_config.yaml"
        )
        _print_field("hf_token", f"{shown}   (from {source})")
    else:
        _print_field("hf_token", "(not set — set one in the config or export $HF_TOKEN)")
    if state["lab_id"]:
        _print_field("uploads to", f"OopsieData-Submissions/{state['lab_id']}")

    print("\nRobot profiles")
    print(f"  ${paths.ENV_PROFILES_DIR} = {os.environ.get(paths.ENV_PROFILES_DIR) or '(not set)'}\n")
    _print_chain(
        state["profile_dirs"],
        profiles_dir if profiles_dir.is_dir() else None,
        _describe_profiles_dir,
    )
    print()
    if profiles_dir.is_dir():
        _print_field("Reading", str(profiles_dir))
        if state["profiles"]:
            wrapped = textwrap.wrap(
                ", ".join(state["profiles"]), width=88, subsequent_indent=" " * _FIELD_INDENT
            )
            _print_field("profiles", wrapped[0])
            for line in wrapped[1:]:
                print(line)
    else:
        print(f"  No profile directory found. A new one belongs in {state['write_profiles_dir']}")
    print(
        "\n  Profiles are usually loaded by explicit path — load_robot_profile(<path>) —\n"
        "  which ignores the lookup above. Paths here are relative to the current directory."
    )

    if not show_token and state["token"]:
        print("\nRe-run with --show-token to print the token in full.")


@cli.command("show-config")
@click.option("--show-token", is_flag=True, help="Print the HuggingFace token in full")
@json_option("the resolved config, both search paths, and what is in effect")
def show_config(show_token, as_json):
    """Show where configs are read from, and the current lab id and token.

    Print every location searched for the contributor config and for robot profiles, which
    one is currently being used, and the lab id and HuggingFace token in effect. Read-only:
    use 'oopsie-data init' to change anything. The token is masked unless --show-token is
    given.

    \b
    Examples:
      oopsie-data show-config
      oopsie-data --config-dir ./cfg show-config   # try a different location
    """
    from oopsie_data_tools.init_wizard import _mask
    from oopsie_data_tools.utils import paths

    state = _config_state()
    if not as_json:
        _print_config_prose(state, show_token)
        return 0

    token, lab_id = state["token"], state["lab_id"]
    _emit_json(
        {
            "credentials": {
                "env_var": paths.ENV_CONFIG_DIR,
                "env_value": os.environ.get(paths.ENV_CONFIG_DIR),
                "search_path": [str(p) for p in state["config_dirs"]],
                "active": str(state["config_path"]) if state["config_path"].is_file() else None,
                "parse_error": state["parse_error"],
                "lab_id": lab_id or None,
                # Masked unless asked, so --json is safe to paste into a bug report.
                "hf_token": (token if show_token else _mask(token)) if token else None,
                "hf_token_source": (
                    None
                    if not token
                    else ("HF_TOKEN" if state["env_token"] else "contributor_config.yaml")
                ),
                "uploads_to": f"OopsieData-Submissions/{lab_id}" if lab_id else None,
            },
            "robot_profiles": {
                "env_var": paths.ENV_PROFILES_DIR,
                "env_value": os.environ.get(paths.ENV_PROFILES_DIR),
                "search_path": [str(p) for p in state["profile_dirs"]],
                "active": (
                    str(state["profiles_dir"]) if state["profiles_dir"].is_dir() else None
                ),
                "profiles": state["profiles"],
                "write_target": str(state["write_profiles_dir"]),
            },
        }
    )
    return 0


# ── new-profile ───────────────────────────────────────────────────────────────


@cli.command("new-profile")
@click.option("--name", default="robot_profile", help="File name without .yaml")
@click.option(
    "--dir",
    "directory",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to write into (default: ./robot_profiles, or $OOPSIE_ROBOT_PROFILES_DIR)",
)
@click.option("--force", is_flag=True, help="Overwrite an existing profile of the same name")
def new_profile(name, directory, force):
    """Write a starter robot profile you can fill in.

    Write a commented robot-profile skeleton into your project, by default
    ./robot_profiles/. Profiles belong next to the robot code that loads them, not in your
    user config directory, and are never read out of the installed package. The skeleton
    does not load until you fill in the required fields — that is deliberate, so a
    half-edited profile cannot stamp placeholder metadata into your recorded episodes.

    \b
    Examples:
      oopsie-data new-profile
      oopsie-data new-profile --name franka_pick --dir ./configs/robot_profiles
    """
    from oopsie_data_tools.utils import paths
    from oopsie_data_tools.utils.robot_profile.template import PROFILE_TEMPLATE

    target_dir = directory if directory is not None else paths.write_profiles_dir()
    target = target_dir / f"{name}.yaml"

    if target.exists() and not force:
        logger.error("%s already exists. Pass --force to overwrite it.", target)
        return 1

    target_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(PROFILE_TEMPLATE, encoding="utf-8")
    logger.info("Wrote %s. Fill in the required fields; it will not load until you do.", target)
    return 0


# ── annotate ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--samples-dir",
    type=click.Path(path_type=Path),
    default=Path("samples"),
    metavar="DIR",
    help="Directory containing saved MP4s (and HDF5 episodes) (default: ./samples)",
)
@click.option(
    "--annotator-name",
    default=None,
    metavar="NAME",
    help="Annotator name to stamp into saved annotations (prompted for if omitted)",
)
@click.option("--port", type=int, default=5001, metavar="N", help="Server port (default: 5001)")
@click.option("--no-browser", is_flag=True, help="Do not open a browser window on start")
@click.option(
    "--with-rollouts",
    is_flag=True,
    help="Enable in-the-loop rollout mode (HDF5 browser + annotation form + rollouts)",
)
def annotate(samples_dir, annotator_name, port, no_browser, with_rollouts):
    """Launch the web annotation UI over a directory of episodes.

    Serve the annotation UI for the episodes in --samples-dir and open it in a browser.
    Annotations are written back into the HDF5 files under your annotator name, so the same
    session can be annotated by several people. Runs until you stop it with Ctrl-C.

    \b
    Examples:
      oopsie-data annotate --samples-dir ./samples --annotator-name alex
      oopsie-data annotate --samples-dir ./samples --port 8080 --no-browser
    """
    from oopsie_data_tools.annotation_tool.annotator_server import run_server

    name = (annotator_name or "").strip()
    if not name:
        # Checked before prompting: click would otherwise write a prompt nobody can answer
        # and abort, which reads as an interrupted run rather than a missing argument.
        if not sys.stdin.isatty():
            logger.error("--annotator-name is required when stdin is not a terminal.")
            return 1
        name = click.prompt("Annotator name (stamped into saved annotations)").strip()
    if not name:
        logger.error("An annotator name is required. Pass --annotator-name.")
        return 1

    return run_server(
        samples_dir=samples_dir,
        annotator_name=name,
        port=port,
        open_browser=not no_browser,
        with_rollouts=with_rollouts,
    )


# ── validate ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--path",
    required=True,
    metavar="PATH",
    help="A single .h5 file, or a session directory containing .h5 files",
)
@click.option(
    "--episode-id",
    default=None,
    metavar="ID",
    help="Zero-padded episode id (e.g. 000001) to validate just that episode inside --path",
)
@click.option("--log-path", default=None, metavar="FILE", help="Also write logs to this file")
@json_option("one record per episode, with the error that rejected it")
def validate(path, episode_id, log_path, as_json):
    """Validate episodes against the oopsiedata schema.

    Check a single .h5 episode, or every episode in a session directory, against the
    oopsiedata_format_v1 schema — the same check 'oopsie-data upload' runs before
    publishing. Every failure is reported with the episode and field that caused it. Exits 0
    if everything passed, 1 otherwise, so it is usable in a script.

    \b
    Examples:
      oopsie-data validate --path ./samples
      oopsie-data validate --path ./samples --episode-id 000001
      oopsie-data validate --path ./samples --log-path validate.log
    """
    from oopsie_data_tools.utils.hf_upload import run_validation

    target = os.path.abspath(path)
    if episode_id:
        target = os.path.join(target, f"{episode_id}.h5")
    # Checked here rather than by click.Path so --json can report it as a document.
    if not os.path.exists(target):
        if as_json:
            _emit_json({"path": target, "error": "path does not exist", "episodes": []})
            return 1
        logger.error("Path does not exist: %s", target)
        return 1

    if not as_json:
        return run_validation(path, episode_id, log_path)

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
        "[restructure] Writing a restructured copy to %s; the original is unchanged.", output
    )

    if run_restructure(source, output, assume_yes=True) != 0:
        logger.error("Aborting: the session could not be restructured.")
        return None

    # A directory holding no episodes cannot be split by episode, so it survives the copy
    # still over the file limit — and uploading it would fail at the Hub.
    if check_folder_size(str(output), suggest_fix=False):
        logger.error(
            "Aborting: %s is still over the file limit after restructuring. "
            "Reorganise the directories listed above by hand.",
            output,
        )
        return None

    logger.info("[restructure] Continuing with %s", output)
    return str(output)


@cli.command()
@click.option(
    "--path",
    required=True,
    type=DIR,
    metavar="DIR",
    help="Session directory containing formatted episode files",
)
@click.option(
    "--episode-id",
    default=None,
    metavar="ID",
    help="Zero-padded episode id to validate; if omitted, every *.h5 in --path is validated",
)
@click.option("--skip-validate", is_flag=True, help="Skip validation before uploading")
@click.option(
    "--skip-upload", is_flag=True, help="Run validation and checks only, do not upload"
)
@click.option(
    "--with-restructure",
    is_flag=True,
    help=(
        "If a directory exceeds the HuggingFace file limit, write a restructured copy to "
        "<path>_restructured and upload that instead of aborting (the original is left "
        "untouched, so this needs room for a second copy)"
    ),
)
@click.option("--log-path", default=None, metavar="FILE", help="Also write logs to this file")
@click.option(
    "--strict-diversity",
    is_flag=True,
    help="Treat low task/annotation diversity warnings as a hard error",
)
def upload(
    path, episode_id, skip_validate, skip_upload, with_restructure, log_path, strict_diversity
):
    """Validate and upload a session directory to HuggingFace.

    Validate a session directory and upload it to your lab's HuggingFace dataset repo,
    OopsieData-Submissions/<lab_id>, creating the repo on first use. Credentials come from
    contributor_config.yaml, with $HF_TOKEN overriding the stored token. Nothing is uploaded
    unless validation passes and the directory layout is within HuggingFace's per-directory
    file limit.

    \b
    Examples:
      oopsie-data upload --path ./samples
      oopsie-data upload --path ./samples --skip-upload    # dry run: checks only
      oopsie-data upload --path ./samples --with-restructure
    """
    from oopsie_data_tools.utils.hf_upload import (
        check_folder_size,
        hf_login,
        require_repo,
        resolve_hf_target,
        run_validation,
        upload_dataset,
    )
    from oopsie_data_tools.utils.validation.diversity import check_diversity

    samples_dir = str(path.resolve())

    # HF enforces a per-directory file limit. Only relevant when something is actually going
    # to be uploaded — with --skip-upload this is a validation run, and a layout HF would
    # reject is not a reason to fail it. Asking for --with-restructure asks for the layout to
    # be fixed, so it runs either way.
    if not skip_upload or with_restructure:
        if check_folder_size(samples_dir, suggest_fix=not with_restructure):
            if not with_restructure:
                return 1
            restructured = _restructure_for_upload(samples_dir)
            if restructured is None:
                return 1
            # Everything downstream must see the copy: validation reads the rewritten video
            # paths, and it is the copy that gets uploaded.
            samples_dir = restructured

    if not skip_validate and run_validation(samples_dir, episode_id, log_path) != 0:
        logger.error("Aborting upload due to validation failure. Fix the dataset format and retry.")
        return 1

    # Low-diversity warning (advisory unless --strict-diversity). Reads attrs only.
    if check_diversity(samples_dir) and strict_diversity:
        logger.error("Aborting: low-diversity warnings present and --strict-diversity is set.")
        return 1

    if skip_upload:
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


@cli.command()
@click.option(
    "--lab-id",
    default=None,
    metavar="ID",
    help="Lab id to query (default: the lab_id in your contributor config)",
)
def submissions(lab_id):
    """Show what your lab has already uploaded to HuggingFace.

    Report episode, video and file counts in OopsieData-Submissions/<lab_id> without
    downloading anything. A repo that does not exist yet is not an error — it is created on
    your first successful upload.
    """
    from oopsie_data_tools.utils.hf_upload import query_submissions

    return query_submissions(lab_id.strip() if lab_id else None)


# ── inspect ───────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path", metavar="FILE")
@json_option("the full group/dataset/attribute tree")
def inspect(path, as_json):
    """Dump the structure of an HDF5 episode.

    Print every group, dataset, shape, dtype and attribute in an HDF5 file. This is a
    debugging aid, not a validator: it makes no assumptions about the schema, so it works
    just as well on a file that 'oopsie-data validate' rejects — which is usually how you
    find out why it was rejected.

    \b
    Examples:
      oopsie-data inspect ./samples/000000.h5
      oopsie-data inspect ./samples/000000.h5 | less
    """
    from oopsie_data_tools.utils.h5_inspect import inspect_h5, inspect_h5_structure

    # Checked here rather than by click.Path so --json can report it as a document.
    if not os.path.isfile(path):
        if as_json:
            _emit_json({"path": path, "error": "not a file"})
            return 1
        logger.error("Not a file: %s", path)
        return 1

    if as_json:
        _emit_json(inspect_h5_structure(path))
    else:
        inspect_h5(path)
    return 0


# ── restructure ───────────────────────────────────────────────────────────────


@cli.command()
@click.option("--source", required=True, type=DIR, metavar="DIR",
              help="Session directory to restructure (its whole tree is scanned)")
@click.option("--output", type=click.Path(path_type=Path), default=None, metavar="DIR",
              help="Destination for the restructured copy (default: <source>_restructured)")
@click.option("--yes", "-y", is_flag=True,
              help="Skip the confirmation prompt (this copies the whole dataset)")
def restructure(source, output, yes):
    """Split an oversized session directory into numbered subfolders.

    Copy a session into a new folder in which every directory that exceeds HuggingFace's
    per-directory limit of {limit:,} files — the one 'oopsie-data upload' refuses on — has
    been split into numbered subfolders of at most {batch} episodes each, wherever in the
    tree that directory sits. Directories already under the limit are copied through
    unchanged.

    Non-destructive: the source is never modified, not even the HDF5 files, whose stored
    video paths are rewritten only in the copy. That means you need room for a second copy
    of the session, and you delete the original yourself once you have checked the output.
    'oopsie-data upload --with-restructure' does all of this as one step.

    \b
    Examples:
      oopsie-data restructure --source ./samples
      oopsie-data restructure --source ./samples --output /big/disk/samples_split
      oopsie-data restructure --source ./samples --yes   # no prompt, for scripts
    """
    from oopsie_data_tools.utils.restructure import run_restructure

    return run_restructure(source, output, assume_yes=yes)


restructure.help = restructure.help.format(limit=FILE_LIMIT, batch=BATCH_SIZE)


# ── install-skill ─────────────────────────────────────────────────────────────


@cli.command("install-skill")
@click.option(
    "--agent",
    type=click.Choice(list(_AGENT_CHOICES)),
    default=_DEFAULT_AGENT,
    help=(
        f"Which agent's skills directory to install into (default: {_DEFAULT_AGENT}); "
        "'agents' writes to shared ./.agents/skills/; 'none' writes to plain ./skills/"
    ),
)
@click.option(
    "--user", is_flag=True,
    help="Install under your home directory instead of the working directory",
)
@click.option("--force", is_flag=True, help="Overwrite an existing installation of the skill")
@click.option(
    "--check",
    is_flag=True,
    help=(
        "Report installed copies and whether they came from this version of the package, "
        "instead of installing anything. Exits 1 if any copy is stale or missing"
    ),
)
def install_skill_cmd(agent, user, force, check):
    """Install the bundled agent skill for oopsie-data.

    Copy the skill that ships with this package into a directory your coding agent scans, so
    it knows how to drive the contributor workflow. SKILL.md is a shared format, so the same
    payload works for Claude Code, Cursor and Codex — --agent only chooses the destination.
    Claude uses .claude/skills/; Cursor and Codex share .agents/skills/. Defaults to
    ./.claude/skills/oopsie-data/; --user installs into your home directory instead.
    Entirely optional: nothing else in oopsie-data needs an agent, and no files are written
    anywhere unless you run this command.

    \b
    Examples:
      oopsie-data install-skill
      oopsie-data install-skill --user             # available in every project
      oopsie-data install-skill --agent cursor
      oopsie-data install-skill --agent codex
      oopsie-data install-skill --agent agents     # shared ./.agents/skills/
      oopsie-data install-skill --agent none       # plain ./skills/
      oopsie-data install-skill --check            # is an installed copy out of date?
    """
    from oopsie_data_tools.utils.claude_skill import check_installations, install_skill

    if check:
        return check_installations()
    return install_skill(user=user, force=force, agent=agent)


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return cli.main(args=argv, prog_name="oopsie-data", standalone_mode=False) or 0
    except click.exceptions.Exit as e:
        return e.exit_code
    except click.Abort:
        logger.info("Interrupted.")
        return 130
    except click.ClickException as e:
        e.show()
        return e.exit_code
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        return 130
    except RuntimeError as e:
        # Config/auth problems already carry an actionable message; don't dump a traceback.
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
