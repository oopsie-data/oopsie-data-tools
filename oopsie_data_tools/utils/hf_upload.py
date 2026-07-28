"""Validation + HuggingFace upload helpers.

Shared by the ``oopsie-data`` CLI (``oopsie_data_tools.cli``) and the standalone
``scripts/validate_and_upload/upload.py`` entry point so both take the exact same
path through validation, repo creation and upload.
"""

from __future__ import annotations

import logging
import os

from oopsie_data_tools.utils.contributor_config import read_contributor_config
from oopsie_data_tools.utils.log import setup_logger
from oopsie_data_tools.utils.validation.validation_utils import (
    validate_h5_file,
    validate_session_dir,
)

logger = logging.getLogger(__name__)

FILE_LIMIT = 10_000
RESTRUCTURE_SCRIPT = "scripts/validate_and_upload/restructure_large_folder.py"


# ── HuggingFace authentication ────────────────────────────────────────────────


def resolve_hf_target() -> tuple[str, str]:
    """Resolve ``(hf_token, repo_id)`` lazily.

    Read only when actually uploading so validation and ``--help`` work on a fresh
    checkout without a filled-in contributor config. ``HF_TOKEN`` in the environment
    overrides the config token.
    """
    lab_id, config_token = read_contributor_config()
    token = os.environ.get("HF_TOKEN", "").strip() or config_token
    return token, f"OopsieData-Submissions/{lab_id}"


def hf_login(token: str) -> str:
    from huggingface_hub import login, whoami

    login(token=token, add_to_git_credential=False)
    info = whoami(token=token)
    logger.info("[auth]  Logged in as: %s", info["name"])
    return info["name"]


# ── Validation ────────────────────────────────────────────────────────────────


def run_validation(base_path: str, episode_id: str | None = None, log_path: str | None = None) -> int:
    """Validate a single episode or a session dir.

    Args:
        base_path: Session directory, or the directory holding ``<episode_id>.h5``.
        episode_id: Optional zero-padded episode id; validates just that episode.
        log_path: Optional file to mirror log output into.

    Returns:
        A shell-style exit code: 0 if everything passed, 1 otherwise.
    """
    # Route this module's pass/fail lines to the log file too, so --log-path captures
    # the single-file path (validate_session_dir already logs through its own logger).
    if log_path is not None:
        setup_logger(__name__, log_path)

    target = os.path.join(base_path, f"{episode_id}.h5") if episode_id else base_path
    if os.path.isfile(target):
        try:
            validate_h5_file(target, strict_annotation_check=True, log_path=log_path)
            logger.info("%s passed", os.path.basename(target))
            return 0
        except AssertionError as e:
            logger.error("Validation failed: %s", e)
            return 1
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            return 1

    if os.path.isdir(target):
        return validate_session_dir(target, strict_annotation_check=True, log_path=log_path)

    logger.error("Path does not exist: %s", target)
    return 1


# ── Repo creation ─────────────────────────────────────────────────────────────


def ensure_repo(api, repo: str) -> None:
    try:
        api.repo_info(repo_id=repo, repo_type="dataset")
        logger.info("[hf]    Repo already exists: https://huggingface.co/datasets/%s", repo)
    except Exception:
        logger.info("[hf]    Creating repo: %s", repo)
        api.create_repo(repo_id=repo, repo_type="dataset", private=False)
        logger.info("[hf]    Created: https://huggingface.co/datasets/%s", repo)


# ── Upload ────────────────────────────────────────────────────────────────────


def check_folder_size(samples_dir: str) -> list[tuple[str, int]]:
    """Find directories under ``samples_dir`` that exceed the HF per-directory file limit.

    Returns:
        A list of ``(directory, file_count)`` pairs; empty if the layout is fine.
        Offending directories are logged with the fix to apply.
    """
    oversized = [
        (dirpath, len(filenames))
        for dirpath, _, filenames in os.walk(samples_dir)
        if len(filenames) > FILE_LIMIT
    ]
    if not oversized:
        return []

    logger.error("[precheck] The following directories exceed %d files:", FILE_LIMIT)
    for d, n in oversized:
        logger.error("             %s  (%d files)", d, n)
    logger.error(
        "[precheck] HuggingFace Hub enforces a per-directory file limit.\n"
        "           Restructure the folder first, then re-run the upload:\n\n"
        "             python %s --source %s\n",
        RESTRUCTURE_SCRIPT,
        samples_dir,
    )
    return oversized


def upload_dataset(api, repo: str, samples_dir: str) -> None:
    logger.info("[upload] Uploading %s → %s", samples_dir, repo)
    logger.info("[upload] Files to upload:")
    total_bytes = 0
    for root, _, files in os.walk(samples_dir):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            rel = os.path.relpath(fpath, samples_dir)
            total_bytes += size
            logger.info("           %s  (%.1f MB)", rel, size / 1e6)

    logger.info("[upload] Total size: %.2f GB", total_bytes / 1e9)
    logger.info("[upload] Uploading (this may take several minutes)...")

    api.upload_large_folder(
        folder_path=samples_dir,
        repo_id=repo,
        repo_type="dataset",
    )

    logger.info("[upload] Done!")
    logger.info("[upload] Dataset URL: https://huggingface.co/datasets/%s", repo)

    # Post-upload confirmation: read the repo back so the user sees their data landed.
    try:
        remote_h5 = [
            f for f in api.list_repo_files(repo_id=repo, repo_type="dataset")
            if f.endswith(".h5")
        ]
        local_h5 = sum(
            1 for _r, _d, files in os.walk(samples_dir) for fn in files if fn.endswith(".h5")
        )
        logger.info(
            "[upload] Confirmed: %d episode(s) now in the repo (from %d local .h5).",
            len(remote_h5), local_h5,
        )
        if local_h5 and len(remote_h5) < local_h5:
            logger.warning(
                "[upload] Remote episode count (%d) is below local (%d) — "
                "re-run the upload if this is unexpected.",
                len(remote_h5), local_h5,
            )
    except Exception as e:
        logger.warning("[upload] Could not confirm upload via repo listing: %s", e)
