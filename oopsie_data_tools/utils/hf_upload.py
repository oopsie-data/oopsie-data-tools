"""Validation + HuggingFace upload helpers.

Backs ``oopsie-data validate``, ``oopsie-data upload`` and ``oopsie-data submissions``
(:mod:`oopsie_data_tools.cli`), which is the only entry point — the parallel
``scripts/validate_and_upload/`` copies of this pipeline are gone.
"""

from __future__ import annotations

import logging
import os
from collections import Counter

from oopsie_data_tools.utils.contributor_config import read_contributor_config
from oopsie_data_tools.utils.hf_limits import FILE_LIMIT
from oopsie_data_tools.utils.log import setup_logger
from oopsie_data_tools.utils.validation.errors import EpisodeValidationError
from oopsie_data_tools.utils.validation.validation_utils import (
    validate_h5_file,
    validate_session_dir,
)

logger = logging.getLogger(__name__)


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
        except EpisodeValidationError as e:
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


def check_folder_size(samples_dir: str, suggest_fix: bool = True) -> list[tuple[str, int]]:
    """Find directories under ``samples_dir`` that exceed the HF per-directory file limit.

    Args:
        samples_dir: Session directory to check, recursively.
        suggest_fix: Log the command that repairs the layout. Set False when the caller is
            about to run it anyway (``upload --with-restructure``), so the output does not
            tell the user to do by hand what is happening on the next line.

    Returns:
        A list of ``(directory, file_count)`` pairs; empty if the layout is fine.
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
    if suggest_fix:
        logger.error(
            "[precheck] HuggingFace Hub enforces a per-directory file limit.\n"
            "           Restructure the folder first, then re-run the upload,\n"
            "           or pass --with-restructure to do both in one step:\n\n"
            "             oopsie-data restructure --source %s\n",
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


# ── Submissions query ─────────────────────────────────────────────────────────


def query_submissions(lab_id: str | None = None) -> int:
    """Report what has landed in ``OopsieData-Submissions/<lab_id>``, downloading nothing.

    Args:
        lab_id: Lab to query. Defaults to the ``lab_id`` in the contributor config.

    Returns:
        A shell-style exit code. A repo that does not exist yet is not an error — it is
        created on the first successful upload — so that case still returns 0.
    """
    from huggingface_hub import HfApi

    if lab_id:
        # An explicit --lab-id must not require a filled-in contributor config, but the
        # token still comes from it unless $HF_TOKEN overrides.
        try:
            _, config_token = read_contributor_config()
        except RuntimeError:
            config_token = ""
    else:
        lab_id, config_token = read_contributor_config()

    hf_token = os.environ.get("HF_TOKEN", "").strip() or config_token
    repo = f"OopsieData-Submissions/{lab_id}"
    api = HfApi(token=hf_token or None)

    try:
        api.repo_info(repo_id=repo, repo_type="dataset")
    except Exception:
        logger.info("No submissions repo found yet at https://huggingface.co/datasets/%s", repo)
        logger.info("(It is created automatically on your first successful upload.)")
        return 0

    files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    h5 = [f for f in files if f.endswith(".h5") or f.endswith(".hdf5")]
    mp4 = [f for f in files if f.endswith(".mp4")]
    by_dir = Counter(f.split("/")[0] if "/" in f else "(root)" for f in h5)

    logger.info("Repo:           https://huggingface.co/datasets/%s", repo)
    logger.info("Episodes (.h5): %d", len(h5))
    logger.info("Videos (.mp4):  %d", len(mp4))
    logger.info("Total files:    %d", len(files))
    if by_dir:
        logger.info("Episodes by top-level folder:")
        for name, count in sorted(by_dir.items()):
            logger.info("  %-32s %d", name, count)
    return 0
