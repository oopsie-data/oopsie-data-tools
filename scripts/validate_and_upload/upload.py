"""
End-to-end script: validate formatted robotic failure data and upload to HuggingFace.

Steps:
    1. Authenticate with HuggingFace
    2. Validate the episode(s)
    3. Create HF dataset repo (if it doesn't exist)
    4. Upload dataset files

Usage:
    python upload.py --samples_dir /path/to/formatted_data          # validate all *.h5, upload whole folder
    python upload.py --samples_dir /path/to/formatted_data --episode_id 000001  # single episode

Environment:
    HF_TOKEN  — override the hardcoded token
"""
from __future__ import annotations

import logging
import sys
import os
import argparse
from pathlib import Path
from oopsie_tools.utils.contributor_config import read_contributor_config
from oopsie_tools.utils.log import setup_logger
from oopsie_tools.utils.validation.diversity import check_diversity
from oopsie_tools.utils.validation.validation_utils import validate_h5_file, validate_session_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Step 1: HuggingFace authentication ────────────────────────────────────────


def _resolve_hf_target() -> tuple[str, str]:
    """Resolve ``(hf_token, repo_id)`` lazily.

    Read only when actually uploading so validation, ``--skip_upload`` and ``--help``
    work on a fresh checkout without a filled-in contributor config. ``HF_TOKEN`` in the
    environment overrides the config token (as documented in the module docstring).
    """
    lab_id, config_token = read_contributor_config()
    token = os.environ.get("HF_TOKEN", "").strip() or config_token
    return token, f"OopsieData-Submissions/{lab_id}"


def hf_login(token: str):
    from huggingface_hub import login, whoami

    login(token=token, add_to_git_credential=False)
    info = whoami(token=token)
    logger.info("[auth]  Logged in as: %s", info["name"])
    return info["name"]


# ── Step 2: validation ────────────────────────────────────────────────────────


def _validate_import_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


def run_validation(base_path: str, episode_id: str, log_path: str | None=None) -> bool:
    # Route this module's pass/fail lines to the log file too, so --log-path captures
    # the single-file path (validate_session_dir already logs through its own logger).
    if log_path is not None:
        setup_logger(__name__, log_path)
    target = os.path.join(base_path, f"{episode_id}.h5") if episode_id else base_path
    if os.path.isfile(target):
        try:
            validate_h5_file(target, strict_annotation_check=True, log_path=log_path)
            logger.info("%s passed", os.path.basename(target))
            return 1
        except AssertionError as e:
            logger.error("Validation failed: %s", e)
            return 0
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            return 0

    if os.path.isdir(target):
        return validate_session_dir(target, strict_annotation_check=True, log_path=log_path)


# ── Step 3: create repo (if needed) ───────────────────────────────────────────


def ensure_repo(api, repo: str):
    try:
        api.repo_info(repo_id=repo, repo_type="dataset")
        logger.info("[hf]    Repo already exists: https://huggingface.co/datasets/%s", repo)
    except Exception:
        logger.info("[hf]    Creating repo: %s", repo)
        api.create_repo(repo_id=repo, repo_type="dataset", private=False)
        logger.info("[hf]    Created: https://huggingface.co/datasets/%s", repo)


# ── Step 4: upload ────────────────────────────────────────────────────────────


def check_folder_size(samples_dir: str) -> None:
    """Abort if any directory under samples_dir exceeds 10 000 files."""
    FILE_LIMIT = 10_000
    oversized = []
    for dirpath, _, filenames in os.walk(samples_dir):
        if len(filenames) > FILE_LIMIT:
            oversized.append((dirpath, len(filenames)))

    if not oversized:
        return

    restructure_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "restructure_large_folder.py"
    )
    logger.error("[precheck] The following directories exceed %d files:", FILE_LIMIT)
    for d, n in oversized:
        logger.error("             %s  (%d files)", d, n)
    logger.error(
        "[precheck] HuggingFace Hub enforces a per-directory file limit.\n"
        "           Restructure the folder first, then re-run the upload:\n\n"
        "             python %s --source %s\n",
        restructure_script,
        samples_dir,
    )
    sys.exit(1)


def upload_dataset(api, repo: str, samples_dir: str):
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


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Validate and upload robotic failure dataset to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--path",
        "-o",
        required=True,
        help="Base directory containing formatted episode files",
    )
    parser.add_argument(
        "--episode_id",
        "-e",
        default=None,
        help="Episode ID (zero-padded, e.g. 000001); if omitted, all *.h5 files in path are validated and uploaded",
    )
    parser.add_argument(
        "--skip_validate",
        action="store_true",
        help="Skip validation step before uploading",
    )
    parser.add_argument(
        "--skip_upload", action="store_true", help="Run validation only, do not upload"
    )
    parser.add_argument(
        "--log-path",
        "-l",
        default=None,
        help="Path to log file"
    )
    parser.add_argument(
        "--strict-diversity",
        action="store_true",
        help="Treat low task/annotation diversity warnings as a hard error (non-zero exit)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Robotic Failure Dataset — End-to-End Upload Pipeline")
    logger.info("=" * 60)

    samples_dir = os.path.abspath(os.path.normpath(args.path))

    # 2. Pre-upload folder-size check
    if not args.skip_upload:
        check_folder_size(samples_dir)

    # 3. Validate
    if not args.skip_validate:
        ok = run_validation(samples_dir, args.episode_id, args.log_path)
        if not ok:
            logger.error("Aborting upload due to validation failure. Fix the dataset format and retry.")
            sys.exit(1)
    else:
        logger.info("[validate] Skipped.")

    # 3b. Low-diversity warning (advisory unless --strict-diversity). Reads attrs only.
    diversity_warnings = check_diversity(samples_dir)
    if diversity_warnings and args.strict_diversity:
        logger.error("Aborting: low-diversity warnings present and --strict-diversity is set.")
        sys.exit(1)

    # 4 + 5. Authenticate, create repo, and upload (config + auth only needed to upload,
    # so validation / --skip_upload / --help work without a filled-in contributor config).
    if not args.skip_upload:
        from huggingface_hub import HfApi

        hf_token, hf_repo = _resolve_hf_target()
        hf_login(hf_token)
        api = HfApi(token=hf_token)
        ensure_repo(api, hf_repo)
        upload_dataset(api, hf_repo, samples_dir)
    else:
        logger.info("[upload] Skipped (--skip_upload).")


if __name__ == "__main__":
    main()
