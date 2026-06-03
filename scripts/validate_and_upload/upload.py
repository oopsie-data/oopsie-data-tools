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
import yaml
import os
import argparse
from pathlib import Path
from oopsie_tools.utils.validation.validation_utils import validate_h5_file, validate_session_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Read lab_id from configs/contributor_config.yaml
try:
    config_path = (
        Path(__file__).resolve().parent.parent.parent
        / "configs"
        / "contributor_config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        lab_id = config.get("lab_id", "").strip()
        huggingface_token = config.get("huggingface_token", "").strip()
        if not lab_id:
            raise ValueError(
                "lab_id must be set in configs/contributor_config.yaml"
            )
except Exception as e:
    raise RuntimeError(
        f"Could not read lab_id from configs/contributor_config.yaml: {e}"
    )

LAB_ID = lab_id
HF_TOKEN = huggingface_token
HF_REPO = f"OopsieData-Submissions/{LAB_ID}"

# ── Step 1: HuggingFace authentication ────────────────────────────────────────


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


def ensure_repo():
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    try:
        api.repo_info(repo_id=HF_REPO, repo_type="dataset")
        logger.info("[hf]    Repo already exists: https://huggingface.co/datasets/%s", HF_REPO)
    except Exception:
        logger.info("[hf]    Creating repo: %s", HF_REPO)
        api.create_repo(repo_id=HF_REPO, repo_type="dataset", private=False)
        logger.info("[hf]    Created: https://huggingface.co/datasets/%s", HF_REPO)


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


def upload_dataset(samples_dir: str, commit_message: str):
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    logger.info("[upload] Uploading %s → %s", samples_dir, HF_REPO)
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
        repo_id=HF_REPO,
        repo_type="dataset",
    )

    logger.info("[upload] Done!")
    logger.info("[upload] Dataset URL: https://huggingface.co/datasets/%s", HF_REPO)


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
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Robotic Failure Dataset — End-to-End Upload Pipeline")
    logger.info("=" * 60)

    # 1. Auth
    hf_login(HF_TOKEN)

    samples_dir = os.path.abspath(os.path.normpath(args.path))
    if args.episode_id is None:
        dir_name = os.path.basename(samples_dir.rstrip(os.sep)) or samples_dir
        commit_message = f"Add {dir_name}"
    else:
        commit_message = f"Add episode {args.episode_id}"

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

    # 4 + 5. Create repo and upload
    if not args.skip_upload:
        ensure_repo()
        upload_dataset(samples_dir, commit_message)
    else:
        logger.info("[upload] Skipped (--skip_upload).")


if __name__ == "__main__":
    main()
