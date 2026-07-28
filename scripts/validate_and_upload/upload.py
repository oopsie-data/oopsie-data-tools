"""
End-to-end script: validate formatted robotic failure data and upload to HuggingFace.

Thin wrapper around the shared pipeline in ``oopsie_data_tools.utils.hf_upload`` —
prefer the CLI:

    oopsie-data upload --path /path/to/formatted_data

Steps:
    1. Authenticate with HuggingFace
    2. Validate the episode(s)
    3. Create HF dataset repo (if it doesn't exist)
    4. Upload dataset files

Usage:
    python upload.py --path /path/to/formatted_data                     # validate all *.h5, upload whole folder
    python upload.py --path /path/to/formatted_data --episode_id 000001 # single episode

Environment:
    HF_TOKEN  — override the token in configs/contributor_config.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from oopsie_data_tools.utils.hf_upload import (  # noqa: F401  (re-exported for existing callers)
    check_folder_size,
    ensure_repo,
    hf_login,
    resolve_hf_target,
    run_validation,
    upload_dataset,
)
from oopsie_data_tools.utils.validation.diversity import check_diversity

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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
    parser.add_argument("--log-path", "-l", default=None, help="Path to log file")
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
    if not args.skip_upload and check_folder_size(samples_dir):
        sys.exit(1)

    # 3. Validate
    if not args.skip_validate:
        if run_validation(samples_dir, args.episode_id, args.log_path) != 0:
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

        hf_token, hf_repo = resolve_hf_target()
        hf_login(hf_token)
        api = HfApi(token=hf_token)
        ensure_repo(api, hf_repo)
        upload_dataset(api, hf_repo, samples_dir)
    else:
        logger.info("[upload] Skipped (--skip_upload).")


if __name__ == "__main__":
    main()
