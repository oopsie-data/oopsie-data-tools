"""Query your lab's OopsieData submissions repo on HuggingFace for quick stats.

Lets a lab confirm what has actually landed in ``OopsieData-Submissions/<lab_id>``
without downloading anything.

Usage:
    python scripts/validate_and_upload/query_submissions.py
    python scripts/validate_and_upload/query_submissions.py --lab-id SomeOtherLab

Environment:
    HF_TOKEN  — override the config token.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter

from oopsie_tools.utils.contributor_config import read_contributor_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query your lab's OopsieData submissions repo for episode counts/stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--lab-id",
        default=None,
        help="Lab id to query (default: lab_id from configs/contributor_config.yaml)",
    )
    args = parser.parse_args()

    config_lab_id, config_token = read_contributor_config()
    lab_id = args.lab_id.strip() if args.lab_id else config_lab_id
    hf_token = os.environ.get("HF_TOKEN", "").strip() or config_token

    from huggingface_hub import HfApi

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


if __name__ == "__main__":
    sys.exit(main())
