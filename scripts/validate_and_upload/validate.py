"""Validate HDF5 episodes against the oopsiedata schema.

Thin wrapper around the shared validation code — prefer the CLI:

    oopsie-data validate --path /path/to/session_dir

Usage:
    python validate.py --path /path/to/session_dir          # all *.h5 in directory
    python validate.py --path /path/to/episode.h5           # single episode file
"""

import argparse
import logging
import os
import sys

from oopsie_data_tools.utils.hf_upload import run_validation

# Re-exported for tests and existing callers that import them from this module.
from oopsie_data_tools.utils.validation.validation_utils import (  # noqa: F401
    validate_h5_file,
    validate_session_dir,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate oopsie episode HDF5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to a single .h5 file or a session directory containing .h5 files",
    )
    args = parser.parse_args()
    return run_validation(os.path.abspath(os.path.normpath(args.path)))


if __name__ == "__main__":
    sys.exit(main())
