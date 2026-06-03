"""Validate HDF5 episodes against the oopsiedata schema.

Usage:
    python validate.py /path/to/session_dir          # all *.h5 in directory
    python validate.py /path/to/episode.h5           # single episode file
"""

import argparse
import logging
import os
import sys

from oopsie_tools.utils.validation.validation_utils import validate_h5_file, validate_session_dir

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
        help="Path to a single .h5 file or a session directory containing .h5 files",
    )
    args = parser.parse_args()
    target = os.path.abspath(os.path.normpath(args.path))

    if os.path.isfile(target):
        try:
            validate_h5_file(target, strict_annotation_check=True)
            logger.info("%s passed", os.path.basename(target))
            return 0
        except AssertionError as e:
            logger.error("Validation failed: %s", e)
            return 1
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            return 1

    if os.path.isdir(target):
        return validate_session_dir(target, strict_annotation_check=True)

    logger.error("Path does not exist: %s", target)
    return 1


if __name__ == "__main__":
    sys.exit(main())
