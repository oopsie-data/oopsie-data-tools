"""Public validation API.

Composes episode_loader (file I/O, schema detection, video loading) and
episode_validator (semantic checks on loaded data) into the entry points
used by the CLI and tests.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

from oopsie_tools.utils.log import setup_logger
from oopsie_tools.utils.validation.episode_loader import load_episode_from_h5
from oopsie_tools.utils.validation.episode_validator import validate_episode

logger = logging.getLogger(__name__)


def validate_h5_file(h5_path: str, strict_annotation_check: bool = False) -> bool:
    """Validate a single HDF5 episode file.

    Args:
        h5_path: Path to the .h5 file.
        strict_annotation_check: If True, require that annotations are present and non-empty.

    Returns:
        True if all checks pass.

    Raises:
        AssertionError: On the first validation failure.
    """
    data = load_episode_from_h5(h5_path)
    validate_episode(data, strict_annotation_check=strict_annotation_check)
    return True


def validate_session_dir(session_dir: str, strict_annotation_check: bool = False, log_path: str | Path | None = None) -> int:
    """Validate every ``*.h5`` / ``*.hdf5`` file in a session directory.

    Returns:
        1 if all files passed, 0 if any failed or the directory is invalid.
    """
    if log_path is not None:
        setup_logger(__name__, log_path)

    session_path = os.path.abspath(os.path.normpath(session_dir))
    if not os.path.isdir(session_path):
        logger.error("Not a directory: %s", session_path)
        return 0

    # find all hdf5 files recursively in the session directory
    h5_files = [
        f
        for ext in ("*.h5", "*.hdf5")
        for f in glob.glob(os.path.join(session_path, "**", ext), recursive=True)
    ]

    if not h5_files:
        logger.error("No .h5 or .hdf5 files found in %s", session_path)
        return 0

    logger.info("Validating %d HDF5 file(s) in: %s", len(h5_files), session_path)
    failures = 0
    for i, path in enumerate(h5_files, 1):
        name = os.path.basename(path)
        logger.info("[%d/%d] %s", i, len(h5_files), name)
        try:
            validate_h5_file(path, strict_annotation_check=strict_annotation_check)
            logger.info("%s passed", name)
        except AssertionError as e:
            failures += 1
            logger.error("%s failed: %s", name, e)
        except Exception as e:
            failures += 1
            logger.error("%s unexpected error: %s", name, e)

    passed = len(h5_files) - failures
    logger.info("Summary: %d/%d passed", passed, len(h5_files))
    return 0 if failures else 1
