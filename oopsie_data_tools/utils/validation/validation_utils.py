"""Public validation API.

Composes episode_loader (file I/O, schema detection, video loading) and
episode_validator (semantic checks on loaded data) into the entry points
used by the CLI and tests.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from oopsie_data_tools.utils.h5 import find_episode_files
from oopsie_data_tools.utils.log import setup_logger
from oopsie_data_tools.utils.validation.episode_loader import load_episode_from_h5
from oopsie_data_tools.utils.validation.episode_validator import validate_episode
from oopsie_data_tools.utils.validation.errors import EpisodeValidationError

logger = logging.getLogger(__name__)


def validate_h5_file(
    h5_path: str,
    strict_annotation_check: bool = False,
    log_path: str | Path | None = None,
) -> bool:
    """Validate a single HDF5 episode file.

    Args:
        h5_path: Path to the .h5 file.
        strict_annotation_check: If True, require that annotations are present and non-empty.
        log_path: Optional file to also write log output to (mirrors ``validate_session_dir``).

    Returns:
        True if all checks pass.

    Raises:
        EpisodeValidationError: On the first validation failure. It subclasses
            ``AssertionError``, so ``except AssertionError`` still catches it.
    """
    if log_path is not None:
        setup_logger(__name__, log_path)
    data = load_episode_from_h5(h5_path)
    validate_episode(data, strict_annotation_check=strict_annotation_check)
    return True


def collect_validation_results(
    target: str, strict_annotation_check: bool = True
) -> list[dict]:
    """Validate a file or a directory and return one result record per episode.

    The same checks ``validate_session_dir`` runs, reported as data rather than log lines,
    so a caller can answer "which episodes failed, and on what" without parsing prose.
    Backs ``oopsie-data validate --json``.

    Each record carries ``episode`` (the file name), ``path``, ``passed``, and on failure
    ``error`` plus an ``error_type`` of ``validation`` for a rejected episode or
    ``unexpected`` for a bug in the validator — the same distinction the prose output
    draws, since only the first is the user's to fix.
    """
    target = os.path.abspath(os.path.normpath(target))
    if os.path.isfile(target):
        paths = [Path(target)]
    elif os.path.isdir(target):
        paths = find_episode_files(target)
    else:
        paths = []

    results = []
    for path in paths:
        record = {"episode": path.name, "path": str(path), "passed": True}
        try:
            validate_h5_file(str(path), strict_annotation_check=strict_annotation_check)
        except EpisodeValidationError as e:
            record.update(passed=False, error=str(e), error_type="validation")
        except Exception as e:
            record.update(passed=False, error=str(e), error_type="unexpected")
        results.append(record)
    return results


def validate_session_dir(session_dir: str, strict_annotation_check: bool = False, log_path: str | Path | None = None) -> int:
    """Validate every ``*.h5`` / ``*.hdf5`` file in a session directory.

    Returns:
        0 if all files passed, 1 if any failed or the directory is invalid.
    """
    if log_path is not None:
        setup_logger(__name__, log_path)

    session_path = os.path.abspath(os.path.normpath(session_dir))
    if not os.path.isdir(session_path):
        logger.error("Not a directory: %s", session_path)
        return 1

    h5_files = [str(p) for p in find_episode_files(session_path)]

    if not h5_files:
        logger.error("No .h5 or .hdf5 files found in %s", session_path)
        return 1

    logger.info("Validating %d HDF5 file(s) in: %s", len(h5_files), session_path)
    failures = 0
    for i, path in enumerate(h5_files, 1):
        name = os.path.basename(path)
        logger.info("[%d/%d] %s", i, len(h5_files), name)
        try:
            validate_h5_file(path, strict_annotation_check=strict_annotation_check)
            logger.info("%s passed", name)
        except EpisodeValidationError as e:
            failures += 1
            logger.error("%s failed: %s", name, e)
        except Exception as e:
            # Anything else is a bug in the validator rather than a bad episode, so it is
            # reported differently on purpose.
            failures += 1
            logger.error("%s unexpected error: %s", name, e)

    passed = len(h5_files) - failures
    logger.info("Summary: %d/%d passed", passed, len(h5_files))
    return 1 if failures else 0
