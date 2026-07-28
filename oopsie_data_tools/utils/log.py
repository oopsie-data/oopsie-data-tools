"""Attach a log file to a named logger, without stacking duplicate handlers."""

from __future__ import annotations

import logging
import os


def setup_logger(name: str, log_file: str | os.PathLike) -> logging.Logger:
    """Add a file handler for ``log_file`` to the named logger, once.

    Pass ``__name__`` of the calling module so the logger stays within the
    ``oopsie_data_tools`` hierarchy and does not affect other packages.

    Calling this repeatedly for the same file is safe — validation calls it once per
    episode. Calling it for a *different* file adds a second handler rather than being
    ignored: the guard used to be "does this logger have any handler at all", so the second
    ``--log-path`` in a process silently wrote nothing.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    target = os.path.abspath(os.fspath(log_file))
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and (
            os.path.abspath(handler.baseFilename) == target
        ):
            return logger

    handler = logging.FileHandler(target)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
    )
    logger.addHandler(handler)
    return logger
