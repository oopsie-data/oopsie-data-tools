import logging


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Add a file handler to a named logger.

    Pass ``__name__`` of the calling module so the logger stays within the
    ``oopsie_tools`` hierarchy and does not affect other packages.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
        )
        logger.addHandler(handler)
    return logger
