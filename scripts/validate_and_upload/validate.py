"""Validate HDF5 episodes against the oopsiedata schema.

Prefer the CLI, which this forwards to verbatim::

    oopsie-data validate --path /path/to/session_dir

Usage:
    python validate.py --path /path/to/session_dir          # all *.h5 in directory
    python validate.py --path /path/to/episode.h5           # single episode file

The re-exports this module used to carry are gone. They existed so tests could reach the
validation helpers through ``sys.path`` manipulation, which the tests no longer do. Import
from ``oopsie_data_tools.utils.validation.validation_utils`` instead.
"""

from __future__ import annotations

import sys

from oopsie_data_tools.cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    return cli_main(["validate", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
