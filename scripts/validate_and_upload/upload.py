"""Validate formatted robotic failure data and upload it to HuggingFace.

Prefer the CLI, which this forwards to verbatim::

    oopsie-data upload --path /path/to/formatted_data

This entry point exists for existing muscle memory and scripts. Every flag it used to
define is already accepted by ``oopsie-data upload``, including the underscore spellings
(``--episode_id``, ``--skip_validate``, ``--skip_upload``) and the ``-o``/``-e``/``-l``
short forms, so arguments are passed straight through rather than re-declared. This file
used to be a hand-maintained copy of the same pipeline, down to identical log strings, and
had already drifted from it.

Usage:
    python upload.py --path /path/to/formatted_data                     # validate + upload
    python upload.py --path /path/to/formatted_data --episode_id 000001 # single episode

Environment:
    HF_TOKEN  — override the token in the contributor config
"""

from __future__ import annotations

import sys

from oopsie_data_tools.cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    return cli_main(["upload", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
