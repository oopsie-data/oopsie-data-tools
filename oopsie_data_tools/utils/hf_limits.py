"""HuggingFace Hub layout limits, and how we split a session to stay under them.

These live apart from :mod:`oopsie_data_tools.utils.restructure` — their natural home —
only so that ``oopsie-data --help`` can quote the numbers without importing h5py, which
costs ~280 ms and is pure waste when all the user asked for was help text. Everything
else imports them from here too, so there is still exactly one definition of each.

Modules bind these names at import time, so a test that patches ``restructure.FILE_LIMIT``
changes the limit only for that module. Patch each module you want affected.
"""

from __future__ import annotations

#: Maximum files HuggingFace Hub accepts in a single directory. Exceeding it fails the
#: upload at the Hub, so ``hf_upload.check_folder_size`` refuses before transferring
#: anything.
FILE_LIMIT = 10_000

#: Episodes per numbered subfolder when splitting an oversized directory. A batch is this
#: many HDF5 files plus their videos, so it stays under FILE_LIMIT as long as episodes have
#: fewer than FILE_LIMIT / BATCH_SIZE - 1 cameras — roughly 19 at the values above.
BATCH_SIZE = 500
