"""Minimal read-only HuggingFace hub access over the stdlib.

Deliberately not ``huggingface_hub``: the pipeline needs exactly two operations — list
blob sizes and GET a file — and the repo's pinned Python 3.8 environment has no
third-party packages installed. Keeping this dependency-free means the download stage
runs before any venv exists.

Every call here is a GET. Nothing in this module can write to the hub.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from oopsie_data_tools.auto_annotate import net

API = "https://huggingface.co/api/datasets"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"

_USER_AGENT = "oopsie-auto-annotate/1.0 (read-only)"


def _request(url: str, token: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"Authorization": "Bearer " + token, "User-Agent": _USER_AGENT},
        method="GET",
    )


def blob_sizes(repo: str, token: str, timeout: int = 300) -> Dict[str, int]:
    """Map every file in the repo to its size in bytes."""
    url = f"{API}/{repo}?blobs=true"
    with urllib.request.urlopen(
        _request(url, token), timeout=timeout, context=net.context()
    ) as response:
        payload = json.load(response)
    return {
        sibling["rfilename"]: sibling.get("size") or 0
        for sibling in payload.get("siblings", [])
    }


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    repo: str,
    path: str,
    dest: Path,
    token: str,
    revision: str = "main",
    expect_size: Optional[int] = None,
    expect_sha256: Optional[str] = None,
    timeout: int = 600,
) -> str:
    """Fetch one file to ``dest``. Returns "skipped", "downloaded", or raises.

    A file already present with the expected size is left alone, so an interrupted run
    resumes cheaply. Content is written to a ``.part`` sibling and moved into place only
    after any checksum check passes, so a truncated transfer never looks complete.
    """
    if dest.exists() and expect_size is not None and dest.stat().st_size == expect_size:
        return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    url = RESOLVE.format(repo=repo, rev=revision, path=urllib.parse.quote(path))

    with urllib.request.urlopen(
        _request(url, token), timeout=timeout, context=net.context()
    ) as response:
        with partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)

    if expect_size is not None and partial.stat().st_size != expect_size:
        actual = partial.stat().st_size
        partial.unlink(missing_ok=True)
        raise IOError(f"{path}: size mismatch, expected {expect_size} got {actual}")

    # The index carries a sha256 for the .h5 files only; videos are checked by size.
    if expect_sha256:
        actual = sha256_of(partial)
        if actual != expect_sha256:
            partial.unlink(missing_ok=True)
            raise IOError(f"{path}: sha256 mismatch, expected {expect_sha256} got {actual}")

    partial.replace(dest)
    return "downloaded"
