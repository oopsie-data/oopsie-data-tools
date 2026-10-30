"""Download an annotator's share of the annotation batches and return the annotated files.

Backs ``oopsie-data annotation download`` and ``oopsie-data annotation upload``
(:mod:`oopsie_data_tools.cli`). The coordinator's dataset repo holds one folder per
annotator, ``<folder>/<batch>/episodes/NNNN.h5`` plus videos, where ``<folder>`` is the
annotator's full name in lowercase ASCII with ``_`` separators. Annotated h5 files go
back as a pull request to a separate returns repo, at the same paths.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import h5py

logger = logging.getLogger(__name__)

BATCH_REPO = "OopsieData-Submissions/oopsie-annotation"
RETURNS_REPO = "OopsieData-Submissions/oopsie-annotation-returns"
STATE_FILE = "annotator.json"


def folder_name(name: str) -> str:
    """Full name as lowercase ASCII with ``_`` separators; must match the coordinator's."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def check_token(token: str) -> str:
    """The HuggingFace user name of ``token``; raises ``RuntimeError`` if it is rejected."""
    from huggingface_hub import whoami

    try:
        return whoami(token=token)["name"]
    except Exception as e:
        raise RuntimeError(f"HuggingFace rejected the token: {e}") from e


def _folder_files(api, repo: str, folder: str) -> list[str]:
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        entries = api.list_repo_tree(repo, path_in_repo=folder, repo_type="dataset", recursive=True)
        return [e.path for e in entries if hasattr(e, "blob_id")]
    except RepositoryNotFoundError as e:
        raise RuntimeError(
            f"Cannot read {repo}. Ask the coordinator to give your HuggingFace account access."
        ) from e
    except EntryNotFoundError:
        return []


def download_share(api, repo: str, name: str, local_dir: Path) -> Path:
    """Download the folder of ``name`` from ``repo`` into ``local_dir``; return that folder.

    Writes ``local_dir/annotator.json`` so ``upload_share`` knows the name and folder.
    """
    folder = folder_name(name)
    files = _folder_files(api, repo, folder) if folder else []
    if not files:
        raise RuntimeError(
            f"{repo} has no folder for {name!r}. Type your full name exactly as you gave it "
            "to the coordinator."
        )
    logger.info("[download] %d files for %s from %s", len(files), name, repo)
    api.snapshot_download(
        repo, repo_type="dataset", allow_patterns=[f"{folder}/*"], local_dir=local_dir
    )
    (local_dir / STATE_FILE).write_text(json.dumps({"name": name, "folder": folder}, indent=2))
    return local_dir / folder


def read_state(local_dir: Path) -> dict | None:
    path = local_dir / STATE_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def batch_dirs(share: Path) -> list[Path]:
    """``<batch>/episodes`` folders of a downloaded share, sorted by batch name."""
    return sorted(p for p in share.glob("*/episodes") if p.is_dir())


def _annotated(path: Path, name: str) -> bool:
    with h5py.File(path, "r") as f:
        ea = f.get("episode_annotations")
        return isinstance(ea, h5py.Group) and isinstance(ea.get(name), h5py.Group)


def annotation_progress(share: Path, name: str) -> dict[str, tuple[list[Path], int]]:
    """Batch -> (h5 files annotated by ``name``, number of h5 files)."""
    out = {}
    for episodes in batch_dirs(share):
        files = sorted(episodes.glob("*.h5"))
        out[episodes.parent.name] = ([p for p in files if _annotated(p, name)], len(files))
    return out


def upload_share(api, repo: str, share: Path, name: str, files: list[Path]) -> str:
    """Open a pull request on ``repo`` adding ``files`` under ``<folder>/``; return its URL."""
    from huggingface_hub.errors import RepositoryNotFoundError

    folder = share.name
    try:
        commit = api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=share,
            path_in_repo=folder,
            allow_patterns=[p.relative_to(share).as_posix() for p in files],
            create_pr=True,
            commit_message=f"Annotations of {name}",
        )
    except RepositoryNotFoundError as e:
        raise RuntimeError(
            f"Cannot open a pull request on {repo}. Ask the coordinator to give your "
            "HuggingFace account access."
        ) from e
    return commit.pr_url
