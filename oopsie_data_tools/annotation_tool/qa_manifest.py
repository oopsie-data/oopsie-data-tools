"""QA bundles: a zip of episodes plus ``qa_manifest.json`` holding one QA note per episode.

A bundle unpacks to ``<bundle_id>/qa_manifest.json`` and ``<bundle_id>/episodes/...``.
Each manifest entry's ``bundle_path`` is the episode's h5 path relative to the bundle
directory; ``dataset_repo`` and ``repo_path`` locate the same file in the source dataset.
QA notes are written only into the manifest, never into the episode files.
"""

from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_NAME = "qa_manifest.json"


class QABundleError(ValueError):
    pass


def unpack_bundle(zip_path: Path, out_dir: Path) -> Path:
    """Extract ``zip_path`` into ``out_dir`` and return the bundle directory.

    An already extracted bundle is reused as is, so a reviewer can resume and keep the
    notes already written into its manifest.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        manifests = [n for n in names if PurePosixPath(n).name == MANIFEST_NAME]
        if len(manifests) != 1:
            raise QABundleError(f"{zip_path} must contain exactly one {MANIFEST_NAME}")
        bundle_rel = PurePosixPath(manifests[0]).parent
        bundle_dir = (out_dir / bundle_rel).resolve()
        if (bundle_dir / MANIFEST_NAME).is_file():
            return bundle_dir

        root = out_dir.resolve()
        for name in names:
            try:
                (root / name).resolve().relative_to(root)
            except ValueError:
                raise QABundleError(f"unsafe path in {zip_path}: {name!r}") from None
        zf.extractall(root)
    return bundle_dir


def find_manifest(bundle_dir: Path) -> Path:
    path = bundle_dir / MANIFEST_NAME
    if not path.is_file():
        raise QABundleError(f"no {MANIFEST_NAME} in {bundle_dir}")
    return path


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write atomically so an interrupted save never truncates the reviewer's notes."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def entry_for(manifest: dict[str, Any], bundle_path: str) -> dict[str, Any] | None:
    for entry in manifest.get("episodes", []):
        if entry.get("bundle_path") == bundle_path:
            return entry
    return None


def set_reviewer(path: Path, reviewer: str) -> None:
    manifest = read_manifest(path)
    manifest["reviewer"] = reviewer
    write_manifest(path, manifest)


def set_note(path: Path, bundle_path: str, notes: str, reviewer: str) -> dict[str, Any]:
    """Store ``notes`` for the episode at ``bundle_path`` and return its updated entry."""
    manifest = read_manifest(path)
    entry = entry_for(manifest, bundle_path)
    if entry is None:
        raise KeyError(bundle_path)
    entry["qa_notes"] = notes
    entry["qa_reviewer"] = reviewer
    entry["qa_updated_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(path, manifest)
    return entry
