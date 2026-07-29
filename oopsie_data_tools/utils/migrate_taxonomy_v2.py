"""Offline migration of annotation attrs from taxonomy v1 to v2.

Deliberately **not** wired into the ``oopsie-data`` CLI. Readers upcast v1 files on the fly
(see :func:`~oopsie_data_tools.annotation_tool.annotation_schema.read_annotation_attrs`), so
nobody *needs* to run this — it exists for the one-off case of normalizing a directory in
place, and making it a supported command would imply an ongoing guarantee we do not want.

Run it ad hoc::

    python -m oopsie_data_tools.utils.migrate_taxonomy_v2 <path>            # dry run
    python -m oopsie_data_tools.utils.migrate_taxonomy_v2 <path> --apply

The mapping tables live in ``annotation_schema`` rather than here, so this and the read-time
upcast cannot disagree about what a v1 value means.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import h5py

from oopsie_data_tools.annotation_tool.annotation_schema import (
    ANNOTATION_SCHEMA_V2,
    FAILURE_CATEGORIES,
    OUTCOME_SUCCESS,
    SEVERITIES,
    TAXONOMY_SCHEMA_V2,
    V1_SUCCESS_CATEGORY_TO_OUTCOME,
    as_value_list,
    decode_attr,
    normalize_category,
    normalize_severity,
    parse_taxonomy,
    success_to_outcome,
)
from oopsie_data_tools.utils.h5 import find_episode_files


def unmapped_values(attrs: Mapping[str, Any]) -> list:
    """v1 category/severity values with no slug, reported so an operator can look at them.

    These are preserved verbatim rather than folded into ``other``: they are real human
    labels, and relabelling one on a guess is not something a later run could undo.
    """
    taxonomy = parse_taxonomy(attrs.get("taxonomy", ""))
    out = []
    raw_categories = taxonomy.get("failure_category", taxonomy.get("side_effect_category"))
    for value in as_value_list(raw_categories):
        if normalize_category(value) not in FAILURE_CATEGORIES:
            out.append(decode_attr(value))
    severity = taxonomy.get("severity")
    if severity is not None and decode_attr(severity):
        if normalize_severity(severity) not in SEVERITIES:
            out.append(decode_attr(severity))
    return out


def is_already_v2(attrs: Mapping[str, Any]) -> bool:
    """Whether this subgroup is already v2, by schema string or by a usable outcome."""
    if decode_attr(attrs.get("schema")) == ANNOTATION_SCHEMA_V2:
        return True
    outcome = decode_attr(parse_taxonomy(attrs.get("taxonomy", "")).get("outcome")).lower()
    return outcome in OUTCOME_SUCCESS


def migrate_annotation_attrs(attrs: Mapping[str, Any]) -> "dict | None":
    """v1 attr set -> the v2 attr set, or ``None`` when there is nothing to do.

    ``None`` covers both "already v2" and "not actually an annotation", which keeps the
    migration idempotent: a second run rewrites nothing.

    Pure — takes and returns plain mappings — so the mapping can be tested without HDF5.
    """
    if is_already_v2(attrs):
        return None
    # A subgroup with neither a success score nor a taxonomy is not an annotation. Inventing
    # a success for it would turn a file that fails strict validation into one that passes.
    if attrs.get("success") is None and not decode_attr(attrs.get("taxonomy", "")):
        return None

    taxonomy = parse_taxonomy(attrs.get("taxonomy", ""))

    outcome = success_to_outcome(attrs.get("success"))
    if outcome is None:
        return None
    legacy_category = decode_attr(taxonomy.get("success_category")).lower()
    if outcome == "success" and legacy_category:
        outcome = V1_SUCCESS_CATEGORY_TO_OUTCOME.get(legacy_category, outcome)

    raw_categories = taxonomy.get("failure_category", taxonomy.get("side_effect_category"))

    out = dict(attrs)
    out["schema"] = ANNOTATION_SCHEMA_V2
    out["taxonomy_schema"] = TAXONOMY_SCHEMA_V2
    # The v1 spelling is dropped rather than left alongside the new one, so no file ever
    # carries both and readers never have to pick.
    out.pop("failure_description", None)
    out["episode_description"] = decode_attr(attrs.get("failure_description", ""))
    out["taxonomy"] = json.dumps(
        {
            "outcome": outcome,
            "failure_category": [
                normalize_category(c) for c in as_value_list(raw_categories)
            ],
            "severity": normalize_severity(taxonomy.get("severity", "")),
        },
        ensure_ascii=False,
    )
    return out


def migrate_h5_file(h5_path: "str | Path", *, apply: bool = True) -> dict:
    """Migrate every annotator subgroup in one episode file.

    Returns a report: ``{"path", "migrated": [...], "skipped": {annotator: reason},
    "unmapped": {annotator: [...]}, "error": str | None}``.
    """
    path = Path(h5_path)
    report: dict = {
        "path": str(path),
        "migrated": [],
        "skipped": {},
        "unmapped": {},
        "error": None,
    }

    try:
        # Read-only first, so a dry run — and a file with nothing to migrate — never opens
        # for write and never touches mtime.
        with h5py.File(path, "r") as f:
            ea = f.get("episode_annotations")
            if not isinstance(ea, h5py.Group):
                report["skipped"]["*"] = "no episode_annotations group"
                return report
            planned = {}
            for name in ea.keys():
                sub = ea[name]
                if not isinstance(sub, h5py.Group):
                    report["skipped"][name] = "not a group"
                    continue
                attrs = dict(sub.attrs)
                new_attrs = migrate_annotation_attrs(attrs)
                if new_attrs is None:
                    report["skipped"][name] = (
                        "already v2" if is_already_v2(attrs) else "not an annotation"
                    )
                    continue
                leftover = unmapped_values(attrs)
                if leftover:
                    report["unmapped"][name] = leftover
                planned[name] = (attrs, new_attrs)

        if not planned or not apply:
            report["migrated"] = sorted(planned)
            return report

        with h5py.File(path, "r+") as f:
            ea = f["episode_annotations"]
            for name, (old_attrs, new_attrs) in planned.items():
                sub = ea[name]
                for key in set(old_attrs) - set(new_attrs):
                    del sub.attrs[key]
                for key, value in new_attrs.items():
                    sub.attrs[key] = value
        report["migrated"] = sorted(planned)
    except Exception as e:  # noqa: BLE001 - one bad file must not stop the walk
        report["error"] = f"{type(e).__name__}: {e}"

    return report


def migrate_tree(root: "str | Path", *, apply: bool = True) -> list:
    """Migrate every episode file under *root*; one report per file."""
    return [migrate_h5_file(p, apply=apply) for p in find_episode_files(Path(root))]
