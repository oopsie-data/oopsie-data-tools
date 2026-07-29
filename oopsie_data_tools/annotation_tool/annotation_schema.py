"""Single definition of the oopsie human-annotation vocabulary and its HDF5 shape.

Everything about the taxonomy that more than one module has to agree on lives here: the
schema strings, the four outcome slugs and their numeric ``success`` mapping, the v1 -> v2
upcast tables, and the read/write pair for an ``episode_annotations/<annotator>`` subgroup.

Kept to json + h5py so the in-the-loop recorder, the converter helpers, the validator, the
web annotator and the offline migration can all share it without importing each other.

Stored values are stable slugs, not the prose the annotation form shows. That split is
deliberate: the UI wording changes far more often than the taxonomy does, and prose on disk
made every rewording a data migration.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import h5py

# ── Schema identifiers ────────────────────────────────────────────────────────
# Written as attrs on episode_annotations/<annotator>. Distinct from the *file* schema
# (OOPSIE_DATA_SCHEMA_V1 in episode_loader.py), which versions the episode layout itself.
ANNOTATION_SCHEMA_V1 = "oopsie_failure_taxonomy_v1"
ANNOTATION_SCHEMA_V2 = "oopsie_failure_taxonomy_v2"
ANNOTATION_SCHEMA_CURRENT = ANNOTATION_SCHEMA_V2

TAXONOMY_SCHEMA_V1 = "oopsiedata_taxonomy_schema_v1"
TAXONOMY_SCHEMA_V2 = "oopsiedata_taxonomy_schema_v2"
TAXONOMY_SCHEMA_CURRENT = TAXONOMY_SCHEMA_V2

# ── Vocabulary ────────────────────────────────────────────────────────────────
SUCCESS_THRESHOLD = 0.5

# The one place the four outcomes and their numeric success value are defined. The float
# stays in the file so converters, uploaders and any external consumer that only knows
# "success" keep working unchanged.
OUTCOME_SUCCESS: dict[str, float] = {
    "success": 1.0,
    "success_suboptimal": 1.0,
    "success_side_effect": 1.0,
    "failure": 0.0,
}
OUTCOMES: tuple[str, ...] = tuple(OUTCOME_SUCCESS)

FAILURE_CATEGORIES: tuple[str, ...] = (
    "reaching",
    "grasp",
    "manipulation",
    "sequencing_semantic",
    "collision",
    "hardware",
    "not_attempted",
    "other",
)

SEVERITIES: tuple[str, ...] = ("low", "medium", "catastrophic")


def outcome_to_success(outcome: Any) -> "float | None":
    """Numeric ``success`` for an outcome slug, or ``None`` if unrecognized.

    ``None`` means "leave the attr unwritten" rather than "0.0": an unrecognized outcome is
    a bug upstream, and inventing a failure for it would silently mislabel the episode.
    """
    return OUTCOME_SUCCESS.get(str(outcome or "").strip().lower())


def success_to_outcome(success: Any) -> "str | None":
    """Coarsest outcome consistent with a numeric ``success``.

    Used to read v1 files and to give converters that only have a float a valid slug. It can
    never return either qualified success -- that distinction does not exist in the float.
    """
    try:
        value = float(success)
    except (TypeError, ValueError):
        return None
    return "success" if value >= SUCCESS_THRESHOLD else "failure"


# ── v1 -> v2 upcast tables ────────────────────────────────────────────────────
# v1 stored the display prose on disk. These tables are the only description of the
# correspondence, shared by the read path below and by migrate_taxonomy_v2.
V1_SUCCESS_CATEGORY_TO_OUTCOME: dict[str, str] = {
    "clean success": "success",
    "success with side-effects": "success_side_effect",
    "suboptimal execution": "success_suboptimal",
}

V1_CATEGORY_TO_SLUG: dict[str, str] = {
    "reaching failure (pre contact)": "reaching",
    "grasp failure (at contact)": "grasp",
    "manipulation failure (post contact)": "manipulation",
    "sequencing or semantic failure": "sequencing_semantic",
    "collision failure": "collision",
    "hardware/mechanical issue": "hardware",
    "task not attempted": "not_attempted",
    "other": "other",
}

# v1 severities were whole sentences ("Low severity - no damage, ..."), so they are matched
# by prefix rather than equality.
V1_SEVERITY_PREFIX_TO_SLUG: tuple[tuple[str, str], ...] = (
    ("low severity", "low"),
    ("medium severity", "medium"),
    ("catastrophic", "catastrophic"),
)


def decode_attr(value: Any) -> str:
    """Normalize an HDF5 attr scalar (bytes / numpy scalar / str) to a trimmed str."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, str):
        return value.strip()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return decode_attr(item())
        except (ValueError, TypeError):
            pass
    return str(value).strip()


def as_value_list(value: Any) -> list:
    """Categories arrive as a list (the form) or a scalar (older hand-written records)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [v for v in value if decode_attr(v)]
    decoded = decode_attr(value)
    return [decoded] if decoded else []


def normalize_category(value: Any) -> str:
    """Map one category value to its slug, passing unknown values through verbatim.

    Unknown values are preserved rather than folded into ``other`` so a stale or hand-rolled
    label stays visible in the data. Silently relabelling human annotations is not reversible.
    """
    text = decode_attr(value)
    if not text:
        return ""
    if text in FAILURE_CATEGORIES:
        return text
    return V1_CATEGORY_TO_SLUG.get(text.lower(), text)


def normalize_severity(value: Any) -> str:
    """Map one severity value to its slug, passing unknown values through verbatim."""
    text = decode_attr(value)
    if not text:
        return ""
    lowered = text.lower()
    if lowered in SEVERITIES:
        return lowered
    for prefix, slug in V1_SEVERITY_PREFIX_TO_SLUG:
        if lowered.startswith(prefix):
            return slug
    return text


def parse_taxonomy(raw: Any) -> dict:
    """Decode a JSON-object attr, tolerating absence and malformed content.

    Used for the ``taxonomy`` attr, and reused by the server for ``robot_profile`` -- both
    are JSON objects stored as attrs with the same "missing is empty" semantics.
    """
    if isinstance(raw, dict):
        return raw
    text = decode_attr(raw)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ── Read ──────────────────────────────────────────────────────────────────────


def read_annotation_attrs(attrs: Mapping[str, Any]) -> dict:
    """Decode a v1 or v2 annotation attr set into the v2 annotation dict.

    This is the only place that knows v1 exists. v1 files are upcast on read and never
    rewritten, so a dataset can hold both versions indefinitely.

    Keys absent from the file stay absent from the result, so callers can tell "not
    annotated" from "annotated and left empty".
    """
    out: dict = {}
    taxonomy = parse_taxonomy(attrs.get("taxonomy", ""))

    outcome = decode_attr(taxonomy.get("outcome")).lower()
    if outcome not in OUTCOME_SUCCESS:
        # v1: rebuild from the success float, then refine with the qualified-success label.
        outcome = success_to_outcome(attrs.get("success")) or ""
        legacy_category = decode_attr(taxonomy.get("success_category")).lower()
        if outcome == "success" and legacy_category:
            outcome = V1_SUCCESS_CATEGORY_TO_OUTCOME.get(legacy_category, outcome)
    if outcome:
        out["outcome"] = outcome

    if "episode_description" in attrs:
        out["episode_description"] = decode_attr(attrs.get("episode_description"))
    elif "failure_description" in attrs:  # v1
        out["episode_description"] = decode_attr(attrs.get("failure_description"))

    # ``side_effect_category`` is an interim v2 spelling of the same field; v1 and current v2
    # both call it ``failure_category``, so it is read as a fallback and never written.
    raw_categories = taxonomy.get("failure_category", taxonomy.get("side_effect_category"))
    if raw_categories is not None:
        out["failure_category"] = [normalize_category(c) for c in as_value_list(raw_categories)]

    if taxonomy.get("severity") is not None:
        out["severity"] = normalize_severity(taxonomy.get("severity"))

    if "additional_notes" in attrs:
        out["additional_notes"] = decode_attr(attrs.get("additional_notes"))

    return out


# ── Write ─────────────────────────────────────────────────────────────────────


def annotation_attrs_dict(annotation: Mapping[str, Any]) -> dict:
    """Build the v2 attr mapping for one annotator subgroup.

    Split out from :func:`write_annotation_attrs` so callers that assemble an
    ``episode_annotations`` dict before opening the file (the recorder) produce the same attr
    set as callers writing straight into a group.
    """
    attrs: dict = {
        "schema": annotation.get("schema") or ANNOTATION_SCHEMA_CURRENT,
        "source": annotation.get("source") or "human",
        "timestamp": annotation.get("annotated_at", annotation.get("timestamp", "")),
        "episode_description": annotation.get("episode_description", "") or "",
        "taxonomy_schema": TAXONOMY_SCHEMA_CURRENT,
        "additional_notes": annotation.get("additional_notes", "") or "",
    }

    outcome = decode_attr(annotation.get("outcome")).lower()
    success = outcome_to_success(outcome)
    if success is not None:
        attrs["success"] = float(success)

    taxonomy = {
        "outcome": outcome,
        "failure_category": [
            normalize_category(c) for c in as_value_list(annotation.get("failure_category"))
        ],
        "severity": normalize_severity(annotation.get("severity", "")),
    }
    attrs["taxonomy"] = json.dumps(taxonomy, ensure_ascii=False)
    return attrs


def write_annotation_attrs(group: h5py.Group, annotation: Mapping[str, Any]) -> None:
    """Write the oopsie annotation attr set onto an ``episode_annotations`` subgroup."""
    for key, value in annotation_attrs_dict(annotation).items():
        group.attrs[key] = value
