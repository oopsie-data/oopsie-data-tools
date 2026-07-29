"""One definition of when an annotation counts as *finished* for its outcome.

Only ``outcome`` is required, so this module no longer speaks for the validator -- a partial
annotation is perfectly valid to upload. What it speaks for is the annotation UI's tick,
which still needs to distinguish "outcome recorded" from "everything this outcome asks
about is filled in", so annotators can find episodes worth revisiting.

The per-outcome field list below mirrors which fields the annotation form actually shows for
each outcome. If one moves, the other must.
"""

from __future__ import annotations

from typing import Any, Mapping

# Mirrors the conditional fields in the annotation form. `success` is a clean run: there is
# nothing further to say about it, so it is complete the moment it is chosen.
OUTCOME_EXPECTED_FIELDS: dict = {
    "success": (),
    "success_suboptimal": ("episode_description",),
    "success_side_effect": ("episode_description", "failure_category", "severity"),
    "failure": ("episode_description", "failure_category", "severity"),
}


def _is_filled(value: Any) -> bool:
    """A scalar field counts as filled when it has non-whitespace content."""
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return bool(str(value).strip())


def _category_is_filled(value: Any) -> bool:
    """``failure_category`` may arrive as a list (the form) or a scalar (older records).

    Blank entries do not count, so ``[""]`` is empty. The stored form drops them on the way
    to disk, and a list that looks filled here but not after a round-trip would make the
    tick flip for no visible reason.
    """
    if isinstance(value, (list, tuple)):
        return any(_is_filled(v) for v in value)
    return _is_filled(value)


def _field_is_filled(annotation: Mapping[str, Any], field: str) -> bool:
    value = annotation.get(field)
    if field == "failure_category":
        return _category_is_filled(value)
    return _is_filled(value)


def completeness_flags(annotation: Mapping[str, Any]) -> dict:
    """Per-field filled flags, restricted to the fields this outcome actually asks about.

    An outcome the vocabulary does not know gets an empty mapping rather than an error: the
    caller's job is to render a tick, not to reject data.
    """
    outcome = str(annotation.get("outcome", "") or "").strip().lower()
    expected = OUTCOME_EXPECTED_FIELDS.get(outcome, ())
    return {field: _field_is_filled(annotation, field) for field in expected}


def is_complete(annotation: Mapping[str, Any]) -> bool:
    """Whether every field this outcome asks about is filled (trivially true for ``success``)."""
    return all(completeness_flags(annotation).values())
