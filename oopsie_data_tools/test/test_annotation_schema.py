"""The v2 annotation schema core: the outcome mapping, v1 reads, and UI/Python drift.

With the questionnaire YAML gone the annotation form is hardcoded markup, so nothing in the
build would notice if the slugs in ``annotator.html`` and the ones in ``annotation_schema``
drifted apart. :func:`test_ui_vocabulary_matches_python` is what notices.
"""

from __future__ import annotations

import json
import re

import h5py
import pytest

from oopsie_data_tools.annotation_tool import ANNOTATION_TOOL_DIR
from oopsie_data_tools.annotation_tool.annotation_schema import (
    OUTCOME_SUCCESS,
    OUTCOMES,
    SEVERITIES,
    FAILURE_CATEGORIES,
    SUCCESS_THRESHOLD,
    outcome_to_success,
    read_annotation_attrs,
    success_to_outcome,
    write_annotation_attrs,
)
from oopsie_data_tools.utils.validation.annotation_completeness import (
    OUTCOME_EXPECTED_FIELDS,
)

_ANNOTATOR_HTML = ANNOTATION_TOOL_DIR / "ui" / "annotator.html"


# ── The outcome <-> success mapping ───────────────────────────────────────────


def test_every_outcome_has_a_success_value():
    assert set(OUTCOME_SUCCESS) == set(OUTCOMES)
    assert all(isinstance(v, float) for v in OUTCOME_SUCCESS.values())


def test_all_three_success_outcomes_score_as_success():
    """Downstream consumers that only read the float must see every success as one."""
    for outcome in ("success", "success_suboptimal", "success_side_effect"):
        assert outcome_to_success(outcome) >= SUCCESS_THRESHOLD
    assert outcome_to_success("failure") < SUCCESS_THRESHOLD


def test_an_unrecognized_outcome_has_no_success_value():
    """None, not 0.0 — guessing "failure" would silently mislabel the episode."""
    assert outcome_to_success("sort_of_worked") is None
    assert outcome_to_success("") is None
    assert success_to_outcome("not a number") is None


def test_success_to_outcome_is_coarse():
    """The float cannot carry the qualified successes, and must not pretend otherwise."""
    assert success_to_outcome(1.0) == "success"
    assert success_to_outcome(0.0) == "failure"
    assert success_to_outcome(0.9) == "success"


def test_an_unrecognized_outcome_leaves_success_unwritten(tmp_path):
    path = tmp_path / "ep.h5"
    with h5py.File(path, "w") as f:
        group = f.require_group("episode_annotations/a")
        write_annotation_attrs(group, {"outcome": "sort_of_worked"})
        assert "success" not in group.attrs


# ── Reading v1 ────────────────────────────────────────────────────────────────


def _v1_attrs(**taxonomy) -> dict:
    success = taxonomy.pop("success", 0.0)
    return {
        "schema": "oopsie_failure_taxonomy_v1",
        "success": success,
        "failure_description": taxonomy.pop("failure_description", ""),
        "taxonomy": json.dumps(taxonomy),
    }


@pytest.mark.parametrize(
    "success,success_category,expected",
    [
        (1.0, None, "success"),
        (1.0, "Clean success", "success"),
        (1.0, "Suboptimal execution", "success_suboptimal"),
        (1.0, "Success with side-effects", "success_side_effect"),
        (0.0, None, "failure"),
    ],
)
def test_v1_outcome_is_reconstructed(success, success_category, expected):
    attrs = _v1_attrs(success=success)
    if success_category:
        attrs["taxonomy"] = json.dumps({"success_category": success_category})

    assert read_annotation_attrs(attrs)["outcome"] == expected


def test_v1_prose_values_read_back_as_slugs():
    attrs = _v1_attrs(
        success=0.0,
        failure_description="dropped it",
        failure_category=["Grasp failure (at contact)", "Collision failure"],
        severity="Low severity - no damage, can be reset and reattempted",
    )
    out = read_annotation_attrs(attrs)

    assert out["episode_description"] == "dropped it"
    assert out["failure_category"] == ["grasp", "collision"]
    assert out["severity"] == "low"


def test_an_unknown_category_value_is_preserved():
    """Stale labels stay visible to the annotator instead of being guessed at."""
    attrs = _v1_attrs(success=0.0, failure_category=["grasp_failure"])

    assert read_annotation_attrs(attrs)["failure_category"] == ["grasp_failure"]


def test_absent_keys_stay_absent():
    """Callers distinguish "not annotated" from "annotated and left empty"."""
    out = read_annotation_attrs({})

    assert "outcome" not in out
    assert "episode_description" not in out


def test_a_malformed_taxonomy_reads_as_empty_rather_than_raising():
    out = read_annotation_attrs({"success": 0.0, "taxonomy": "{not json"})

    assert out["outcome"] == "failure"
    assert "failure_category" not in out


def test_the_legacy_v1_fixture_reads_as_v2(legacy_v1_episode):
    """The on-disk v1 fixture, opened through the normal read path."""
    with h5py.File(legacy_v1_episode, "r") as f:
        out = read_annotation_attrs(f["episode_annotations"]["test_annotator"].attrs)

    assert out["outcome"] == "failure"
    assert out["failure_category"] == ["grasp", "collision"]
    assert out["severity"] == "medium"
    assert out["episode_description"].startswith("Gripper closed early")


# ── Drift guards ──────────────────────────────────────────────────────────────


def _html_values(field: str) -> list:
    """Every ``value="…"`` on inputs named *field* in the annotation form."""
    html = _ANNOTATOR_HTML.read_text(encoding="utf-8")
    pattern = re.compile(
        r'<input[^>]*name="' + re.escape(field) + r'"[^>]*value="([^"]*)"'
        r"|"
        r'<input[^>]*value="([^"]*)"[^>]*name="' + re.escape(field) + r'"'
    )
    return [a or b for a, b in pattern.findall(html)]


@pytest.mark.parametrize(
    "field,expected",
    [
        ("outcome", OUTCOMES),
        ("failure_category", FAILURE_CATEGORIES),
        ("severity", SEVERITIES),
    ],
)
def test_ui_vocabulary_matches_python(field, expected):
    """The hardcoded form and the schema must offer exactly the same slugs.

    Nothing else connects them since the questionnaire YAML was removed, so a slug typo in
    the markup would otherwise only show up as data that quietly fails to round-trip.
    """
    assert _html_values(field) == list(expected)


def test_expected_fields_match_the_outcomes():
    assert set(OUTCOME_EXPECTED_FIELDS) == set(OUTCOMES)


def test_the_form_marks_only_outcome_as_required():
    """Only the outcome is required; a required marker anywhere else contradicts that."""
    html = _ANNOTATOR_HTML.read_text(encoding="utf-8")
    required_groups = re.findall(r'data-question-id="([^"]+)"[^>]*data-required="1"', html)

    assert required_groups == ["outcome"]
