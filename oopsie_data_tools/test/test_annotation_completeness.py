"""The validator and the annotation UI must agree on what "filled in" means.

They ask different questions of the same three fields — "acceptable to upload?" versus
"fully annotated?" — and each used to carry its own copy of the field-level rule. The
thresholds differ on purpose; the rule underneath must not.
"""

from __future__ import annotations

import json

import pytest

from oopsie_data_tools.annotation_tool.annotator_server import _annotation_tick_level
from oopsie_data_tools.utils.validation.annotation_completeness import failure_trio_flags
from oopsie_data_tools.utils.validation.episode_validator import _failure_trio_fill_flags

NONE, PARTIAL, COMPLETE = 0, 1, 2


def _h5_attrs(category, description, severity) -> dict:
    """Annotation as the HDF5 stores it: category and severity inside a taxonomy blob."""
    return {
        "failure_description": description,
        "taxonomy": json.dumps({"failure_category": category, "severity": severity}),
    }


def _ui_dict(category, description, severity, binary_success="Failure") -> dict:
    """The same annotation as the questionnaire hands it over: flat keys."""
    return {
        "binary_success": binary_success,
        "failure_category": category,
        "failure_description": description,
        "severity": severity,
    }


@pytest.mark.parametrize(
    "category,description,severity",
    [
        ([], "", ""),
        (["Grasp"], "slipped", "Low"),
        ([], "slipped", ""),
        (["Grasp"], "", "Low"),
        ([""], "  ", "  "),  # whitespace and an empty string are not "filled"
        ("Grasp", "slipped", "Low"),  # category as a scalar, from older records
    ],
)
def test_both_shapes_agree_on_which_fields_are_filled(category, description, severity):
    """The stored and in-flight representations must resolve to the same three flags."""
    from_h5 = _failure_trio_fill_flags(_h5_attrs(category, description, severity))
    direct = failure_trio_flags(
        failure_category=category, failure_description=description, severity=severity
    )

    assert from_h5 == direct


def test_whitespace_does_not_count_as_filled():
    assert failure_trio_flags(
        failure_category=[], failure_description="   ", severity="\t"
    ) == (False, False, False)


def test_an_empty_category_list_is_not_filled():
    assert failure_trio_flags(
        failure_category=[], failure_description="x", severity="x"
    ) == (False, True, True)


# ── The thresholds, which are deliberately different ───────────────────────────


def _validator_accepts(success: float, category, description, severity) -> bool:
    if success >= 0.5:
        return True
    return sum(_failure_trio_fill_flags(_h5_attrs(category, description, severity))) in (0, 3)


@pytest.mark.parametrize(
    "label,success,category,description,severity,accepted,tick",
    [
        # A failure with nothing filled is uploadable but not finished: the validator's rule
        # is all-or-nothing, while the UI keeps prompting. This asymmetry is intended.
        ("no taxonomy at all", 0.0, [], "", "", True, PARTIAL),
        ("full trio", 0.0, ["Grasp"], "slipped", "Low", True, COMPLETE),
        ("only a description", 0.0, [], "slipped", "", False, PARTIAL),
        ("success needs nothing else", 1.0, [], "", "", True, COMPLETE),
    ],
)
def test_documented_thresholds(
    label, success, category, description, severity, accepted, tick
):
    assert _validator_accepts(success, category, description, severity) is accepted, label

    ui = _ui_dict(
        category,
        description,
        severity,
        binary_success="Success" if success >= 0.5 else "Failure",
    )
    assert _annotation_tick_level(ui) == tick, label


def test_an_unannotated_episode_ticks_none():
    assert _annotation_tick_level({"binary_success": ""}) == NONE
