"""What is stored on disk and what the form hands over must mean the same thing.

Validation no longer has an opinion here — only ``outcome`` is required, and every taxonomy
field is optional. What remains is the UI's tick, which tells an annotator whether the
outcome they picked still has blank fields. That answer must not depend on whether the
annotation came out of an HDF5 file or straight off the form, so both shapes are run through
:func:`read_annotation_attrs` and compared.
"""

from __future__ import annotations

import json

import pytest

from oopsie_data_tools.annotation_tool.annotation_schema import (
    OUTCOMES,
    read_annotation_attrs,
    write_annotation_attrs,
)
from oopsie_data_tools.annotation_tool.annotator_server import _annotation_tick_level
from oopsie_data_tools.utils.validation.annotation_completeness import (
    OUTCOME_EXPECTED_FIELDS,
    completeness_flags,
    is_complete,
)

NONE, PARTIAL, COMPLETE = 0, 1, 2


def _h5_attrs(outcome, category, description, severity) -> dict:
    """Annotation as the HDF5 stores it: category and severity inside a taxonomy blob."""
    return {
        "episode_description": description,
        "taxonomy": json.dumps(
            {
                "outcome": outcome,
                "side_effect_category": category,
                "severity": severity,
            }
        ),
    }


def _form_dict(outcome, category, description, severity) -> dict:
    """The same annotation as the form hands it over: flat keys."""
    return {
        "outcome": outcome,
        "side_effect_category": category,
        "episode_description": description,
        "severity": severity,
    }


@pytest.mark.parametrize(
    "category,description,severity",
    [
        ([], "", ""),
        (["grasp"], "slipped", "low"),
        ([], "slipped", ""),
        (["grasp"], "", "low"),
        ([""], "  ", "  "),  # whitespace and an empty string are not "filled"
        ("grasp", "slipped", "low"),  # category as a scalar, from older records
    ],
)
def test_both_shapes_agree_on_which_fields_are_filled(category, description, severity):
    """The stored and in-flight representations must resolve to the same flags."""
    stored = read_annotation_attrs(_h5_attrs("failure", category, description, severity))
    direct = _form_dict("failure", category, description, severity)

    assert completeness_flags(stored) == completeness_flags(direct)


def test_whitespace_does_not_count_as_filled():
    flags = completeness_flags(_form_dict("failure", [], "   ", "\t"))
    assert flags == {
        "episode_description": False,
        "side_effect_category": False,
        "severity": False,
    }


def test_an_empty_category_list_is_not_filled():
    flags = completeness_flags(_form_dict("failure", [], "x", "x"))
    assert flags["side_effect_category"] is False
    assert flags["episode_description"] is True


# ── The tick, per outcome ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "label,outcome,category,description,severity,tick",
    [
        # A clean success asks for nothing else, so it is complete the moment it is picked.
        ("clean success", "success", [], "", "", COMPLETE),
        # Anything set on a clean success is ignored: those fields are never shown for it.
        ("clean success ignores extras", "success", ["grasp"], "x", "low", COMPLETE),
        ("suboptimal without a description", "success_suboptimal", [], "", "", PARTIAL),
        ("suboptimal with a description", "success_suboptimal", [], "clumsy", "", COMPLETE),
        # Category and severity are not shown for suboptimal, so they cannot hold it back.
        ("suboptimal ignores category", "success_suboptimal", [], "clumsy", "", COMPLETE),
        ("side-effect, nothing filled", "success_side_effect", [], "", "", PARTIAL),
        ("side-effect, partly filled", "success_side_effect", ["collision"], "", "", PARTIAL),
        (
            "side-effect, fully filled",
            "success_side_effect",
            ["collision"],
            "clipped a cup",
            "low",
            COMPLETE,
        ),
        # Previously this combination was rejected outright; now it is simply unfinished.
        ("failure, nothing filled", "failure", [], "", "", PARTIAL),
        ("failure, only a description", "failure", [], "slipped", "", PARTIAL),
        ("failure, fully filled", "failure", ["grasp"], "slipped", "low", COMPLETE),
    ],
)
def test_tick_level_per_outcome(label, outcome, category, description, severity, tick):
    form = _form_dict(outcome, category, description, severity)
    assert _annotation_tick_level(form) == tick, label

    stored = read_annotation_attrs(_h5_attrs(outcome, category, description, severity))
    assert _annotation_tick_level(stored) == tick, f"{label} (via stored attrs)"


def test_an_unannotated_episode_ticks_none():
    assert _annotation_tick_level({"outcome": ""}) == NONE
    assert _annotation_tick_level({}) == NONE


def test_an_unrecognized_outcome_ticks_none():
    """A slug the vocabulary does not know is not a partial annotation — it is no signal."""
    assert _annotation_tick_level({"outcome": "sort_of_worked"}) == NONE


def test_expected_fields_cover_every_outcome():
    """The tick's field table and the schema's outcome list must not drift apart."""
    assert set(OUTCOME_EXPECTED_FIELDS) == set(OUTCOMES)


def test_a_round_trip_through_hdf5_preserves_completeness(tmp_path):
    """Whatever the writer stores must read back with the same tick it went in with."""
    import h5py

    form = _form_dict("success_side_effect", ["collision"], "clipped a cup", "low")
    path = tmp_path / "round_trip.h5"
    with h5py.File(path, "w") as f:
        write_annotation_attrs(f.require_group("episode_annotations/a"), form)
    with h5py.File(path, "r") as f:
        stored = read_annotation_attrs(f["episode_annotations/a"].attrs)

    assert is_complete(stored) is True
    assert _annotation_tick_level(stored) == COMPLETE
