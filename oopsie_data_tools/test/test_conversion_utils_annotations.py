"""``write_episode_annotations`` — the public API third-party converters write against.

It is the one annotation writer that is not driven by the annotation tool, so it has its
own argument shape and its own consistency rules. The rules changed in v2: the failure
trio is no longer all-or-nothing, and ``outcome`` joined the numeric ``success``.
"""

from __future__ import annotations

import json

import h5py
import pytest

from oopsie_data_tools.utils.conversion_utils import write_episode_annotations


def _write(tmp_path, **kwargs) -> dict:
    path = tmp_path / "ep.h5"
    with h5py.File(path, "w") as f:
        write_episode_annotations(f, annotator_name="conv", **kwargs)
    with h5py.File(path, "r") as f:
        return dict(f["episode_annotations"]["conv"].attrs)


def test_a_partial_taxonomy_is_accepted(tmp_path):
    """v1 raised here; v2 lets a converter record whatever it actually knows."""
    attrs = _write(tmp_path, success=0.0, severity="medium")

    assert json.loads(attrs["taxonomy"])["severity"] == "medium"
    assert attrs["episode_description"] == ""


def test_outcome_defaults_to_the_coarse_reading_of_success(tmp_path):
    assert json.loads(_write(tmp_path, success=1.0)["taxonomy"])["outcome"] == "success"
    assert json.loads(_write(tmp_path, success=0.0)["taxonomy"])["outcome"] == "failure"


def test_a_qualified_success_needs_an_explicit_outcome(tmp_path):
    """The float alone cannot express the two qualified successes."""
    attrs = _write(
        tmp_path,
        success=1.0,
        outcome="success_side_effect",
        failure_category=["collision"],
        severity="low",
    )
    taxonomy = json.loads(attrs["taxonomy"])

    assert taxonomy["outcome"] == "success_side_effect"
    assert taxonomy["failure_category"] == ["collision"]
    assert float(attrs["success"]) == 1.0


def test_a_fractional_success_survives(tmp_path):
    """The exact float is authoritative for magnitude, the slug only for the branch."""
    attrs = _write(tmp_path, success=0.75, outcome="success_suboptimal")

    assert float(attrs["success"]) == 0.75
    assert json.loads(attrs["taxonomy"])["outcome"] == "success_suboptimal"


def test_an_outcome_contradicting_the_success_float_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="disagrees with success"):
        _write(tmp_path, success=1.0, outcome="failure")


def test_an_unrecognized_outcome_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outcome must be one of"):
        _write(tmp_path, success=0.0, outcome="sort_of_worked")


def test_success_out_of_range_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"success must be in \[0.0, 1.0\]"):
        _write(tmp_path, success=1.5)
