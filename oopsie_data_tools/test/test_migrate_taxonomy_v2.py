"""The v1 -> v2 migration, and the read-time upcast it has to agree with.

Two paths turn a v1 file into v2 data: reading it (which upcasts in memory and leaves the
file alone) and migrating it (which rewrites the attrs). If they ever disagreed, the same
episode would mean different things depending on whether anyone had run the migration —
so the central test here is that the two produce identical results.
"""

from __future__ import annotations

import json

import h5py
import pytest

from oopsie_data_tools.annotation_tool.annotation_schema import (
    ANNOTATION_SCHEMA_V2,
    FAILURE_CATEGORIES,
    TAXONOMY_SCHEMA_V2,
    V1_CATEGORY_TO_SLUG,
    read_annotation_attrs,
)
from oopsie_data_tools.test.fixtures.make_valid import write_valid_episode
from oopsie_data_tools.utils.migrate_taxonomy_v2 import (
    migrate_annotation_attrs,
    migrate_h5_file,
    migrate_tree,
)
from oopsie_data_tools.utils.validation.validation_utils import validate_h5_file


def _v1_attrs(
    *,
    success: float = 0.0,
    failure_description: str = "",
    failure_category=(),
    severity: str = "",
    success_category: str = "",
) -> dict:
    taxonomy: dict = {
        "failure_category": list(failure_category),
        "severity": severity,
    }
    if success_category:
        taxonomy["success_category"] = success_category
    return {
        "schema": "oopsie_failure_taxonomy_v1",
        "source": "human",
        "timestamp": "2026-04-21T10:00:00+00:00",
        "success": success,
        "failure_description": failure_description,
        "taxonomy_schema": "oopsiedata_taxonomy_schema_v1",
        "taxonomy": json.dumps(taxonomy),
        "additional_notes": "",
    }


def _outcome_of(attrs: dict) -> str:
    return json.loads(attrs["taxonomy"])["outcome"]


# ── The outcome mapping ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "success,success_category,expected",
    [
        (1.0, "", "success"),
        (1.0, "Clean success", "success"),
        (1.0, "Suboptimal execution", "success_suboptimal"),
        (1.0, "Success with side-effects", "success_side_effect"),
        (0.0, "", "failure"),
        # A qualified-success label on a failure is nonsense; the float wins.
        (0.0, "Clean success", "failure"),
    ],
)
def test_outcome_is_derived_from_success_and_category(success, success_category, expected):
    out = migrate_annotation_attrs(
        _v1_attrs(success=success, success_category=success_category)
    )
    assert _outcome_of(out) == expected


@pytest.mark.parametrize("prose,slug", sorted(V1_CATEGORY_TO_SLUG.items()))
def test_every_v1_category_maps_to_a_known_slug(prose, slug):
    """Parametrized over the whole table, so a new category cannot be added untested."""
    out = migrate_annotation_attrs(_v1_attrs(failure_category=[prose]))

    assert json.loads(out["taxonomy"])["failure_category"] == [slug]
    assert slug in FAILURE_CATEGORIES


@pytest.mark.parametrize(
    "prose,slug",
    [
        ("Low severity - no damage, can be reset and reattempted", "low"),
        (
            "Medium severity - some damage or risk of damage or significant reset "
            "required, but can be reattempted",
            "medium",
        ),
        (
            "Catastrophic - significant damage or risk of damage, cannot be "
            "reattempted without repair",
            "catastrophic",
        ),
        ("", ""),
    ],
)
def test_severity_prose_maps_to_a_slug(prose, slug):
    out = migrate_annotation_attrs(_v1_attrs(severity=prose))
    assert json.loads(out["taxonomy"])["severity"] == slug


def test_description_is_renamed_and_the_old_key_removed():
    out = migrate_annotation_attrs(_v1_attrs(failure_description="dropped it"))

    assert out["episode_description"] == "dropped it"
    # Leaving both would force every reader to decide which one wins.
    assert "failure_description" not in out


def test_schema_strings_are_bumped():
    out = migrate_annotation_attrs(_v1_attrs())

    assert out["schema"] == ANNOTATION_SCHEMA_V2
    assert out["taxonomy_schema"] == TAXONOMY_SCHEMA_V2


def test_unmapped_values_are_preserved_verbatim():
    """Stale labels stay visible rather than being guessed into ``other``."""
    out = migrate_annotation_attrs(
        _v1_attrs(failure_category=["grasp_failure", "transport_failure"], severity="major")
    )
    taxonomy = json.loads(out["taxonomy"])

    assert taxonomy["failure_category"] == ["grasp_failure", "transport_failure"]
    assert taxonomy["severity"] == "major"


# ── Nothing-to-do cases ───────────────────────────────────────────────────────


def test_an_already_v2_annotation_is_left_alone():
    v2 = migrate_annotation_attrs(_v1_attrs(success=1.0))
    assert migrate_annotation_attrs(v2) is None


def test_a_subgroup_that_is_not_an_annotation_is_skipped():
    """No success and no taxonomy: inventing one would make an invalid file pass."""
    assert migrate_annotation_attrs({"source": "human"}) is None


# ── Whole files ───────────────────────────────────────────────────────────────


def test_migration_and_read_time_upcast_agree(tmp_path):
    """The load-bearing test: migrating must not change what the file already means."""
    h5_path = write_valid_episode(tmp_path, stem="ep")
    with h5py.File(h5_path, "r+") as f:
        g = f["episode_annotations"]["test_annotator"]
        for key in list(g.attrs):
            del g.attrs[key]
        for key, value in _v1_attrs(
            success=0.0,
            failure_description="Gripper closed early.",
            failure_category=["Grasp failure (at contact)", "Collision failure"],
            severity="Low severity - no damage, can be reset and reattempted",
        ).items():
            g.attrs[key] = value

    with h5py.File(h5_path, "r") as f:
        before = read_annotation_attrs(f["episode_annotations"]["test_annotator"].attrs)

    migrate_h5_file(h5_path, apply=True)

    with h5py.File(h5_path, "r") as f:
        after = read_annotation_attrs(f["episode_annotations"]["test_annotator"].attrs)

    assert before == after
    assert after["outcome"] == "failure"
    assert after["failure_category"] == ["grasp", "collision"]
    assert after["severity"] == "low"


def test_a_dry_run_writes_nothing(tmp_path):
    h5_path = write_valid_episode(tmp_path, stem="ep")
    with h5py.File(h5_path, "r+") as f:
        g = f["episode_annotations"]["test_annotator"]
        for key in list(g.attrs):
            del g.attrs[key]
        for key, value in _v1_attrs(success=1.0).items():
            g.attrs[key] = value
    before = h5_path.read_bytes()

    report = migrate_h5_file(h5_path, apply=False)

    assert report["migrated"] == ["test_annotator"]
    assert h5_path.read_bytes() == before


def test_migration_is_idempotent(tmp_path):
    h5_path = write_valid_episode(tmp_path, stem="ep")
    with h5py.File(h5_path, "r+") as f:
        g = f["episode_annotations"]["test_annotator"]
        for key in list(g.attrs):
            del g.attrs[key]
        for key, value in _v1_attrs(success=1.0).items():
            g.attrs[key] = value

    first = migrate_h5_file(h5_path, apply=True)
    after_first = h5_path.read_bytes()
    second = migrate_h5_file(h5_path, apply=True)

    assert first["migrated"] == ["test_annotator"]
    assert second["migrated"] == []
    assert second["skipped"]["test_annotator"] == "already v2"
    assert h5_path.read_bytes() == after_first


def test_a_v2_file_is_untouched(tmp_path):
    """write_valid_episode already writes v2, so there is nothing to do."""
    h5_path = write_valid_episode(tmp_path, stem="ep")
    before = h5_path.read_bytes()

    report = migrate_h5_file(h5_path, apply=True)

    assert report["migrated"] == []
    assert h5_path.read_bytes() == before


def test_a_migrated_file_still_validates(tmp_path):
    h5_path = write_valid_episode(tmp_path, stem="ep")
    with h5py.File(h5_path, "r+") as f:
        g = f["episode_annotations"]["test_annotator"]
        for key in list(g.attrs):
            del g.attrs[key]
        for key, value in _v1_attrs(
            success=0.0, failure_description="dropped it", severity="major"
        ).items():
            g.attrs[key] = value

    migrate_h5_file(h5_path, apply=True)

    assert validate_h5_file(str(h5_path), strict_annotation_check=True) is True


def test_migrate_tree_reports_one_entry_per_file(tmp_path):
    write_valid_episode(tmp_path, stem="a")
    write_valid_episode(tmp_path, stem="b")

    reports = migrate_tree(tmp_path, apply=False)

    assert len(reports) == 2
    assert all(r["error"] is None for r in reports)
