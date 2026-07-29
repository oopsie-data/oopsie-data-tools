"""Tests for the annotator Flask server's HDF5 sample endpoint.

These use Flask's test client (no real socket / browser). Focus: videos are read
from the correct ``observations/video_paths`` group — a regression guard for the
bug where the server read a non-existent ``image_observations`` group and only
rendered videos by accident via the ``<stem>_<cam>.mp4`` filename fallback.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from oopsie_data_tools.annotation_tool.annotation_schema import write_annotation_attrs
from oopsie_data_tools.annotation_tool.annotator_server import app, configure_runtime
from oopsie_data_tools.test.fixtures.make_valid import (
    _write_base_h5,
    _write_video,
    write_valid_episode,
)


@pytest.fixture
def client(tmp_path: Path):
    configure_runtime(
        samples_dir=tmp_path, annotator_name="test_annotator", browse_only=True
    )
    app.config.update(TESTING=True)
    return app.test_client()


def _get_sample(client, rel_path: str) -> dict:
    resp = client.get(f"/api/h5/sample?path={rel_path}")
    assert resp.status_code == 200, resp.data
    return resp.get_json()


def test_video_urls_from_video_paths_group_nested_layout(
    client, tmp_path: Path
) -> None:
    """Video sits in a subdir with no ``<stem>_<cam>.mp4`` sibling, so only a
    correct ``observations/video_paths`` read (not the glob fallback) finds it."""
    nested = tmp_path / "nested"
    (nested / "cam_videos").mkdir(parents=True)
    _write_video(nested / "cam_videos" / "front.mp4", color=(10, 20, 30))
    with h5py.File(nested / "ep.h5", "w") as f:
        _write_base_h5(
            f,
            episode_id="ep",
            language_instruction="do the thing",
            camera_video_paths={"front": "cam_videos/front.mp4"},
        )

    data = _get_sample(client, "nested/ep.h5")

    assert list(data["video_urls"].keys()) == ["front"]
    assert data["video_urls"]["front"].endswith("cam_videos/front.mp4")


def test_video_urls_resolved_for_flat_layout(client, tmp_path: Path) -> None:
    """A standard flat episode still resolves its camera video."""
    write_valid_episode(tmp_path, stem="flat")

    data = _get_sample(client, "flat.h5")

    assert "front" in data["video_urls"]


def test_sample_returns_episode_fields(client, tmp_path: Path) -> None:
    """/api/h5/sample summarizes all logged fields for the visualizer (#30)."""
    write_valid_episode(tmp_path, stem="ep")

    fields = _get_sample(client, "ep.h5")["episode_fields"]

    assert fields["attributes"]["lab_id"] == "test_lab"
    assert fields["robot_profile"]["policy_name"] == "test_policy"
    assert fields["robot_states"]["joint_position"]["shape"] == [20, 7]
    assert fields["robot_states"]["joint_position"]["empty"] is False
    # Unused action keys are stored as h5py.Empty and flagged empty.
    assert fields["actions"]["joint_velocity"]["empty"] is False
    assert fields["actions"]["cartesian_position"]["empty"] is True
    assert fields["trajectory_length"] == 20


def test_set_instruction_persists(client, tmp_path: Path) -> None:
    """POST /api/h5/instruction overwrites the language_instruction attr (#31)."""
    write_valid_episode(tmp_path, stem="ep")

    resp = client.post(
        "/api/h5/instruction?path=ep.h5",
        json={"instruction": "pour the water carefully"},
    )
    assert resp.status_code == 200, resp.data

    data = _get_sample(client, "ep.h5")
    assert data["metadata"]["language_instruction"] == "pour the water carefully"
    assert data["episode_fields"]["attributes"]["language_instruction"] == "pour the water carefully"


def test_set_instruction_rejects_empty(client, tmp_path: Path) -> None:
    write_valid_episode(tmp_path, stem="ep")

    resp = client.post("/api/h5/instruction?path=ep.h5", json={"instruction": "  "})

    assert resp.status_code == 400


def test_save_outcome_roundtrip(client, tmp_path: Path) -> None:
    """A qualified success stores its outcome in the taxonomy and reads back (#29)."""
    write_valid_episode(tmp_path, stem="ep")

    resp = client.post(
        "/api/h5/annotations?path=ep.h5",
        json={
            "outcome": "success_side_effect",
            "severity": "low",
            "side_effect_category": ["collision"],
            "episode_description": "clipped a nearby cup",
            "additional_notes": "no damage",
        },
    )
    assert resp.status_code == 200, resp.data

    data = _get_sample(client, "ep.h5")
    # All three success_* outcomes still store success=1.0, so existing converters and
    # uploaders keep reading the file unchanged.
    assert data["metadata"]["success"] == 1.0
    ann = data["existing_annotation"]
    assert ann["outcome"] == "success_side_effect"
    assert ann["side_effect_category"] == ["collision"]
    assert ann["severity"] == "low"


def test_save_rejects_unknown_vocabulary(client, tmp_path: Path) -> None:
    """Slugs are opaque, so a typo must be caught before it reaches disk."""
    write_valid_episode(tmp_path, stem="ep")

    resp = client.post(
        "/api/h5/annotations?path=ep.h5",
        json={"outcome": "success_side_effect", "side_effect_category": ["grasp_failure"]},
    )
    assert resp.status_code == 400
    assert "grasp_failure" in resp.get_json()["error"]

    resp = client.post("/api/h5/annotations?path=ep.h5", json={"outcome": "sort_of"})
    assert resp.status_code == 400


def test_save_accepts_a_partial_taxonomy(client, tmp_path: Path) -> None:
    """Only the outcome is required — a failure with just a severity saves cleanly."""
    write_valid_episode(tmp_path, stem="ep")

    resp = client.post(
        "/api/h5/annotations?path=ep.h5",
        json={"outcome": "failure", "severity": "low"},
    )
    assert resp.status_code == 200, resp.data

    ann = _get_sample(client, "ep.h5")["existing_annotation"]
    assert ann["outcome"] == "failure"
    assert ann["episode_description"] == ""


def test_recent_annotations_returns_distinct(client, tmp_path: Path) -> None:
    """/api/annotations/recent surfaces the annotator's distinct prior failures (#27).

    The endpoint feeds the taxonomy autofill picker, so clean successes are filtered
    out and identical labels are deduplicated.
    """
    write_valid_episode(tmp_path, stem="a")  # clean success by test_annotator — not offered
    for stem, description in (("b", "dropped the object"), ("c", "dropped the object"), ("d", "missed the grasp")):
        write_valid_episode(tmp_path, stem=stem)
        resp = client.post(
            f"/api/h5/annotations?path={stem}.h5",
            json={
                "outcome": "failure",
                "side_effect_category": ["other"],
                "episode_description": description,
                "severity": "low",
            },
        )
        assert resp.status_code == 200, resp.data

    items = client.get("/api/annotations/recent?limit=10").get_json()

    assert {i["outcome"] for i in items} == {"failure"}
    # b and c share a label; only one of them is offered.
    assert sorted(i["episode_description"] for i in items) == [
        "dropped the object",
        "missed the grasp",
    ]


def test_recent_annotations_includes_qualified_successes(client, tmp_path: Path) -> None:
    """A suboptimal-execution description is as worth copying as a failure's."""
    write_valid_episode(tmp_path, stem="a")  # clean success — still filtered out
    write_valid_episode(tmp_path, stem="b")
    resp = client.post(
        "/api/h5/annotations?path=b.h5",
        json={"outcome": "success_suboptimal", "episode_description": "took a long detour"},
    )
    assert resp.status_code == 200, resp.data

    items = client.get("/api/annotations/recent?limit=10").get_json()

    assert [i["episode_description"] for i in items] == ["took a long detour"]


def test_list_reports_other_human_annotator(client, tmp_path: Path) -> None:
    """api_h5_list flags episodes annotated by a different human (#26)."""
    write_valid_episode(tmp_path, stem="ep")  # annotated by test_annotator
    with h5py.File(tmp_path / "ep.h5", "r+") as f:
        g = f["episode_annotations"].require_group("someone_else")
        write_annotation_attrs(
            g,
            {
                "outcome": "failure",
                "source": "human",
                "side_effect_category": ["other"],
                "episode_description": "x",
                "severity": "low",
            },
        )

    entry = next(e for e in client.get("/api/h5/list").get_json() if e["rel_path"] == "ep.h5")
    assert entry["annotated_by_others"] is True


def test_list_ignores_nonhuman_annotator(client, tmp_path: Path) -> None:
    """VLM/automated subgroups do not count as 'another annotator' (#26)."""
    write_valid_episode(tmp_path, stem="ep")
    with h5py.File(tmp_path / "ep.h5", "r+") as f:
        g = f["episode_annotations"].require_group("cosmos-7b")
        g.attrs["source"] = "cosmos-7b"
        g.attrs["success"] = 0.0

    entry = next(e for e in client.get("/api/h5/list").get_json() if e["rel_path"] == "ep.h5")
    assert entry["annotated_by_others"] is False
