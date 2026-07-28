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

from oopsie_tools.annotation_tool.annotation_schema import write_annotation_attrs
from oopsie_tools.annotation_tool.annotator_server import app, configure_runtime
from oopsie_tools.test.fixtures.make_valid import (
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


def test_save_success_category_roundtrip(client, tmp_path: Path) -> None:
    """A qualified success stores success_category in the taxonomy and reads back (#29)."""
    write_valid_episode(tmp_path, stem="ep")

    resp = client.post(
        "/api/h5/annotations?path=ep.h5",
        json={
            "binary_success": "Success",
            "success_category": "Success with side-effects",
            "severity": "Low severity - no damage, can be reset and reattempted",
            "failure_category": [],
            "failure_description": "",
            "additional_notes": "clipped a nearby cup",
        },
    )
    assert resp.status_code == 200, resp.data

    data = _get_sample(client, "ep.h5")
    assert data["metadata"]["success"] == 1.0
    ann = data["existing_annotation"]
    assert ann["binary_success"] == "Success"
    assert ann["success_category"] == "Success with side-effects"
    assert ann["severity"].startswith("Low severity")


def test_recent_annotations_returns_distinct(client, tmp_path: Path) -> None:
    """/api/annotations/recent surfaces the annotator's distinct prior labels (#27)."""
    write_valid_episode(tmp_path, stem="a")  # success by test_annotator
    write_valid_episode(tmp_path, stem="b")
    client.post(
        "/api/h5/annotations?path=b.h5",
        json={
            "binary_success": "Failure",
            "failure_category": ["Other"],
            "failure_description": "dropped the object",
            "severity": "Low severity - no damage, can be reset and reattempted",
        },
    )

    items = client.get("/api/annotations/recent?limit=10").get_json()
    kinds = {i["binary_success"] for i in items}

    assert "Success" in kinds and "Failure" in kinds


def test_list_reports_other_human_annotator(client, tmp_path: Path) -> None:
    """api_h5_list flags episodes annotated by a different human (#26)."""
    write_valid_episode(tmp_path, stem="ep")  # annotated by test_annotator
    with h5py.File(tmp_path / "ep.h5", "r+") as f:
        g = f["episode_annotations"].require_group("someone_else")
        write_annotation_attrs(
            g,
            {
                "binary_success": "Failure",
                "source": "human",
                "failure_category": ["Other"],
                "failure_description": "x",
                "severity": "Low severity - no damage, can be reset and reattempted",
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
