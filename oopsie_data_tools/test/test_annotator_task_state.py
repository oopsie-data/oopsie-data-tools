"""The rollout loop's own endpoints: ``/api/task/done`` and the instruction history.

``done`` is the one transition the browser can trigger at a moment of its own choosing —
the operator clicks "Start next rollout" — so it races the rollout driver, which posts the
same endpoint itself once an annotation arrives. These pin what a mistimed or repeated
click may do to a rollout that is queued or executing: nothing.

``/api/instructions/recent`` feeds the instruction card's picker, which replaced a "repeat
last" button that could only reach the previous task.
"""

from __future__ import annotations

import os

import h5py
import pytest

from oopsie_data_tools.annotation_tool import annotator_server
from oopsie_data_tools.test.fixtures.make_valid import write_valid_episode


@pytest.fixture
def client(tmp_path):
    """An in-the-loop server (browse_only=False), which is what has a task state at all."""
    annotator_server.configure_runtime(
        samples_dir=tmp_path, annotator_name="tester", browse_only=False
    )
    annotator_server.app.config.update(TESTING=True)
    with annotator_server.app.test_client() as test_client:
        yield test_client


def _state(client) -> dict:
    return client.get("/api/task/state").get_json()


def _annotating(client, sample_id: str = "000000") -> None:
    client.post(
        "/api/task/annotating",
        json={"sample_id": sample_id, "video_urls": {"cam": "/videos-path/a.mp4"}},
    )


def test_done_while_pending_leaves_the_instruction_claimable(client):
    """The driver claims an instruction only while the status reads 'pending'.

    Resetting to idle here would stand the rollout loop up forever: the browser has already
    handed over an instruction it will not send again, and nothing would ever run it.
    """
    client.post("/api/task/submit", json={"instruction": "pick up the red block"})

    client.post("/api/task/done", json={})

    state = _state(client)
    assert state["status"] == "pending"
    assert state["pending_instruction"] == "pick up the red block"


def test_done_while_running_does_not_cancel_the_rollout(client):
    client.post("/api/task/submit", json={"instruction": "pick up the red block"})
    client.post("/api/task/start", json={"instruction": "pick up the red block"})

    client.post("/api/task/done", json={})

    assert _state(client)["status"] == "running"


def test_done_while_annotating_finishes_the_episode(client):
    """The transition it does own still works, and still records the skip marker."""
    _annotating(client)

    client.post("/api/task/done", json={})

    state = _state(client)
    assert state["status"] == "idle"
    assert state["current_sample"] is None
    skipped = client.get("/api/annotations/000000").get_json()
    assert skipped == {"__annotation_skipped__": True}


def test_repeated_done_does_not_disturb_the_next_rollout(client):
    """The click that got the tool stuck: finish, then keep clicking into the next task."""
    _annotating(client)
    client.post("/api/task/done", json={})

    client.post("/api/task/submit", json={"instruction": "next task"})
    for _ in range(3):
        client.post("/api/task/done", json={})

    state = _state(client)
    assert state["status"] == "pending"
    assert state["pending_instruction"] == "next task"


def test_a_saved_annotation_is_not_overwritten_by_the_skip_marker(client):
    """Save then click: the human's annotation must survive the operator's click."""
    _annotating(client)
    client.post("/api/annotations", json={"sample_id": "000000", "outcome": "failure"})

    client.post("/api/task/done", json={})

    saved = client.get("/api/annotations/000000").get_json()
    assert saved["outcome"] == "failure"
    assert "__annotation_skipped__" not in saved


# ── /api/instructions/recent ──────────────────────────────────────────────────


def _episode_with_instruction(dir_path, stem: str, instruction: str, age_seconds: float):
    """A valid episode carrying *instruction*, aged so mtime ordering is deterministic."""
    h5_path = write_valid_episode(dir_path, stem)
    with h5py.File(h5_path, "r+") as f:
        f.attrs["language_instruction"] = instruction
    stamp = 1_700_000_000 - age_seconds
    os.utime(h5_path, (stamp, stamp))
    return h5_path


def test_recent_instructions_are_newest_first_and_unique(client, tmp_path):
    _episode_with_instruction(tmp_path, "a", "open the drawer", age_seconds=300)
    _episode_with_instruction(tmp_path, "b", "pick up the red block", age_seconds=200)
    _episode_with_instruction(tmp_path, "c", "open the drawer", age_seconds=100)

    instructions = client.get("/api/instructions/recent").get_json()

    assert instructions == ["open the drawer", "pick up the red block"]


def test_recent_instructions_dedupe_ignores_case(client, tmp_path):
    """Two rows differing only in capitalization are the same task to the operator."""
    _episode_with_instruction(tmp_path, "a", "Open The Drawer", age_seconds=200)
    _episode_with_instruction(tmp_path, "b", "open the drawer", age_seconds=100)

    instructions = client.get("/api/instructions/recent").get_json()

    assert instructions == ["open the drawer"]


def test_recent_instructions_honour_the_limit(client, tmp_path):
    for i in range(6):
        _episode_with_instruction(tmp_path, f"e{i}", f"task {i}", age_seconds=100 * (6 - i))

    instructions = client.get("/api/instructions/recent?limit=3").get_json()

    assert instructions == ["task 5", "task 4", "task 3"]


def test_recent_instructions_skip_episodes_without_one(client, tmp_path):
    _episode_with_instruction(tmp_path, "a", "", age_seconds=100)
    _episode_with_instruction(tmp_path, "b", "   ", age_seconds=200)

    assert client.get("/api/instructions/recent").get_json() == []


def test_recent_instructions_on_an_empty_samples_dir(client):
    """A first session has no history; the picker hides itself on an empty list."""
    assert client.get("/api/instructions/recent").get_json() == []
