"""Tests for the attr-only diversity heuristics (issue #40)."""

from __future__ import annotations

from pathlib import Path

import h5py

from oopsie_tools.test.fixtures.make_valid import write_valid_episode
from oopsie_tools.utils.validation.diversity import check_diversity


def test_below_min_episodes_no_warning(tmp_path: Path) -> None:
    for i in range(3):
        write_valid_episode(tmp_path, stem=f"ep{i}")
    assert check_diversity(str(tmp_path)) == []


def test_identical_instructions_warns(tmp_path: Path) -> None:
    # write_valid_episode uses a single fixed language_instruction for every episode.
    for i in range(6):
        write_valid_episode(tmp_path, stem=f"ep{i}")
    warnings = check_diversity(str(tmp_path))
    assert any("task diversity" in w.lower() for w in warnings)


def test_diverse_instructions_no_task_warning(tmp_path: Path) -> None:
    for i in range(6):
        h5_path = write_valid_episode(tmp_path, stem=f"ep{i}")
        with h5py.File(h5_path, "r+") as f:
            f.attrs["language_instruction"] = f"task number {i}"
    warnings = check_diversity(str(tmp_path))
    assert not any("task diversity" in w.lower() for w in warnings)
