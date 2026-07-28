"""Tests for the shared contributor-config reader (issue #20)."""

from __future__ import annotations

from pathlib import Path

import pytest

from oopsie_tools.utils.contributor_config import read_contributor_config


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "contributor_config.yaml"
    path.write_text(content)
    return path


def test_valid_config_returns_lab_and_token(tmp_path: Path) -> None:
    path = _write(tmp_path, "lab_id: MyLab\nhuggingface_token: hf_abc\n")
    assert read_contributor_config(path) == ("MyLab", "hf_abc")


def test_blank_lab_id_gives_clear_error(tmp_path: Path) -> None:
    # `lab_id:` parses to None; must not crash with `None.strip()` (the original #20 bug).
    path = _write(tmp_path, "lab_id:\nhuggingface_token:\n")
    with pytest.raises(RuntimeError, match="lab_id is not set"):
        read_contributor_config(path)


def test_placeholder_lab_id_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "lab_id: your_lab_id\n")
    with pytest.raises(RuntimeError, match="placeholder"):
        read_contributor_config(path)


def test_missing_file_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not found"):
        read_contributor_config(tmp_path / "does_not_exist.yaml")
