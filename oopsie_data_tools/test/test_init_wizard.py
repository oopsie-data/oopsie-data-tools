"""Tests for ``oopsie-data init`` (``oopsie_data_tools/init_wizard.py``).

The wizard is driven by scripted answers: ``builtins.input`` is replaced with a queue and
``sys.stdin.isatty`` is forced True, which is what the prompt helpers gate on.

Note: ``conftest._test_contributor_config`` rebinds ``read_contributor_config`` for the whole
session, so assertions here read the written YAML directly rather than through that name.
"""

from __future__ import annotations

import sys

import pytest
import yaml

from oopsie_data_tools import cli
from oopsie_data_tools.utils import paths


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Send every config read and write into a scratch directory."""
    target = tmp_path / "config"
    monkeypatch.setenv(paths.ENV_CONFIG_DIR, str(target))
    return target


@pytest.fixture
def answers(monkeypatch):
    """Script the wizard's prompts; returns the list to fill in per test."""
    queue: list[str] = []

    def _fake_input(prompt: str = "") -> str:
        if not queue:
            raise AssertionError(f"wizard asked an unexpected question: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    return queue


def _read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ── credentials ───────────────────────────────────────────────────────────────


def test_credentials_written_and_readable(config_dir, answers):
    answers.extend(["MyLab", "hf_secret_token"])

    assert cli.main(["init", "--no-verify-token"]) == 0

    path = config_dir / "contributor_config.yaml"
    assert _read_yaml(path) == {"lab_id": "MyLab", "huggingface_token": "hf_secret_token"}
    assert path.stat().st_mode & 0o777 == 0o600, "the token file must not be world readable"


def test_credentials_from_flags_need_no_prompts(config_dir, answers):
    # `answers` is empty: any prompt raises. Fully-flagged runs must be unattended.
    assert (
        cli.main(
            ["init", "--lab-id", "FlagLab", "--hf-token", "t", "--no-verify-token"]
        )
        == 0
    )
    assert _read_yaml(config_dir / "contributor_config.yaml")["lab_id"] == "FlagLab"


def test_placeholder_lab_id_is_rejected(config_dir, answers):
    answers.extend(["your_lab_id", "RealLab", ""])  # re-asked after the placeholder

    assert cli.main(["init", "--no-verify-token"]) == 0
    assert _read_yaml(config_dir / "contributor_config.yaml")["lab_id"] == "RealLab"


def test_placeholder_lab_id_from_flag_aborts(config_dir, answers):
    assert cli.main(["init", "--lab-id", "your_lab_id"]) == 1
    assert not (config_dir / "contributor_config.yaml").exists()


def test_existing_config_is_not_overwritten_without_consent(config_dir, answers):
    config_dir.mkdir(parents=True)
    path = config_dir / "contributor_config.yaml"
    path.write_text("lab_id: Original\nhuggingface_token: keep\n")
    answers.append("n")  # "Update it?" -> no

    assert cli.main(["init", "--no-verify-token"]) == 0
    assert _read_yaml(path)["lab_id"] == "Original"


def test_existing_config_without_force_fails_non_interactively(config_dir, monkeypatch):
    """Nothing was written, so the exit code must say so rather than report success."""
    config_dir.mkdir(parents=True)
    path = config_dir / "contributor_config.yaml"
    path.write_text("lab_id: Original\nhuggingface_token: keep\n")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    assert cli.main(["init", "--lab-id", "New", "--no-verify-token"]) == 1
    assert _read_yaml(path)["lab_id"] == "Original"


def test_force_overwrites_without_asking(config_dir, answers):
    config_dir.mkdir(parents=True)
    path = config_dir / "contributor_config.yaml"
    path.write_text("lab_id: Original\nhuggingface_token: keep\n")

    assert (
        cli.main(
            ["init", "--force", "--lab-id", "New", "--hf-token", "t2",
             "--no-verify-token"]
        )
        == 0
    )
    assert _read_yaml(path)["lab_id"] == "New"


def test_token_is_verified_when_asked_for(config_dir, answers, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "huggingface_hub.whoami", lambda token=None: seen.update(token=token) or {"name": "someone"}
    )

    assert cli.main(["init", "--lab-id", "L", "--hf-token", "good"]) == 0
    assert seen["token"] == "good"


def test_token_verification_failure_is_not_fatal(config_dir, answers, monkeypatch):
    def _boom(token=None):
        raise RuntimeError("Invalid user token.")

    monkeypatch.setattr("huggingface_hub.whoami", _boom)

    assert cli.main(["init", "--lab-id", "L", "--hf-token", "bad"]) == 0
    assert _read_yaml(config_dir / "contributor_config.yaml")["huggingface_token"] == "bad"


# ── persisting a custom config dir ────────────────────────────────────────────


def test_config_dir_flag_prints_the_export_line(tmp_path, monkeypatch, capsys, caplog):
    """--config-dir only lives for one process, so init must say how to keep it."""
    custom = tmp_path / "custom"
    monkeypatch.delenv(paths.ENV_CONFIG_DIR, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    assert (
        cli.main(
            ["--config-dir", str(custom), "init", "--lab-id", "L", "--hf-token", "t",
             "--no-verify-token"]
        )
        == 0
    )
    logged = caplog.text + capsys.readouterr().err
    assert f'export {paths.ENV_CONFIG_DIR}="{custom}"' in logged


def test_exported_env_dir_needs_no_advice(config_dir, monkeypatch, capsys, caplog):
    """A dir already exported in the environment is found next run; stay quiet."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    assert (
        cli.main(["init", "--lab-id", "L", "--hf-token", "t", "--no-verify-token"]) == 0
    )
    assert f"export {paths.ENV_CONFIG_DIR}" not in (caplog.text + capsys.readouterr().err)


# ── non-interactive behaviour ─────────────────────────────────────────────────


def test_non_tty_without_flags_exits_one(config_dir, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input", lambda prompt="": pytest.fail("must not prompt without a terminal")
    )

    assert cli.main(["init"]) == 1
    assert not (config_dir / "contributor_config.yaml").exists()


def test_fully_flagged_run_needs_no_terminal(config_dir, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input", lambda prompt="": pytest.fail("must not prompt without a terminal")
    )

    assert (
        cli.main(["init", "--lab-id", "L", "--hf-token", "t", "--no-verify-token"]) == 0
    )
    assert (config_dir / "contributor_config.yaml").is_file()
