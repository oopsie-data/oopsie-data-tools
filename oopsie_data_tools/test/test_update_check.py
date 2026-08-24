"""Tests for the optional PyPI update notification."""

from __future__ import annotations

import builtins
import json

import pytest

from oopsie_data_tools import cli
from oopsie_data_tools.utils import update_check


def test_newer_release_is_reported_and_cached(tmp_path, monkeypatch):
    cache = tmp_path / "update-check.json"
    monkeypatch.setattr(update_check.metadata, "version", lambda _: "0.9.3")
    monkeypatch.setattr(update_check, "_fetch_latest_version", lambda: "0.9.4")

    assert update_check.available_update(cache_path=cache, now=1000) == ("0.9.3", "0.9.4")
    assert json.loads(cache.read_text(encoding="utf-8")) == {
        "checked_at": 1000,
        "latest_version": "0.9.4",
    }


def test_fresh_cache_avoids_the_network(tmp_path, monkeypatch):
    cache = tmp_path / "update-check.json"
    cache.write_text(
        json.dumps({"checked_at": 1000, "latest_version": "0.9.4"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(update_check.metadata, "version", lambda _: "0.9.3")

    def unexpected_request():
        raise AssertionError("fresh cache should avoid PyPI")

    monkeypatch.setattr(update_check, "_fetch_latest_version", unexpected_request)

    assert update_check.available_update(cache_path=cache, now=1001) == ("0.9.3", "0.9.4")


def test_current_release_does_not_warn(tmp_path, monkeypatch):
    monkeypatch.setattr(update_check.metadata, "version", lambda _: "0.9.4")
    monkeypatch.setattr(update_check, "_fetch_latest_version", lambda: "0.9.4")

    assert update_check.available_update(cache_path=tmp_path / "cache", now=1000) is None


def test_network_failure_is_silent_and_cached(tmp_path, monkeypatch):
    cache = tmp_path / "update-check.json"
    monkeypatch.setattr(update_check.metadata, "version", lambda _: "0.9.3")

    def unavailable():
        raise OSError("offline")

    monkeypatch.setattr(update_check, "_fetch_latest_version", unavailable)

    assert update_check.available_update(cache_path=cache, now=1000) is None
    assert json.loads(cache.read_text(encoding="utf-8")) == {
        "checked_at": 1000,
        "latest_version": None,
    }


def test_check_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv(update_check.DISABLE_ENV, "1")

    def unexpected_version_lookup(_):
        raise AssertionError("disabled check should do no work")

    monkeypatch.setattr(update_check.metadata, "version", unexpected_version_lookup)

    assert update_check.available_update(cache_path=tmp_path / "cache", now=1000) is None


def test_warning_includes_requested_guidance(monkeypatch):
    monkeypatch.setattr(
        update_check,
        "available_update",
        lambda: ("0.9.3", "0.9.4"),
    )

    warning = update_check.update_warning_message()

    assert warning is not None
    assert "0.9.3 -> 0.9.4" in warning
    assert "important updates and fixes for logging, converting, and contributing data" in warning
    assert "All updates are backwards compatible and won't require you to change any code" in warning
    assert "pip install --upgrade oopsie-data-tools" in warning


@pytest.mark.parametrize(
    "argv",
    [
        ["show-config"],
        ["--help"],
        ["--version"],
    ],
    ids=["subcommand", "help", "version"],
)
def test_every_cli_invocation_prints_warning_at_the_end(argv, monkeypatch):
    emitted = []
    monkeypatch.setattr(cli, "_interactive_update_warning", lambda: "update notice")
    monkeypatch.setattr(cli, "_emit_update_warning", emitted.append)

    assert cli.main(argv) == 0

    assert emitted == ["update notice"]


def test_update_notice_uses_the_shared_warning_logger(monkeypatch):
    calls = []

    def fake_warning(message):
        calls.append(message)

    monkeypatch.setattr(cli.logger, "warning", fake_warning)

    cli._emit_update_warning("update notice")

    assert calls == ["update notice"]


def test_update_check_import_failure_does_not_break_cli(monkeypatch):
    class InteractiveStderr:
        @staticmethod
        def isatty():
            return True

    real_import = builtins.__import__

    def fail_update_check_import(name, *args, **kwargs):
        if name == "oopsie_data_tools.utils.update_check":
            raise ImportError("broken optional module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(cli.sys, "stderr", InteractiveStderr())
    monkeypatch.setattr(builtins, "__import__", fail_update_check_import)

    assert cli._interactive_update_warning() is None


def test_ctrl_c_during_update_check_uses_cli_interrupt_handler(monkeypatch, caplog):
    def interrupted():
        raise KeyboardInterrupt

    emitted = []
    caplog.set_level("INFO")
    monkeypatch.setattr(cli, "_interactive_update_warning", interrupted)
    monkeypatch.setattr(cli, "_emit_update_warning", emitted.append)

    assert cli.main(["--version"]) == 130
    assert "Interrupted." in caplog.text
    assert emitted == []
