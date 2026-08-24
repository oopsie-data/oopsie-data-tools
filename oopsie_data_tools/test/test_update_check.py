"""Tests for the optional PyPI update notification."""

from __future__ import annotations

import json

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


def test_warning_includes_versions_and_upgrade_command(monkeypatch, caplog):
    monkeypatch.setattr(
        update_check,
        "available_update",
        lambda: ("0.9.3", "0.9.4"),
    )

    update_check.warn_if_outdated()

    assert "0.9.3 -> 0.9.4" in caplog.text
    assert "pip install --upgrade oopsie-data-tools" in caplog.text
