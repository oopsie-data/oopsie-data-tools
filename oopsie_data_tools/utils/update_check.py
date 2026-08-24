"""Best-effort notification when a newer ``oopsie-data-tools`` release exists."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGE_NAME = "oopsie-data-tools"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
CHECK_INTERVAL_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 1.0
DISABLE_ENV = "OOPSIE_DISABLE_UPDATE_CHECK"

logger = logging.getLogger(__name__)


def _cache_path() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "oopsie-data" / "update-check.json"


def _release_tuple(version: str) -> tuple[int, int, int] | None:
    """Return a comparable tuple for the stable ``X.Y.Z`` versions this project ships."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version.strip())
    return tuple(int(part) for part in match.groups()) if match else None


def _read_cache(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_cache(path: Path, checked_at: float, latest_version: str | None) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps({"checked_at": checked_at, "latest_version": latest_version}),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        # A read-only home directory should not turn an optional update check into a failure.
        return


def _fetch_latest_version() -> str:
    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers={"User-Agent": f"{PACKAGE_NAME} update-check"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.load(response)
    return str(payload["info"]["version"])


def available_update(
    *, cache_path: Path | None = None, now: float | None = None
) -> tuple[str, str] | None:
    """Return ``(installed, latest)`` when PyPI has a newer stable release.

    Network and cache errors deliberately return ``None``. A failed request is cached as an
    attempt too, so an offline machine does not absorb the timeout on every invocation.
    """
    if os.environ.get(DISABLE_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return None

    try:
        installed = metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return None

    current_time = time.time() if now is None else now
    path = _cache_path() if cache_path is None else cache_path
    cached = _read_cache(path)
    checked_at = cached.get("checked_at")
    latest = cached.get("latest_version")
    is_fresh = (
        isinstance(checked_at, (int, float))
        and current_time - checked_at < CHECK_INTERVAL_SECONDS
        and current_time >= checked_at
    )

    if not is_fresh:
        try:
            latest = _fetch_latest_version()
        except (OSError, ValueError, KeyError, TypeError):
            latest = latest if isinstance(latest, str) else None
        _write_cache(path, current_time, latest)

    if not isinstance(latest, str):
        return None
    installed_release = _release_tuple(installed)
    latest_release = _release_tuple(latest)
    if installed_release is None or latest_release is None or latest_release <= installed_release:
        return None
    return installed, latest


def warn_if_outdated() -> None:
    """Log an actionable update notice, without ever disrupting the command."""
    try:
        update = available_update()
    except Exception:  # pragma: no cover - the update check must remain strictly optional
        return
    if update is None:
        return
    installed, latest = update
    logger.warning(
        "A newer oopsie-data release is available: %s -> %s. "
        "Upgrade with: pip install --upgrade oopsie-data-tools",
        installed,
        latest,
    )
