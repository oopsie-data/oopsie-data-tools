"""Run the QA checks over local episodes and write the report behind ``oopsie-data qa``.

The report is one YAML document: a ``summary`` (counts, and the episodes flagged per check),
then one record per episode with its findings. Every finding carries the check's one-line
``description`` next to the values that tripped it, so the file reads without the source.
"""

from __future__ import annotations

import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml

from oopsie_data_tools.qa.action_checks import (
    load_action_state_arrays,
    profile_control_freq,
    run_action_qa_on_arrays,
)
from oopsie_data_tools.qa.config import CHECK_DESCRIPTIONS, DEFAULT_CHECKS, build_checks
from oopsie_data_tools.qa.video_checks import run_video_checks
from oopsie_data_tools.utils.h5 import find_episode_files
from oopsie_data_tools.utils.video_paths import read_video_paths, resolve_from_episode

REPORT_SCHEMA = "oopsie_qa_report_v1"
CAMERA_GROUP = "observations/video_paths/"

STATUS_FLAGGED = "flagged"
STATUS_CLEAN = "clean"
STATUS_NOT_CHECKED = "not_checked"
STATUS_ERROR = "error"


def episode_cameras(h5_path: Path) -> Dict[str, Path]:
    """Map camera name -> mp4 path, from the episode's ``observations/video_paths``."""
    cameras = {}
    for ref, stored in read_video_paths(h5_path).items():
        if ref.startswith(CAMERA_GROUP) and stored:
            cameras[ref[len(CAMERA_GROUP):]] = resolve_from_episode(stored, h5_path.parent)
    return cameras


def _merge(a: Optional[dict], b: dict) -> dict:
    if a is None:
        return b
    return {key: a[key] + b[key] for key in ("findings", "skips", "checked")}


def check_episode(h5_path: Path, config: dict) -> dict:
    """Action and video checks for one episode, as ``{"findings", "skips", "checked"}``.

    Raises on an unreadable file; :func:`run_qa` records that as an errored episode.
    """
    with h5py.File(h5_path, "r") as f:
        actions, states, profile = load_action_state_arrays(f)
    result = run_action_qa_on_arrays(actions, states, profile, config)
    video_cfg = config["video"]
    if any(cfg["enabled"] for cfg in video_cfg.values()):
        cameras = episode_cameras(h5_path)
        result = _merge(
            result,
            run_video_checks(cameras, states, profile_control_freq(profile), video_cfg, actions),
        )
    return result or {"findings": [], "skips": [], "checked": []}


def _check_job(job: Tuple[Path, dict]) -> Tuple[Optional[dict], Optional[str]]:
    path, config = job
    try:
        return check_episode(path, config), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _plain(value):
    """``value`` with numpy types and tuples turned into what ``yaml.safe_dump`` accepts."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return float(f"{value:.4g}") if np.isfinite(value) else str(value)
    return value


def _tool_version() -> str:
    try:
        return metadata.version("oopsie-data-tools")
    except metadata.PackageNotFoundError:
        return "unknown"


def _check_name(category: str, test: str) -> str:
    return f"{category}.{test}"


def _finding_record(finding: dict) -> dict:
    category, test = finding["category"], finding["test"]
    details = {k: v for k, v in finding.items() if k not in ("category", "test", "key")}
    return {
        "check": _check_name(category, test),
        "key": finding["key"],
        "description": CHECK_DESCRIPTIONS.get((category, test), ""),
        "details": _plain(details),
    }


def _skip_record(skip: dict) -> dict:
    return {
        "check": _check_name(skip["category"], skip["test"]),
        "key": skip["key"],
        "reason": skip["reason"],
    }


def _episode_record(path: Path, root: Path, result: Optional[dict], error: Optional[str]) -> dict:
    record = {"episode": path.relative_to(root).as_posix()}
    if error is not None:
        record.update(status=STATUS_ERROR, error=error)
        return record
    if result["findings"]:
        status = STATUS_FLAGGED
    elif result["checked"]:
        status = STATUS_CLEAN
    else:
        status = STATUS_NOT_CHECKED
    record["status"] = status
    record["checks_run"] = len(result["checked"])
    if result["findings"]:
        record["flagged_checks"] = sorted(
            {_check_name(f["category"], f["test"]) for f in result["findings"]}
        )
        record["findings"] = [_finding_record(f) for f in result["findings"]]
    if result["skips"]:
        record["skipped"] = [_skip_record(s) for s in result["skips"]]
    return record


def _config_overrides(config: dict) -> dict:
    """The values of ``config`` that differ from the defaults, nested like the config."""
    defaults = build_checks({})
    out: dict = {}
    for category, checks in config.items():
        for check, params in checks.items():
            changed = {k: v for k, v in params.items() if defaults[category][check][k] != v}
            if changed:
                out.setdefault(category, {})[check] = changed
    return out


def select_episodes(
    paths: List[Path], limit: Optional[int], seed: Optional[int]
) -> List[Path]:
    """All of ``paths``, or a random sample of ``limit`` of them in their original order."""
    if not limit or limit >= len(paths):
        return paths
    chosen = set(random.Random(seed).sample(range(len(paths)), limit))
    return [p for i, p in enumerate(paths) if i in chosen]


def run_qa(
    target: Path,
    config: dict,
    *,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    workers: int = 1,
    progress=None,
) -> dict:
    """Check the episode file ``target``, or every episode under the directory ``target``.

    ``progress`` is called with ``(done, total, record)`` after each episode. Returns the
    report as a plain dict; :func:`write_report` saves it.
    """
    target = Path(target).resolve()
    if target.is_file():
        found, root = [target], target.parent
    else:
        found, root = find_episode_files(target), target
    paths = select_episodes(found, limit, seed)

    jobs = [(p, config) for p in paths]
    if workers > 1 and len(jobs) > 1:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_check_job, jobs, chunksize=2)
    else:
        pool = None
        results = map(_check_job, jobs)

    episodes = []
    try:
        for done, (path, (result, error)) in enumerate(zip(paths, results), 1):
            record = _episode_record(path, root, result, error)
            episodes.append(record)
            if progress is not None:
                progress(done, len(paths), record)
    finally:
        if pool is not None:
            pool.shutdown()

    status_counts = Counter(e["status"] for e in episodes)
    flagged_per_check: Counter = Counter()
    for e in episodes:
        flagged_per_check.update(e.get("flagged_checks", []))
    enabled = [
        _check_name(category, test)
        for category, tests in DEFAULT_CHECKS.items()
        for test in tests
        if config[category][test]["enabled"]
    ]

    summary = {
        "schema": REPORT_SCHEMA,
        "tool_version": _tool_version(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "path": str(target),
        "episodes_found": len(found),
        "episodes_checked": len(episodes),
        "episodes_flagged": status_counts[STATUS_FLAGGED],
        "episodes_clean": status_counts[STATUS_CLEAN],
        "episodes_not_checked": status_counts[STATUS_NOT_CHECKED],
        "episodes_errored": status_counts[STATUS_ERROR],
        "flagged_episodes_per_check": dict(flagged_per_check.most_common()),
        "checks_enabled": enabled,
        "config_overrides": _plain(_config_overrides(config)),
    }
    return {"summary": summary, "episodes": episodes}


def write_report(report: dict, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(report, fh, sort_keys=False, allow_unicode=True, width=100)
