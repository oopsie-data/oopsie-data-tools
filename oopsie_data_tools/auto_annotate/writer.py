"""Persist an annotation as a JSON sidecar and into a copy of the episode HDF5.

The sidecar is the record of what the model actually said: it holds the full
``annotation_requirement.md`` answer including the per-segment fields the stored taxonomy
has no room for.

The HDF5 write is deliberately conservative:

* it writes into a *copy* under ``annotated/``, so the verified ``raw/`` download keeps
  matching the sha256 the release index publishes;
* it writes under its own annotator name, so the human annotations already present in
  these files are never touched;
* it stamps ``source: "model"``, since the repo's writer defaults that field to "human".

Episode-level ``failure_category`` and ``severity`` are not things the model is asked for
directly -- the requirement puts them on segments. They are aggregated here, and how is
written down in :func:`aggregate_taxonomy` rather than left implicit.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import h5py

from oopsie_data_tools.annotation_tool.annotation_schema import SEVERITIES
from oopsie_data_tools.auto_annotate import config, schema
from oopsie_data_tools.utils.conversion_utils import write_episode_annotations

_SEVERITY_RANK = {name: rank for rank, name in enumerate(SEVERITIES)}


def aggregate_taxonomy(segments: List[dict]) -> Dict[str, Any]:
    """Roll per-segment labels up to the one episode-level slot the HDF5 provides.

    Categories are the union over segments, mapped to repo slugs. Severity is the worst
    single segment, not an average: an episode containing one catastrophic segment is a
    catastrophic episode regardless of how many harmless ones accompany it.
    """
    categories: List[str] = []
    for segment in segments:
        categories.extend(segment.get("failure_categories", []))

    severity = ""
    for segment in segments:
        candidate = segment.get("severity", "")
        if candidate in _SEVERITY_RANK and (
            not severity or _SEVERITY_RANK[candidate] > _SEVERITY_RANK[severity]
        ):
            severity = candidate

    return {"failure_category": schema.to_repo_categories(categories), "severity": severity}


def sidecar_path(episode: dict) -> Path:
    return config.ANNOT_DIR / episode["source"] / f"{episode['episode_id']}.json"


def write_sidecar(episode: dict, record: Dict[str, Any]) -> Path:
    path = sidecar_path(episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


def write_h5(episode: dict, record: Dict[str, Any]) -> Path:
    """Copy the episode and write the annotation into the copy. Returns the copy's path."""
    source_h5 = config.RAW_DIR / episode["release_path"]
    dest = config.ANNOTATED_DIR / episode["release_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(source_h5, dest)

    annotation = record["episode"]
    segments = record["segments"]
    rolled = aggregate_taxonomy(segments)

    with h5py.File(dest, "a") as handle:
        if config.ANNOTATOR_NAME in handle.get("episode_annotations", {}):
            del handle["episode_annotations"][config.ANNOTATOR_NAME]

        write_episode_annotations(
            handle,
            annotator_name=config.ANNOTATOR_NAME,
            success=schema.outcome_to_success(annotation["outcome"]),
            outcome=annotation["outcome"],
            episode_description=annotation["language_task"],
            failure_category=rolled["failure_category"],
            severity=rolled["severity"],
            additional_notes=annotation["rationale"],
            timestamp=record["annotated_at"],
        )

        group = handle["episode_annotations"][config.ANNOTATOR_NAME]
        # The repo writer hard-defaults this to "human"; these labels are not human.
        group.attrs["source"] = "model"
        group.attrs["model"] = record["model"]
        group.attrs["sidecar"] = str(sidecar_path(episode))

        # Per-segment detail has no place in the stored taxonomy, so it lives in a
        # subgroup of this annotator's group -- not of episode_annotations itself, which
        # the episode loader treats as a list of annotators.
        segment_group = group.require_group("segments")
        segment_group.attrs["count"] = len(segments)
        segment_group.attrs["schema"] = "oopsie_failure_segments_v0"
        for index, segment in enumerate(segments):
            entry = segment_group.require_group(f"{index:03d}")
            entry.attrs["start_timestep"] = int(segment["start_timestep"])
            entry.attrs["end_timestep"] = int(segment["end_timestep"])
            # Present in video mode: what the model said before conversion to timesteps.
            if "start_seconds" in segment:
                entry.attrs["start_seconds"] = float(segment["start_seconds"])
                entry.attrs["end_seconds"] = float(segment["end_seconds"])
            entry.attrs["what_happened"] = segment["what_happened"]
            entry.attrs["how_to_recover"] = segment["how_to_recover"]
            entry.attrs["failure_categories"] = json.dumps(segment["failure_categories"])
            entry.attrs["severity"] = segment["severity"]
            entry.attrs["resetability"] = segment["resetability"]
    return dest


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
