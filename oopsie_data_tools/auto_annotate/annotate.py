"""Annotate one episode: episode-level first, then failure segments.

The two tasks run as one two-turn conversation rather than two independent calls. The
requirement states the order matters -- a segment is defined as a period not progressing
toward *the episode's* task -- so the second turn carries the first turn's answer and
cannot score segments against a task it invented. Sending the episode once and continuing
the conversation costs the same as two calls here (the gateway reports no prompt caching),
so the correctness argument decides it.

Two input modes:

``video``   the whole episode, normalised by :mod:`clip`. The gateway samples it
            server-side, so the model can only refer to elapsed seconds, which are
            converted back to timesteps here.
``frames``  a fixed set of frames with timestep indices drawn on them. Coarser in time but
            exact in index, and far cheaper for long episodes.

Model output is schema-constrained, so enums are not re-validated. What is checked is what
a schema cannot express: that bounds fall inside the episode, that they are ordered, and
that the segments agree with the episode-level label. Disagreements are recorded as
warnings rather than repaired -- a "success" episode carrying three failure segments is a
signal worth seeing, not a bug to paper over.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import h5py

from oopsie_data_tools.auto_annotate import client as client_module
from oopsie_data_tools.auto_annotate import clip, config, frames, prompts, schema, writer


def language_instruction(h5_path) -> Optional[str]:
    with h5py.File(h5_path, "r") as handle:
        value = handle.attrs.get("language_instruction")
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return str(value) if value else None


def seconds_to_timestep(seconds: float, duration: float, timestep_count: int) -> int:
    """Map a moment in the video onto a trajectory index.

    Normalisation preserves duration, so this holds for the normalised clip too. Note the
    container's frame rate is not consulted: several episodes here disagree with their own
    ``source_video_fps`` attr, and duration is the quantity both sides agree on.
    """
    if duration <= 0:
        return 0
    fraction = min(max(seconds / duration, 0.0), 1.0)
    return int(round(fraction * (timestep_count - 1)))


def check_segments(
    segments: List[dict], outcome: str, last_timestep: int
) -> Tuple[List[dict], List[str]]:
    """Return ``(clean_segments, warnings)``; bounds are clamped, not dropped."""
    warnings: List[str] = []
    clean: List[dict] = []

    for index, segment in enumerate(segments):
        start = int(segment["start_timestep"])
        end = int(segment["end_timestep"])

        if start < 0 or end > last_timestep:
            warnings.append(
                f"segment {index}: bounds [{start}, {end}] outside [0, {last_timestep}]"
            )
        start = max(0, min(start, last_timestep))
        end = max(0, min(end, last_timestep))

        if end < start:
            warnings.append(f"segment {index}: end {end} before start {start}; swapped")
            start, end = end, start

        entry = dict(segment)
        entry["start_timestep"] = start
        entry["end_timestep"] = end
        clean.append(entry)

    clean.sort(key=lambda s: (s["start_timestep"], s["end_timestep"]))

    for first, second in zip(clean, clean[1:]):
        if second["start_timestep"] < first["end_timestep"]:
            warnings.append(
                f"segments overlap: [{first['start_timestep']}, {first['end_timestep']}] "
                f"and [{second['start_timestep']}, {second['end_timestep']}]"
            )

    if outcome == "success" and clean:
        warnings.append(f"outcome 'success' but {len(clean)} failure segment(s) reported")
    if outcome != "success" and not clean:
        warnings.append(f"outcome {outcome!r} but no failure segments reported")

    return clean, warnings


def _build_input(episode: dict, mode: str, n_frames: int, budget: Optional[int],
                 save_frames: bool) -> Tuple[List[Dict[str, Any]], dict]:
    """The content parts describing the episode, plus a record of what was sent."""
    video_rel = frames.choose_camera(episode["videos"], episode["episode_id"])
    source_video = config.RAW_DIR / video_rel
    probe = frames.probe(source_video)
    duration = probe["frames"] / probe["fps"] if probe["fps"] > 0 else 0.0

    meta = {
        "mode": mode,
        "video": video_rel,
        "camera": frames.camera_of(video_rel, episode["episode_id"]),
        "duration_seconds": round(duration, 3),
        "source_resolution": [probe["width"], probe["height"]],
        "source_fps": probe["fps"],
        "timestep_count": episode["timestep_count"],
    }

    if mode == "video":
        normalised = clip.normalize(
            source_video,
            episode["episode_id"],
            episode["source"],
            duration=duration,
            budget_tokens=budget,
            source_long_side=max(probe["width"], probe["height"]),
        )
        meta["clip"] = str(normalised.relative_to(config.CLIP_DIR))
        meta["clip_bytes"] = normalised.stat().st_size
        return [client_module.Client.video_part(normalised)], meta

    frame_dir = (
        config.FRAME_DIR / episode["source"] / episode["episode_id"] if save_frames else None
    )
    records = frames.extract(
        source_video, episode["timestep_count"], n_frames=n_frames, out_dir=frame_dir
    )
    meta["n_frames"] = len(records)
    meta["timesteps"] = [record["timestep"] for record in records]
    return [client_module.Client.image_part(r["jpeg_b64"]) for r in records], meta


def annotate_episode(
    episode: dict,
    model_client: "client_module.Client",
    mode: str = "video",
    n_frames: int = 16,
    budget_tokens: Optional[int] = 24000,
    save_frames: bool = False,
) -> Dict[str, Any]:
    """Run both tasks for one episode and return the full record."""
    h5_path = config.RAW_DIR / episode["release_path"]
    media, meta = _build_input(episode, mode, n_frames, budget_tokens, save_frames)
    instruction = language_instruction(h5_path)
    last_timestep = episode["timestep_count"] - 1
    unit = "seconds" if mode == "video" else "timestep"

    # ── Turn 1: episode level ─────────────────────────────────────────────────
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": prompts.SYSTEM},
        {
            "role": "user",
            "content": [client_module.Client.text_part(prompts.episode_prompt(instruction))]
            + media,
        },
    ]
    first = model_client.complete_json(history, schema.EPISODE_SCHEMA, max_tokens=8000)
    episode_annotation = first["parsed"]

    # ── Turn 2: failure segments ──────────────────────────────────────────────
    extent = meta["duration_seconds"] if unit == "seconds" else last_timestep
    history = history + [
        {"role": "assistant", "content": first["content"]},
        {
            "role": "user",
            "content": prompts.segment_prompt(
                episode_annotation["language_task"],
                episode_annotation["outcome"],
                unit,
                extent,
            ),
        },
    ]
    second = model_client.complete_json(
        history, schema.segment_schema(unit), max_tokens=12000
    )

    raw_segments = second["parsed"]["segments"]
    if unit == "seconds":
        for segment in raw_segments:
            segment["start_timestep"] = seconds_to_timestep(
                segment["start_seconds"], meta["duration_seconds"], episode["timestep_count"]
            )
            segment["end_timestep"] = seconds_to_timestep(
                segment["end_seconds"], meta["duration_seconds"], episode["timestep_count"]
            )
    segments, warnings = check_segments(
        raw_segments, episode_annotation["outcome"], last_timestep
    )

    return {
        "episode_id": episode["episode_id"],
        "source": episode["source"],
        "release_path": episode["release_path"],
        "model": model_client.model,
        "annotated_at": writer.now(),
        "input": dict(meta, provided_instruction=instruction),
        "episode": episode_annotation,
        "segments": segments,
        "warnings": warnings,
        "usage": {"episode_call": first["usage"], "segment_call": second["usage"]},
    }
