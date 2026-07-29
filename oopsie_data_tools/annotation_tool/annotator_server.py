#!/usr/bin/env python3
"""Minimal web UI for instruction -> rollout -> annotate -> save.

Flow for in the loop annotation (with rollouts):
1) Browser submits a language instruction.
2) External rollout loop polls the instruction, runs policy, saves MP4s & HDF5, then
   signals this server to show the videos + annotation form.
3) Browser fills the annotation form + clicks Save.
4) Rollout loop polls the saved annotation and writes it into the episode HDF5
   (via EpisodeRecorder).

Tool can also be launched as a standalone module to browse and annotate existing HDF5 episodes, without any rollout loop.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import h5py
from flask import Flask, abort, jsonify, request, send_file

from oopsie_data_tools.annotation_tool.annotation_schema import (
    ANNOTATION_SCHEMA_CURRENT,
    FAILURE_CATEGORIES,
    OUTCOME_SUCCESS,
    OUTCOMES,
    SEVERITIES,
    parse_taxonomy,
    read_annotation_attrs,
    write_annotation_attrs,
)
from oopsie_data_tools.utils.validation.annotation_completeness import is_complete

app = Flask(__name__)


def _json_payload() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_annotation_payload(payload: dict[str, Any]) -> str:
    """Return an error message if the payload uses vocabulary the schema does not know.

    Stored values are opaque slugs, so a typo would otherwise reach disk unnoticed — where
    v1's prose values at least looked obviously wrong. Only ``outcome`` is required; every
    other field is optional and is checked only if present.
    """
    outcome = str(payload.get("outcome", "") or "").strip().lower()
    if outcome not in OUTCOME_SUCCESS:
        return f"outcome must be one of {list(OUTCOMES)}, got {payload.get('outcome')!r}"

    categories = payload.get("failure_category") or []
    if not isinstance(categories, (list, tuple)):
        categories = [categories]
    unknown = [c for c in categories if str(c).strip() not in FAILURE_CATEGORIES]
    if unknown:
        return f"unrecognized failure_category: {unknown}"

    severity = str(payload.get("severity", "") or "").strip()
    if severity and severity not in SEVERITIES:
        return f"severity must be one of {list(SEVERITIES)}, got {severity!r}"

    return ""


@dataclass
class ServerConfig:
    samples_dir: Path
    annotator_name: str
    # HDF5 browser + annotation form only (no instruction / rollout cards).
    browse_only: bool = False


class Runtime:
    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self.task_state: dict[str, Any] = {
            "status": "idle",  # idle | pending | running | annotating
            "pending_instruction": None,
            "last_instruction": "",
            "current_sample": None,  # dict: {sample_id, video_urls, language_instruction}
        }
        self.annotations: dict[str, Any] = {}  # sample_id -> annotation dict

    def set_pending_instruction(self, instruction: str) -> None:
        with self._lock:
            self.task_state["pending_instruction"] = instruction
            self.task_state["status"] = "pending"

    def start_task(self, instruction: str) -> None:
        with self._lock:
            self.task_state["last_instruction"] = instruction
            self.task_state["pending_instruction"] = None
            self.task_state["status"] = "running"

    def set_annotating(
        self, sample_id: str, video_urls: dict[str, str], language_instruction: str
    ) -> None:
        with self._lock:
            self.task_state["current_sample"] = {
                "sample_id": sample_id,
                "video_urls": video_urls,
                "language_instruction": language_instruction,
            }
            self.task_state["status"] = "annotating"

    def mark_done(self) -> None:
        with self._lock:
            if self.task_state.get("status") == "annotating":
                cs = self.task_state.get("current_sample")
                if isinstance(cs, dict):
                    sid = str(cs.get("sample_id", "")).strip()
                    if sid and sid not in self.annotations:
                        # Lets rollout poll observe "skip" (Back) vs still waiting.
                        self.annotations[sid] = {"__annotation_skipped__": True}
            self.task_state["status"] = "idle"
            self.task_state["current_sample"] = None

    def stamp_annotation(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Attach annotator, timestamp and schema to an annotation payload.

        Split out from :meth:`save_annotation` because the two annotation routes persist
        very differently: the rollout route parks the result in memory for the rollout loop
        to collect, while the HDF5 browser writes straight into the file and needs no
        in-memory entry at all.
        """
        reserved = {"annotated_at", "annotator", "source", "schema", "__annotation_skipped__"}
        annotation = {k: v for k, v in payload.items() if k not in reserved}
        annotation.update(
            {
                "annotated_at": _now_iso(),
                "annotator": self.cfg.annotator_name,
                "source": "human",
                "schema": ANNOTATION_SCHEMA_CURRENT,
            }
        )
        return annotation

    def save_annotation(self, sample_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Stamp an annotation and hold it for the rollout loop polling ``sample_id``."""
        annotation = self.stamp_annotation(payload)
        with self._lock:
            self.annotations[sample_id] = annotation
        return annotation

    def get_annotation(self, sample_id: str) -> dict[str, Any] | None:
        with self._lock:
            ann = self.annotations.get(sample_id)
            return ann if isinstance(ann, dict) else None


def _get_runtime() -> Runtime:
    rt = app.extensions.get("annotator_runtime")
    if isinstance(rt, Runtime):
        return rt
    raise RuntimeError("Runtime not configured. Call configure_runtime().")


def configure_runtime(
    samples_dir: Path,
    annotator_name: str,
    *,
    browse_only: bool = False,
) -> Runtime:
    cfg = ServerConfig(
        samples_dir=samples_dir.resolve(),
        annotator_name=annotator_name,
        browse_only=browse_only,
    )
    rt = Runtime(cfg)
    app.extensions["annotator_runtime"] = rt
    return rt


@app.get("/")
def index() -> str:
    html = _load_template("annotator.html")
    browse = "true" if _get_runtime().cfg.browse_only else "false"
    name_js = json.dumps(_get_runtime().cfg.annotator_name)
    html = html.replace(
        "<script>",
        f"<script>\n      const ANNOTATOR_BROWSE_ONLY = {browse};\n"
        f"      const ANNOTATOR_NAME = {name_js};\n",
        1,
    )
    return html


@app.get("/api/task/state")
def api_task_state():
    rt = _get_runtime()
    # Let the UI distinguish rollout idle vs annotation-only (browse) idle.
    return jsonify({**rt.task_state, "browse_only": rt.cfg.browse_only})


@app.post("/api/task/submit")
def api_task_submit():
    instruction = str(_json_payload().get("instruction", "")).strip()
    if not instruction:
        return jsonify({"error": "instruction is required"}), 400
    _get_runtime().set_pending_instruction(instruction)
    return jsonify({"status": "ok"})


@app.post("/api/task/start")
def api_task_start():
    instruction = str(_json_payload().get("instruction", "")).strip()
    if not instruction:
        return jsonify({"error": "instruction is required"}), 400
    _get_runtime().start_task(instruction)
    return jsonify({"status": "ok"})


@app.post("/api/task/annotating")
def api_task_annotating():
    payload = _json_payload()
    sample_id = str(payload.get("sample_id", "")).strip()
    video_urls = payload.get("video_urls", {})
    language_instruction = str(payload.get("language_instruction", "")).strip()
    if not sample_id:
        return jsonify({"error": "sample_id is required"}), 400
    if not isinstance(video_urls, dict) or not video_urls:
        return jsonify({"error": "video_urls must be a non-empty object"}), 400
    _get_runtime().set_annotating(sample_id, video_urls, language_instruction)
    return jsonify({"status": "ok"})


@app.post("/api/task/done")
def api_task_done():
    _get_runtime().mark_done()
    return jsonify({"status": "ok"})


@app.get("/api/annotations/<sample_id>")
def api_get_annotation(sample_id: str):
    ann = _get_runtime().get_annotation(sample_id)
    return jsonify(ann or {})


@app.post("/api/annotations")
def api_save_annotation_json() -> Any:
    """Save annotation with ``sample_id`` in the JSON body (avoids slashes in URL paths)."""
    payload = _json_payload()
    sample_id = str(payload.get("sample_id", "")).strip()
    if not sample_id:
        return jsonify({"error": "sample_id is required"}), 400
    answers = {k: v for k, v in payload.items() if k != "sample_id"}
    error = validate_annotation_payload(answers)
    if error:
        return jsonify({"error": error}), 400
    ann = _get_runtime().save_annotation(sample_id, answers)
    return jsonify({"status": "saved", "annotation": ann})


@app.get("/videos-path/<path:video_path>")
def serve_video_by_path(video_path: str):
    rt = _get_runtime()
    samples_root = rt.cfg.samples_dir.resolve()
    target_path = (samples_root / video_path).resolve()
    try:
        target_path.relative_to(samples_root)
    except ValueError:
        abort(403)
    if not target_path.exists() or not target_path.is_file():
        abort(404)
    if target_path.suffix.lower() == ".mp4":
        return send_file(target_path, mimetype="video/mp4")
    return send_file(target_path)


def _relative_to_samples_dir(samples_root: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(samples_root.resolve()).as_posix()
    except ValueError:
        return None


def _resolve_video_path(
    samples_root: Path, raw_path: str, h5_path: Path
) -> Path | None:
    candidate = Path(raw_path)

    if candidate.is_absolute():
        if candidate.exists():
            return candidate
    else:
        local_candidate = (h5_path.parent / candidate).resolve()
        if local_candidate.exists():
            return local_candidate

    basename_candidate = (h5_path.parent / candidate.name).resolve()
    if basename_candidate.exists():
        return basename_candidate

    samples_root_candidate = (samples_root.resolve() / candidate.name).resolve()
    if samples_root_candidate.exists():
        return samples_root_candidate

    return None


def _decode_h5_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")

    if hasattr(value, "shape") and getattr(value, "shape", None) == ():
        try:
            return _decode_h5_value(value.item())
        except Exception:
            return value

    if hasattr(value, "tolist"):
        try:
            listed = value.tolist()
            if isinstance(listed, list):
                return [_decode_h5_value(v) for v in listed]
            return _decode_h5_value(listed)
        except Exception:
            pass

    return value


def _read_h5_attr(group: h5py.Group | h5py.File, key: str, default: Any = "") -> Any:
    return _decode_h5_value(group.attrs.get(key, default))


class H5PathError(ValueError):
    """Invalid or unsafe HDF5 path from query string; carries HTTP status for API responses."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@app.errorhandler(H5PathError)
def _h5_path_error(e: H5PathError):
    return jsonify({"error": str(e)}), e.status


def _safe_h5_path_from_query(samples_root: Path) -> Path:
    raw = str(request.args.get("path", "")).strip()
    if not raw:
        raise H5PathError("path query parameter is required", 400)
    rel = Path(raw)
    target = (samples_root.resolve() / rel).resolve()
    try:
        target.relative_to(samples_root.resolve())
    except ValueError as e:
        raise H5PathError("path escapes samples directory", 403) from e
    if target.suffix.lower() != ".h5":
        raise H5PathError("path must refer to an .h5 file", 400)
    if not target.exists() or not target.is_file():
        raise H5PathError("HDF5 file not found", 404)
    return target


def _annotator_subgroup_looks_annotated(fa: h5py.Group) -> bool:
    src = str(_read_h5_attr(fa, "source", "") or "").strip().lower()
    if src == "human":
        return True
    if _read_h5_attr(fa, "success", None) is not None:
        return True
    # episode_description is v2; failure_description is the v1 spelling.
    for key in ("episode_description", "failure_description", "taxonomy"):
        if str(_read_h5_attr(fa, key, "") or "").strip():
            return True
    return False


def _read_existing_annotation_dict(ea: h5py.Group, annotator_name: str) -> dict[str, Any]:
    """Same episode_annotations extraction as ``/api/h5/sample`` (subgroup first, else JSON).

    Decoding — including the v1 upcast — is :func:`read_annotation_attrs`' job, so the form
    sees v2 keys and slugs regardless of which schema the file on disk uses. Nothing is
    rewritten here: a v1 file stays v1 until the annotator actually saves.
    """
    existing_annotation: dict[str, Any] = {}
    if annotator_name in ea.keys() and isinstance(ea[annotator_name], h5py.Group):
        existing_annotation = read_annotation_attrs(ea[annotator_name].attrs)

    if not existing_annotation:
        raw_failure = _read_h5_attr(ea, "failure_annotation", "")
        raw_s = str(raw_failure or "").strip()
        if raw_s:
            try:
                parsed = json.loads(raw_s)
                if isinstance(parsed, dict) and parsed:
                    # The pre-subgroup layout stored the flat annotation dict as JSON.
                    # Route it through the same upcast so one definition serves both.
                    existing_annotation = read_annotation_attrs(
                        {
                            "success": parsed.get("success"),
                            "failure_description": parsed.get("failure_description", ""),
                            "additional_notes": parsed.get("additional_notes", ""),
                            "taxonomy": json.dumps(
                                {
                                    "failure_category": parsed.get("failure_category", []),
                                    "severity": parsed.get("severity", ""),
                                    "success_category": parsed.get("success_category", ""),
                                }
                            ),
                        }
                    )
            except Exception:
                pass

    return existing_annotation


def _annotation_tick_level(ann: dict[str, Any]) -> int:
    """0 = not annotated, 1 = outcome only, 2 = complete for that outcome.

    Validation no longer requires anything beyond ``outcome``, so this is purely a prompt:
    it flags episodes where the chosen outcome asks about fields the annotator left blank.
    Which fields each outcome asks about lives in :mod:`annotation_completeness`, alongside
    the mirror of the form's conditional logic.
    """
    if str(ann.get("outcome", "")).strip().lower() not in OUTCOME_SUCCESS:
        return 0
    return 2 if is_complete(ann) else 1


def _h5_annotation_tick_level(h5f: h5py.File, annotator_name: str) -> int:
    ea = h5f.get("episode_annotations")
    if not isinstance(ea, h5py.Group):
        return 0
    ann = _read_existing_annotation_dict(ea, annotator_name)
    level = _annotation_tick_level(ann)
    if level > 0:
        return level
    sub = ea.get(annotator_name)
    if isinstance(sub, h5py.Group) and _annotator_subgroup_looks_annotated(sub):
        return 1
    return 0


def _dataset_summary(ds: h5py.Dataset) -> dict[str, Any]:
    """Shape/dtype summary for a dataset, tolerant of ``h5py.Empty`` (null) datasets."""
    shape = ds.shape
    is_tuple = isinstance(shape, tuple)
    empty = shape is None or (is_tuple and (len(shape) == 0 or 0 in shape))
    return {
        "shape": list(shape) if is_tuple else None,
        "dtype": str(ds.dtype),
        "empty": bool(empty),
    }


def _summarize_episode_fields(h5f: h5py.File) -> dict[str, Any]:
    """Compact, read-only summary of everything logged in an episode (issue #30).

    Reads only attrs + dataset shapes/dtypes (no array or video decoding) so the
    annotator can double-check that all fields were captured correctly.
    """
    attr_keys = ("schema", "episode_id", "operator_name", "lab_id", "language_instruction", "timestamp")
    fields: dict[str, Any] = {
        "attributes": {k: _read_h5_attr(h5f, k, "") for k in attr_keys},
        # parse_taxonomy is reused here as a generic "JSON object attr, missing is empty"
        # decoder — robot_profile is stored the same way the taxonomy is.
        "robot_profile": parse_taxonomy(_read_h5_attr(h5f, "robot_profile", "")),
        "robot_states": {},
        "actions": {},
        "video_paths": {},
    }

    obs = h5f.get("observations")
    if isinstance(obs, h5py.Group):
        rs = obs.get("robot_states")
        if isinstance(rs, h5py.Group):
            for key in rs.keys():
                if isinstance(rs[key], h5py.Dataset):
                    fields["robot_states"][key] = _dataset_summary(rs[key])
        vp = obs.get("video_paths")
        if isinstance(vp, h5py.Group):
            for cam in vp.keys():
                try:
                    fields["video_paths"][cam] = _decode_h5_value(vp[cam][()])
                except Exception:
                    continue

    ag = h5f.get("actions")
    if isinstance(ag, h5py.Group):
        for key in ag.keys():
            if isinstance(ag[key], h5py.Dataset):
                fields["actions"][key] = _dataset_summary(ag[key])

    fields["trajectory_length"] = next(
        (s["shape"][0] for s in fields["robot_states"].values() if s.get("shape")),
        None,
    )
    ea = h5f.get("episode_annotations")
    fields["annotators"] = list(ea.keys()) if isinstance(ea, h5py.Group) else []
    return fields


def _has_other_human_annotation(h5f: h5py.File, annotator_name: str) -> bool:
    """Whether a human other than ``annotator_name`` has annotated this episode (#26)."""
    ea = h5f.get("episode_annotations")
    if not isinstance(ea, h5py.Group):
        return False
    for name in ea.keys():
        if name == annotator_name:
            continue
        sub = ea[name]
        if not isinstance(sub, h5py.Group):
            continue
        source = str(_read_h5_attr(sub, "source", "") or "").strip().lower()
        if source and source != "human":
            continue  # VLM / automated annotators don't count as "another annotator"
        if _annotator_subgroup_looks_annotated(sub):
            return True
    return False


_H5_META_CACHE: dict[tuple[str, str], tuple[tuple[float, int], dict[str, Any]]] = {}
_H5_META_LOCK = threading.Lock()
_H5_META_CACHE_MAX = 5000


def _h5_meta(path: Path, annotator_name: str) -> dict[str, Any]:
    """Annotation summary of one episode, cached on (mtime, size).

    The browser polls ``/api/h5/list`` every 1.5 s per open tab, so without this every
    poll would reopen every episode in the tree.
    """
    try:
        st = path.stat()
    except OSError:
        return {"tick_level": 0, "annotated_by_others": False, "annotation": None}
    key = (str(path), annotator_name)
    stamp = (st.st_mtime, st.st_size)
    with _H5_META_LOCK:
        hit = _H5_META_CACHE.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]

    meta: dict[str, Any] = {"tick_level": 0, "annotated_by_others": False, "annotation": None}
    try:
        with h5py.File(path, "r") as h5f:
            meta["tick_level"] = _h5_annotation_tick_level(h5f, annotator_name)
            meta["annotated_by_others"] = _has_other_human_annotation(h5f, annotator_name)
            ea = h5f.get("episode_annotations")
            # Only the current annotator's OWN subgroup — never the legacy
            # episode-level failure_annotation fallback, which isn't annotator-scoped
            # and could surface someone else's labels (#27).
            if isinstance(ea, h5py.Group) and isinstance(ea.get(annotator_name), h5py.Group):
                meta["annotation"] = _read_existing_annotation_dict(ea, annotator_name)
    except Exception:
        meta = {"tick_level": 0, "annotated_by_others": False, "annotation": None}

    with _H5_META_LOCK:
        if len(_H5_META_CACHE) >= _H5_META_CACHE_MAX:
            _H5_META_CACHE.clear()
        _H5_META_CACHE[key] = (stamp, meta)
    return meta


@app.get("/api/h5/list")
def api_h5_list():
    rt = _get_runtime()
    root = rt.cfg.samples_dir.resolve()
    entries: list[dict[str, Any]] = []
    for p in root.rglob("*.h5"):
        if not p.is_file():
            continue
        rel = p.resolve().relative_to(root).as_posix()
        meta = _h5_meta(p, rt.cfg.annotator_name)
        tick_level = meta["tick_level"]
        others = meta["annotated_by_others"]
        annotated = tick_level >= 1
        entries.append(
            {
                "id": rel.replace("/", "__"),
                "rel_path": rel,
                "display_name": p.name,
                "mtime": p.stat().st_mtime,
                "annotated": annotated,
                "annotation_tick_level": tick_level,
                "annotated_by_others": others,
            }
        )
    entries.sort(key=lambda e: e.get("rel_path") or "")
    return jsonify(entries)


@app.get("/api/annotations/recent")
def api_recent_annotations():
    """The current annotator's recent distinct annotations, for the autofill picker (#27)."""
    rt = _get_runtime()
    root = rt.cfg.samples_dir.resolve()
    ann_name = rt.cfg.annotator_name
    try:
        limit = int(request.args.get("limit", 15))
    except (TypeError, ValueError):
        limit = 15
    limit = max(1, min(limit, 100))

    files = [p for p in root.rglob("*.h5") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    seen: set[str] = set()
    recent: list[dict[str, Any]] = []
    for p in files:
        if len(recent) >= limit:
            break
        ann = _h5_meta(p, ann_name)["annotation"]
        if ann is None:
            continue
        # Anything but a clean success has details worth copying — a suboptimal-execution
        # description is as reusable as a failure's.
        if str(ann.get("outcome", "")).strip().lower() in ("", "success"):
            continue
        key = json.dumps(
            {
                "failure_category": sorted(
                    str(x) for x in (ann.get("failure_category") or [])
                ),
                "severity": str(ann.get("severity", "")).strip(),
                "episode_description": str(ann.get("episode_description", "")).strip(),
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        rel = p.resolve().relative_to(root).as_posix()
        recent.append({**ann, "__source_rel_path": rel, "__source_name": p.stem})
    return jsonify(recent)


@app.get("/api/h5/sample")
def api_h5_sample():
    rt = _get_runtime()
    samples_root = rt.cfg.samples_dir.resolve()
    h5_path = _safe_h5_path_from_query(samples_root)

    video_urls: dict[str, str] = {}
    existing_annotation: dict[str, Any] = {}
    episode_fields: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "rel_path": h5_path.resolve().relative_to(samples_root).as_posix()
    }

    with h5py.File(h5_path, "r") as h5f:
        metadata["language_instruction"] = (
            str(_read_h5_attr(h5f, "language_instruction", "")) or ""
        )
        metadata["episode_id"] = str(_read_h5_attr(h5f, "episode_id", "")) or ""
        metadata["operator_name"] = str(_read_h5_attr(h5f, "operator_name", "")) or ""
        episode_fields = _summarize_episode_fields(h5f)

        ea = h5f.get("episode_annotations")
        if isinstance(ea, h5py.Group):
            ann_name = rt.cfg.annotator_name
            if ann_name in ea.keys() and isinstance(ea[ann_name], h5py.Group):
                fa = ea[ann_name]
                metadata["success"] = _read_h5_attr(fa, "success", None)
            existing_annotation = _read_existing_annotation_dict(ea, ann_name)

        video_paths_group = h5f.get("observations/video_paths")
        if isinstance(video_paths_group, h5py.Group):
            for cam in video_paths_group.keys():
                try:
                    raw_path = _decode_h5_value(video_paths_group[cam][()])
                except Exception:
                    continue
                if not isinstance(raw_path, str) or not raw_path:
                    continue
                vp = _resolve_video_path(samples_root, raw_path, h5_path)
                if vp is None:
                    continue
                rel = _relative_to_samples_dir(samples_root, vp)
                if rel is None:
                    continue
                video_urls[str(cam)] = f"/videos-path/{quote(rel, safe='/')}"

    # Fallback: if HDF5 doesn't store `observations/video_paths/*`, try to infer
    # videos from files next to the H5: `<stem>_<cam>.mp4`.
    if not video_urls:
        stem = h5_path.stem
        for mp4 in sorted(h5_path.parent.glob(f"{stem}_*.mp4")):
            cam = mp4.stem[len(stem) + 1 :]  # after "<stem>_"
            if not cam:
                continue
            rel = _relative_to_samples_dir(samples_root, mp4)
            if rel is None:
                continue
            video_urls[cam] = f"/videos-path/{quote(rel, safe='/')}"

    return jsonify(
        {
            "rel_path": metadata["rel_path"],
            "metadata": metadata,
            "video_urls": video_urls,
            "existing_annotation": existing_annotation,
            "episode_fields": episode_fields,
        }
    )


@app.post("/api/h5/annotations")
def api_h5_save_annotation():
    rt = _get_runtime()
    samples_root = rt.cfg.samples_dir.resolve()
    h5_path = _safe_h5_path_from_query(samples_root)

    payload = _json_payload()
    error = validate_annotation_payload(payload)
    if error:
        return jsonify({"error": error}), 400

    # Stamp only. The annotation's home is the HDF5 file written below; keying the
    # in-memory rollout dict by an absolute path would put two unrelated kinds of key
    # into it for no reader.
    ann = rt.stamp_annotation(payload)

    try:
        with h5py.File(h5_path, "r+") as f:
            ea = f.require_group("episode_annotations")
            ag = ea.require_group(ann["annotator"])
            write_annotation_attrs(ag, ann)
    except Exception as e:
        return jsonify({"error": f"could not write annotation: {e}"}), 500

    return jsonify({"status": "saved", "annotation": ann})


@app.post("/api/h5/instruction")
def api_h5_set_instruction():
    """Overwrite an episode's ``language_instruction`` root attr (issue #31)."""
    rt = _get_runtime()
    samples_root = rt.cfg.samples_dir.resolve()
    h5_path = _safe_h5_path_from_query(samples_root)

    instruction = str(_json_payload().get("instruction", "")).strip()
    if not instruction:
        return jsonify({"error": "instruction is required"}), 400

    try:
        with h5py.File(h5_path, "r+") as f:
            f.attrs["language_instruction"] = instruction
    except Exception as e:
        return jsonify({"error": f"could not write instruction: {e}"}), 500

    return jsonify({"status": "saved", "language_instruction": instruction})


TEMPLATE_DIR = Path(__file__).parent / "ui"


def _load_template(filename: str) -> str:
    path = (TEMPLATE_DIR / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing template: {path}")
    return path.read_text(encoding="utf-8")


def _port_in_use(port: int, host: str = "localhost") -> bool:
    """Whether something is already listening on ``host:port``.

    Connecting rather than test-binding: a bind probe also trips on sockets in TIME_WAIT,
    which werkzeug's SO_REUSEADDR would have accepted anyway.
    """
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def run_server(
    samples_dir: Path,
    annotator_name: str,
    port: int = 5001,
    open_browser: bool = True,
    with_rollouts: bool = False,
) -> int:
    """Configure the runtime and serve the annotation UI (blocks until interrupted).

    Backs the ``oopsie-data annotate`` CLI command.
    """
    samples_dir = Path(samples_dir).resolve()
    samples_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime(
        samples_dir=samples_dir,
        annotator_name=annotator_name.strip(),
        browse_only=bool(not with_rollouts),
    )

    # Checked before the browser opens: werkzeug would otherwise fail its own bind a
    # moment later, leaving a tab pointed at whatever already owns the port.
    if _port_in_use(port):
        logging.getLogger(__name__).error(
            "Port %d is already in use — another annotator is probably still running "
            "(open http://localhost:%d/ to check). Stop it, or pass --port with a free port.",
            port,
            port,
        )
        return 1

    if open_browser:
        webbrowser.open(f"http://localhost:{port}/")

    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    app.run(host="localhost", port=port)
    return 0
