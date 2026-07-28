"""Single definition of how a human annotation is written into an episode HDF5.

Kept in a dependency-light module (json + h5py only) so both the in-the-loop
recorder patch and the web annotator server can share it without either pulling in
the other's heavier imports.
"""

from __future__ import annotations

import json
from typing import Any

import h5py


def write_annotation_attrs(group: h5py.Group, annotation: dict[str, Any]) -> None:
    """Write the oopsie annotation attribute set onto an ``episode_annotations`` subgroup.

    ``binary_success`` maps to the numeric ``success`` attr (1.0/0.0); the failure
    taxonomy plus the optional qualified-success category live in the ``taxonomy`` JSON.
    Missing keys fall back to safe defaults, and a ``None`` success (an unrecognized
    ``binary_success``) is left unwritten rather than crashing h5py.
    """
    bs = str(annotation.get("binary_success", "")).strip().lower()
    success = 1.0 if bs == "success" else 0.0 if bs == "failure" else None

    group.attrs["schema"] = annotation.get("schema", "oopsie_failure_taxonomy_v1")
    group.attrs["source"] = annotation.get("source", "human")
    group.attrs["timestamp"] = annotation.get("annotated_at", annotation.get("timestamp", ""))
    if success is not None:
        group.attrs["success"] = float(success)
    group.attrs["failure_description"] = annotation.get("failure_description", "")

    taxonomy: dict[str, Any] = {
        "failure_category": annotation.get("failure_category", []),
        "severity": annotation.get("severity", ""),
    }
    success_category = str(annotation.get("success_category", "") or "").strip()
    if success_category:
        taxonomy["success_category"] = success_category
    group.attrs["taxonomy_schema"] = "oopsiedata_taxonomy_schema_v1"
    group.attrs["taxonomy"] = json.dumps(taxonomy, ensure_ascii=False)
    group.attrs["additional_notes"] = annotation.get("additional_notes", "")
