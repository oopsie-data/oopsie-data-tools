"""Schema-agnostic episode representation used by loader and validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile

# oopsie-data episode format versions.
#   v1 → original layout.
#   v2 → strict superset of v1: adds optional ``temporal_segments`` attrs on
#        ``episode_annotations/<annotator>`` subgroups. All v1 files remain valid.
# New writers stamp CURRENT_OOPSIE_DATA_SCHEMA; readers accept every supported version.
OOPSIE_DATA_SCHEMA_V1 = "oopsiedata_format_v1"
OOPSIE_DATA_SCHEMA_V2 = "oopsiedata_format_v2"
CURRENT_OOPSIE_DATA_SCHEMA = OOPSIE_DATA_SCHEMA_V2
SUPPORTED_OOPSIE_DATA_SCHEMAS = (OOPSIE_DATA_SCHEMA_V1, OOPSIE_DATA_SCHEMA_V2)


@dataclass
class VideoInfo:
    """Camera video metadata, constructable from an MP4 file or in-memory frames."""

    frame_count: int
    fps: float
    width: int
    height: int
    path: Optional[str] = None
    crf: Optional[float] = None

    @classmethod
    def from_frames(
        cls, frames: list[np.ndarray], fps: float, crf: Optional[float] = None
    ) -> VideoInfo:
        """Build VideoInfo from in-memory frame buffers (recorder pre-save path)."""
        if not frames:
            raise ValueError("Cannot build VideoInfo from empty frame list")
        h, w = frames[0].shape[:2]
        return cls(frame_count=len(frames), fps=fps, width=w, height=h, crf=crf)


@dataclass
class EpisodeData:
    """Normalized episode data, independent of source schema.

    Populated by episode_loader (from HDF5) or constructed directly by the
    recorder for pre-save validation. Both always supply a robot profile:
    episode_loader rejects any schema that does not embed one.
    """

    robot_profile: RobotProfile
    language_instruction: str
    episode_id: str
    lab_id: str
    operator_name: str
    trajectory_length: int
    control_freq: float
    # Eagerly-loaded arrays; shape (T, ...) for each key.
    observations: dict[str, np.ndarray]
    # Only non-empty action datasets; shape (T, ...) for each key.
    actions: dict[str, np.ndarray]
    # Per-camera metadata.
    videos: dict[str, VideoInfo]
    # Optional: annotator_name → {attr_key: attr_val}.
    annotations: Optional[dict[str, dict[str, Any]]] = field(default=None)
    # Additional sensors stored as arrays; shape (T, ...) for each key.
    additional_data: dict[str, np.ndarray] = field(default_factory=dict)
    # Additional sensors stored as MP4.
    additional_videos: dict[str, VideoInfo] = field(default_factory=dict)
