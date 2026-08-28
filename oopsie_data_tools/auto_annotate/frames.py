"""Turn an episode's video into a small set of labelled frames the model can reason over.

Two decisions here carry the whole segment-annotation task:

*Which camera.* A wrist or tactile view cannot show whether the robot knocked something
over, so an exterior view is preferred and wrist views are used only as a fallback.

*How the model refers to time.* The timestep index is drawn onto each frame. The model is
asked to answer with those indices, which removes any dependence on it estimating elapsed
seconds — and seconds are genuinely ambiguous in this release, where a container can report
15 fps while the episode's own ``source_video_fps`` attr says 10.

Frame count maps to timestep by position, not by assuming the two are equal: most episodes
have ``frame_count == timestep_count``, but ``source_alignment: trim-to-shortest`` means
some do not.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional, Sequence

import cv2

# Preferred camera names, best first. Matched as substrings of the camera token.
EXTERIOR_HINTS = (
    "exterior", "front", "scene", "overhead", "top", "head", "side", "high",
    "left", "right", "cam_2", "camera_0", "image", "main", "base",
)
# Views that cannot show scene-level consequences; used only if nothing else exists.
CLOSEUP_HINTS = ("wrist", "tactile", "gripper", "ego")


def camera_of(video_path: str, episode_id: str) -> str:
    """The camera token in a ``...<episode_id>_<camera>.mp4`` filename.

    Some labs prefix the video with a session timestamp the episode_id does not carry
    (``20260508_221406_episode_24_top.mp4`` for ``episode_24``), so the id is located
    anywhere in the stem rather than assumed to start it.
    """
    stem = Path(video_path).stem
    marker = episode_id + "_"
    position = stem.find(marker)
    if position >= 0:
        return stem[position + len(marker):]
    return stem.rsplit("_", 1)[-1]


def choose_camera(videos: Sequence[str], episode_id: str) -> str:
    """Pick the most informative video for judging task progress."""
    if not videos:
        raise ValueError("episode has no videos")
    names = [(camera_of(v, episode_id).lower(), v) for v in videos]

    for hint in EXTERIOR_HINTS:
        for name, path in names:
            if hint in name and not any(c in name for c in CLOSEUP_HINTS):
                return path
    for name, path in names:
        if not any(c in name for c in CLOSEUP_HINTS):
            return path
    return videos[0]


def _label(frame, text: str):
    """Draw a legible timestep tag in the top-left corner."""
    scale, thickness = 0.6, 2
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(frame, (0, 0), (w + 12, h + 14), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, h + 6), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def extract(
    video: Path,
    timestep_count: int,
    n_frames: int = 16,
    max_side: int = 448,
    quality: int = 72,
    out_dir: Optional[Path] = None,
) -> List[dict]:
    """Uniformly sample ``n_frames`` labelled frames.

    Returns one record per frame with its timestep index and base64 JPEG. Frames are read
    by seeking, which is exact enough for the short clips in this release and far cheaper
    than decoding every frame of a 2000-step episode.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise IOError(f"cannot open video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise IOError(f"video reports no frames: {video}")

    count = min(n_frames, frame_count)
    indices = [round(i * (frame_count - 1) / max(count - 1, 1)) for i in range(count)]

    records: List[dict] = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        # Position within the video maps to position within the trajectory.
        timestep = round(frame_index * (timestep_count - 1) / max(frame_count - 1, 1))

        height, width = frame.shape[:2]
        if max(height, width) > max_side:
            ratio = max_side / max(height, width)
            frame = cv2.resize(frame, (int(width * ratio), int(height * ratio)),
                               interpolation=cv2.INTER_AREA)
        frame = _label(frame, f"t={timestep}")

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            continue
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"t{timestep:06d}.jpg").write_bytes(buffer.tobytes())
        records.append(
            {
                "timestep": timestep,
                "frame_index": frame_index,
                "jpeg_b64": base64.b64encode(buffer.tobytes()).decode("ascii"),
            }
        )
    capture.release()
    if not records:
        raise IOError(f"decoded no frames from {video}")
    return records


def probe(video: Path) -> dict:
    capture = cv2.VideoCapture(str(video))
    info = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return info
