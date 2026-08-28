"""Normalise an episode video before sending it to the model.

Sending source video straight to the gateway works but costs wildly unpredictable amounts:
one 1280x720 15fps clip in this dataset billed 105,600 video tokens for 120 seconds, while
a 256x256 clip of similar length billed under 2,000. Cost tracks resolution and frame rate,
not just duration, and the source videos here span 192x192 to 1280x720.

Re-encoding to a fixed frame rate and a capped long side makes the cost a predictable
function of duration alone. Measured on that same 120s clip:

    source 1280x720 @15fps   105,600 tokens   112s to upload
    2fps, long side 448       13,440 tokens     4s
    2fps, long side 336        7,200 tokens     2s

Frame rate above 2fps buys nothing: 2fps and 4fps bill identically, so the gateway is
resampling to its own cadence and anything denser is wasted upload.

Duration is preserved, so a timestamp in the normalised clip refers to the same moment in
the original episode.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from oopsie_data_tools.auto_annotate import config

FPS = 2
LONG_SIDE = 336
CRF = 28

# Roughly 60 video tokens per second at LONG_SIDE=336, measured. Used to decide when a long
# episode needs a smaller frame to stay inside its budget.
TOKENS_PER_SECOND = 60.0


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def plan(duration: float, budget_tokens: Optional[int], source_long_side: int = 0) -> int:
    """The largest long side whose cost still fits the budget, capped at the source.

    Token count scales with pixel area, so the long side scales with the square root of the
    budget. Resolution is what decides whether fine detail survives: at 336px a peg-in-hole
    insertion was billed 1,440 tokens against a 24,000 budget and the model reported the
    peg as seated when it never went in. Spending the budget that was available is not an
    optimisation here, it is the difference between seeing the outcome and guessing it.

    Upscaling past the source resolution invents no detail and bills for it, so the source
    is a hard ceiling. The floor of 192 is the smallest source resolution in this release.
    """
    if not budget_tokens or duration <= 0:
        return min(LONG_SIDE, source_long_side) if source_long_side else LONG_SIDE
    estimate = duration * TOKENS_PER_SECOND
    scale = (budget_tokens / estimate) ** 0.5
    target = max(192, int(LONG_SIDE * scale) // 2 * 2)
    return min(target, source_long_side) if source_long_side else target


def normalize(
    source: Path,
    episode_id: str,
    source_name: str,
    duration: float,
    budget_tokens: Optional[int] = None,
    fps: int = FPS,
    source_long_side: int = 0,
) -> Path:
    """Return a path to the normalised clip, encoding it if not already cached."""
    long_side = plan(duration, budget_tokens, source_long_side)
    out = config.CLIP_DIR / source_name / f"{episode_id}_{fps}fps_{long_side}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".part.mp4")
    # scale='min(N,iw)':-2 never upscales, and -2 keeps the height even for libx264.
    subprocess.run(
        [
            _ffmpeg(), "-y", "-loglevel", "error", "-i", str(source),
            "-vf", f"fps={fps},scale='min({long_side},iw)':-2",
            "-an", "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p",
            str(partial),
        ],
        check=True,
    )
    partial.replace(out)
    return out
