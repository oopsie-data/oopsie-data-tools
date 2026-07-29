"""Scaffolding shared by the converter examples in this directory.

Everything schema-shaped — root attrs, robot states, actions, video paths, annotations —
lives in :mod:`oopsie_data_tools.utils.conversion_utils` and is used from there. What is
left over is the part that is the same in every converter but is not about the format:
the CLI shape, the output directory layout, and a batch loop that keeps going when one
episode fails.

A converter in this directory is then three things:

1. ``build_profile(...)`` — the :class:`RobotProfile` describing the source robot;
2. ``convert_one(item, episode_id, out)`` — read one source episode, write one HDF5;
3. a ``main`` that discovers items and hands them to :func:`run_batch`.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

from tqdm import tqdm

T = TypeVar("T")


class Skip(Exception):
    """Raised by a converter to drop one source episode without counting it as a failure.

    Use it for episodes that are legitimately out of scope — too short for the validator's
    duration bounds, missing a camera — and let real errors propagate as themselves.
    """


@dataclass(frozen=True)
class ConversionOutput:
    """The ``{output_dir}/{episode_id}.h5`` + ``{output_dir}/videos/`` layout.

    Videos are *copied* rather than symlinked: a symlink survives local validation and then
    uploads as a dangling path.
    """

    output_dir: Path
    videos_dir: Path

    @classmethod
    def create(cls, output_dir: Path | str) -> "ConversionOutput":
        output_dir = Path(output_dir).resolve()
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        return cls(output_dir=output_dir, videos_dir=videos_dir)

    def episode_h5(self, episode_id: str) -> Path:
        return self.output_dir / f"{episode_id}.h5"

    def video_path(self, episode_id: str, camera: str) -> Path:
        """Absolute destination for one camera's MP4.

        Pass it straight to ``write_video_paths``, which relativizes against the .h5 file.
        """
        return self.videos_dir / f"{episode_id}_{camera}.mp4"

    def copy_video(self, source: Path, episode_id: str, camera: str) -> Path:
        if not Path(source).is_file():
            raise Skip(f"missing source video for camera {camera!r}")
        destination = self.video_path(episode_id, camera)
        shutil.copy2(source, destination)
        return destination


def add_common_args(parser: argparse.ArgumentParser, *, source_help: str) -> None:
    """Add the arguments every converter in this directory takes."""
    parser.add_argument("--source", "-s", type=Path, required=True, help=source_help)
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory for the HDF5 files and their videos/ subdirectory.",
    )
    parser.add_argument(
        "--lab-id",
        required=True,
        help="Your registered lab id. Written as a root attr; 'your_lab_id' is rejected.",
    )
    parser.add_argument(
        "--operator-name",
        required=True,
        help="Who ran the episodes. Written as a root attr.",
    )
    parser.add_argument(
        "--annotator-name",
        required=True,
        help="Name of the episode_annotations/<annotator>/ subgroup the labels land in.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Starting numeric counter for episode IDs (default: 0).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Stop after writing this many episodes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite output files that already exist (default: leave them alone).",
    )


def default_episode_id(_item: object, counter: int) -> str:
    return f"episode_{counter:06d}"


def run_batch(
    items: Sequence[T] | Iterable[T],
    convert_one: Callable[[T, str, ConversionOutput], None],
    *,
    out: ConversionOutput,
    episode_id_for: Callable[[T, int], str] = default_episode_id,
    label: Callable[[T], str] = str,
    start_id: int = 0,
    desc: str = "Converting",
    overwrite: bool = False,
    max_episodes: int | None = None,
) -> dict[str, int]:
    """Convert every item, counting outcomes instead of stopping at the first failure.

    ``convert_one`` writes ``out.episode_h5(episode_id)``; raising :class:`Skip` drops the
    item. The counter behind ``episode_id_for`` advances for every item, skipped ones
    included, so an id stays tied to its source episode across reruns with different
    filters.
    """
    items = list(items)
    counts = {"written": 0, "exists": 0, "skipped": 0, "failed": 0}

    with tqdm(items, desc=desc, unit="ep") as progress:
        for offset, item in enumerate(progress):
            if max_episodes is not None and counts["written"] >= max_episodes:
                break
            episode_id = episode_id_for(item, start_id + offset)
            progress.set_postfix_str(label(item)[:48])

            if out.episode_h5(episode_id).exists() and not overwrite:
                counts["exists"] += 1
                continue
            try:
                convert_one(item, episode_id, out)
            except Skip as skip:
                tqdm.write(f"  – {episode_id}: {skip}")
                counts["skipped"] += 1
            except Exception as exc:  # noqa: BLE001 — one bad episode must not end the run
                tqdm.write(f"  ✗ {episode_id}: {exc}")
                counts["failed"] += 1
            else:
                counts["written"] += 1

    return counts


def report(counts: dict[str, int]) -> None:
    tqdm.write(
        f"\nDone: {counts['written']} written, {counts['exists']} already existed, "
        f"{counts['skipped']} skipped, {counts['failed']} failed.\n"
        "Next: oopsie-data validate --path <output-dir>"
    )
