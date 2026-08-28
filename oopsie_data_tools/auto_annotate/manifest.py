"""Choose the episodes this run annotates, and record that choice.

The release index (``episodes.csv``) is the only thing consulted: it names every episode's
source lab, its ``.h5`` release path and the videos that belong to it. Selection is
deterministic given a seed so a run is reproducible and re-runnable without re-downloading.

"Most important sources" is resolved as the largest sources by episode count, which is the
only ranking the index supports. Within a source, episodes are spread across that lab's
task directories rather than taken in file order, so ten episodes are not ten attempts at
one task.
"""

from __future__ import annotations

import csv
import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from oopsie_data_tools.auto_annotate import config


def load_index(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def usable(row: dict) -> bool:
    """Rows the pipeline can actually annotate.

    An episode whose videos are not all present cannot be shown to a vision model, and an
    unknown schema would not round-trip through the annotation writer.
    """
    return (
        row.get("schema") == "oopsiedata_format_v1"
        and str(row.get("all_video_references_present", "")).lower() == "true"
        and int(row.get("camera_count") or 0) > 0
        and int(row.get("timestep_count") or 0) > 0
    )


def rank_sources(rows: Sequence[dict]) -> List[str]:
    """Source labs ordered by episode count, descending; ties broken by name."""
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[source_of(row)] += 1
    return [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def source_of(row: dict) -> str:
    return row["source_repo"].split("/")[-1]


def _task_of(row: dict) -> str:
    """The lab's own grouping for an episode — the directory holding the .h5."""
    return str(Path(row["release_path"]).parent)


def _spread(rows: Sequence[dict], count: int, rng: random.Random) -> List[dict]:
    """Take ``count`` rows, round-robin over task directories.

    Each task group is shuffled, then groups are visited in turn. This yields task variety
    when a lab has many tasks and degrades gracefully to a plain sample when it has one.
    """
    groups: Dict[str, List[dict]] = OrderedDict()
    for row in sorted(rows, key=lambda r: r["release_path"]):
        groups.setdefault(_task_of(row), []).append(row)
    for group in groups.values():
        rng.shuffle(group)

    order = sorted(groups)
    rng.shuffle(order)

    picked: List[dict] = []
    depth = 0
    while len(picked) < count:
        progressed = False
        for name in order:
            group = groups[name]
            if depth < len(group):
                picked.append(group[depth])
                progressed = True
                if len(picked) == count:
                    break
        if not progressed:  # every group exhausted
            break
        depth += 1
    return picked


def videos_of(row: dict) -> List[str]:
    return list(json.loads(row["video_reference_paths_json"] or "[]"))


def select(
    rows: Sequence[dict],
    n_sources: int = 10,
    per_source: int = 10,
    seed: int = 42,
) -> List[dict]:
    """Sampled episode records, ``per_source`` from each of the top ``n_sources`` labs."""
    good = [row for row in rows if usable(row)]
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for row in good:
        by_source[source_of(row)].append(row)

    chosen: List[dict] = []
    for source in rank_sources(good)[:n_sources]:
        available = by_source[source]
        if len(available) < per_source:
            raise ValueError(
                f"source {source} has {len(available)} usable episodes, need {per_source}"
            )
        # Seed per source so adding or reordering sources cannot change another's sample.
        rng = random.Random(f"{seed}:{source}")
        for row in _spread(available, per_source, rng):
            chosen.append(
                {
                    "source": source,
                    "episode_id": row["episode_id"],
                    "release_path": row["release_path"],
                    "videos": videos_of(row),
                    "timestep_count": int(row["timestep_count"]),
                    "camera_count": int(row["camera_count"]),
                    "sha256": row["release_file_sha256"] or "",
                    "h5_bytes": int(row["release_file_size_bytes"] or 0),
                }
            )
    return chosen


def build(
    index_csv: Path = config.INDEX_CSV,
    out: Path = config.MANIFEST,
    n_sources: int = 10,
    per_source: int = 10,
    seed: int = 42,
) -> List[dict]:
    episodes = select(load_index(index_csv), n_sources, per_source, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "repo": config.RELEASE_REPO,
                "revision": config.RELEASE_REVISION,
                "seed": seed,
                "n_sources": n_sources,
                "per_source": per_source,
                "episodes": episodes,
            },
            indent=2,
        )
    )
    return episodes


def load(path: Path = config.MANIFEST) -> List[dict]:
    return json.loads(path.read_text())["episodes"]
