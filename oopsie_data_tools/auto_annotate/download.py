"""Fetch the manifest's episodes from the release repo into the local data root.

Read-only: every transfer is a GET against ``resolve/``. The local tree mirrors the
release's ``labs/<source>/<task>/`` layout so a local path maps back to its release path
by inspection.

Files are verified as they land — ``.h5`` against the sha256 the index carries, videos
against their expected byte size — and an already-complete file is skipped, so the stage
is safe to re-run after an interruption.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from oopsie_data_tools.auto_annotate import config, hub, manifest


def _planned_files(episodes: List[dict], sizes: Dict[str, int]) -> List[Tuple[str, int, str]]:
    """``(release_path, expected_size, expected_sha256)`` for everything to fetch."""
    planned: List[Tuple[str, int, str]] = []
    for episode in episodes:
        planned.append((episode["release_path"], episode["h5_bytes"], episode["sha256"]))
        for video in episode["videos"]:
            planned.append((video, sizes.get(video, 0), ""))
    return planned


def run(workers: int = 8, dry_run: bool = False) -> int:
    episodes = manifest.load()
    token = config.hf_token()
    sizes = hub.blob_sizes(config.RELEASE_REPO, token)
    planned = _planned_files(episodes, sizes)

    total = sum(size for _, size, _ in planned)
    print(f"episodes: {len(episodes)}  files: {len(planned)}  bytes: {total / 1e9:.2f} GB")
    print(f"destination: {config.RAW_DIR}")
    if dry_run:
        return 0

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "skipped": 0}
    failures: List[str] = []

    def fetch(item: Tuple[str, int, str]) -> Tuple[str, str]:
        path, size, digest = item
        status = hub.download(
            config.RELEASE_REPO,
            path,
            config.RAW_DIR / path,
            token,
            revision=config.RELEASE_REVISION,
            expect_size=size or None,
            expect_sha256=digest or None,
        )
        return path, status

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, item): item[0] for item in planned}
        for done, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                _, status = future.result()
                counts[status] += 1
            except Exception as error:  # noqa: BLE001 - one bad file must not kill the run
                failures.append(f"{path}: {error}")
            if done % 25 == 0 or done == len(planned):
                print(f"  [{done}/{len(planned)}] {counts['downloaded']} new, "
                      f"{counts['skipped']} present, {len(failures)} failed")

    for line in failures:
        print("FAILED " + line, file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-j", "--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
