"""Run the annotation pipeline over the manifest.

Usage:
    python -m oopsie_data_tools.auto_annotate.run --limit 1 --save-frames   # smoke test
    python -m oopsie_data_tools.auto_annotate.run -j 4                     # full run

Already-annotated episodes are skipped unless ``--force`` is given, so an interrupted run
resumes without re-spending tokens. One episode failing does not stop the others; failures
are listed at the end and the exit code reflects them.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from oopsie_data_tools.auto_annotate import annotate, client as client_module
from oopsie_data_tools.auto_annotate import config, manifest, writer


def _selected(episodes: List[dict], sources, limit, per_source) -> List[dict]:
    if sources:
        wanted = set(sources)
        episodes = [e for e in episodes if e["source"] in wanted]
    if per_source:
        counts: dict = {}
        kept = []
        for episode in episodes:
            seen = counts.get(episode["source"], 0)
            if seen < per_source:
                counts[episode["source"]] = seen + 1
                kept.append(episode)
        episodes = kept
    return episodes[:limit] if limit else episodes


def run(args) -> int:
    episodes = _selected(manifest.load(), args.source, args.limit, args.per_source)
    if not args.force:
        pending = [e for e in episodes if not writer.sidecar_path(e).exists()]
        skipped = len(episodes) - len(pending)
        episodes = pending
    else:
        skipped = 0

    print(f"episodes to annotate: {len(episodes)} (skipped {skipped} already done)")
    detail = f"frames/episode: {args.frames}" if args.mode == "frames" else f"video budget: {args.budget} tok"
    print(f"model: {config.MODEL}  mode: {args.mode}  {detail}  workers: {args.workers}")
    print(f"sidecars -> {config.ANNOT_DIR}")
    print(f"annotated h5 -> {config.ANNOTATED_DIR}")
    if args.dry_run or not episodes:
        for episode in episodes[:10]:
            print(f"  would annotate {episode['source']}/{episode['episode_id']}")
        return 0

    model_client = client_module.Client(
        api_key=config.api_key(), base_url=config.base_url(), model=config.MODEL
    )

    failures: List[str] = []
    warned = 0
    tokens = 0

    def work(episode: dict) -> dict:
        record = annotate.annotate_episode(
            episode,
            model_client,
            mode=args.mode,
            n_frames=args.frames,
            budget_tokens=args.budget,
            save_frames=args.save_frames,
        )
        writer.write_sidecar(episode, record)
        if not args.no_h5:
            writer.write_h5(episode, record)
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, e): e for e in episodes}
        for done, future in enumerate(as_completed(futures), 1):
            episode = futures[future]
            label = f"{episode['source']}/{episode['episode_id']}"
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - one episode must not kill the run
                failures.append(f"{label}: {error}")
                if args.traceback:
                    traceback.print_exc()
                print(f"  [{done}/{len(episodes)}] FAIL {label}: {error}")
                continue
            for call in record["usage"].values():
                tokens += call.get("total_tokens", 0)
            flag = f"  ({len(record['warnings'])} warning)" if record["warnings"] else ""
            warned += 1 if record["warnings"] else 0
            print(
                f"  [{done}/{len(episodes)}] {label}: {record['episode']['outcome']}, "
                f"{len(record['segments'])} segment(s){flag}"
            )

    print(f"\ndone: {len(episodes) - len(failures)} annotated, {len(failures)} failed, "
          f"{warned} with warnings, {tokens} tokens")
    for line in failures:
        print("FAILED " + line, file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", help="restrict to a source lab (repeatable)")
    parser.add_argument("--limit", type=int, help="annotate at most N episodes overall")
    parser.add_argument("--per-source", type=int, help="annotate at most N per source")
    parser.add_argument("--mode", choices=("video", "frames"), default="video",
                        help="send the whole episode as video, or as labelled frames")
    parser.add_argument("--budget", type=int, default=24000,
                        help="soft cap on video tokens per pass; shrinks the clip to fit")
    parser.add_argument("--frames", type=int, default=16,
                        help="frames shown per episode in --mode frames")
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="re-annotate finished episodes")
    parser.add_argument("--no-h5", action="store_true", help="write sidecars only")
    parser.add_argument("--save-frames", action="store_true", help="keep the sampled jpegs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--traceback", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
