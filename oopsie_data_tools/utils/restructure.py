"""Split a session directory with too many files into numbered subfolders.

Backs ``oopsie-data restructure``. HuggingFace Hub enforces a per-directory file limit
(:data:`FILE_LIMIT`), and a long recording session blows past it easily — one HDF5 plus
one MP4 per camera, per episode. ``hf_upload.check_folder_size`` refuses the upload and
points here.

This is a NON-DESTRUCTIVE operation: files are COPIED to a new output folder and the source
is never modified — not even the HDF5 files, whose stored video paths are rewritten only in
the copy. After verifying the output you must delete the original yourself.

The output is a complete copy of the source tree in which **every** oversized directory has
been split: a directory holding 12,000 episodes appears in the output at the same relative
path but as ``0000/``, ``0500/``, … underneath it, while directories that are already small
enough are copied through unchanged. Nesting therefore works, including re-running this on
output it produced.

Detection and repair used to disagree — oversized directories were found recursively but
only the source *root* was ever split, so a source with episodes at the root and an
oversized subdirectory would restructure the wrong directory and report success, leaving
the upload just as blocked. Both halves walk the tree now.

Video paths stored inside the HDF5 files are resolved to absolute paths before copying, so
relative paths, absolute paths, and paths containing ``..`` all work. Each video is copied
flat into the same subfolder as the HDF5 that references it, and the path stored in the
HDF5 *copy* is rewritten to match.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import h5py

from oopsie_data_tools.utils.hf_limits import BATCH_SIZE, FILE_LIMIT

logger = logging.getLogger(__name__)


def files_per_dir(root: Path) -> dict[Path, int]:
    """Count files (non-recursively) in every directory under *root*."""
    return {Path(dirpath): len(filenames) for dirpath, _, filenames in os.walk(root)}


def oversized_dirs(root: Path) -> list[tuple[Path, int]]:
    return [(d, n) for d, n in files_per_dir(root).items() if n > FILE_LIMIT]


def collect_h5_files(directory: Path) -> list[Path]:
    """Return a sorted list of the HDF5 files directly in *directory*."""
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".h5", ".hdf5")
    )


def read_video_paths(h5_path: Path) -> dict[str, str]:
    """Return ``{camera: stored_path_string}`` from ``observations/video_paths``."""
    paths: dict[str, str] = {}
    try:
        with h5py.File(h5_path, "r") as f:
            vp = f.get("observations/video_paths")
            if vp is None:
                return paths
            for cam in vp.keys():
                raw = vp[cam][()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                paths[cam] = str(raw).strip()
    except Exception as exc:
        logger.warning("  ! Could not read video paths from %s: %s", h5_path.name, exc)
    return paths


def resolve_video_path(stored_path: str, h5_dir: Path) -> Path:
    """Resolve a stored video path to an absolute Path."""
    p = Path(stored_path)
    return p.resolve() if p.is_absolute() else (h5_dir / p).resolve()


def write_video_paths(h5_path: Path, new_paths: dict[str, str]) -> None:
    """Overwrite the video path datasets in the HDF5 file at *h5_path*."""
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "r+") as f:
        vp = f["observations/video_paths"]
        for cam, rel in new_paths.items():
            if cam in vp:
                del vp[cam]
            vp.create_dataset(cam, data=rel, dtype=str_dtype)


def dirs_to_split(source: Path) -> tuple[list[Path], list[Path]]:
    """Split the oversized directories under *source* into fixable and unfixable.

    Batching is keyed on HDF5 files, so an oversized directory holding none of them cannot
    be split by this command — a videos-only folder, say. Those are returned separately so
    the caller can say so out loud rather than silently leaving them oversized.

    Returns:
        ``(splittable, unfixable)``, both sorted.
    """
    splittable, unfixable = [], []
    for directory, _count in sorted(oversized_dirs(source)):
        (splittable if collect_h5_files(directory) else unfixable).append(directory)
    return splittable, unfixable


def estimate_bytes(source: Path, h5_files: list[Path]) -> int:
    """Total bytes the output would occupy: the whole source tree, plus any video it
    references from outside it.

    The output is a full copy of *source*, not just of the episodes being split, so
    estimating from the HDF5 files alone would badly undercount a tree that also holds
    directories small enough to pass through untouched.

    Each unique absolute path is counted once even if several HDF5 files reference it.
    """
    total = 0
    seen: set[Path] = set()

    for dirpath, _dirnames, filenames in os.walk(source):
        for name in filenames:
            p = Path(dirpath) / name
            if p not in seen and p.is_file():
                total += p.stat().st_size
                seen.add(p)

    # Videos can live outside the source tree entirely (stored paths may contain "..").
    for h5 in h5_files:
        for stored in read_video_paths(h5).values():
            abs_v = resolve_video_path(stored, h5.parent)
            if abs_v not in seen and abs_v.is_file():
                total += abs_v.stat().st_size
                seen.add(abs_v)
    return total


def subfolder_name(start_idx: int, n_total: int) -> str:
    """Zero-padded folder name for the batch starting at *start_idx*.

    All names are padded to the same width so the directories sort correctly.
    Examples for 10 000 files: 0000, 0500, 1000, …, 9500.
    """
    max_start = ((n_total - 1) // BATCH_SIZE) * BATCH_SIZE
    width = max(3, len(str(max_start)))
    return str(start_idx).zfill(width)


def _unique_video_dest(
    abs_video: Path,
    cam: str,
    h5_stem: str,
    used: dict[str, Path],
) -> str:
    """Return a filename (no directory) for *abs_video* that is unique within *used*.

    Episode stems are second-resolution timestamps, so two sessions can easily contribute
    the same ``<stem>_<cam>.mp4`` to one batch. This used to return the colliding name
    regardless, silently overwriting the first episode's video during the copy.

    Re-asking for a video already registered returns the same name, so the function is safe
    to call more than once per file.
    """
    stem = f"{h5_stem}_{cam}"
    suffix = abs_video.suffix

    candidate = f"{stem}{suffix}"
    attempt = 1
    while used.get(candidate, abs_video) != abs_video:
        attempt += 1
        candidate = f"{stem}_{attempt}{suffix}"

    used[candidate] = abs_video
    return candidate


def restructure(output: Path, h5_files: list[Path]) -> set[Path]:
    """Batch *h5_files* into numbered subfolders under *output*.

    Returns:
        The absolute source paths consumed — the HDF5 files and every video they
        reference. :func:`copy_through` skips these so nothing is copied twice.
    """
    consumed: set[Path] = set()
    n = len(h5_files)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        "\n[restructure] %d HDF5 file(s) → %d subfolder(s) of up to %d each",
        n, n_batches, BATCH_SIZE,
    )

    for batch_start in range(0, n, BATCH_SIZE):
        batch = h5_files[batch_start : batch_start + BATCH_SIZE]
        sub = output / subfolder_name(batch_start, n)
        sub.mkdir(parents=True, exist_ok=True)
        logger.info("\n  Subfolder %s/  (%d HDF5 files)", sub.name, len(batch))

        # Which filenames this subfolder already holds, so collisions across HDF5 files in
        # the same batch are detected rather than silently overwriting.
        used_filenames: dict[str, Path] = {}

        for i, h5_src in enumerate(batch, 1):
            h5_dir = h5_src.parent
            stored_video_paths = read_video_paths(h5_src)

            # Copy the HDF5 first so the destination exists before we patch it.
            h5_dst = sub / h5_src.name
            shutil.copy2(h5_src, h5_dst)
            consumed.add(h5_src)

            # Resolve each video to an absolute path, copy it flat into the subfolder, and
            # record what the new relative path will be.
            new_rel_paths: dict[str, str] = {}
            for cam, stored in stored_video_paths.items():
                abs_video = resolve_video_path(stored, h5_dir)
                if not abs_video.exists():
                    logger.warning(
                        "    ! video not found, skipping: %s  (ref'd by %s)",
                        abs_video, h5_src.name,
                    )
                    # Leave the original path; validation will catch this later.
                    new_rel_paths[cam] = stored
                    continue

                dest_name = _unique_video_dest(abs_video, cam, h5_src.stem, used_filenames)
                shutil.copy2(abs_video, sub / dest_name)
                consumed.add(abs_video)
                # Stored relative to the HDF5 location, which is now the same folder.
                new_rel_paths[cam] = dest_name

            # Always write the updated paths back into the HDF5 copy so it is
            # self-consistent regardless of how the original paths were formatted.
            if new_rel_paths:
                write_video_paths(h5_dst, new_rel_paths)

            logger.info("    [%d/%d] %s", i, len(batch), h5_src.name)

    return consumed


def copy_through(source: Path, output: Path, consumed: set[Path]) -> int:
    """Copy every file under *source* that the split did not already place into *output*.

    Preserves each file's path relative to *source*, so directories that were already small
    enough come through untouched and the output is a complete dataset rather than only the
    episodes that needed splitting.

    Returns:
        The number of files copied.
    """
    copied = 0
    for dirpath, dirnames, filenames in os.walk(source):
        here = Path(dirpath)
        # Never descend into the output, in case it was placed inside the source.
        dirnames[:] = [d for d in dirnames if (here / d).resolve() != output]
        for name in filenames:
            src = here / name
            if src in consumed:
                continue
            dst = output / src.relative_to(source)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    return copied


def run_restructure(source: Path, output: Path | None, assume_yes: bool) -> int:
    """Full interactive flow: check, estimate, confirm, copy. Returns an exit code."""
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        logger.error("Source is not a directory: %s", source)
        return 1

    output = (
        Path(output).expanduser().resolve()
        if output
        else source.parent / (source.name + "_restructured")
    )

    logger.info("=" * 60)
    logger.info("  Large-Folder Restructure Utility")
    logger.info("=" * 60)

    # ── Step 1: does any directory exceed the file limit? ─────────────────────
    logger.info("\n[check] Scanning %s ...", source)
    over = oversized_dirs(source)
    if not over:
        logger.info("[check] No directory exceeds %d files. No restructuring needed.", FILE_LIMIT)
        return 0

    logger.info("[check] Director(ies) exceeding %d files:", FILE_LIMIT)
    for d, n in over:
        logger.info("          %s  (%d files)", d, n)

    # ── Step 2: work out which of them we can actually split ──────────────────
    splittable, unfixable = dirs_to_split(source)
    if unfixable:
        logger.warning(
            "[check] These are oversized but hold no HDF5 files, so they cannot be split\n"
            "        by episode and are copied through as they are:"
        )
        for d in unfixable:
            logger.warning("          %s", d)
    if not splittable:
        logger.error(
            "[check] No oversized directory under %s contains HDF5 files, so there is\n"
            "        nothing to split. Reorganise these by hand.",
            source,
        )
        return 1

    h5_by_dir = {d: collect_h5_files(d) for d in splittable}
    all_h5 = [h5 for files in h5_by_dir.values() for h5 in files]
    for d, files in h5_by_dir.items():
        logger.info("[check] %d HDF5 file(s) to split in %s", len(files), d)

    if output == source or source in output.parents:
        logger.error(
            "Output must not be inside the source (%s is under %s); "
            "pick a destination outside it.",
            output, source,
        )
        return 1

    # ── Step 3: estimate copy size ────────────────────────────────────────────
    logger.info(
        "\n[size]  Estimating copy size "
        "(resolving video paths from all HDF5 files — may take a moment) ..."
    )
    total_bytes = estimate_bytes(source, all_h5)
    logger.info("[size]  Estimated copy size: %.2f GB", total_bytes / 1e9)

    # ── Step 4: explicit consent ──────────────────────────────────────────────
    n_batches = sum((len(f) + BATCH_SIZE - 1) // BATCH_SIZE for f in h5_by_dir.values())
    logger.info("\n" + "=" * 60)
    logger.info("  ACTION REQUIRED — please read carefully before proceeding")
    logger.info("=" * 60)
    logger.info(
        "\nThis will CREATE a new folder:\n"
        "  %s\n"
        "\nIt will hold a complete copy of the source tree. Each of the %d\n"
        "oversized director(ies) above is replaced by %d subfolder(s) in total,\n"
        "named by starting HDF5 index (e.g. 000/, 500/, 1000/, …), each holding\n"
        "up to %d HDF5 files and the video files they reference. Directories\n"
        "already under the limit are copied through unchanged.\n"
        "\nVideo paths stored inside each HDF5 file will be rewritten to\n"
        "point to the copied video location.\n"
        "\nEstimated disk space needed:  %.2f GB\n"
        "\nThe SOURCE FOLDER WILL NOT BE MODIFIED. Once you have verified\n"
        "that the restructured copy is complete and valid, you must\n"
        "MANUALLY DELETE the original source to reclaim space:\n"
        "  %s\n"
        "  (approx. %.2f GB to free)",
        output, len(splittable), n_batches, BATCH_SIZE,
        total_bytes / 1e9,
        source, total_bytes / 1e9,
    )

    if output.exists() and any(output.iterdir()):
        logger.warning(
            "\n[warn]  Output folder already exists and is not empty:\n"
            "          %s\n"
            "        Existing files with the same names will be overwritten.",
            output,
        )
    logger.info(
        "\nThis operation is not robust to interruptions! Please ensure that it\n"
        "runs in full. If it is interrupted, delete the output folder and restart."
    )

    if not assume_yes:
        confirm = input('\nType exactly "yes" to proceed, anything else to abort: ').strip()
        if confirm != "yes":
            logger.info("\nAborted. No files were written.")
            return 0

    # ── Step 5: split each oversized directory, then copy the rest through ────
    output.mkdir(parents=True, exist_ok=True)
    consumed: set[Path] = set()
    for directory, files in h5_by_dir.items():
        # Each split lands at the directory's own position in the output tree, so the
        # result mirrors the source rather than flattening it.
        logger.info("\n[restructure] %s", directory)
        consumed |= restructure(output / directory.relative_to(source), files)

    copied = copy_through(source, output, consumed)
    logger.info("\n[copy]  %d file(s) copied through unchanged.", copied)

    logger.info("\n" + "=" * 60)
    logger.info("  Done!")
    logger.info("=" * 60)
    logger.info("\nRestructured data written to:\n  %s", output)
    logger.info(
        "\nREMINDER: Verify the restructured folder before deleting the original.\n"
        "To delete the original once satisfied:\n"
        "  rm -rf %s\n"
        "  (will free approx. %.2f GB)",
        source, total_bytes / 1e9,
    )
    return 0
