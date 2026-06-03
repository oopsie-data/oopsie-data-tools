"""Restructure a dataset folder with >10,000 files into numbered subfolders.

This is a NON-DESTRUCTIVE operation: files are COPIED to a new output folder.
The source folder is never modified. After verifying the output you must
manually delete the original.

Video paths stored inside HDF5 files are resolved to absolute paths before
copying, so the script handles relative paths, absolute paths, and paths that
cross directory boundaries (e.g. containing "..").  Each video is copied flat
into the same subfolder as the HDF5 that references it, and the path stored in
the HDF5 copy is updated accordingly.

Usage:
    python restructure_large_folder.py --source /path/to/session_dir
    python restructure_large_folder.py --source /path/to/session_dir --output /path/to/output
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import h5py

FILE_LIMIT = 10_000
BATCH_SIZE = 500

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def files_per_dir(root: Path) -> dict[Path, int]:
    """Count files (non-recursively) in every directory under root."""
    counts: dict[Path, int] = {}
    for dirpath, _, filenames in os.walk(root):
        counts[Path(dirpath)] = len(filenames)
    return counts


def oversized_dirs(root: Path) -> list[tuple[Path, int]]:
    return [(d, n) for d, n in files_per_dir(root).items() if n > FILE_LIMIT]


def collect_h5_files(directory: Path) -> list[Path]:
    """Return sorted list of all HDF5 files directly in *directory*."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in (".h5", ".hdf5")
    )


def read_video_paths(h5_path: Path) -> dict[str, str]:
    """Return {camera: stored_path_string} from observations/video_paths."""
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
    if p.is_absolute():
        return p.resolve()
    return (h5_dir / p).resolve()


def write_video_paths(h5_path: Path, new_paths: dict[str, str]) -> None:
    """Overwrite video path datasets in the HDF5 file at *h5_path*."""
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "r+") as f:
        vp = f["observations/video_paths"]
        for cam, rel in new_paths.items():
            if cam in vp:
                del vp[cam]
            vp.create_dataset(cam, data=rel, dtype=str_dtype)


def estimate_bytes(h5_files: list[Path]) -> int:
    """Return total bytes that would be copied (HDF5 files + referenced videos).

    Each unique absolute path is counted only once even if referenced by
    multiple HDF5 files.
    """
    total = 0
    seen: set[Path] = set()
    for h5 in h5_files:
        if h5.exists() and h5 not in seen:
            total += h5.stat().st_size
            seen.add(h5)
        h5_dir = h5.parent
        for stored in read_video_paths(h5).values():
            abs_v = resolve_video_path(stored, h5_dir)
            if abs_v.exists() and abs_v not in seen:
                total += abs_v.stat().st_size
                seen.add(abs_v)
    return total


# ── Subfolder naming ───────────────────────────────────────────────────────────


def subfolder_name(start_idx: int, n_total: int) -> str:
    """Return zero-padded folder name for the batch starting at *start_idx*.

    All names are padded to the same width so directories sort correctly.
    Examples for 10 000 files: 0000, 0500, 1000, …, 9500.
    """
    max_start = ((n_total - 1) // BATCH_SIZE) * BATCH_SIZE
    width = max(3, len(str(max_start)))
    return str(start_idx).zfill(width)


# ── Core restructure ───────────────────────────────────────────────────────────


def _unique_video_dest(
    abs_video: Path,
    cam: str,
    h5_stem: str,
    used: dict[str, Path],
) -> str:
    """Return a filename (no directory) for *abs_video* that is unique in *used*."""
    basename = abs_video.name
    
    fallback = f"{h5_stem}_{cam}{abs_video.suffix}"
    used[fallback] = abs_video
    return fallback


def restructure(source: Path, output: Path, h5_files: list[Path]) -> None:
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

        # Track which filenames are already used in this subfolder to detect
        # collisions across HDF5 files in the same batch.
        used_filenames: dict[str, Path] = {}

        for i, h5_src in enumerate(batch, 1):
            h5_dir = h5_src.parent
            stored_video_paths = read_video_paths(h5_src)

            # Copy HDF5 first so the destination file exists before we patch it.
            h5_dst = sub / h5_src.name
            shutil.copy2(h5_src, h5_dst)

            # Resolve each video to an absolute path, copy it flat into the
            # subfolder, and record what the new relative path will be.
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

                dest_name = _unique_video_dest(
                    abs_video, cam, h5_src.stem, used_filenames
                )
                dst_video = sub / dest_name
                shutil.copy2(abs_video, dst_video)
                # Path is stored with forward slashes, relative to HDF5 location.
                new_rel_paths[cam] = dest_name

            # Always write the updated paths back into the HDF5 copy so it is
            # self-consistent regardless of how the original paths were formatted.
            if new_rel_paths:
                write_video_paths(h5_dst, new_rel_paths)

            logger.info("    [%d/%d] %s", i, len(batch), h5_src.name)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Restructure a large folder into numbered subfolders "
            f"of up to {BATCH_SIZE} HDF5 files each."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", "-s", required=True,
        help="Source folder to restructure (must contain .h5 / .hdf5 files at its root)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help=(
            "Destination folder for the restructured copy "
            "(default: <source>_restructured next to the source folder)"
        ),
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        logger.error("Source is not a directory: %s", source)
        sys.exit(1)

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else source.parent / (source.name + "_restructured")
    )

    logger.info("=" * 60)
    logger.info("  Large-Folder Restructure Utility")
    logger.info("=" * 60)

    # ── Step 1: Check whether any directory exceeds the file limit ─────────────
    logger.info("\n[check] Scanning %s ...", source)
    over = oversized_dirs(source)

    if not over:
        logger.info(
            "[check] No directory exceeds %d files. No restructuring needed.",
            FILE_LIMIT,
        )
        sys.exit(0)

    logger.info("[check] Director(ies) exceeding %d files:", FILE_LIMIT)
    for d, n in over:
        logger.info("          %s  (%d files)", d, n)

    # ── Step 2: Collect HDF5 files ─────────────────────────────────────────────
    # Only files directly inside the source root are processed.
    # If the oversized directory is a subdirectory, run the script on that
    # subdirectory directly.
    h5_files = collect_h5_files(source)
    if not h5_files:
        logger.error(
            "[check] No HDF5 files found at the root of %s\n"
            "        If an oversized subdirectory needs restructuring, "
            "pass that subdirectory as --source.",
            source,
        )
        sys.exit(1)

    logger.info("[check] Found %d HDF5 file(s) at root of %s", len(h5_files), source)

    # ── Step 3: Estimate copy size ─────────────────────────────────────────────
    logger.info(
        "\n[size]  Estimating copy size "
        "(resolving video paths from all HDF5 files — may take a moment) ..."
    )
    total_bytes = estimate_bytes(h5_files)
    logger.info("[size]  Estimated copy size: %.2f GB", total_bytes / 1e9)

    # ── Step 4: Explicit user consent ─────────────────────────────────────────
    n_batches = (len(h5_files) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info("\n" + "=" * 60)
    logger.info("  ACTION REQUIRED — please read carefully before proceeding")
    logger.info("=" * 60)
    logger.info(
        "\nThis script will CREATE a new folder:\n"
        "  %s\n"
        "\nIt will contain %d subfolder(s) named by starting HDF5 index\n"
        "(e.g. 000/, 500/, 1000/, …), each holding up to %d HDF5 files\n"
        "and the video files they reference.\n"
        "\nVideo paths stored inside each HDF5 file will be rewritten to\n"
        "point to the copied video location.\n"
        "\nEstimated disk space needed:  %.2f GB\n"
        "\nThe SOURCE FOLDER WILL NOT BE MODIFIED. Once you have verified\n"
        "that the restructured copy is complete and valid, you must\n"
        "MANUALLY DELETE the original source to reclaim space:\n"
        "  %s\n"
        "  (approx. %.2f GB to free)",
        output, n_batches, BATCH_SIZE,
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
    logger.info("\nThis operation is not robust to interruptions! Please \n"
            "ensure that the program runs in full. If it is interrupted \n"
            "delete the output folder and restart.")
    confirm = input('\nType exactly "yes" to proceed, anything else to abort: ').strip()
    if confirm != "yes":
        logger.info("\nAborted. No files were written.")
        sys.exit(0)

    # ── Step 5: Restructure ────────────────────────────────────────────────────
    output.mkdir(parents=True, exist_ok=True)
    restructure(source, output, h5_files)

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


if __name__ == "__main__":
    main()
