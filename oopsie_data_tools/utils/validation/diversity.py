"""Lightweight, attribute-only diversity heuristics for a dataset (issue #40).

Warns (never blocks by default) when a dataset shows very low task or annotation
diversity — a nudge to catch copy-pasted annotations or an inattentive annotator.
Reads only HDF5 attrs (never opens videos or decodes arrays) so it is cheap to run
over a whole session directory, and it never raises.
"""

from __future__ import annotations

import glob
import logging
import os
from collections import Counter
from typing import Any

import h5py

logger = logging.getLogger(__name__)

# Below this many episodes the ratios are too noisy to be meaningful.
MIN_EPISODES_FOR_CHECK = 5
# A single description accounting for more than this fraction looks copy-pasted.
DUP_DESCRIPTION_FRACTION = 0.8


def check_diversity(samples_dir: str) -> list[str]:
    """Return (and WARN-log) low-diversity messages for a session directory.

    Args:
        samples_dir: Directory containing ``*.h5`` / ``*.hdf5`` episodes.

    Returns:
        A list of human-readable warning strings (empty if nothing notable).
    """
    files = glob.glob(os.path.join(samples_dir, "**", "*.h5"), recursive=True)
    files += glob.glob(os.path.join(samples_dir, "**", "*.hdf5"), recursive=True)
    if len(files) < MIN_EPISODES_FOR_CHECK:
        return []

    instructions: list[str] = []
    descriptions: list[str] = []
    for path in files:
        try:
            with h5py.File(path, "r") as f:
                instructions.append(_attr_str(f, "language_instruction"))
                ea = f.get("episode_annotations")
                if isinstance(ea, h5py.Group):
                    for name in ea.keys():
                        sub = ea[name]
                        if not isinstance(sub, h5py.Group):
                            continue
                        desc = _attr_str(sub, "failure_description").lower()
                        if desc:
                            descriptions.append(desc)
        except Exception:
            continue

    warnings: list[str] = []
    n = len(instructions)
    distinct_instructions = len({i for i in instructions if i})
    if n >= MIN_EPISODES_FOR_CHECK and distinct_instructions <= 1:
        warnings.append(
            f"Low task diversity: all {n} episodes share a single language_instruction."
        )
    elif n >= 10 and distinct_instructions / n < 0.1:
        warnings.append(
            f"Low task diversity: only {distinct_instructions} distinct instruction(s) "
            f"across {n} episodes."
        )

    if len(descriptions) >= MIN_EPISODES_FOR_CHECK:
        top_desc, top_n = Counter(descriptions).most_common(1)[0]
        if top_n / len(descriptions) > DUP_DESCRIPTION_FRACTION:
            warnings.append(
                f"Possible copied annotations: {top_n}/{len(descriptions)} failure "
                f'descriptions are identical ("{top_desc[:60]}...").'
            )

    for w in warnings:
        logger.warning("[diversity] %s", w)
    return warnings


def _attr_str(group: h5py.Group | h5py.File, key: str) -> str:
    value: Any = group.attrs.get(key, "")
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip()
