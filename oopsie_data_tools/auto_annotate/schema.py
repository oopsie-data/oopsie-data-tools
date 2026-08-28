"""The two output schemas, and how they map onto the repo's stored vocabulary.

The schemas here follow ``annotation_requirement.md``, which is a superset of what
``oopsie_data_tools.annotation_tool.annotation_schema`` can store:

  * the requirement's ``sudden_termination`` category has no slug in the repo taxonomy
  * per-segment annotation has no place in the HDF5 layout at all
  * ``resetability`` is not part of the stored taxonomy

Nothing is silently dropped to resolve that. The JSON sidecar keeps the full requirement
answer, the HDF5 keeps whatever the repo vocabulary can legally hold, and the mapping that
bridges them is written down below rather than buried in the writer.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ── Vocabulary, exactly as annotation_requirement.md defines it ────────────────
OUTCOMES = ("success", "failure", "success_suboptimal", "success_side_effect")

CATEGORIES = (
    "reaching",             # pre-contact: never reached the target
    "grasp",                # at contact: slip, drop, wrong grasp; includes a near-miss
    "manipulation",         # post-contact: grasped but not manipulated as intended
    "sequencing_semantic",  # planning/ordering error, e.g. right action on wrong object
    "collision",            # hit an obstacle or the environment
    "not_attempted",        # no discernible attempt, e.g. stalling
    "sudden_termination",   # e-stop or max-length cutoff
    "hardware",             # visible hardware/mechanical problem
    "other",                # anything else; must be described in the free text
)

SEVERITIES = ("low", "medium", "catastrophic")

RESETABILITY = (
    "immediate",     # no intervention needed
    "minor_reset",   # quick, straightforward intervention
    "hard_reset",    # substantial intervention, multiple steps
    "unresettable",  # needs repair or replacement
)

# ── Bridge to the repo's stored taxonomy ──────────────────────────────────────
# Every requirement category except this one is already a repo slug. A sudden termination
# is not describable in the stored vocabulary, so it stores as "other" and the sidecar
# keeps the precise label. Recorded as a table so the loss is visible, not incidental.
CATEGORY_TO_REPO = {name: name for name in CATEGORIES}
CATEGORY_TO_REPO["sudden_termination"] = "other"

# Fields the requirement defines that the HDF5 taxonomy has nowhere to put.
UNSTORED_FIELDS = ("resetability", "segments", "how_to_recover")


def to_repo_categories(categories: List[str]) -> List[str]:
    """Requirement categories -> repo slugs, de-duplicated, order preserved."""
    seen: Dict[str, None] = {}
    for name in categories:
        seen.setdefault(CATEGORY_TO_REPO.get(name, "other"), None)
    return list(seen)


def outcome_to_success(outcome: str) -> float:
    """The float the repo stores alongside the outcome slug."""
    return 0.0 if outcome == "failure" else 1.0


# ── Task 1: episode-level ─────────────────────────────────────────────────────
EPISODE_SCHEMA: Dict[str, Any] = {
    "name": "episode_annotation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["language_task", "language_task_source", "outcome", "rationale"],
        "properties": {
            "language_task": {
                "type": "string",
                "description": "One sentence or phrase for what the robot tried to do, "
                               "regardless of whether it succeeded.",
            },
            "language_task_source": {
                "type": "string",
                "enum": ["copied_existing", "inferred_from_video"],
                "description": "copied_existing only if a provided instruction was present "
                               "and is consistent with the video.",
            },
            "outcome": {"type": "string", "enum": list(OUTCOMES)},
            "rationale": {
                "type": "string",
                "description": "Two or three sentences citing what in the frames decided "
                               "the outcome.",
            },
        },
    },
}

# ── Task 2: failure segments ──────────────────────────────────────────────────
# The unit the model answers in depends on what it was shown. Given whole video it can only
# refer to elapsed seconds, because the frames it sees are sampled server-side and carry no
# labels. Given a labelled frame set it answers in exact timestep indices. Both are offered
# rather than forcing one, since seconds must be converted back to timesteps and that
# conversion is only as good as the container's duration.
_UNIT_FIELDS = {
    "seconds": (
        "start_seconds",
        "end_seconds",
        {"type": "number"},
        "Elapsed seconds into the video.",
    ),
    "timestep": (
        "start_timestep",
        "end_timestep",
        {"type": "integer"},
        "Timestep index, using the t= labels drawn on the frames.",
    ),
}


def segment_schema(unit: str = "seconds") -> Dict[str, Any]:
    """The Task 2 schema, expressed in ``seconds`` or ``timestep``."""
    start, end, kind, note = _UNIT_FIELDS[unit]
    bound = dict(kind)
    return {
        "name": "failure_segments",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["segments"],
            "properties": {
                "segments": {
                    "type": "array",
                    "description": "One entry per continuous run of bad actions. Empty "
                                   "for a cleanly successful episode.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            start, end, "what_happened", "how_to_recover",
                            "failure_categories", "severity", "resetability",
                        ],
                        "properties": {
                            start: dict(bound, description=
                                        "Earliest point the failure is observable. " + note),
                            end: dict(bound, description=
                                      "Where the robot begins recovering, another failure "
                                      "begins, or the episode ends. " + note),
                            "what_happened": {
                                "type": "string",
                                "description": "What went wrong and why the robot failed.",
                            },
                            "how_to_recover": {
                                "type": "string",
                                "description": "What the robot should do to recover.",
                            },
                            "failure_categories": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "enum": list(CATEGORIES)},
                            },
                            "severity": {"type": "string", "enum": list(SEVERITIES)},
                            "resetability": {"type": "string", "enum": list(RESETABILITY)},
                        },
                    },
                }
            },
        },
    }
