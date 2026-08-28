"""Prompt text for the two annotation tasks.

Kept apart from the schemas because these get reworded far more often than the taxonomy
changes. The wording is close to ``annotation_requirement.md`` on purpose: the definitions
there are what a human annotator is held to, and paraphrasing them invites the model to
apply a different standard than the humans whose labels sit in the same files.
"""

from __future__ import annotations

from typing import Optional

SYSTEM = (
    "You annotate robot manipulation episodes for a failure dataset. You are shown one "
    "episode, either as a video or as frames sampled in order and labelled t=<n>. "
    "Judge only what is visible. Do not assume an episode succeeded because it looks "
    "routine, and do not invent detail you cannot see."
)

# ── Task 1: episode level ─────────────────────────────────────────────────────
EPISODE_TASK = """\
Do the episode-level annotation.

1. Free-form language task: one sentence or phrase describing what the robot was trying to
   accomplish, whether or not it succeeded.
   {instruction_clause}
   If it is ambiguous what the robot was attempting, use your best judgement to infer the
   intent, and label success or failure against that inferred task.

2. Categorical success label, exactly one of:
   - success: the task was achieved cleanly.
   - failure: the task was not achieved.
   - success_suboptimal: the task was achieved, but not efficiently or directly -- stalling,
     moving the wrong way before correcting, or repeated attempts before succeeding.
   - success_side_effect: the core task was achieved, but the robot caused unintended side
     effects -- knocking over or manipulating task-unrelated objects, bumping into objects
     or walls, or damaging anything unrelated to the task.

Give a short rationale citing what in the frames decided the label.
"""

_HAS_INSTRUCTION = (
    "The episode ships with this instruction: \"{instruction}\". Check it against the "
    "frames; if it is sane, copy it. Only rewrite it if the video plainly shows something "
    "else."
)
_NO_INSTRUCTION = "This episode has no recorded instruction; infer the task from the frames."


def episode_prompt(instruction: Optional[str]) -> str:
    clause = (
        _HAS_INSTRUCTION.format(instruction=instruction.strip())
        if instruction and instruction.strip()
        else _NO_INSTRUCTION
    )
    return EPISODE_TASK.format(instruction_clause=clause)


# ── Task 2: failure segments ──────────────────────────────────────────────────
SEGMENT_TASK = """\
Now identify every failure segment in this episode.

The episode-level annotation you just produced:
  task:    {task}
  outcome: {outcome}

A failure segment is a continuous period during which the robot takes "bad" actions -- ones
that do not progress toward the task above. Continuous bad actions form one segment.
Bad actions separated by good actions form separate segments.

A cleanly successful episode has no segments; return an empty list. An episode labelled
failure, success_suboptimal, or success_side_effect has at least one.

For each segment give:

- {start_field}: the earliest point at which the failure is observable.
- {end_field}: where the robot begins recovering, where a different failure begins, or the
  end of the episode if it never recovers.
  {extent}
- what_happened: what went wrong and why the robot failed.
- how_to_recover: what the robot should do to recover.
- failure_categories: one or more of
    reaching            -- never reached the target, no contact made. A grasp that barely
                           misses counts as grasp, not reaching.
    grasp               -- target not grasped properly: slipping, dropping, wrong grasp.
    manipulation        -- grasped but not manipulated as intended, e.g. holding a handle
                           but failing to open the door.
    sequencing_semantic -- planning or ordering error, e.g. a related but wrong action such
                           as picking the wrong object.
    collision           -- collision with an obstacle or the environment. May co-occur with
                           other categories; list all that apply.
    not_attempted       -- no discernible attempt at the task, e.g. stalling.
    sudden_termination  -- the episode ends abruptly, e.g. an e-stop or a length cutoff.
    hardware            -- visible hardware or mechanical problem, e.g. the arm falls limp.
    other               -- a failure mode none of the above covers; describe it in the text.
- severity:
    low           -- no meaningful damage or risk; only this attempt is spoiled.
    medium        -- some damage, disruption, or real risk of damage, but the environment
                     stays usable and the task could still be completed.
    catastrophic  -- significant damage, or damage that stops the task or environment being
                     used again without repair or replacement.
- resetability:
    immediate     -- no intervention needed, can be retried at once.
    minor_reset   -- quick, straightforward intervention, e.g. standing a cup back up.
    hard_reset    -- substantial intervention: significant time, effort, or several steps.
    unresettable  -- cannot be restored without repair or replacement.
"""

_EXTENT = {
    "seconds": "Give both in elapsed seconds; the video runs from 0.0s to {last:.1f}s.",
    "timestep": "Use only the t= values labelled on the frames; the episode runs from "
                "t=0 to t={last:.0f}.",
}


def segment_prompt(task: str, outcome: str, unit: str, last: float) -> str:
    """Task 2 wording for the unit the model can actually answer in."""
    start = "start_seconds" if unit == "seconds" else "start_timestep"
    end = "end_seconds" if unit == "seconds" else "end_timestep"
    return SEGMENT_TASK.format(
        task=task,
        outcome=outcome,
        start_field=start,
        end_field=end,
        extent=_EXTENT[unit].format(last=last),
    )
