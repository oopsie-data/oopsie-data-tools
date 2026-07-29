# Troubleshooting

`oopsie-data validate --path <dir> --json` is the fastest way to see which episodes failed and
on what: one record per episode with `passed`, `error`, and an `error_type` of `validation` (the
episode is wrong) or `unexpected` (the validator is wrong — report it rather than working around
it). `oopsie-data inspect <file.h5>` dumps one file's structure, even if `validate` rejects it.

## Errors you will actually see

**`lab_id` unset, blank, or still `your_lab_id`.** A `RuntimeError` pointing at the registration
form, from `EpisodeRecorder.__init__` and from `oopsie-data upload`. Capitalization must match
the value you were given exactly.

**Config edited in the wrong place.** Editing the clone's `configs/` while `$OOPSIE_CONFIG_DIR`
or `~/.config/oopsie-data` also exists — the earlier entry in the chain wins. The error names
the file that was actually read; `oopsie-data show-config` shows the whole chain.

**Action dict keys not matching `action_space`.** Raised at `record_step`. The two must be equal
as sets, not merely overlapping.

**"missing robot state keys required by its action_space".** At `load_robot_profile`. Add
`joint_position` to `robot_state_keys` for a joint action space, or `cartesian_position` for a
Cartesian one. `gripper_position` is required either way.

**"robot_state_joint_names is required when joint_position is included in robot_state_keys".**
Also at `load_robot_profile`. Name every joint, in the order the array uses. The key is needed
*only* in that case — drop it from a purely Cartesian profile rather than leaving it blank. A
length that disagrees with the recorded DOF surfaces later, inside `finish_rollout`.

**`cartesian_position` recorded but `orientation_representation` unset.** The value is stored
unconverted and rejected unless it already is `[x, y, z, qx, qy, qz, qw]` with a unit quaternion.
A representation that is set but does not match what the policy emits is reported by width, e.g.
`QUAT orientation expects 4 value(s), got 3`.

**Episode duration out of range.** `[1, 600]` seconds, computed as
`trajectory_length / control_freq` — not a step count. Short test rollouts trip this constantly.

**"Annotations dict is empty, must be provided for upload".** The episode has no
`episode_annotations` group. Expected between recording and annotation — run `oopsie-data
annotate` — not a malformed file.

**Undeclared keys.** `observations/robot_states contains N key(s) the robot profile does not
declare`. Add them to the profile or stop recording them.

**"Too many files in a directory" on upload.** HuggingFace caps a directory at 10,000 files.
`oopsie-data restructure --source ./samples` writes a split copy alongside the original, or
`upload --with-restructure` does it inline. Both need room for a second copy.

**Low-diversity warnings on upload.** Near-identical tasks or annotations warn by default and
fail with `--strict-diversity`. It is a signal about dataset value, not a bug to silence.

**Flag gotchas.** `--config-dir` belongs to `oopsie-data` itself and must come *before* the
subcommand. `inspect` takes a positional path, not `--path`. Exit codes: 1 on a handled failure,
2 on a usage error or a path that does not exist, 130 on Ctrl-C.

## Mistakes that pass validation silently

These cannot be caught by the toolkit. Ask the user to confirm each.

**Actions in delta rather than absolute coordinates.** Nothing detects this, and deltas cannot be
used downstream because the base offset is not recorded.

**Wrong quaternion component order.** Poses must be scalar-last, `(x, y, z, w)`. A scalar-first
pose only produces a `logger.warning` from a heuristic, never an error.

**An action chunk passed instead of a per-step action.** Only `cartesian_position` is
shape-checked. For joint action spaces a `(T, chunk, dof)` array satisfies both the DOF and
trajectory-length checks and is recorded silently.

**`cartesian_velocity` of any shape.** Recorded exactly as given, with no conversion and no shape
check.

**Annotation *quality*.** Presence and shape are checked; whether a description actually
describes what happened, or the category fits, is not.
