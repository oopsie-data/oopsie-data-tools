# Troubleshooting

## Errors you will actually see

**`lab_id` unset, blank (`lab_id:`), or still `your_lab_id`.** A `RuntimeError` pointing at the
registration form, raised from `EpisodeRecorder.__init__` and from `oopsie-data upload`.
Capitalization must match the value you were given exactly.

**Config edited in the wrong place.** Editing the clone's `configs/` while `$OOPSIE_CONFIG_DIR`
or `~/.config/oopsie-data` also exists — the earlier entry in the chain wins. The error message
names the file that was actually read; `oopsie-data show-config` shows the whole chain.

**Action dict keys not matching `action_space`.** Raised at `record_step`. The two must be equal
as sets, not merely overlapping.

**`robot_state_joint_names` length ≠ `joint_position` DOF.** `EpisodeValidationError` inside
`finish_rollout`, before any file is written.

**`cartesian_position` recorded but `orientation_representation` unset.** The value is stored
unconverted and then rejected unless it already is `[x, y, z, qx, qy, qz, qw]` with a unit
quaternion. A representation that is set but does not match what the policy emits is reported by
width, e.g. `QUAT orientation expects 4 value(s), got 3`.

**Episode duration out of range.** `[1, 600]` seconds, computed as `trajectory_length /
control_freq` — not a step count. Short test rollouts trip this constantly.

**Undeclared keys.** `observations/robot_states contains N key(s) the robot profile does not
declare`. Add them to the profile or stop recording them; an extra diagnostic channel is not
allowed through.

**"Too many files in a directory" on upload.** HuggingFace caps a directory at 10,000 files.
`oopsie-data restructure --source ./samples` writes a split copy alongside the original
(non-destructive, rewrites video paths inside the copied HDF5 files), or `oopsie-data upload
--with-restructure` does it inline. Both need room for a second copy.

**Low-diversity warnings on upload.** A submission of near-identical tasks or annotations warns
by default and fails with `--strict-diversity`. It is a signal about dataset value, not a bug to
silence. The check runs even under `--skip-validate`.

**`--config-dir` rejected.** It is a flag on `oopsie-data` itself and must come *before* the
subcommand.

**`oopsie-data inspect --path ...` fails.** `inspect` takes a positional path.

**`--robot-profile` missing** when running an `examples/inference_examples/` script. It is
required and deliberately has no default. Those examples also need `uv sync --extra droid`.

Exit codes: 1 on a handled failure, 2 for a bare invocation or argparse error, 130 on Ctrl-C.

## Mistakes that pass validation silently

These cannot be caught by the toolkit. Ask the user to confirm each.

**Actions in delta rather than absolute coordinates.** Nothing detects this, and deltas cannot be
used downstream because the base offset is not recorded.

**Wrong quaternion component order.** Poses must be scalar-last, `(x, y, z, w)`. A scalar-first
pose only produces a `logger.warning` from a heuristic, never an error.

**An action chunk passed instead of a per-step action.** Only `cartesian_position` is
shape-checked, to `(7,)` or `(14,)`. For joint action spaces a `(T, chunk, dof)` array satisfies
both the DOF and trajectory-length checks and is recorded silently.

**Annotation content.** `strict_annotation_check` is off by default, so a plain `validate` or
`upload` run never inspects annotation quality — including whether failure descriptions actually
describe what happened.

**`cartesian_velocity` of any shape.** It is recorded exactly as given, with no conversion and
no shape check.

**A base action declared with `uses_mobile_base: false`.** Only the other direction is checked.
