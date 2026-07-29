# The `oopsiedata_format_v1` data format

One episode per HDF5 file, with its MP4 videos alongside. Run `oopsie-data inspect <file.h5>` to
see what a file actually contains before theorizing about it — it works even on files `validate`
rejects.

## Layout

Root attributes — `schema`, `episode_id`, `language_instruction`, `lab_id`, `operator_name` and
`robot_profile` (the whole profile, JSON-serialized) are **required**. `timestamp` (float epoch
seconds) is written by the recorder but not required by the loader.

```
observations/
  robot_states/<key>      one float64 dataset per profile.robot_state_keys, shape (T, ...)
  video_paths/<camera>    one string dataset per profile.camera_names, holding a path
                          relative to the .h5 file
actions/<key>             one dataset per VALID_ACTION_KEYS
episode_annotations/<annotator_name>/    optional; annotation fields are HDF5 attrs on the
                          subgroup, not datasets
```

Note that state datasets live under `observations/robot_states/`, not directly under
`observations/`.

## Actions

The recorder writes a dataset for **all nine** valid action keys. Keys outside the profile's
`action_space` are written as `h5py.Empty` placeholders; **every key in `action_space` must be a
real tensor** — an `h5py.Empty` there is rejected with `actions/<key> is in profile.action_space
but stored as h5py.Empty`. Since a valid action space always has at least one arm and one gripper
key, at least two `actions/` datasets are always real.

The reverse also holds: a non-empty action dataset the profile does not declare is rejected.

Actions are **unnormalized and absolute**. Bi-arm setups concatenate left and right.

## What the validator checks

Metadata:

- `language_instruction`, `episode_id`, `lab_id`, `operator_name` all non-empty; `lab_id` must
  not still be `your_lab_id`.
- `control_freq > 0`.
- **Episode duration, not step count**: `trajectory_length / control_freq` must be between 1 and
  600 seconds. A 5-step episode at 10 Hz is rejected.

Profile consistency — the profile documents the episode, so both directions are enforced:

- Every `robot_state_keys`, `action_space` and `camera_names` entry must be present.
- Nothing may be present that the profile does not declare, for either
  `observations/robot_states` or `actions`.
- `len(robot_state_joint_names)` must equal the last axis of `observations/joint_position`, when
  `joint_position` is recorded at all.
- `len(action_joint_names)` must equal the last axis of `actions/joint_position` /
  `joint_velocity`, when set.
- `cartesian_position` (state or action) must be exactly 7 DOF, or 14 when `is_biarm` — it is
  `[x, y, z, qx, qy, qz, qw]` per arm. Joint counts are deliberately **not** constrained by
  `is_biarm`: two arms need not share a DOF count. Gripper DOF is not checked at all, since
  nothing in the profile declares it.

Trajectories:

- Every observation and action array must share the same leading dimension `T`.

Videos:

- Each side between 180 and 1280 px.
- Frame count within `max(5, 0.1 * T)` of `T`.
- Video duration within 0.5 s of `T / control_freq`.
- Frame counts across cameras within 1 of each other.

Annotations — **always checked by `validate` and `upload`.** The `strict_annotation_check`
*parameter* defaults to `False`, but `run_validation` — what both CLI commands call — passes
`True` unconditionally, so an episode with no `episode_annotations` group fails with
`Annotations dict is empty, must be provided for upload`. Record an episode first and annotate
it second; it is only between those two steps that an unannotated episode is legal.

- Every annotator subgroup needs a numeric, non-NaN `success` in `[0.0, 1.0]`.
- A present `taxonomy` attr must parse as a JSON object.
- If that object carries an `outcome`, it must be one of `success`, `success_suboptimal`,
  `success_side_effect`, `failure`, and must agree in sign with `success` (`failure` iff
  `success < 0.5`).
- Nothing else is required. `episode_description`, `side_effect_category` and `severity` are
  all optional in every branch, so a partial annotation — including a failure with no taxonomy
  at all — is valid. Taxonomy v1 files, which carry no `outcome`, skip the outcome check and
  remain valid unchanged.

## Recording-time checks

`record_step` requires:

- `observation` to be a dict with both `robot_state` and `image_observation`;
- `robot_state` to contain every `robot_state_keys` entry, and `image_observation` a key named
  exactly `<cam>` for every `camera_names` entry — the alternative spellings `image_<cam>` and
  `<cam>_image` are recognized when the frame is read, but the presence check runs first and
  rejects the step, so they are not a usable substitute for the plain name;
- `action` keys to equal `action_space` exactly, with no `None` values.

`cartesian_position` is converted to `(x, y, z, qx, qy, qz, qw)` via
`orientation_representation`, then shape-checked to `(7,)` or `(14,)` with a unit quaternion in
`[3:7]` (tolerance 1e-2). Everything else is recorded as given.

`finish_rollout` validates *before* writing videos or HDF5, so a rejected episode leaves nothing
on disk. It runs the same checks as `validate` **except** the annotation ones — this is the one
path that uses the lenient default — which is why `finish_rollout(instruction=...)` without
`success` is allowed, and why the resulting episode does not pass `validate` until annotated. It raises `EpisodeValidationError`, which subclasses `AssertionError` — so an existing
`except AssertionError` still catches it. One exception: a camera with zero buffered frames
raises a plain `ValueError` from `VideoInfo.from_frames`, which neither catches.

Episode file names are second-resolution timestamps with `_2`, `_3` suffixes on collision, so do
not parse them as pure timestamps.
