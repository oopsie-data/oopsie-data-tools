# The `oopsiedata_format_v1` data format

One episode per HDF5 file, with its MP4 videos alongside. Run `oopsie-data inspect <file.h5>` to
see what a file actually contains before theorizing about it — it works even on files `validate`
rejects.

## Layout

Root attributes — `schema`, `episode_id`, `language_instruction`, `lab_id`, `operator_name` and
`robot_profile` (the whole profile, JSON-serialized) are **required**. `timestamp` is written by
the recorder but not required.

```
observations/
  robot_states/<key>      one float64 dataset per profile.robot_state_keys, shape (T, ...)
  video_paths/<camera>    one string dataset per profile.camera_names, holding a path
                          relative to the .h5 file
actions/<key>             one dataset per VALID_ACTION_KEYS
episode_annotations/<annotator_name>/    annotation fields are HDF5 attrs on the subgroup,
                          not datasets
```

State datasets live under `observations/robot_states/`, not directly under `observations/`.

## Actions

A dataset is written for **all nine** valid action keys. Keys outside the profile's
`action_space` are `h5py.Empty` placeholders; **every key in `action_space` must be a real
tensor** — an `h5py.Empty` there is rejected. The reverse also holds: a non-empty action dataset
the profile does not declare is rejected.

Actions are **unnormalized and absolute**. Bi-arm setups concatenate left and right.

## What the validator checks

Metadata:

- `language_instruction`, `episode_id`, `lab_id`, `operator_name` non-empty; `lab_id` not still
  `your_lab_id`.
- `control_freq` finite and greater than zero.
- **Episode duration, not step count**: `trajectory_length / control_freq` between 1 and 600
  seconds. A 5-step episode at 10 Hz is rejected.

Profile consistency — the profile documents the episode, so both directions are enforced:

- Every `robot_state_keys`, `action_space` and `camera_names` entry must be present, and nothing
  may be present that the profile does not declare.
- `len(robot_state_joint_names)` must equal the last axis of `observations/joint_position`, and
  `len(action_joint_names)` the last axis of `actions/joint_position` / `joint_velocity`, when
  those are recorded.
- `cartesian_position` (state or action) must be exactly 7 DOF, or 14 when `is_biarm` — it is
  `[x, y, z, qx, qy, qz, qw]` per arm, with a unit quaternion for every arm. Joint counts are
  deliberately **not** constrained by `is_biarm`.
- `gripper_binary` contains only `0` or `1`. A single arm accepts `(T,)` or `(T, 1)`; a biarm
  profile also accepts `(T, 2)` for independent per-arm commands.

Trajectories: every observation and action array must be real numeric and finite, and must share
the same leading dimension `T`.

Videos:

- The stored path must be relative to the `.h5`; an absolute path resolves only on the machine
  that recorded the episode, and is rejected.
- After resolving `..` and symlinks, the video must remain inside the submitted directory.
  Parent-relative paths are allowed when they still resolve inside that directory.
- Each side between 180 and 1280 px.
- Frame count within `max(5, 0.1 * T)` of `T`, and duration within 0.5 s of `T / control_freq`.
- Frame counts across cameras within 1 of each other.

## Annotations

**Always checked by `validate` and `upload`** — an episode with no `episode_annotations` group
fails with `Annotations dict is empty, must be provided for upload`. Record first, annotate
second; it is only between those two steps that an unannotated episode is legal.

Each `episode_annotations/<annotator_name>/` subgroup carries these attrs:

| attr | value |
| --- | --- |
| `schema` | `oopsie_failure_taxonomy_v2` |
| `taxonomy_schema` | `oopsiedata_taxonomy_schema_v2` |
| `source` | `human`, or the name of whatever produced it |
| `timestamp` | ISO 8601 string |
| `success` | float in `[0.0, 1.0]` |
| `episode_description` | free text |
| `additional_notes` | free text |
| `taxonomy` | JSON object: `outcome`, `failure_category`, `severity` |

Stored `failure_category` and `severity` values are stable slugs, not the prose the form shows.
The vocabularies are defined once in `annotation_tool/annotation_schema.py`:

- `outcome` — `success`, `success_suboptimal`, `success_side_effect`, `failure`. All three
  `success_*` outcomes write `success = 1.0`.
- `failure_category` (a list) — `reaching`, `grasp`, `manipulation`, `sequencing_semantic`,
  `collision`, `hardware`, `not_attempted`, `other`.
- `severity` — `low`, `medium`, `catastrophic`.

The validator requires a numeric, finite `success` in `[0.0, 1.0]`, and — if a `taxonomy` attr
is present — that it parses as a JSON object whose `outcome` is one of the four above and agrees
in sign with `success` (`failure` iff `success < 0.5`). Category and severity remain optional,
but every provided non-empty value must come from the vocabulary above. A partial annotation,
including a failure with no taxonomy at all, is valid.

Files from an older release carry `oopsie_failure_taxonomy_v1`, no `outcome`, and
`failure_description` instead of `episode_description`. They are upcast on read and never
rewritten. Known v1 prose vocabulary remains valid; unknown legacy category or severity values
stay visible but are rejected by strict upload validation.

## Recording-time checks

`record_step` requires `observation` to be a dict with both `robot_state` and
`image_observation`; `robot_state` to contain every `robot_state_keys` entry and
`image_observation` a key named exactly `<cam>` for every `camera_names` entry; and `action` keys
to equal `action_space` exactly, with no `None` values.

`cartesian_position` is converted to `(x, y, z, qx, qy, qz, qw)` via
`orientation_representation`, then shape-checked to `(7,)` or `(14,)` with a unit quaternion in
every arm slice. Quaternion component order is taken from the declared representation; it
cannot be inferred reliably from component values. Robot-state and action values must be real
and finite, and `gripper_binary` is shape/domain checked before the step is buffered.

`finish_rollout` validates *before* writing videos or HDF5, so a rejected episode leaves nothing
on disk. It runs the same checks as `validate` **except** the annotation ones — which is why
`finish_rollout(instruction=...)` without `success` is allowed, and why the resulting episode
does not pass `validate` until annotated.
