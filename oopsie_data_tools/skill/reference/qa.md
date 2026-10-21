# Data-quality checks (`oopsie-data qa`)

`validate` checks that an episode follows the schema. `qa` looks for data that passes
`validate` but is probably wrong: actions in the wrong units, normalized or stored as deltas,
commands that do not match the robot state, swapped arms, and videos that are dark, frozen or
out of step with the trajectory. These are the checks the Oopsie Data team runs on submitted
data, so running them before `upload` catches the same problems earlier.

```bash
oopsie-data qa --path ./samples                       # every episode, report in ./qa_report.yaml
oopsie-data qa --path ./samples --limit 50 --seed 0   # a reproducible random sample of 50
oopsie-data qa --path ./samples --episode-id 000001   # one episode
oopsie-data qa --path ./samples --no-video            # action checks only, much faster
```

Each camera video is decoded in full, about 2 CPU-seconds per episode; `--workers N` (default
`min(8, CPUs)`) runs episodes in parallel. For a large session start with `--limit`.

Findings are advisory. The command exits 0 when episodes are flagged and 1 only if the path is
missing or no episode could be checked; `--strict` also exits 1 on any flagged or errored
episode. `--json` prints the same report on stdout.

Keep the report out of the session directory: `upload` publishes everything in it. The command
warns when `--output` points inside `--path`.

## The report

One YAML file (`--output`, default `qa_report.yaml`), schema `oopsie_qa_report_v1`:

```yaml
summary:
  episodes_checked: 4
  episodes_flagged: 1        # also: episodes_clean, episodes_not_checked, episodes_errored
  flagged_episodes_per_check:
    video.resolution: 1      # most frequent first
  checks_enabled: [...]
  config_overrides: {}       # values that differ from the defaults
episodes:
- episode: session_a/000001.h5    # relative to --path
  status: flagged                 # flagged | clean | not_checked | error
  checks_run: 15
  flagged_checks: [video.resolution]
  findings:
  - check: video.resolution       # <category>.<check>
    key: front                    # the action key, camera, or "all cameras"
    description: A camera video's shorter side is below the minimum resolution
    details:
      issue: 224x224 (short side < 360)
  skipped:                        # checks that could not run, and why
  - check: video.periodic_skips
    key: all cameras
    reason: no_position_state
```

`details` holds the values that tripped the check; video findings put a readable sentence in
`details.issue`. `not_checked` means no check applied (no declared action space and the video
checks off). `error` carries the exception in `error`: the file could not be read, which
`validate` will explain better.

When summarizing a report for a user, start from `flagged_episodes_per_check`. A check that
flags every episode points at the recording setup or robot profile, not at single episodes.

## The checks

Each check only runs for the action spaces it belongs to (`general` runs for all). Thresholds
are the defaults.

| Check | Flags when |
|---|---|
| `general.length_mismatch` | an action and its matching robot state have different lengths |
| `general.action_copies_state` | ≥ 90% of joint/Cartesian position actions equal the same or the next state: actions were rebuilt from states, not logged commands |
| `general.gripper_binary_present` | no non-empty `gripper_binary` action is recorded |
| `general.gripper_range` | `gripper_binary` leaves [0, 1] by more than 0.05 (e.g. −1/+1) |
| `general.gripper_command_range` | `gripper_position` commands exceed the gripper state's range by more than 25% of its scale |
| `joint_position.joint_names` | action/state joint names missing, mismatched, or out of base-to-wrist order |
| `joint_position.dof` | number of joint values ≠ number of joint names |
| `joint_position.radians` | a joint value beyond 6.5: degrees or a −100..100 range |
| `joint_position.step_jump` | a joint moves faster than 6.28 rad/s between two steps |
| `joint_position.gripper_in_joints` | a joint name contains "gripper" |
| `joint_position.normalized_range` | joint actions peak at exactly ±1 |
| `joint_position.delta_detection` | actions look like `state[t+1] - state[t]` rather than absolute positions |
| `joint_position.tracking` | median per-joint error between `action[t]` and `state[t+1]` above 0.2 rad |
| `joint_velocity.dof` | number of joint velocities ≠ number of joint names |
| `joint_velocity.speed_limit` | a joint commanded faster than 3.14 rad/s |
| `joint_velocity.normalized_range` | velocities peak at exactly ±1 |
| `joint_velocity.integration` | velocity × step time vs. actual joint motion: ratio outside 0.25–8 or correlation below 0.3 (`suspected` names deg/s or per-step deltas) |
| `joint_velocity.looks_like_position` | the "velocity" sits on the joint position state |
| `joint_velocity.smoothness` | > 30% of steps reverse direction |
| `cartesian_position.quaternion_shape` | pose is not `(x, y, z, qx, qy, qz, qw)` per arm, or a quaternion is not unit length |
| `cartesian_position.position_units` | xyz beyond 3 m or ≫ the state's: mm/cm or a far-away frame (`action_suspected_unit` / `state_suspected_unit`) |
| `cartesian_position.quaternion_sign_flip` | consecutive quaternions with negative dot product (q → −q) |
| `cartesian_position.rotation_jump` | orientation turns faster than 360°/s between steps |
| `cartesian_position.normalized_range` | xyz actions peak at exactly ±1 |
| `cartesian_position.pose_tracking` | median error between action and next state above 0.10 m or 20° |
| `cartesian_position.delta_detection` | xyz actions look like deltas |
| `cartesian_position.biarm_ordering` | swapping left/right arms halves the tracking error |
| `cartesian_velocity.shape` | not 6 values per arm (3 linear, 3 angular) |
| `cartesian_velocity.speed_limit` | linear above 2 m/s or angular above 6.28 rad/s |
| `cartesian_velocity.normalized_range` | velocities peak at exactly ±1 |
| `cartesian_velocity.integration` | as for joint velocity, separately for linear and angular parts |
| `cartesian_velocity.linear_angular_order` | swapping the halves matches the real motion clearly better |
| `cartesian_velocity.looks_like_position` | the linear part sits on the state's xyz |
| `cartesian_velocity.smoothness` | > 30% of steps reverse direction |
| `video.resolution` | short side below 360 px |
| `video.decoding` | file missing, ffmpeg errors, nothing decodes, or frame count off by more than 1 |
| `video.brightness` | median brightness below 30 or above 225 (0–255), or ≥ 10% near-black/white frames |
| `video.speed` | fps ≠ `control_freq` by > 5%, video duration ≠ `T / control_freq` by more than max(0.5 s, 5%), irregular frame times, or cameras disagreeing in duration |
| `video.heavy_artifacts_and_desync` | ≥ 1 s frozen or ≥ 20% duplicate frames while the robot moves, sudden image jumps, single corrupted frames |
| `video.periodic_skips` | the recording pauses at a regular period (typically each action-chunk inference): robot state and video jump together at each gap |

Off by default, because they misfire on legitimate data: `general.duplicate_pairs`,
`general.scale_consistency`, `general.tracking_residual`, `general.delta_detection`,
`cartesian_position.quaternion_scalar_order`.

Skip reasons name the missing precondition, e.g. `declared_but_empty` (the profile declares a
key that holds no data), `no_matching_state` / `no_robot_state` (the state needed for the
comparison was not recorded), `no_control_freq`, `too_short`, `too_little_motion`,
`non_finite_values`, `no_video_files` (no camera reference resolves). A skip is not a failure,
but a check skipped on every episode checks nothing.

## Reading findings

Most findings trace back to one recording-side cause, so fix the cause and record again rather
than patching files:

- **`radians`, `position_units`, `normalized_range`, `speed_limit`** — the values are in the
  wrong unit or were normalized by the policy. Record the robot's physical command (radians,
  metres, rad/s), not the policy's normalized output.
- **`delta_detection`, `integration` with `suspected: per_step_delta`** — a delta action was
  recorded under a position key, or a per-step change under a velocity key. Either convert to
  the declared space or declare the space that matches.
- **`action_copies_state`** — the action was filled from the observation. Record the command
  sent to the robot.
- **`tracking`, `pose_tracking`** — action and state are in different frames, conventions or
  orders, or are misaligned in time. `robot-profile.md` covers frames and orientation.
- **`quaternion_sign_flip`** — the same orientation written as q and −q. Harmless to the robot,
  but bad for learning on raw values; keep the sign continuous (`dot(q[t], q[t-1]) >= 0`).
- **`video.speed`** — `control_freq` in the profile does not match the rate `record_step` is
  called at, or frames are dropped.
- **`periodic_skips`** — recording pauses while the policy plans the next chunk. Keep calling
  `record_step` at `control_freq` during inference.

A flagged episode is not rejected by `upload`. If a finding is a false positive for the
user's setup, say so and note it in the episode or submission, rather than changing data to
silence it.

## Overriding thresholds

`--config FILE` takes YAML nested like `<category>: <check>: <parameter>`; anything left out
keeps its default, and an unknown name is an error. The report records every override under
`summary.config_overrides`.

```yaml
video:
  resolution:
    min_short_side: 240
general:
  gripper_binary_present:
    enabled: false
```

Only change a threshold when the user can say why their setup differs. Do not loosen checks
to make a report clean.
