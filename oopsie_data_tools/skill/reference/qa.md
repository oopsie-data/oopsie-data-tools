# Data-quality checks (`oopsie-data qa`)

`validate` checks that an episode follows the schema. `qa` looks for data that passes
`validate` but is probably wrong: actions in the wrong units, normalized or stored as deltas,
commands that do not match the robot state, swapped arms, placeholder names, and videos that
are frozen, choppy or out of step with the trajectory. These are the checks the Oopsie Data team runs on submitted
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
  flagged_episodes_per_check:          # most frequent first
    general.placeholder_metadata:
      episodes: 1
      confidence: high_confidence
      suggested_action: fix_and_resubmit
  checks_enabled: [...]
  config_overrides: {}       # values that differ from the defaults
episodes:
- episode: session_a/000001.h5    # relative to --path
  status: flagged                 # flagged | clean | not_checked | error
  checks_run: 15
  flagged_checks: [general.placeholder_metadata]
  findings:
  - check: general.placeholder_metadata   # <category>.<check>
    key: operator_name                     # action key, state key, metadata field, camera or "all cameras"
    description: Operator, policy, robot or gripper name is blank or a template placeholder
    confidence: high_confidence
    suggested_action: fix_and_resubmit
    details:
      value: test_operator
      reason: placeholder
  skipped:                        # checks that could not run, and why
  - check: video.periodic_skips
    key: all cameras
    reason: no_position_state
```

`details` holds the values that tripped the check; video findings put a readable sentence in
`details.issue`. `not_checked` means no check applied (no declared action space and the video
checks off). `error` carries the exception in `error`: the file could not be read, which
`validate` will explain better.

Every finding carries the two labels the Oopsie Data team uses in its feedback to labs:

- `confidence`: `high_confidence` means a flagged episode almost certainly has the issue;
  `lower_confidence` means the check is a heuristic that can flag fine episodes.
- `suggested_action`: `fix_and_resubmit` (correct the data, then re-record or reconvert),
  `confirm_and_discuss` (check whether the issue is real and tell the team), or
  `informational` (not acted on by the team). An episode whose only findings are
  `informational` counts as `clean`.

When summarizing a report for a user, start from `flagged_episodes_per_check`, high-confidence
`fix_and_resubmit` checks first. A check that flags every episode points at the recording setup,
the converter or the robot profile, not at single episodes.

## The checks

Each check only runs for the action spaces it belongs to (`general` runs for all). Thresholds are
the defaults. *On* is whether the check runs by default (the same set the team runs); *Conf.*
and *Action* abbreviate the `confidence` and `suggested_action` labels.

| Check | On | Conf. | Action | Flags when |
|---|---|---|---|---|
| `general.duplicate_pairs` | yes | high | fix | two declared action arrays of the same shape hold the same values (copy-paste bug) |
| `general.length_mismatch` | yes | high | fix | an action and its matching robot state have different lengths |
| `general.action_chunk_shape` | yes | high | fix | an arm action has a third axis: the policy's whole action chunk was recorded at every step |
| `general.constant_action` | yes | high | confirm | an arm action never changes over the episode |
| `general.command_rate` | yes | lower | confirm | each new arm action is held for the same N ≥ 2 steps (in ≥ 80% of ≥ 10 holds): commands arrive at `control_freq / N` |
| `general.required_state_keys` | yes | high | fix | a robot state the action space needs is missing (`gripper_position` always, plus the arm's position state) |
| `general.placeholder_metadata` | yes | high | fix | `operator_name`, `policy_name`, `robot_name` or `gripper_name` is blank, a placeholder (`unknown`, `test`, `pending_operator`, ...), a template (`your_...`), or an operator that is not a person (`..._policy`, `autoeval_...`) |
| `general.action_copies_state` | yes | lower | confirm | ≥ 90% of joint/Cartesian position actions equal the same or the next state: actions were rebuilt from states, not logged commands |
| `general.scale_consistency` | off | lower | info | `gripper_position` action is ≥ 5× or ≤ 0.2× the state's scale |
| `general.tracking_residual` | off | lower | info | `gripper_position` action misses the next state by more than half its own size |
| `general.delta_detection` | off | lower | info | `gripper_position` actions look like step-to-step changes |
| `general.gripper_binary_present` | yes | high | fix | no non-empty `gripper_binary` action is recorded |
| `general.gripper_range` | yes | high | fix | `gripper_binary` leaves [0, 1] by more than 0.05 (e.g. −1/+1) |
| `general.gripper_command_range` | yes | lower | confirm | `gripper_position` commands exceed the gripper state's range by more than 25% of its scale |
| `joint_position.delta_detection` | yes | high | fix | actions look like `state[t+1] - state[t]` rather than absolute positions |
| `joint_position.tracking` | yes | lower | confirm | median per-joint error between `action[t]` and `state[t+1]` above 0.2 rad |
| `joint_position.joint_names` | yes | lower | confirm | action/state joint names missing, mismatched, or out of base-to-wrist order |
| `joint_position.dof` | yes | high | fix | number of joint values ≠ number of joint names |
| `joint_position.radians` | yes | high | fix | a joint value beyond 6.5: degrees or a −100..100 range |
| `joint_position.step_jump` | yes | lower | confirm | a joint moves faster than 20 rad/s between two steps |
| `joint_position.gripper_in_joints` | yes | high | fix | a joint name contains "gripper" |
| `joint_position.normalized_range` | yes | lower | confirm | joint actions peak at exactly ±1 |
| `joint_velocity.dof` | yes | high | fix | number of joint velocities ≠ number of joint names |
| `joint_velocity.tracking` | yes | lower | confirm | velocity × step time vs. actual joint motion: ratio outside 0.25–8 or correlation below 0.3 (`suspected` names deg/s or per-step deltas) |
| `joint_velocity.looks_like_position` | yes | high | fix | the "velocity" sits on the joint position state |
| `joint_velocity.normalized_range` | yes | lower | confirm | velocities peak at exactly ±1 |
| `joint_velocity.speed_limit` | yes | lower | confirm | a joint commanded faster than 3.14 rad/s |
| `joint_velocity.smoothness` | yes | lower | info | > 30% of steps reverse direction |
| `cartesian_position.biarm_ordering` | yes | lower | confirm | swapping left/right arms halves the tracking error |
| `cartesian_position.delta_detection` | yes | high | fix | xyz actions look like deltas |
| `cartesian_position.pose_tracking` | yes | lower | confirm | median error between action and next state above 0.10 m or 20° |
| `cartesian_position.position_units` | yes | high | fix | xyz beyond 3 m or ≫ the state's: mm/cm or a far-away frame (`action_suspected_unit` / `state_suspected_unit`) |
| `cartesian_position.quaternion_shape` | yes | high | fix | pose is not `(x, y, z, qx, qy, qz, qw)` per arm, or a quaternion is not unit length |
| `cartesian_position.quaternion_sign_flip` | off | lower | info | consecutive quaternions with negative dot product (q → −q) |
| `cartesian_position.rotation_jump` | yes | lower | confirm | orientation turns faster than 360°/s between steps |
| `cartesian_position.normalized_range` | yes | lower | confirm | xyz actions peak at exactly ±1 |
| `cartesian_position.quaternion_scalar_order` | off | lower | info | heuristic: `qw` looks like it was written first |
| `cartesian_velocity.shape` | yes | high | fix | not 6 values per arm (3 linear, 3 angular) |
| `cartesian_velocity.speed_limit` | yes | lower | confirm | linear above 2 m/s or angular above 6.28 rad/s |
| `cartesian_velocity.normalized_range` | yes | lower | confirm | velocities peak at exactly ±1 |
| `cartesian_velocity.tracking` | yes | lower | confirm | as `joint_velocity.tracking`, separately for linear and angular parts |
| `cartesian_velocity.linear_angular_order` | yes | lower | confirm | swapping the halves matches the real motion clearly better |
| `cartesian_velocity.looks_like_position` | yes | high | fix | the linear part sits on the state's xyz |
| `cartesian_velocity.smoothness` | yes | lower | info | > 30% of steps reverse direction |
| `video.decoding` | yes | high | fix | file missing, ffmpeg errors, nothing decodes, or frame count off by more than 1 |
| `video.corrupted_frames` | yes | lower | confirm | a single frame unlike both its neighbours, which match each other |
| `video.state_jump` | yes | lower | confirm | the robot state jumps in one step while no camera does: a state glitch, or state and video out of step |
| `video.video_jump` | yes | lower | confirm | a sudden image change (≥ 20× the surrounding second) while the robot state moves smoothly |
| `video.recording_gap` | yes | lower | confirm | robot state (≥ 10× the surrounding steps) and a camera (≥ 5×) jump at the same moment: time is missing from the recording |
| `video.periodic_skips` | yes | lower | confirm | the recording pauses at a regular period (typically each action-chunk inference): robot state and video jump together at each gap |
| `video.choppy` | yes | lower | confirm | ≥ 20% duplicate frames while the robot moves |
| `video.frozen` | yes | lower | confirm | ≥ 1 s of static frames while the robot moves |
| `video.resolution` | off | lower | info | short side below 360 px |
| `video.brightness` | off | lower | info | median brightness below 30 or above 225 (0–255), or ≥ 10% near-black/white frames |
| `video.fps` | yes | high | confirm | container fps differs from `control_freq` by > 5% |
| `video.duration` | yes | high | confirm | video duration differs from `T / control_freq` by more than max(0.5 s, 5%): frame *i* is no longer step *i* |
| `video.camera_duration_mismatch` | yes | high | confirm | the cameras of one episode differ in duration by > 0.5 s |

Skip reasons name the missing precondition, e.g. `declared_but_empty` (the profile declares a
key that holds no data), `no_matching_state` / `no_robot_state` (the state needed for the
comparison was not recorded), `no_control_freq`, `too_short`, `too_few_changes`,
`too_little_motion`, `non_finite_values`, `fewer_than_two_cameras`, `ignored_camera`,
`no_video_files` (no camera reference resolves). A skip is not a failure, but a check skipped on
every episode checks nothing.

## Reading findings

Most findings trace back to one recording-side cause, so fix the cause and record again rather
than patching files:

- **`placeholder_metadata`** — set the real operator name in the recorder call and the real
  policy, robot and gripper names in the robot profile. Never invent them; ask the user.
- **`radians`, `position_units`, `normalized_range`, `speed_limit`** — the values are in the
  wrong unit or were normalized by the policy. Record the robot's physical command (radians,
  metres, rad/s), not the policy's normalized output.
- **`delta_detection`, velocity `tracking` with `suspected: per_step_delta`** — a delta action
  was recorded under a position key, or a per-step change under a velocity key. Either convert
  to the declared space or declare the space that matches.
- **`action_copies_state`** — the action was filled from the observation. Record the command
  sent to the robot.
- **`action_chunk_shape`, `command_rate`** — the policy's chunk was recorded instead of the one
  action executed at each step, or each chunk entry was repeated for several steps. Record the
  action actually sent at each `record_step`, and set `control_freq` to the rate it is called at.
- **`constant_action`** — the action never changed: a stuck policy, or a converter that wrote one
  placeholder action.
- **`required_state_keys`** — the profile's `robot_state_keys` must include the state each action
  space is observed through; `robot-profile.md` lists the rule.
- **position `tracking`, `pose_tracking`** — action and state are in different frames,
  conventions or orders, or are misaligned in time. `robot-profile.md` covers frames and
  orientation.
- **`video.fps`, `video.duration`** — `control_freq` in the profile does not match the rate
  `record_step` is called at, or frames are dropped.
- **`periodic_skips`, `recording_gap`** — recording pauses, typically while the policy plans the
  next chunk. Keep calling `record_step` at `control_freq` during inference.

A flagged episode is not rejected by `upload`. If a finding is a false positive for the
user's setup, say so and note it in the episode or submission, rather than changing data to
silence it.

## Overriding thresholds

`--config FILE` takes YAML nested like `<category>: <check>: <parameter>`; anything left out
keeps its default, and an unknown name is an error. A category also takes `enabled: false`, which
switches off all its checks, and `video` takes `ignore_cameras` (camera names left out of every
video check, e.g. a tactile sensor recorded as a camera) and `moving_fraction_of_p95` (when the
robot counts as moving). The report records every override under `summary.config_overrides`.

```yaml
video:
  ignore_cameras: [tactile_left]
  resolution:
    enabled: true
    min_short_side: 240
general:
  gripper_binary_present:
    enabled: false
```

Only change a threshold when the user can say why their setup differs. Do not loosen checks
to make a report clean.
