# Robot profile

A robot profile captures hardware and policy metadata. It is serialized to JSON and stored as an
HDF5 root attribute on every episode, so it is the episode's documentation — the validator checks
the recorded data against it in both directions.

Start from `oopsie-data new-profile --name <name>`, which writes a commented skeleton into
`<profiles-dir>/`. Every required field is blank, so the profile will not load until a human
fills it in.

Ask the questions below **step by step**, explaining the constrained choices as they come up. A
full list up front easily overwhelms.

## Required fields

All nine must be present or `load_robot_profile` raises `Robot profile missing keys: [...]`.

| Question | YAML key | Example / rule |
|---|---|---|
| What is the policy name? | `policy_name` | `pi0.5`, `act_plus_plus` |
| What is the robot name? | `robot_name` | `franka_droid`, `aloha` |
| What is the gripper name? | `gripper_name` | `robotiq_2f_85`, `aloha_gripper` |
| Is this a bimanual setup? | `is_biarm` | `true` / `false` |
| Does the robot use a mobile base? | `uses_mobile_base` | `true` / `false` |
| Control frequency (Hz)? | `control_freq` | `10`, `50` — must be > 0 |
| Camera names? | `camera_names` | `[left, right, wrist]` |
| Which robot state keys are recorded? | `robot_state_keys` | see below |
| What does the policy output? | `action_space` | see below |

### `action_space`

Not a free choice (`is_valid_action_space`):

- **at least one** arm action from `joint_position`, `joint_velocity`, `cartesian_position`,
  `cartesian_velocity`
- **at least one** gripper action from `gripper_position`, `gripper_velocity`, `gripper_binary`
- **at most one** base action from `base_velocity`, `base_position`
- no other keys

Declaring two arm or two gripper actions is legal — but every declared key must then be recorded
as a real array in every episode. `uses_mobile_base: true` requires a base action. The dict
passed to `record_step` must have keys equal to `action_space` **exactly**.

### `robot_state_keys`

`gripper_position` is **always mandatory**. Beyond that, the state must observe whatever space
the action controls — checked at profile load, as a union, so an `action_space` mixing joint and
Cartesian actions needs both keys:

| If `action_space` contains | `robot_state_keys` must contain |
|---|---|
| `joint_position` or `joint_velocity` | `joint_position` |
| `cartesian_position` or `cartesian_velocity` | `cartesian_position` |

Velocity control requires *position* state: you observe where the arm is, you command how fast it
moves. `base_position` is an optional addition. Nothing else may be recorded — an observation key
the profile does not declare fails validation, because it has no joint names, units or expected
DOF and nothing downstream can interpret it.

## Conditionally required

| Question | YAML key | Rule |
|---|---|---|
| Joint names for the robot state, in order? | `robot_state_joint_names` | Required whenever `joint_position` is in `robot_state_keys`; a purely Cartesian profile omits it entirely. Length is checked against the recorded DOF. |
| Joint names for arm actions? | `action_joint_names` | Required whenever `joint_position` or `joint_velocity` is in `action_space`. Same order as the action vector; length checked against the recorded DOF. |
| Orientation representation for cartesian actions? | `orientation_representation` | Needed whenever `cartesian_position` is in `action_space` and the policy does not already emit scalar-last quaternions. |
| Orientation representation for cartesian state? | `robot_state_orientation_representation` | Same, for `cartesian_position` in `robot_state_keys`. |

### Orientation representation values

`quat` (scalar-last, `(4,)`), `matrix` (`(3, 3)`), `rot6d` (first two columns of the rotation
matrix, flattened, `(6,)` — what openpi uses), `rotvec` (axis-angle, `(3,)`), or `euler_<order>`
where order is one of `xyz`, `zyx`, `xyx`, `XYZ`, `ZYX`, `XYX`. **Case is meaningful**: lowercase
orders are extrinsic, uppercase intrinsic.

These strings are **not validated when the YAML is parsed**. A bad value surfaces as a plain
`ValueError` from `EpisodeRecorder.__init__` — at construction, not mid-rollout, but not as a
profile error either.

Conversion applies to `cartesian_position` only. `cartesian_velocity` is recorded exactly as
given, with no conversion and no shape check.

## Optional

Stored for reproducibility, never validated — ask explicitly whether the user wants to provide
them: `controller` (e.g. `OSC`, `joint_position`, `joint_velocity`), `gains` (see the skeleton for
the expected nesting), and `intrinsic_calibration_matrix` / `extrinsic_calibration_matrix`, keyed
by camera name — the spaced spelling (`intrinsic calibration matrix`) is also read, but the
underscored one is canonical.

## Additional data

Optional sensor streams beyond robot state and cameras — force/torque, IMU, tactile. Ask whether
the setup records any. Each entry is keyed by the name the data is stored under:

```yaml
additional_data:
  wrist_ft:
    sensor: ATI Mini45                # required: the device recording the data
    sensor_info:                      # optional: a string or a mapping, free-form
      units: N, Nm
      frame: sensor
  tactile_left:
    sensor: GelSight Mini
    format: video                     # "array" (default) or "video"
```

- Keys may contain letters, digits, `_` and `-`, and must not repeat a camera name or each other, ignoring case.
- `sensor` must be non-empty; any other field than `sensor`, `sensor_info` and `format` is
  rejected at profile load, so a typo does not silently drop metadata.
- `format: array` stores the per-step values as a `(T, ...)` dataset with its numeric dtype
  kept. The sum over all array keys is capped at 100 MiB per episode (uncompressed).
- `format: video` takes one `(H, W, 3)` uint8 frame per step and stores it as an MP4, like a
  camera. Use it for anything image-like; raw images as arrays hit the cap within minutes.
  Frames must be at least 16×16; a coarse taxel grid (e.g. 4×4) belongs under `format: array`.
- Sensor readings must go through `additional_data`. Extra keys in the observation's
  `robot_state` or `image_observation` are ignored, not recorded.

Once declared, every key must be passed on every step:
`record_step(observation, action, additional_data={"wrist_ft": ft, "tactile_left": frame})`,
with the same shape each step. Passing `additional_data` to a profile that declares none is an
error.
