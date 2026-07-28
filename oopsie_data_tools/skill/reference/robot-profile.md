# Robot profile

A robot profile captures hardware and policy metadata. It is serialized to JSON and stored as an
HDF5 root attribute on every episode, so it is the episode's documentation — the validator
checks the recorded data against it in both directions.

Start from `oopsie-data new-profile --name <name>`, which writes a commented skeleton into
`<profiles-dir>/`. Every required field is blank, so the profile will not load until a human
fills it in. Ask the user the questions below; where the choice is constrained to a set, list the
options and explain them if asked.

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

### `robot_state_keys`

`gripper_position` is **always mandatory**. Beyond that, the state must observe whatever space
the action controls — checked at profile load:

| If `action_space` contains | `robot_state_keys` must contain |
|---|---|
| `joint_position` or `joint_velocity` | `joint_position` |
| `cartesian_position` or `cartesian_velocity` | `cartesian_position` |

Velocity control requires *position* state: you observe where the arm is, you command how fast
it moves. The requirement is a union over the action space, so an `action_space` mixing joint
and Cartesian actions needs both keys. `base_position` is an optional addition — no base action
currently requires it.

`base_position` aside, nothing else may be recorded: an observation key the profile does not
declare fails validation, because it has no joint names, units or expected DOF and nothing
downstream can interpret it.

### `robot_state_joint_names`

**Required when — and only when — `joint_position` is in `robot_state_keys`**; a profile whose
robot is controlled purely in Cartesian space omits it. When present it must have exactly as
many entries as the last axis of the recorded `joint_position` array. The emptiness is caught at
profile load; the length mismatch at validation.

### `action_space`

Not a free choice. The rule (`is_valid_action_space`):

- **at least one** arm action from `joint_position`, `joint_velocity`, `cartesian_position`,
  `cartesian_velocity`
- **at least one** gripper action from `gripper_position`, `gripper_velocity`, `gripper_binary`
- **at most one** base action from `base_velocity`, `base_position`
- no other keys

Declaring two arm actions or two gripper actions is legal — but every declared key must then be
recorded as a real array in every episode. A profile with `uses_mobile_base: true` must include a
base action. (The converse is not checked: a base action with `uses_mobile_base: false` passes.)

The dict passed to `record_step` must have keys equal to `action_space` **exactly** — not a
subset, not a superset.

## Conditionally required

| Question | YAML key | Rule |
|---|---|---|
| Joint names for the robot state, in order? | `robot_state_joint_names` | **Required** whenever `joint_position` is in `robot_state_keys` — see above. Not one of the nine unconditional keys, so a purely Cartesian profile omits it entirely. |
| Joint names for arm actions? | `action_joint_names` | **Required** whenever `joint_position` or `joint_velocity` is in `action_space`. Same order as the action vector; its length is checked against the recorded DOF. |
| Orientation representation for cartesian actions? | `orientation_representation` | Needed whenever `cartesian_position` is in `action_space` and the policy does not already emit scalar-last quaternions. |
| Orientation representation for cartesian state? | `robot_state_orientation_representation` | Same, for `cartesian_position` in `robot_state_keys`. |

### Orientation representation values

`quat` (scalar-last, shape `(4,)`), `matrix` (`(3, 3)`), `rot6d` (first two columns of the
rotation matrix, flattened, `(6,)` — what openpi uses), `rotvec` (axis-angle, `(3,)`), or
`euler_<order>` where order is one of `xyz`, `zyx`, `xyx`, `XYZ`, `ZYX`, `XYX`. **Case is
meaningful**: lowercase orders are extrinsic rotations, uppercase intrinsic.

These strings are **not validated when the YAML is parsed**. A bad value is stored verbatim and
surfaces as a plain `ValueError` from `EpisodeRecorder.__init__` — at construction, not
mid-rollout, but not as a profile error either.

Conversion applies to `cartesian_position` only. `cartesian_velocity` is recorded exactly as
given, with no conversion and no shape check.

## Optional

Stored for reproducibility, never validated:

- `controller` — e.g. `OSC`, `joint_position`, `joint_velocity`
- `gains` — controller gains; see the skeleton for the expected nesting
- `intrinsic_calibration_matrix` / `extrinsic_calibration_matrix`, keyed by camera name. Both
  the underscored and the spaced spelling (`intrinsic calibration matrix`) are read; the
  underscored one is canonical.
