# oopsiedata_format_v1 — Full Schema Reference

Use this skill whenever writing or debugging a converter that produces Oopsie HDF5 episode files. It describes the canonical `oopsiedata_format_v1` schema as enforced by `oopsie_data_tools`.

`reference/format.md` is the same schema from the reading side; this page is the writing side. Where they disagree, the validator source wins — check it rather than guessing.

You follow these steps:

1. Create a new robot profile with `oopsie-data new-profile`, which writes into `./robot_profiles/` by default. Leave it where the command put it — profiles are looked up at `$OOPSIE_ROBOT_PROFILES_DIR`, then `./robot_profiles` or `./configs/robot_profiles`, so a bare `./configs/` is not found. Double check with the user that this robot profile is correct and no unverified information has been entered.
2. Create the conversion script. Follow DRY and make sure to reuse utilities from `oopsie_data_tools.utils.conversion_utils`. Make sure to read all relevant information before attempting to write the conversion script.
3. Wait for the user to run the conversion script with the full correct input path.

## Validation entry points (`oopsie_data_tools`)

```
oopsie_data_tools/utils/validation/episode_loader.py    # HDF5 → EpisodeData
oopsie_data_tools/utils/validation/episode_validator.py # semantic checks
oopsie_data_tools/utils/validation/validation_utils.py  # public API
oopsie_data_tools/utils/robot_profile/robot_profile.py  # RobotProfile
```

Run validation:
```bash
oopsie-data validate --path /path/to/directory
oopsie-data inspect  /path/to/episode.h5    # structure dump, works on files validate rejects
```

---

## HDF5 File Structure

### Root attributes (all required)

| Attribute | Type | Notes |
|---|---|---|
| `schema` | str | Must be `"oopsiedata_format_v1"` |
| `episode_id` | str | Non-empty unique identifier |
| `language_instruction` | str | Non-empty task description |
| `lab_id` | str | Non-empty; not `"your_lab_id"` |
| `operator_name` | str | Non-empty |
| `robot_profile` | str | JSON-serialized `RobotProfile` (use `robot_profile_to_json`) |

### `/observations/` group

**`/observations/video_paths/`** — one string dataset per camera:
- Key = camera name (must match `profile.camera_names`)
- Value = relative path to MP4 from the HDF5 file's directory

**`/observations/robot_states/`** — one dataset per robot state key:
- Keys must equal `profile.robot_state_keys` exactly — every declared key present, and nothing
  undeclared (an extra key is rejected, not ignored)
- Shape: `(T, D)` float64
- `joint_position`: `(T, n_joints)` — names from `robot_state_joint_names`, and the count must match
- `gripper_position`: `(T, 1)` or `(T, n_fingers)` — DOF is not checked
- `cartesian_position`: **`(T, 7)`**, or `(T, 14)` when `is_biarm` — `[x, y, z, qx, qy, qz, qw]`
  per arm, scalar-last. Not euler: convert before writing, or the episode is rejected.

`gripper_position` is the only unconditionally required state key. `joint_position` is required
only when `action_space` holds `joint_position`/`joint_velocity`, and `cartesian_position` only
when it holds `cartesian_position`/`cartesian_velocity` — the state must observe the space the
action controls. Mixing joint and Cartesian actions requires both.

### `/actions/` group

- Keys in `profile.action_space`: stored as `(T, D)` float64 arrays (non-empty)
- All other canonical action keys: stored as `h5py.Empty(dtype=np.float64)`

Canonical action keys (write Empty for any not in action_space):
```
cartesian_position, cartesian_velocity,
joint_position, joint_velocity,
base_position, base_velocity,
gripper_velocity, gripper_position, gripper_binary
```

### `/episode_annotations/` group (required by both `validate` and `upload`)

Structure: `episode_annotations/{annotator_name}/` — a group with HDF5 attributes:

| Attr | Type | Notes |
|---|---|---|
| `success` | float | Required; must be in [0.0, 1.0] |
| `episode_description` | str | Optional free text |
| `taxonomy` | str | JSON: `{"outcome": "...", "failure_category": [...], "severity": "..."}` |

`outcome` is one of four slugs — `success`, `success_suboptimal`, `success_side_effect`,
`failure` — and is the only taxonomy field that matters to validation. It must agree in sign
with `success` (`failure` iff `success < 0.5`); all three `success_*` outcomes write
`success = 1.0`, so a consumer that only reads the float sees them alike.

`failure_category` and `severity` are stored as stable slugs, not prose:

| Field | Allowed values |
|---|---|
| `failure_category` | `reaching`, `grasp`, `manipulation`, `sequencing_semantic`, `collision`, `hardware`, `not_attempted`, `other` |
| `severity` | `low`, `medium`, `catastrophic` |

Every field except `outcome` is optional — a partial annotation is valid, and so is a failure
with no taxonomy at all.

Files written by an older release carry `schema = oopsie_failure_taxonomy_v1`,
`failure_description` instead of `episode_description`, prose values instead of slugs, and no
`outcome`. Readers upcast those on the fly and never rewrite them, so both versions can sit in
one dataset. `oopsie_data_tools/utils/migrate_taxonomy_v2.py` converts them in place if you
want them normalized.

A converter that emits no `episode_annotations` produces structurally valid files that still fail `oopsie-data validate` with `Annotations dict is empty, must be provided for upload`. Either carry the source dataset's labels across, or plan to run `oopsie-data annotate` afterwards.

---

## RobotProfile fields

Defined in `oopsie_data_tools/utils/robot_profile/robot_profile.py`. Serialized as JSON into `f.attrs["robot_profile"]`.

**Required fields** — these nine, and no others, are what `REQUIRED_KEYS` enforces:

| Field | Type | Example |
|---|---|---|
| `policy_name` | str | `"pi0_droid"` |
| `robot_name` | str | `"franka_research_3"` |
| `gripper_name` | str | `"robotiq_2f_85"` |
| `is_biarm` | bool | `false` |
| `uses_mobile_base` | bool | `false` |
| `control_freq` | int | `10` |
| `camera_names` | list[str] | `["left", "right", "wrist"]` |
| `robot_state_keys` | list[str] | Must include `"gripper_position"`, plus whatever `action_space` implies (see above) |
| `action_space` | list[str] | ≥1 arm key + ≥1 gripper key (see below) |

**`action_space` validity rules** (`is_valid_action_space`):
- **at least one** arm key from: `{joint_position, joint_velocity, cartesian_position, cartesian_velocity}`
- **at least one** gripper key from: `{gripper_position, gripper_velocity, gripper_binary}`
- **at most one** base key from: `{base_velocity, base_position}`
- No other keys allowed

Declaring two arm keys or two gripper keys is legal — but every declared key must then be written as a real (non-Empty) array. `uses_mobile_base: true` requires a base key; the converse is not checked.

**Optional fields:**

| Field | Notes |
|---|---|
| `robot_state_joint_names` | Required if `joint_position` is in `robot_state_keys` (enforced at profile load). One entry per joint DOF; the count is checked against the recorded array |
| `action_joint_names` | Required if `action_space` includes `joint_position` or `joint_velocity` |
| `robot_state_orientation_representation` | Declares how `cartesian_position` state encodes orientation. Options: `"quat"`, `"matrix"`, `"rot6d"`, `"rotvec"`, `"euler_<order>"` where order ∈ `xyz, zyx, xyx, XYZ, ZYX, XYX` (**case matters**: lowercase extrinsic, uppercase intrinsic) |
| `orientation_representation` | Same, for `cartesian_position` **actions** only — `cartesian_velocity` is never converted |
| `controller` | str — `"OSC"`, `"joint_position"`, `"joint_velocity"` |

---

## Validation constraints

| Check | Limit |
|---|---|
| Video min dimension | 180 px |
| Video max dimension | 1280 px |
| Episode duration (trajectory_length / control_freq) | 1 – 600 seconds |
| Frame count vs trajectory_length tolerance | max(5, 10% of T) |
| Video duration vs expected duration | ≤ 0.5 s |
| All obs/action arrays same length T | strict |
| Frame counts across cameras | within 1 of each other |

**Orientation conversion does not happen here.** `orientation_representation` is applied by
`EpisodeRecorder.record_step`, not by the validator or the loader. A converter writing HDF5
directly bypasses that entirely, so it must write `cartesian_position` as scalar-last
quaternions itself. Declaring `orientation_representation: euler_xyz` and then writing euler
angles produces a file that is rejected on width (6 ≠ 7) — or, worse, silently mislabels data
if the widths happen to line up.

`to_quaternion_poses(poses, "euler_xyz", is_biarm=...)` does the conversion `record_step`
would have done, for a whole `(T, D)` trajectory at once. Declare the *result* in the profile
(`"quat"`), not the source format — the profile describes what is on disk. Frames that exceed
the 1280 px limit go through `resize_frames` for the same reason: the check is downstream, so
the fix has to be upstream.

Worked examples of all of this live in `examples/conversion_script_examples/`.

---

## Minimal write pattern (Python)

Prefer the helpers in `oopsie_data_tools.utils.conversion_utils` over hand-rolling any of this —
they are written against the definitions the validator uses, so they cannot drift from it.

```python
import h5py, numpy as np
from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile
from oopsie_data_tools.utils.conversion_utils import (
    write_root_attrs, write_video_paths, write_robot_states, write_actions,
    write_episode_annotations, to_quaternion_poses, resize_frames,
)

profile = RobotProfile(
    policy_name="my_policy",
    robot_name="franka_research_3",
    gripper_name="robotiq_2f_85",
    is_biarm=False,
    uses_mobile_base=False,
    control_freq=10,
    camera_names=["left", "right", "wrist"],
    robot_state_keys=["joint_position", "gripper_position"],
    robot_state_joint_names=[f"joint_{i}" for i in range(1, 8)],
    action_space=["joint_velocity", "gripper_position"],
    action_joint_names=[f"joint_{i}" for i in range(1, 8)],
)

h5_path = "episode.h5"

with h5py.File(h5_path, "w") as f:
    write_root_attrs(
        f,
        episode_id="episode_000001",
        language_instruction="Pick up the cup",
        lab_id="UT Austin",          # the real one from registration — never invent it
        operator_name="Alice",
        robot_profile=profile,
    )

    # Paths are stored relative to the episode file; pass them however you have them.
    write_video_paths(
        f,
        {cam: f"videos/episode_000001_{cam}.mp4" for cam in profile.camera_names},
        h5_path,
    )

    # Keys must equal profile.robot_state_keys exactly — write_robot_states enforces that
    # in both directions rather than letting the validator find it later.
    write_robot_states(
        f,
        {
            "joint_position": joint_pos_array,      # (T, 7)
            "gripper_position": gripper_pos_array,  # (T, 1)
        },
        profile.robot_state_keys,
    )

    # Fills in h5py.Empty for every canonical key outside action_space.
    write_actions(
        f,
        {"joint_velocity": joint_vel_array, "gripper_position": gripper_act_array},
        profile.action_space,
    )

    # Goes into episode_annotations/<annotator_name>/ — attrs on the parent group are
    # invisible to the loader, and the episode then fails as unannotated.
    write_episode_annotations(f, annotator_name="my_annotator", success=1.0)
```

For a failure, add whatever you know — none of these are required:

```python
    write_episode_annotations(
        f,
        annotator_name="my_annotator",
        success=0.0,
        episode_description="Robot grasped the cup but dropped it in transit.",
        failure_category=["grasp"],
        severity="medium",
    )
```

`outcome` defaults to the coarse reading of `success`, which can only ever be `success` or
`failure`. Pass it explicitly to record one of the two qualified successes:

```python
    write_episode_annotations(
        f,
        annotator_name="my_annotator",
        success=1.0,
        outcome="success_side_effect",
        episode_description="Completed the task but knocked over a nearby cup.",
        failure_category=["collision"],
        severity="low",
    )
```

## For reference
Example conversion scripts can be found at https://github.com/oopsie-data/oopsie-data-tools/tree/main/examples/conversion_script_examples