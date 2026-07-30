# Converting a dataset into `oopsiedata_format_v1`

For writing or debugging a converter that produces Oopsie HDF5 episode files.
`reference/format.md` is the same schema from the reading side and holds the full field-by-field
rules; this page covers what a converter has to get right. Where they disagree, the validator
source wins.

Steps:

1. Create a robot profile with `oopsie-data new-profile` (writes into `./robot_profiles/` —
   leave it there, a bare `./configs/` is not on the lookup chain). Confirm with the user that
   every field is real and nothing was guessed.
2. Write the conversion script, reusing `oopsie_data_tools.utils.conversion_utils` rather than
   hand-rolling HDF5 writes. Read the relevant reference first.
3. Let the user run it against the real input path.

Worked examples: `examples/conversion_script_examples/` in the repo.

## What a converter must produce

```
root attrs      schema, episode_id, language_instruction, lab_id, operator_name, robot_profile
observations/robot_states/<key>   (T, D) float64, keys equal to profile.robot_state_keys exactly
observations/video_paths/<cam>    string, path relative to the .h5 file
actions/<key>                     (T, D) float64 for keys in action_space;
                                  h5py.Empty(dtype=np.float64) for every other canonical key
episode_annotations/<annotator>/  annotation fields as attrs on the subgroup
```

Canonical action keys: `cartesian_position`, `cartesian_velocity`, `joint_position`,
`joint_velocity`, `base_position`, `base_velocity`, `gripper_velocity`, `gripper_position`,
`gripper_binary`.

`gripper_position` is the only unconditionally required state key. `joint_position` is required
when `action_space` holds `joint_position`/`joint_velocity`, `cartesian_position` when it holds
`cartesian_position`/`cartesian_velocity` — the state must observe the space the action controls,
as a union.

For annotations, `outcome` is one of `success`, `success_suboptimal`, `success_side_effect`,
`failure`, and must agree in sign with `success` (`failure` iff `success < 0.5`).
`failure_category` and `severity` are optional, but provided values must be recognized slugs —
see `reference/format.md` for the vocabularies. A converter that emits no `episode_annotations`
produces structurally valid files that still fail `validate` with `Annotations dict is empty,
must be provided for upload`: either carry the source labels across, or plan to run
`oopsie-data annotate` afterwards.

## The robot profile

`reference/robot-profile.md` has the full field reference and the questions to ask the user.
What a converter needs inline — the nine keys `REQUIRED_KEYS` enforces:

| Field | Type | Example |
|---|---|---|
| `policy_name` | str | `"pi0_droid"` |
| `robot_name` | str | `"franka_research_3"` |
| `gripper_name` | str | `"robotiq_2f_85"` |
| `is_biarm` | bool | `false` |
| `uses_mobile_base` | bool | `false` |
| `control_freq` | int | `10` |
| `camera_names` | list[str] | `["left", "right", "wrist"]` |
| `robot_state_keys` | list[str] | `gripper_position` plus whatever `action_space` implies |
| `action_space` | list[str] | ≥1 arm key + ≥1 gripper key, ≤1 base key, nothing else |

Arm keys are `joint_position`, `joint_velocity`, `cartesian_position`, `cartesian_velocity`;
gripper keys `gripper_position`, `gripper_velocity`, `gripper_binary`; base keys `base_velocity`,
`base_position`. Declaring two arm or two gripper keys is legal, but every declared key must then
be written as a real (non-Empty) array.

Conditionally required: `robot_state_joint_names` when `joint_position` is in
`robot_state_keys`, and `action_joint_names` when `action_space` holds
`joint_position`/`joint_velocity`. Both are one entry per DOF, in the array's own order, and
their length is checked against the recorded arrays.

Optional but worth setting: `robot_state_orientation_representation` and
`orientation_representation` (state and action `cartesian_position` respectively) and
`controller`. Note the next section — for a converter these describe what you *wrote*, which is
not necessarily what the source dataset held.

## The two things converters get wrong

**Orientation is not converted for you.** `orientation_representation` is applied by
`EpisodeRecorder.record_step`; writing HDF5 directly bypasses it entirely. `cartesian_position`
must be written as `[x, y, z, qx, qy, qz, qw]` per arm — shape `(T, 7)`, or `(T, 14)` when
`is_biarm`. Declaring `euler_xyz` and then writing euler angles is rejected on width (6 ≠ 7), or
worse, can mislabel data if the widths line up: component order cannot be inferred from values
alone. Every written quaternion is norm-checked. Use
`to_quaternion_poses(poses, "euler_xyz", is_biarm=...)` for a whole trajectory at once, and
declare the *result* (`"quat"`) in the profile — the profile describes what is on disk.

**Video and duration limits are checked downstream, so the fix is upstream.** Each side must be
180–1280 px (`resize_frames` handles the ceiling), episode duration
(`trajectory_length / control_freq`) must be 1–600 s, all arrays must share the same leading `T`,
all robot arrays must contain finite real numbers, and frame counts must land within
`max(5, 0.1 * T)` of `T`. Video paths must remain inside the submitted directory after resolving
`..` and symlinks.

Check the result with `oopsie-data validate --path <dir>`, or `oopsie-data inspect <file.h5>`
for a structure dump that works even on files `validate` rejects.

## Write pattern

Prefer these helpers over hand-rolling: they are written against the definitions the validator
uses, so they cannot drift from it.

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

    # Enforces key equality with profile.robot_state_keys in both directions.
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

    # Must go into episode_annotations/<annotator_name>/ — attrs on the parent group are
    # invisible to the loader, and the episode then fails as unannotated.
    write_episode_annotations(
        f,
        annotator_name="my_annotator",
        success=0.0,
        # everything below is optional
        episode_description="Robot grasped the cup but dropped it in transit.",
        failure_category=["grasp"],
        severity="medium",
    )
```

`outcome` defaults to the coarse reading of `success`, so it can only come out as `success` or
`failure`. Pass `outcome="success_side_effect"` (or `"success_suboptimal"`) explicitly to record
a qualified success.
